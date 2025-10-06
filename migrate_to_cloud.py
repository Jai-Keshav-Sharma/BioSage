from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import os
import sys
import time
import json
import signal
from typing import Optional, List, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from threading import Thread
from datetime import datetime

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, use system env vars

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **kw: x  # fallback


# Config (env fallbacks)
SOURCE_URL = os.environ.get("SOURCE_QDRANT_URL")
CLOUD_URL = os.environ.get("CLOUD_URL")
CLOUD_API_KEY = os.environ.get("CLOUD_API_KEY")
COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION", "space-biology-papers")

# Tunables (optimized for speed, reduce if you hit errors)
SCROLL_BATCH = int(os.environ.get("SCROLL_BATCH", "256"))  # Reduced for faster startup
UPSERT_SUB_BATCH = int(os.environ.get("UPSERT_SUB_BATCH", "64"))  # Reduced for better progress
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "8"))
PARALLEL_UPLOADS = int(os.environ.get("PARALLEL_UPLOADS", "4"))  # Reduced to 4 for stability
USE_GRPC = os.environ.get("USE_GRPC", "true").lower() in ("1", "true", "yes")  # gRPC for better throughput
PREFETCH_BATCHES = int(os.environ.get("PREFETCH_BATCHES", "1"))  # Reduced prefetch
CHECKPOINT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)) if __file__ else os.getcwd(), f".migrate_{COLLECTION_NAME}.checkpoint.json")

# Finalization / polling
WAIT_FOR_UPSERTS = os.environ.get("MIGRATE_WAIT_FOR_UPSERTS", "false").lower() in ("1", "true", "yes")
FINAL_WAIT_SECS = int(os.environ.get("MIGRATE_FINAL_WAIT_SECS", "30"))  # Reduced
POLL_INTERVAL = float(os.environ.get("MIGRATE_POLL_INTERVAL", "5.0"))  # Increased
CHECKPOINT_EVERY = int(os.environ.get("CHECKPOINT_EVERY", "5"))  # More frequent for visibility


_shutdown_requested = False


def _signal_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    print("\n⚠️  Shutdown requested. Will finish current work and save checkpoint...")


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def load_checkpoint() -> dict:
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_checkpoint(data: dict):
    tmp = CHECKPOINT_FILE + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, CHECKPOINT_FILE)
    except Exception as e:
        print(f"❌ Failed to write checkpoint: {e}")


def infer_vector_size(client: QdrantClient, collection: str) -> int:
    resp = client.scroll(collection_name=collection, limit=1, with_vectors=True, with_payload=False)
    
    # Handle tuple response (gRPC) or object response (HTTP)
    if isinstance(resp, tuple):
        points, _ = resp
    else:
        points = getattr(resp, "points", resp)
    
    if not points:
        raise RuntimeError("No points in source collection to infer vector size.")
    p = points[0]
    
    # Try different ways to access the vector
    vec = None
    if hasattr(p, "vector"):
        vec = p.vector
    elif isinstance(p, dict) and "vector" in p:
        vec = p["vector"]
    
    # Vector might be a named vector dict
    if isinstance(vec, dict) and len(vec) > 0:
        # Get first named vector
        vec = next(iter(vec.values()))
    
    if not vec or not isinstance(vec, (list, tuple)):
        raise RuntimeError(f"Sample point has no valid vector. Got: {type(vec)}, value: {vec}")
    return len(vec)


def ensure_cloud_collection(cloud: QdrantClient, collection: str, vector_size: int, distance: Distance = Distance.COSINE):
    try:
        cloud.get_collection(collection_name=collection)
        print(f"✅ Cloud collection '{collection}' exists.")
    except Exception:
        print(f"➕ Creating cloud collection '{collection}' (size={vector_size})...")
        cloud.create_collection(collection_name=collection, vectors_config=VectorParams(size=vector_size, distance=distance))
        time.sleep(1)


def chunk_list(lst, n):
    """Split list into chunks of size n."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def convert_points_fast(points) -> List[PointStruct]:
    """Fast point conversion with minimal overhead."""
    result = []
    for p in points:
        # Fast attribute access
        if hasattr(p, 'id'):
            pid, vec, payload = p.id, p.vector, p.payload
        else:
            pid = p.get("id")
            vec = p.get("vector")
            payload = p.get("payload")
        
        # Handle named vectors (dict format)
        if isinstance(vec, dict) and len(vec) > 0:
            vec = next(iter(vec.values()))
        
        if vec is not None and isinstance(vec, (list, tuple)):
            result.append(PointStruct(id=pid, vector=vec, payload=payload))
    return result


def upload_batch_with_retry(cloud: QdrantClient, collection: str, points: List[PointStruct], 
                            wait: bool, max_retries: int, batch_idx: int) -> int:
    """Upload a single batch with retry logic. Returns number of points uploaded."""
    attempt = 0
    while True:
        attempt += 1
        try:
            cloud.upsert(collection_name=collection, points=points, wait=wait)
            return len(points)
        except Exception as e:
            if attempt >= max_retries:
                raise Exception(f"Batch {batch_idx} failed after {max_retries} attempts: {e}")
            backoff = min(60, 2 ** attempt)
            time.sleep(backoff)


def prefetch_scroll(local: QdrantClient, collection: str, batch_size: int, 
                    scroll_id: Optional[str], queue: Queue, prefetch_count: int):
    """Prefetch scroll batches in background thread."""
    try:
        current_scroll_id = scroll_id
        for _ in range(prefetch_count):
            if current_scroll_id == "DONE":
                break
            
            resp = local.scroll(
                collection_name=collection,
                limit=batch_size,
                offset=current_scroll_id,
                with_payload=True,
                with_vectors=True,
            )
            
            # Handle tuple response (gRPC) or object response (HTTP)
            if isinstance(resp, tuple):
                points, next_scroll_id = resp
            else:
                points = getattr(resp, "points", resp)
                next_scroll_id = getattr(resp, "next_page_offset", None)
            
            if not points:
                queue.put(("DONE", None, None))
                break
            
            queue.put(("DATA", points, next_scroll_id))
            current_scroll_id = next_scroll_id
            
            if next_scroll_id is None or len(points) < batch_size:
                queue.put(("DONE", None, None))
                break
    except Exception as e:
        queue.put(("ERROR", str(e), None))


def poll_destination_count(cloud: QdrantClient, collection: str, expected: Optional[int], timeout: int = FINAL_WAIT_SECS) -> Optional[int]:
    """Poll cloud count until it reaches expected or timeout expires. Returns last observed count or None."""
    waited = 0.0
    last = None
    while waited < timeout:
        try:
            resp = cloud.count(collection_name=collection)
            # resp may be object or dict
            count = getattr(resp, "count", None) or (resp.get("count") if isinstance(resp, dict) else None)
            if count is not None:
                last = int(count)
                if expected is not None and last >= expected:
                    return last
            else:
                # attempt get_collection fallback
                info = cloud.get_collection(collection_name=collection)
                last = getattr(info, "points_count", None) or getattr(info, "result", {}).get("points_count", None)
                if last is not None and expected is not None and int(last) >= expected:
                    return int(last)
        except Exception:
            pass
        time.sleep(POLL_INTERVAL)
        waited += POLL_INTERVAL
    return last


def migrate(source_url: str, cloud_url: str, api_key: str) -> bool:
    if not cloud_url or not api_key:
        raise ValueError("CLOUD_URL and CLOUD_API_KEY are required.")

    print(f"🔌 Connecting to source: {source_url}")
    # Local: use HTTP only (Docker Qdrant usually doesn't expose gRPC)
    local = QdrantClient(
        url=source_url, 
        timeout=60,
        prefer_grpc=False,  # Force HTTP for local
        https=False
    )
    print(f"☁️  Connecting to cloud: {cloud_url}")
    # Cloud: use gRPC for better performance
    cloud = QdrantClient(
        url=cloud_url, 
        api_key=api_key, 
        timeout=120,
        prefer_grpc=USE_GRPC,  # gRPC on cloud
        https=True
    )

    # Load checkpoint
    ck = load_checkpoint()
    scroll_id = ck.get("scroll_id", None)
    migrated = ck.get("migrated", 0)
    batch_counter = 0  # Track batches for checkpoint optimization

    # Source info
    try:
        src_info = local.get_collection(collection_name=COLLECTION_NAME)
        src_count = getattr(src_info, "points_count", None) or getattr(src_info, "result", {}).get("points_count", None)
    except Exception as e:
        print(f"❌ Failed to read source collection info: {e}")
        return False

    try:
        vec_size = infer_vector_size(local, COLLECTION_NAME)
        ensure_cloud_collection(cloud, COLLECTION_NAME, vec_size)
    except Exception as e:
        print(f"❌ Setup error: {e}")
        return False

    print(f"🔁 Resuming migration: scroll_id={scroll_id}, already migrated={migrated}, total source={src_count}")
    total = migrated
    start_time = time.time()
    last_update_time = start_time
    last_update_count = migrated

    # Initialize progress bar if tqdm available
    pbar = None
    if src_count and tqdm != (lambda x, **kw: x):
        pbar = tqdm(
            total=src_count, 
            initial=migrated, 
            desc="Migrating", 
            unit="pts",
            unit_scale=True,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )

    try:
        while True:
            if _shutdown_requested:
                print("\n⚠️  Shutdown requested; saving checkpoint and exiting.")
                save_checkpoint({"scroll_id": scroll_id, "migrated": total})
                if pbar:
                    pbar.close()
                return False

            resp = local.scroll(
                collection_name=COLLECTION_NAME,
                limit=SCROLL_BATCH,
                offset=scroll_id,
                with_payload=True,
                with_vectors=True,
            )
            
            # Handle tuple response (gRPC) or object response (HTTP)
            if isinstance(resp, tuple):
                points, scroll_id = resp
            else:
                points = getattr(resp, "points", resp)
                scroll_id = getattr(resp, "next_page_offset", None)
            
            if not points:
                print("\n✅ No more points to migrate.")
                break

            # Fast point conversion
            entries = convert_points_fast(points)

            # Parallel upsert with ThreadPoolExecutor for speed
            sub_batches = list(chunk_list(entries, UPSERT_SUB_BATCH))
            
            try:
                with ThreadPoolExecutor(max_workers=PARALLEL_UPLOADS) as executor:
                    futures = {
                        executor.submit(
                            upload_batch_with_retry, 
                            cloud, 
                            COLLECTION_NAME, 
                            sub, 
                            WAIT_FOR_UPSERTS, 
                            MAX_RETRIES,
                            idx
                        ): idx for idx, sub in enumerate(sub_batches)
                    }
                    
                    for future in as_completed(futures):
                        try:
                            uploaded_count = future.result()
                            total += uploaded_count
                            if pbar:
                                pbar.update(uploaded_count)
                        except Exception as e:
                            print(f"\n❌ Upload batch failed: {type(e).__name__}: {e}")
                            save_checkpoint({"scroll_id": scroll_id, "migrated": total})
                            if pbar:
                                pbar.close()
                            return False
            except Exception as e:
                print(f"\n❌ Parallel upload error: {type(e).__name__}: {e}")
                save_checkpoint({"scroll_id": scroll_id, "migrated": total})
                if pbar:
                    pbar.close()
                return False

            # Persist checkpoint every N batches (optimization to reduce I/O)
            batch_counter += 1
            if batch_counter % CHECKPOINT_EVERY == 0:
                save_checkpoint({"scroll_id": scroll_id, "migrated": total})

            # Calculate and display speed metrics
            current_time = time.time()
            elapsed = current_time - last_update_time
            if elapsed > 5:  # Update every 5 seconds
                points_since_last = total - last_update_count
                speed = points_since_last / elapsed if elapsed > 0 else 0
                eta_seconds = (src_count - total) / speed if speed > 0 else 0
                eta_minutes = eta_seconds / 60
                
                if not pbar:
                    if src_count:
                        pct = (total / src_count) * 100
                        print(f"📦 {total}/{src_count} ({pct:.1f}%) | Speed: {speed:.0f} pts/s | ETA: {eta_minutes:.1f} min")
                    else:
                        print(f"📦 {total} points | Speed: {speed:.0f} pts/s")
                
                last_update_time = current_time
                last_update_count = total

            # if scroll_id is None, we've reached the end
            if scroll_id is None or len(points) < SCROLL_BATCH:
                break

    except Exception as e:
        print(f"\n❌ Error during migration loop: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        save_checkpoint({"scroll_id": scroll_id, "migrated": total})
        if pbar:
            pbar.close()
        return False

    # Final checkpoint save
    save_checkpoint({"scroll_id": scroll_id, "migrated": total})
    
    # Calculate final statistics
    total_time = time.time() - start_time
    avg_speed = total / total_time if total_time > 0 else 0
    
    if pbar:
        pbar.close()
    
    print(f"\n📊 Migration Statistics:")
    print(f"   Total points: {total}")
    print(f"   Total time: {total_time/60:.2f} minutes")
    print(f"   Average speed: {avg_speed:.0f} points/sec")

    # Finalization: if non-blocking upserts used, poll until expected count or timeout
    expected = total if src_count is None else src_count
    if not WAIT_FOR_UPSERTS:
        print("\n⏳ Waiting for Cloud to finish indexing uploaded points...")
        observed = poll_destination_count(cloud, COLLECTION_NAME, expected=expected, timeout=FINAL_WAIT_SECS)
    else:
        # if wait=True used, we expect counts to be close already; still poll a bit
        observed = poll_destination_count(cloud, COLLECTION_NAME, expected=expected, timeout=min(FINAL_WAIT_SECS, 30))

    # Final reporting
    try:
        dst_info = cloud.get_collection(collection_name=COLLECTION_NAME)
        dst_count = getattr(dst_info, "points_count", None) or getattr(dst_info, "result", {}).get("points_count", None)
    except Exception:
        dst_count = observed

    print(f"✅ Finished. Source: {src_count}, Destination: {dst_count}, Migrated: {total}")
    if dst_count is not None and src_count is not None and int(dst_count) == int(src_count):
        try:
            os.remove(CHECKPOINT_FILE)
        except Exception:
            pass
        print("🎉 Migration complete and verified.")
        return True
    else:
        print("⚠️  Migration finished but counts differ or verification unavailable.")
        print("You can re-run the script; it will resume from the checkpoint.")
        return False


if __name__ == "__main__":
    print("="*60)
    print("Qdrant Migration Tool: Local -> Cloud")
    print("="*60)
    
    # interactive fallback for env vars
    src = os.environ.get("SOURCE_QDRANT_URL") or input(f"Source Qdrant URL [{SOURCE_URL}]: ").strip() or SOURCE_URL
    cloud_url = os.environ.get("CLOUD_URL") or input("Qdrant Cloud URL: ").strip()
    key = os.environ.get("CLOUD_API_KEY") or input("Qdrant Cloud API Key: ").strip()

    if not cloud_url or not key:
        print("❌ CLOUD_URL and CLOUD_API_KEY are required.")
        sys.exit(1)

    print(f"\n📋 Collection: {COLLECTION_NAME}")
    print(f"📦 Scroll batch size: {SCROLL_BATCH}")
    print(f"📦 Upsert sub-batch size: {UPSERT_SUB_BATCH}")
    print(f"🔄 Max retries: {MAX_RETRIES}")
    print(f"⚡ Parallel uploads: {PARALLEL_UPLOADS} threads")
    print(f"� Using gRPC: {USE_GRPC}")
    print(f"🔮 Prefetch batches: {PREFETCH_BATCHES}")
    print(f"�💾 Checkpoint every: {CHECKPOINT_EVERY} batches")
    print(f"💾 Checkpoint file: {CHECKPOINT_FILE}\n")

    ok = migrate(src, cloud_url, key)
    sys.exit(0 if ok else 1)