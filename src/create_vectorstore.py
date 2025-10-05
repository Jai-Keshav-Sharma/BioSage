from pathlib import Path
from typing import List, Union
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import io
import base64
from langchain.docstore.document import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_core.embeddings import Embeddings
from dotenv import load_dotenv


from src.config import (
    DOCUMENTS_PATH,
    QDRANT_PATH,
    QDRANT_MODE,
    QDRANT_URL,
    COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    BATCH_SIZE
)


load_dotenv(override=True)


def initialize_qdrant_collection():
    """
    Initialize an empty Qdrant collection before processing any documents.
    This prevents timeout issues when creating collection with large datasets.
    
    Returns:
        bool: True if successful, False otherwise
    """
    print("\n🔄 Initializing CLIP embeddings...")
    embeddings = CLIPEmbeddings(model_name="openai/clip-vit-base-patch32")
    
    # Get embedding dimension
    sample_embedding = embeddings.embed_query("test")
    vector_size = len(sample_embedding)
    print(f"📊 Vector dimension: {vector_size}")
    
    # Initialize Qdrant client based on mode
    if QDRANT_MODE == "docker":
        print(f"\n🐳 Connecting to Qdrant Docker instance at {QDRANT_URL}...")
        try:
            # Simpler connection with api_key=None to avoid auth issues
            client = QdrantClient(
                host="localhost",
                port=6333,
                timeout=120,  # 2 minute timeout
                prefer_grpc=False  # Use REST API instead of gRPC
            )
            # Test connection
            collections = client.get_collections()
            print(f"✅ Connected successfully! Found {len(collections.collections)} existing collections")
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    else:
        print(f"\n🔄 Connecting to local Qdrant (file-based storage)...")
        client = QdrantClient(path=str(QDRANT_PATH))
    
    # Delete existing collection if it exists
    try:
        client.delete_collection(collection_name=COLLECTION_NAME)
        print(f"🗑️  Deleted existing collection '{COLLECTION_NAME}'")
    except Exception:
        pass  # Collection doesn't exist, that's fine
    
    # Create collection with proper vector configuration
    print(f"\n🔄 Creating Qdrant collection '{COLLECTION_NAME}'...")
    try:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )
        print(f"✅ Collection '{COLLECTION_NAME}' created successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to create collection: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_qdrant_store_batched(chunks: List[Document], batch_size: int = 100):
    """
    Add documents to existing Qdrant collection in batches.
    This is called multiple times during the build process.
    
    Args:
        chunks: List of document chunks (can be text or images)
        batch_size: Number of chunks to process at once (default: 100)
        
    Returns:
        bool: True if successful, False if errors occurred
    """
    
    print(f"\n🔄 Loading CLIP embeddings...")
    embeddings = CLIPEmbeddings(model_name="openai/clip-vit-base-patch32")
    
    # Initialize Qdrant client based on mode
    if QDRANT_MODE == "docker":
        client = QdrantClient(
            host="localhost",
            port=6333,
            timeout=120,
            prefer_grpc=False
        )
    else:
        client = QdrantClient(path=str(QDRANT_PATH))
    
    # Separate text and image documents
    text_chunks = [doc for doc in chunks if doc.metadata.get("type") != "image"]
    image_chunks = [doc for doc in chunks if doc.metadata.get("type") == "image"]
    
    print(f"📦 Processing documents:")
    print(f"   📝 Text chunks: {len(text_chunks)}")
    print(f"   🖼️  Image chunks: {len(image_chunks)}")
    print(f"   📊 Total: {len(chunks)}")
    print(f"   Batch size: {batch_size}\n")
    
    has_errors = False
    
    # Process text documents with standard LangChain method
    if text_chunks:
        total_batches = (len(text_chunks) + batch_size - 1) // batch_size
        print("📝 Embedding text documents...")
        
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=embeddings
        )
        
        for batch_idx in range(0, len(text_chunks), batch_size):
            batch = text_chunks[batch_idx:batch_idx + batch_size]
            batch_num = (batch_idx // batch_size) + 1
            
            try:
                print(f"   ⚡ Text batch {batch_num}/{total_batches} ({len(batch)} chunks)...")
                vector_store.add_documents(batch)
            except Exception as e:
                print(f"   ⚠️  Error in text batch {batch_num}: {e}")
                has_errors = True
    
    # Process image documents with custom embeddings
    if image_chunks:
        print(f"\n🖼️  Embedding image documents...")
        image_batches = (len(image_chunks) + batch_size - 1) // batch_size
        
        # Track point IDs to avoid collisions
        existing_point_ids = set()
        successful_images = 0
        failed_images = 0
        
        for batch_idx in range(0, len(image_chunks), batch_size):
            batch = image_chunks[batch_idx:batch_idx + batch_size]
            batch_num = (batch_idx // batch_size) + 1
            
            print(f"   ⚡ Image batch {batch_num}/{image_batches} ({len(batch)} images)...")
            
            # Manually embed images and add to vector store
            for idx, doc in enumerate(batch):
                try:
                    # Get image bytes from metadata
                    image_bytes = doc.metadata.get("image_bytes")
                    if not image_bytes:
                        failed_images += 1
                        continue
                        
                    # Create embedding using CLIP image encoder
                    image_embedding = embeddings.embed_image(image_bytes)
                    
                    # Generate a unique positive integer ID with collision detection
                    image_id = doc.metadata.get("image_id", f"img_{batch_idx}_{idx}")
                    unique_id = abs(hash(image_id)) % (2**63 - 1)
                    
                    # Handle potential ID collision (very rare but possible)
                    salt = 0
                    while unique_id in existing_point_ids:
                        salt += 1
                        unique_id = abs(hash(f"{image_id}_salt_{salt}")) % (2**63 - 1)
                        if salt > 100:  # Safety limit
                            raise ValueError(f"Could not generate unique ID after 100 attempts for {image_id}")
                    
                    # Track this ID to avoid future collisions
                    existing_point_ids.add(unique_id)
                    
                    # Prepare metadata without image bytes
                    clean_metadata = {k: v for k, v in doc.metadata.items() if k != "image_bytes"}
                    
                    # Add to vector store with embedding
                    client.upsert(
                        collection_name=COLLECTION_NAME,
                        points=[
                            PointStruct(
                                id=unique_id,
                                vector=image_embedding,
                                payload={
                                    "page_content": doc.page_content,
                                    "metadata": clean_metadata
                                }
                            )
                        ]
                    )
                    successful_images += 1
                        
                except Exception as e:
                    failed_images += 1
                    has_errors = True
                    continue
        
        if image_chunks:
            print(f"\n   ✅ Successfully embedded: {successful_images} images")
            if failed_images > 0:
                print(f"   ⚠️  Failed: {failed_images} images")
    
    return not has_errors


# Custom CLIP Embedding Wrapper for LangChain
class CLIPEmbeddings(Embeddings):
    """Custom CLIP embedding class compatible with LangChain - handles both text and images."""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """Initialize CLIP model and processor."""
        print(f"🔄 Loading CLIP model: {model_name}")
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        print(f"✅ CLIP model loaded successfully")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using CLIP text encoder."""
        embeddings = []
        for text in texts:
            embedding = self._embed_text(text)
            embeddings.append(embedding.tolist())
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a query using CLIP text encoder."""
        return self._embed_text(text).tolist()
    
    def embed_image(self, image: Union[Image.Image, str, bytes]) -> List[float]:
        """Embed an image using CLIP image encoder.
        
        Args:
            image: PIL Image, file path (str), or image bytes
            
        Returns:
            List of floats representing the image embedding
        """
        return self._embed_image(image).tolist()
    
    def _embed_text(self, text: str) -> np.ndarray:
        """Internal method to embed text using CLIP."""
        inputs = self.processor(
            text=text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77  # CLIP's max token length
        )
        with torch.no_grad():
            features = self.model.get_text_features(**inputs)
            # Normalize embeddings to unit vector
            features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().cpu().numpy()
    
    def _embed_image(self, image: Union[Image.Image, str, bytes]) -> np.ndarray:
        """Internal method to embed image using CLIP."""
        # Convert to PIL Image if needed
        if isinstance(image, str):
            # File path
            pil_image = Image.open(image).convert("RGB")
        elif isinstance(image, bytes):
            # Image bytes
            pil_image = Image.open(io.BytesIO(image)).convert("RGB")
        else:
            # Already PIL Image
            pil_image = image.convert("RGB") if image.mode != "RGB" else image
        
        inputs = self.processor(
            images=pil_image,
            return_tensors="pt"
        )
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
            # Normalize embeddings to unit vector
            features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().cpu().numpy()



# Create Qdrant Vector Store with CLIP Embeddings (Text + Images)
def create_qdrant_store(chunks: List[Document], batch_size: int = 100):
    """
    Create Qdrant Vector Store with CLIP embeddings for both text and images.
    
    Args:
        chunks: List of document chunks (can be text or images)
        batch_size: Number of chunks to process at once (default: 100)
    """
    
    print("\n🔄 Initializing CLIP embeddings...")
    embeddings = CLIPEmbeddings(model_name="openai/clip-vit-base-patch32")
    
    # Get embedding dimension (CLIP ViT-B/32 produces 512-dim vectors)
    sample_embedding = embeddings.embed_query("test")
    vector_size = len(sample_embedding)
    print(f"📊 Vector dimension: {vector_size}")
    
    # Initialize Qdrant client based on mode
    if QDRANT_MODE == "docker":
        print(f"\n🐳 Connecting to Qdrant Docker instance at {QDRANT_URL}...")
        client = QdrantClient(url=QDRANT_URL)
    else:
        print(f"\n🔄 Connecting to local Qdrant (file-based storage)...")
        client = QdrantClient(path=str(QDRANT_PATH))
    
    # Delete existing collection if it exists
    try:
        client.delete_collection(collection_name=COLLECTION_NAME)
        print(f"🗑️  Deleted existing collection '{COLLECTION_NAME}'")
    except Exception:
        pass  # Collection doesn't exist, that's fine
    
    # Create collection with proper vector configuration
    print(f"\n🔄 Creating Qdrant collection '{COLLECTION_NAME}'...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE
        )
    )
    
    # Separate text and image documents
    text_chunks = [doc for doc in chunks if doc.metadata.get("type") != "image"]
    image_chunks = [doc for doc in chunks if doc.metadata.get("type") == "image"]
    
    print(f"📦 Processing documents:")
    print(f"   📝 Text chunks: {len(text_chunks)}")
    print(f"   🖼️  Image chunks: {len(image_chunks)}")
    print(f"   📊 Total: {len(chunks)}")
    print(f"   Batch size: {batch_size}\n")
    
    # Process text documents with standard LangChain method
    total_batches = (len(text_chunks) + batch_size - 1) // batch_size
    
    if text_chunks:
        print("📝 Embedding text documents...")
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=embeddings
        )
        
        for batch_idx in range(0, len(text_chunks), batch_size):
            batch = text_chunks[batch_idx:batch_idx + batch_size]
            batch_num = (batch_idx // batch_size) + 1
            
            print(f"   ⚡ Text batch {batch_num}/{total_batches} ({len(batch)} chunks)...")
            vector_store.add_documents(batch)
    else:
        # No text docs, just create empty store
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=embeddings
        )
    
    # Process image documents with custom embeddings
    if image_chunks:
        print(f"\n🖼️  Embedding image documents...")
        image_batches = (len(image_chunks) + batch_size - 1) // batch_size
        
        # Track point IDs to avoid collisions
        existing_point_ids = set()
        successful_images = 0
        failed_images = 0
        
        for batch_idx in range(0, len(image_chunks), batch_size):
            batch = image_chunks[batch_idx:batch_idx + batch_size]
            batch_num = (batch_idx // batch_size) + 1
            
            print(f"   ⚡ Image batch {batch_num}/{image_batches} ({len(batch)} images)...")
            
            # Manually embed images and add to vector store
            for idx, doc in enumerate(batch):
                try:
                    # Get image bytes from metadata
                    image_bytes = doc.metadata.get("image_bytes")
                    if not image_bytes:
                        print(f"      ⚠️  Skipping image {doc.metadata.get('image_id', 'unknown')}: No image bytes")
                        failed_images += 1
                        continue
                        
                    # Create embedding using CLIP image encoder
                    image_embedding = embeddings.embed_image(image_bytes)
                    
                    # Generate a unique positive integer ID with collision detection
                    image_id = doc.metadata.get("image_id", f"img_{batch_idx}_{idx}")
                    unique_id = abs(hash(image_id)) % (2**63 - 1)
                    
                    # Handle potential ID collision (very rare but possible)
                    salt = 0
                    while unique_id in existing_point_ids:
                        salt += 1
                        unique_id = abs(hash(f"{image_id}_salt_{salt}")) % (2**63 - 1)
                        if salt > 100:  # Safety limit
                            raise ValueError(f"Could not generate unique ID after 100 attempts for {image_id}")
                    
                    # Track this ID to avoid future collisions
                    existing_point_ids.add(unique_id)
                    
                    # Prepare metadata without image bytes
                    clean_metadata = {k: v for k, v in doc.metadata.items() if k != "image_bytes"}
                    
                    # Add to vector store with embedding
                    client.upsert(
                        collection_name=COLLECTION_NAME,
                        points=[
                            PointStruct(
                                id=unique_id,
                                vector=image_embedding,
                                payload={
                                    "page_content": doc.page_content,
                                    "metadata": clean_metadata
                                }
                            )
                        ]
                    )
                    successful_images += 1
                        
                except Exception as e:
                    failed_images += 1
                    print(f"      ⚠️  Error embedding image {doc.metadata.get('image_id', f'img_{batch_idx}_{idx}')}: {e}")
                    continue
        
        print(f"\n   ✅ Successfully embedded: {successful_images} images")
        if failed_images > 0:
            print(f"   ⚠️  Failed: {failed_images} images")
    
    print(f"\n{'='*80}")
    print(f"✅ Multimodal Qdrant collection created successfully!")
    if QDRANT_MODE == "docker":
        print(f"   Mode: Docker")
        print(f"   URL: {QDRANT_URL}")
    else:
        print(f"   Mode: Local file-based")
        print(f"   Location: {QDRANT_PATH}")
    print(f"   Collection: {COLLECTION_NAME}")
    print(f"   Embedding Model: CLIP (openai/clip-vit-base-patch32)")
    print(f"   Vector Dimension: {vector_size}")
    print(f"   📝 Text vectors: {len(text_chunks)}")
    print(f"   🖼️  Image vectors: {len(image_chunks)}")
    print(f"   📊 Total vectors: {len(chunks)}")
    print(f"   Distance Metric: COSINE")
    print(f"{'='*80}\n")
    
    return vector_store
