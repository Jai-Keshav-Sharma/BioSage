

import fitz
from pathlib import Path
from typing import Dict, List 
import re
from tqdm import tqdm
from qdrant_client.http.exceptions import UnexpectedResponse
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.docstore.document import Document 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from dotenv import load_dotenv




from src.config import (
    DOCUMENTS_PATH,
    QDRANT_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    BATCH_SIZE
)


load_dotenv(override=True)

# Create Qdrant Vector Store 
def create_qdrant_store(chunks: List[Document], batch_size: int = 100):
    """
    Create Qdrant Vector Store with embeddings using batching
    Qdrant runs locally (no Docker needed!)

    Args:
        chunks: List of document chunks to embed
        batch_size: Number of chunks to process at once (default: 100)
    """

    print("\n🔄 Loading embedding model...")
    print("   (First time will download ~400MB model)")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    print(f"✅ Embedding model loaded: {EMBEDDING_MODEL}")

    import gc
    gc.collect()  # Force garbage collection to close any lingering connections

    # Initialize Qdrant client
    client = QdrantClient(url="http://localhost:6333")


    # Delete existing collection safely
    try:
        client.delete_collection(collection_name=COLLECTION_NAME)
        print(f"🗑️  Deleted existing collection '{COLLECTION_NAME}'")
    except Exception:  # ✅ Catch any exception (not just UnexpectedResponse)
        pass  # Collection doesn't exist, that's fine

    print(f"\n🔄 Creating Qdrant collection '{COLLECTION_NAME}'...")
    print(f"   Processing {len(chunks)} chunks in batches of {batch_size}\n")

    # Initialize vector store with first batch
    vector_store = None
    batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]

    for batch_idx, batch in enumerate(tqdm(batches, desc="Creating embeddings", unit="batch")):
        try:
            if batch_idx == 0:
                # First batch: create collection
                vector_store = Qdrant.from_documents(
                    documents=batch,
                    embedding=embeddings,
                    url="http://localhost:6333",
                    collection_name=COLLECTION_NAME,
                    force_recreate=True
                )
            else:
                # Subsequent batches: add to existing collection
                vector_store.add_documents(batch)

        except Exception as e:
            print(f"\n❌ Error in batch {batch_idx + 1}: {e}")
            raise

    print(f"\n{'='*80}")
    print(f"✅ Qdrant collection created successfully!")
    print(f"   Location: {QDRANT_PATH}")
    print(f"   Collection: {COLLECTION_NAME}")
    print(f"   Total vectors: {len(chunks)}")
    print(f"   Batches processed: {len(batches)}")
    print(f"{'='*80}")

    return vector_store
