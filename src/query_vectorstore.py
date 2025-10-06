"""
Query the Qdrant Vector Store using CLIP embeddings
Supports local file-based, Docker-based, and Cloud-based Qdrant instances
"""

from functools import lru_cache
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from src.create_vectorstore import CLIPEmbeddings
from src.config import QDRANT_PATH, QDRANT_MODE, QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    """
    Get cached Qdrant client connection.
    Supports local, docker, and cloud modes.
    This is cached to avoid creating multiple connections.
    
    Returns:
        QdrantClient: Cached Qdrant client instance
    """
    if QDRANT_MODE == "cloud":
        # Cloud mode with API key
        if not QDRANT_URL or not QDRANT_API_KEY:
            raise ValueError(
                "QDRANT_URL and QDRANT_API_KEY are required for cloud mode. "
                "Please set them in your .env file."
            )
        
        print(f"☁️  Connecting to Qdrant Cloud at {QDRANT_URL}...")
        client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=60,
            prefer_grpc=True,  # Use gRPC for better performance
            https=True
        )
        
    elif QDRANT_MODE == "docker":
        # Docker mode
        if not QDRANT_URL:
            raise ValueError("QDRANT_URL is required for docker mode")
        
        print(f"🐳 Connecting to Qdrant Docker instance at {QDRANT_URL}...")
        client = QdrantClient(
            url=QDRANT_URL,
            timeout=60,
            prefer_grpc=False,
            https=False
        )
        
    else:
        # Local file-based mode
        print(f"📁 Connecting to local Qdrant at {QDRANT_PATH}...")
        client = QdrantClient(path=str(QDRANT_PATH))
    
    return client


@lru_cache(maxsize=1)
def get_clip_embeddings() -> CLIPEmbeddings:
    """
    Get cached CLIP embeddings model.
    This is cached to avoid reloading the model multiple times.
    
    Returns:
        CLIPEmbeddings: Cached CLIP embeddings instance
    """
    print("🔄 Loading CLIP embeddings...")
    return CLIPEmbeddings(model_name="openai/clip-vit-base-patch32")


def load_vectorstore():
    """Load existing Qdrant vector store with CLIP embeddings."""
    
    # Use cached CLIP embeddings
    embeddings = get_clip_embeddings()
    
    # Use cached Qdrant client (now supports cloud!)
    client = get_qdrant_client()
    
    # Check if collection exists
    try:
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]
    except Exception as e:
        raise RuntimeError(
            f"Failed to connect to Qdrant ({QDRANT_MODE} mode). "
            f"URL: {QDRANT_URL}. "
            f"Error: {e}"
        ) from e
    
    if COLLECTION_NAME not in collection_names:
        raise ValueError(
            f"Collection '{COLLECTION_NAME}' not found! "
            f"Available collections: {collection_names}\n"
            f"Mode: {QDRANT_MODE}, URL: {QDRANT_URL}"
        )
    
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings
    )
    
    print(f"✅ Vector store loaded: {COLLECTION_NAME} (mode: {QDRANT_MODE})\n")
    return vector_store


def search_documents(query: str, k: int = 5):
    """
    Search for relevant documents using semantic similarity.
    
    Args:
        query: Search query text
        k: Number of results to return
    """
    
    vector_store = load_vectorstore()
    
    print(f"🔍 Query: {query}")
    print(f"📊 Retrieving top {k} results...\n")
    print("-" * 80)
    
    results = vector_store.similarity_search(query, k=k)
    
    for i, doc in enumerate(results, 1):
        doc_type = doc.metadata.get('type', 'text')
        
        if doc_type == 'image':
            print(f"\n🖼️  Result {i} [IMAGE]:")
            print(f"   Image ID: {doc.metadata.get('image_id', 'Unknown')}")
            print(f"   Source: {doc.metadata.get('filename', 'Unknown')}")
            print(f"   Page: {doc.metadata.get('page', 'N/A')}")
            print(f"   Description: {doc.page_content}")
        else:
            print(f"\n📄 Result {i} [TEXT]:")
            print(f"   Source: {doc.metadata.get('filename', 'Unknown')}")
            print(f"   PMC ID: {doc.metadata.get('pmc_id', 'Unknown')}")
            print(f"   Content Preview: {doc.page_content[:200]}...")
        
        print(f"   Organisms: {doc.metadata.get('organisms', [])}")
        print(f"   Research Types: {doc.metadata.get('research_types', [])}")
    
    print("\n" + "-" * 80)
    return results


# No changes needed to the main block
if __name__ == "__main__":
    query = "effects of microgravity on plant root development"
    results = search_documents(query, k=3)