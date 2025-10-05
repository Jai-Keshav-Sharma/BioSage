"""
Query the Qdrant Vector Store using CLIP embeddings
Supports both local file-based and Docker-based Qdrant instances
"""

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from src.create_vectorstore import CLIPEmbeddings
from src.config import QDRANT_PATH, QDRANT_MODE, QDRANT_URL, COLLECTION_NAME


def load_vectorstore():
    """Load existing Qdrant vector store with CLIP embeddings."""
    
    print("🔄 Loading CLIP embeddings...")
    embeddings = CLIPEmbeddings(model_name="openai/clip-vit-base-patch32")
    
    # Connect to Qdrant based on mode
    if QDRANT_MODE == "docker":
        print(f"🐳 Connecting to Qdrant Docker instance at {QDRANT_URL}...")
        client = QdrantClient(url=QDRANT_URL)
    else:
        print(f"🔄 Connecting to local Qdrant at {QDRANT_PATH}...")
        client = QdrantClient(path=str(QDRANT_PATH))
    
    # Check if collection exists
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]
    
    if COLLECTION_NAME not in collection_names:
        raise ValueError(
            f"Collection '{COLLECTION_NAME}' not found! "
            f"Available collections: {collection_names}\n"
            f"Please run build_vectorstore.py first."
        )
    
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings
    )
    
    print(f"✅ Vector store loaded: {COLLECTION_NAME}\n")
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
        modality = doc.metadata.get('modality', 'text')
        
        if doc_type == 'image' or modality == 'image':
            print(f"\n�️  Result {i} [IMAGE]:")
            print(f"   Image ID: {doc.metadata.get('image_id', 'Unknown')}")
            print(f"   Source: {doc.metadata.get('filename', 'Unknown')}")
            print(f"   Page: {doc.metadata.get('page', 'N/A')}")
            print(f"   Description: {doc.page_content}")
        else:
            print(f"\n📄 Result {i} [TEXT]:")
            print(f"   Source: {doc.metadata.get('filename', 'Unknown')}")
            print(f"   Page: {doc.metadata.get('page', 'N/A')}")
            print(f"   Organisms: {', '.join(doc.metadata.get('organisms', ['Unknown']))}")
            print(f"   Research Types: {', '.join(doc.metadata.get('research_types', ['Unknown']))}")
            print(f"   Section: {doc.metadata.get('section_position', 'Unknown')}")
            print(f"\n   Preview: {doc.page_content[:200]}...")
        print("-" * 80)
    
    return results


def search_with_score(query: str, k: int = 5):
    """
    Search with similarity scores.
    
    Args:
        query: Search query text
        k: Number of results to return
    """
    
    vector_store = load_vectorstore()
    
    print(f"🔍 Query: {query}")
    print(f"📊 Retrieving top {k} results with scores...\n")
    print("-" * 80)
    
    results = vector_store.similarity_search_with_score(query, k=k)
    
    for i, (doc, score) in enumerate(results, 1):
        doc_type = doc.metadata.get('type', 'text')
        modality = doc.metadata.get('modality', 'text')
        
        if doc_type == 'image' or modality == 'image':
            print(f"\n�️  Result {i} [IMAGE] (Score: {score:.4f}):")
            print(f"   Image ID: {doc.metadata.get('image_id', 'Unknown')}")
            print(f"   Source: {doc.metadata.get('filename', 'Unknown')[:60]}...")
            print(f"   Page: {doc.metadata.get('page', 'N/A')}")
            print(f"\n   Description: {doc.page_content}")
        else:
            print(f"\n�📄 Result {i} [TEXT] (Score: {score:.4f}):")
            print(f"   Source: {doc.metadata.get('filename', 'Unknown')[:60]}...")
            print(f"   Organisms: {', '.join(doc.metadata.get('organisms', ['Unknown'])[:3])}")
            print(f"   Research Types: {', '.join(doc.metadata.get('research_types', ['Unknown']))}")
            print(f"\n   Preview: {doc.page_content[:150]}...")
        print("-" * 80)
    
    return results


if __name__ == "__main__":
    # Example queries designed to return both text and images
    # Scientific images include: microscopy, graphs, diagrams, biofilm structures, etc.
    queries = [
        "microscopy images of cells in space",
        "biofilm morphology and structure",
        "bone tissue microstructure changes",
        "graph showing gene expression data",
        "retinal photoreceptor cell morphology",
        "fungal hyphae and spore formation patterns",
        "muscle fiber cross-section imaging",
        "plant root architecture and growth patterns",
        "bacterial colony morphology on surfaces",
        "tissue histology sections"
    ]
    
    print("=" * 80)
    print("🔮 NASA Space Biology Papers - Multimodal Semantic Search Demo")
    print("🖼️  Queries optimized to retrieve both TEXT and IMAGE results")
    print("=" * 80)
    print()
    
    # Run first 3 queries with more results to show images
    for query in queries[:3]:
        search_with_score(query, k=5)  # Increased k to 5 for better chance of images
        print("\n" + "=" * 80 + "\n")
