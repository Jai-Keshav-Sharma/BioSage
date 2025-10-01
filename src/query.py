from typing import List, Dict, Optional
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from src.config import COLLECTION_NAME, EMBEDDING_MODEL, QDRANT_PATH

def load_vector_store():
    """Load existing qdrant vector store from Docker instance."""

    print("🔄 Loading embedding model...")
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}, 
        encode_kwargs={'normalize_embeddings': True}
    )

    print("🔄 Connecting to Qdrant server...")
    client = QdrantClient(path=str(QDRANT_PATH))

    vector_store = Qdrant(
        client=client, 
        collection_name=COLLECTION_NAME, 
        embeddings=embeddings
    )

    print(f"✅ Connected to collection: {COLLECTION_NAME}")
    return vector_store


def search(query: str, k: int = 5, filters: Optional[Dict] = None):
    """
    Search the vector store 

    Args: 
        query: Search query
        k: Number of results to return
        filters: Optional metadata filters (e.g. {"organisms": "Arabidopsis"})

    Example: 
        search("microgravity effects", k = 5)
        search("gene expression", k=3, filters={"organisms": "Arabidopsis"})
    """

    vector_store = load_vector_store()

    print(f"\n🔍 Query: '{query}'")
    if filters:
        print(f"🎯 Filters: {filters}")

    if filters:
        results = vector_store.similarity_search(query, k=k, filter=filters)
    else:
        results = vector_store.similarity_search(query, k=k)

    print(f"\n📊 Found {len(results)} results:\n")
    print("="*80)

    for i, doc in enumerate(results, 1):
        print(f"\n【 Result {i} 】")
        print(f"📄 Paper: {doc.metadata.get('title', 'Unknown')[:70]}")
        print(f"🔖 PMC ID: {doc.metadata.get('pmc_id', 'Unknown')}")
        print(f"🧬 Organisms: {', '.join(doc.metadata.get('organisms', ['Unknown']))}")
        print(f"🔬 Types: {', '.join(doc.metadata.get('research_types', ['Unknown']))}")
        print(f"📍 Section: {doc.metadata.get('section_position', 'Unknown')}")
        print(f"\n💬 Content Preview:")
        print(f"   {doc.page_content[:300].strip()}...")
        print("-"*80)
    
    return results


def search_with_scores(query: str, k: int = 5):
    """Search with similarity scores"""

    vector_store = load_vector_store()

    results = vector_store.similarity_search_with_score(query, k=k)

    for i, (doc, score) in enumerate(results, 1):
        print(f"\n【 Result {i} 】 Similarity: {1-score:.4f}")
        print(f"📄 {doc.metadata.get('title', 'Unknown')[:60]}")
        print(f"🔖 PMC: {doc.metadata.get('pmc_id')}")
        print(f"   {doc.page_content[:200].strip()}...")
        print("-"*80)
    
    return results



if __name__ == "__main__":
    print("\n" + "="*80)
    print("🧪 TESTING QUERY SYSTEM")
    print("="*80)
    
    # Test 1: General search
    search("What are the effects of microgravity on muscle and bone?", k=3)
    
    print("\n\n")
    
    # Test 2: Search with filter
    search("biofilm formation", k=3, filters={"research_types": "microbiology"})
    
    print("\n\n")
    
    # Test 3: Search with scores
    search_with_scores("gene expression in space", k=3)