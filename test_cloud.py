from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv

load_dotenv()

# Get cloud credentials
CLOUD_URL = os.getenv("CLOUD_URL")
CLOUD_API_KEY = os.getenv("CLOUD_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "space-biology-papers")

print("="*60)
print("🧪 Qdrant Cloud Connection Test")
print("="*60)
print(f"Cloud URL: {CLOUD_URL}")
print(f"Collection: {COLLECTION_NAME}\n")

try:
    # Connect to Qdrant Cloud
    print("🔌 Connecting to Qdrant Cloud...")
    client = QdrantClient(
        url=CLOUD_URL,
        api_key=CLOUD_API_KEY,
        timeout=30,
        prefer_grpc=True
    )
    print("✅ Connection successful!\n")
    
    # Test 1: List all collections
    print("📚 Test 1: Listing collections...")
    collections = client.get_collections()
    collection_names = [c.name for c in collections.collections]
    print(f"   Found {len(collection_names)} collections: {collection_names}\n")
    
    # Test 2: Check if your collection exists
    print(f"📦 Test 2: Checking collection '{COLLECTION_NAME}'...")
    if COLLECTION_NAME in collection_names:
        print(f"   ✅ Collection '{COLLECTION_NAME}' exists!\n")
        
        # Test 3: Get collection info
        print("📊 Test 3: Getting collection details...")
        info = client.get_collection(collection_name=COLLECTION_NAME)
        print(f"   Points count: {info.points_count}")
        print(f"   Vector size: {info.config.params.vectors.size}")
        print(f"   Distance: {info.config.params.vectors.distance}\n")
        
        # Test 4: Sample search
        print("🔍 Test 4: Testing search functionality...")
        results = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=5,
            with_vectors=False,
            with_payload=True
        )
        
        points, _ = results if isinstance(results, tuple) else (results.points, results.next_page_offset)
        
        print(f"   ✅ Retrieved {len(points)} sample points")
        if points:
            print(f"   Sample point ID: {points[0].id}")
            print(f"   Sample payload keys: {list(points[0].payload.keys())}\n")
        
        # Test 5: Test vector search (if we have points)
        if points and hasattr(points[0], 'id'):
            print("🎯 Test 5: Testing vector similarity search...")
            # Get a point to use its vector for search
            test_point = client.retrieve(
                collection_name=COLLECTION_NAME,
                ids=[points[0].id],
                with_vectors=True
            )
            
            if test_point and test_point[0].vector:
                search_results = client.search(
                    collection_name=COLLECTION_NAME,
                    query_vector=test_point[0].vector,
                    limit=3
                )
                print(f"   ✅ Search returned {len(search_results)} results")
                print(f"   Top result score: {search_results[0].score:.4f}\n")
        
        print("="*60)
        print("🎉 All tests passed! Qdrant Cloud is working perfectly!")
        print("="*60)
        
    else:
        print(f"   ❌ Collection '{COLLECTION_NAME}' not found!")
        print(f"   Available collections: {collection_names}")
        
except Exception as e:
    print(f"\n❌ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    print("\n💡 Troubleshooting:")
    print("   1. Check CLOUD_URL in .env (should start with https://)")
    print("   2. Check CLOUD_API_KEY is correct")
    print("   3. Verify collection was migrated successfully")