from qdrant_client import QdrantClient
from src.config import COLLECTION_NAME, QDRANT_URL, QDRANT_MODE

def reset_qdrant_docker():
    """Manually delete the collection in Docker Qdrant."""
    
    print(f"🐳 Connecting to Qdrant Docker at {QDRANT_URL}...")
    print(f"📋 Mode: {QDRANT_MODE}")
    print(f"🎯 Target collection: {COLLECTION_NAME}")
    
    try:
        # Connect to Docker Qdrant
        client = QdrantClient(
            host="localhost",
            port=6333,
            timeout=120,
            prefer_grpc=False
        )
        
        # Test connection first
        try:
            info = client.get_collections()
            print("✅ Successfully connected to Qdrant!")
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            return False
        
        # List existing collections
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]
        print(f"📋 Found {len(collections.collections)} collections:")
        for collection in collections.collections:
            print(f"   - {collection.name}")
        
        # Check if target collection exists
        if COLLECTION_NAME not in collection_names:
            print(f"⚠️  Collection '{COLLECTION_NAME}' not found. Nothing to delete.")
            return True
        
        # Delete the specific collection
        try:
            client.delete_collection(collection_name=COLLECTION_NAME)
            print(f"🗑️  Deleted collection '{COLLECTION_NAME}' successfully!")
        except Exception as e:
            print(f"❌ Failed to delete collection '{COLLECTION_NAME}': {e}")
            return False
        
        # Verify deletion
        collections_after = client.get_collections()
        remaining_names = [c.name for c in collections_after.collections]
        print(f"📋 Collections after deletion: {len(collections_after.collections)}")
        for collection in collections_after.collections:
            print(f"   - {collection.name}")
        
        # Double-check the target collection is gone
        if COLLECTION_NAME in remaining_names:
            print(f"⚠️  WARNING: Collection '{COLLECTION_NAME}' still exists after deletion!")
            return False
        else:
            print(f"✅ Confirmed: Collection '{COLLECTION_NAME}' successfully removed!")
        
        print("✅ Qdrant reset completed!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to connect to Qdrant Docker: {e}")
        print("Make sure your Qdrant Docker container is running:")
        print("docker run -p 6333:6333 qdrant/qdrant")
        return False

if __name__ == "__main__":
    success = reset_qdrant_docker()
    if success:
        print("\n🎉 Ready to build fresh vector store!")
        print("Next step: Run 'python src/build_vectorstore.py'")
    else:
        print("\n💥 Reset failed. Please check the error messages above.")