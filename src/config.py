from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Paths
DOCUMENTS_PATH = PROJECT_ROOT / "documents"
QDRANT_PATH = PROJECT_ROOT / "qdrant_db_new"

QDRANT_MODE = "docker"
QDRANT_URL = "http://192.168.137.1:6333"

# Collection settings
COLLECTION_NAME = "space-biology-papers"
EMBEDDING_MODEL = "openai/clip-vit-base-patch32"

# Chunking settings
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Batch settings
BATCH_SIZE = 100

# Print paths for verification
if __name__ == "__main__":
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Documents Path: {DOCUMENTS_PATH}")
    print(f"Documents Exists: {DOCUMENTS_PATH.exists()}")
    print(f"Qdrant Path: {QDRANT_PATH}")
