import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Paths
DOCUMENTS_PATH = PROJECT_ROOT / "documents"
QDRANT_PATH = PROJECT_ROOT / "qdrant_db_new"

QDRANT_MODE = os.getenv("QDRANT_MODE", "cloud")  # "local" or "docker" or "cloud"
QDRANT_URL = os.getenv("QDRANT_URL")   
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)  

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
    print(f"QDRANT_MODE: {QDRANT_MODE}")
    print(f"QDRANT_URL: {QDRANT_URL}")
    print(f"QDRANT_API_KEY set: {bool(QDRANT_API_KEY)}")
