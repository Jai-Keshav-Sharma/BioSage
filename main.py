from src.preprocess_docs import load_all_documents, chunk_documents
from src.create_vectorstore import create_qdrant_store

def main():
    documents = load_all_documents()
    chunks = chunk_documents(documents)
    vector_store = create_qdrant_store(chunks)


if __name__ == "__main__":
    main()
