import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_all_documents
from src.vector_store import ChromaDBVectorStore


def build_vector_store(
    data_dir: str = "data",
    collection_name: str = "pdf_documents",
    persist_dir: str = "chromaDB_store",
    embedding_model: str = "all-MiniLM-L6-v2",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
):
    data_path = Path(data_dir).resolve()
    if not data_path.exists():
        print(f"[ERROR] Data directory not found: {data_path}")
        sys.exit(1)
    
    print(f"[INFO] Loading documents from: {data_path}")
    documents = load_all_documents(str(data_path))
    
    if not documents:
        print("[ERROR] No documents found in data directory")
        sys.exit(1)
    
    print(f"[INFO] Building vector store...")
    vectorstore = ChromaDBVectorStore(
        collection_name=collection_name,
        persist_dir=persist_dir,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    vectorstore.build_from_documents(documents)
    
    print(f"[INFO] Vector store built successfully!")
    print(f"[INFO] Total documents in store: {vectorstore.collection.count()}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build vector store from documents")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory path")
    parser.add_argument("--collection", type=str, default="pdf_documents", help="ChromaDB collection name")
    parser.add_argument("--persist-dir", type=str, default="chromaDB_store", help="ChromaDB persist directory")
    parser.add_argument("--embedding-model", type=str, default="all-MiniLM-L6-v2", help="Embedding model name")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap")
    
    args = parser.parse_args()
    
    build_vector_store(
        data_dir=args.data_dir,
        collection_name=args.collection,
        persist_dir=args.persist_dir,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
