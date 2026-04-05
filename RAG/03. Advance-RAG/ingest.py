
from pathlib import Path
import sys
import argparse

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
    
    documents = load_all_documents(data_dir)
    
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

    
    parser = argparse.ArgumentParser(description="Build vector store from documents")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory path")
    parser.add_argument("--collection", type=str, default="pdf_documents", help="ChromaDB collection name")
    parser.add_argument("--persist-dir", type=str, default="chromaDB_store", help="ChromaDB persist directory")
    parser.add_argument("--embedding-model", type=str, default="all-mpnet-base-v2", help="Embedding model name")
    parser.add_argument("--chunk-size", type=int, default=600, help="Chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=100, help="Chunk overlap")
    
    args = parser.parse_args()
    
    build_vector_store(
        data_dir=args.data_dir,
        collection_name=args.collection,
        persist_dir=args.persist_dir,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
