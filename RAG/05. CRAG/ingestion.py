import sys
from ingestion_pipeline.vector_store import ChromaDBVectorStore
from ingestion_pipeline.data_loader import load_all_documents

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]


vector_store = ChromaDBVectorStore(
    collection_name="corrective-rag",
    chunk_size=500,
    chunk_overlap=100,
)


def ingestion():
    documents = load_all_documents(urls)
    if not documents:
        print("[ERROR] No documents found in data directory")
        sys.exit(1)

    print(f"[INFO] Building vector store...")

    vector_store.build_from_documents(documents)

    print(f"[INFO] Vector store built successfully!")
    print(f"[INFO] Total documents in store: {vector_store.collection.count()}")


if __name__ == "__main__":
    ingestion()
