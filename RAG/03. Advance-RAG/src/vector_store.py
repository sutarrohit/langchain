import os
import numpy as np
from typing import List, Any
import uuid

from src.embeddings import EmbeddingPipeline
import chromadb
from chromadb.config import Settings


class ChromaDBVectorStore:
    def __init__(
        self,
        collection_name: str = "pdf_documents",
        persist_dir: str = "chromaDB_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.embedding_model = embedding_model
        self.model = None
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.client = None
        self.collection = None
        self._load_embedding_model()
        self._initialize_store()

    def _load_embedding_model(self):
        emb_pipe = EmbeddingPipeline(embedding_model=self.embedding_model)
        self.model = emb_pipe.model

    def _initialize_store(self):
        self.client = chromadb.PersistentClient(
            path=self.persist_dir, settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "PDF document embedding for RAG"},
        )
        print(f"[INFO] ChromaDB initialized. Collection: {self.collection_name}")
        print(f"[INFO] Existing documents in collection: {self.collection.count()}")

    def build_from_documents(self, documents: List[Any]):
        print(f"[INFO] Building vector store from {len(documents)} raw documents...")
        emb_pipe = EmbeddingPipeline(
            embedding_model=self.embedding_model,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        chunks = emb_pipe.chunk_documents(documents)
        embeddings = emb_pipe.embed_chunks(chunks)
        metadatas = [{"text": chunk.page_content} for chunk in chunks]
        texts = [chunk.page_content for chunk in chunks]
        self.add_documents(texts, embeddings, metadatas)
        print(f"[INFO] Vector store built and saved to {self.persist_dir}")

    def add_documents(
        self, documents: List[str], embeddings: np.ndarray, metadatas: List[Any] = None
    ):
        ids = [f"doc_{uuid.uuid4().hex[:8]}_{i}" for i in range(len(documents))]
        embeddings_list = embeddings.tolist()

        self.collection.add(
            ids=ids,
            embeddings=embeddings_list,
            metadatas=metadatas,
            documents=documents,
        )
        print(
            f"[INFO] Added {len(documents)} documents to ChromaDB. Total: {self.collection.count()}"
        )

    def query(self, query_text: str, top_k: int = 5):
        print(f"[INFO] Querying: '{query_text}'")
        query_emb = self.model.encode([query_text]).astype("float32")

        results = self.collection.query(
            query_embeddings=query_emb.tolist(), n_results=top_k
        )

        formatted_results = []
        for i in range(len(results["documents"][0])):
            formatted_results.append(
                {
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": (
                        results["distances"][0][i] if "distances" in results else None
                    ),
                }
            )
        return formatted_results


if __name__ == "__main__":
    from data_loader import load_all_documents

    docs = load_all_documents("data")
    store = ChromaDBVectorStore("pdf_documents", "chromaDB_store")
    store.build_from_documents(docs)
    print(store.query("What is attention mechanism?", top_k=3))
