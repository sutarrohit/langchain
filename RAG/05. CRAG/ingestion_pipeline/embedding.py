import os
from typing import Any, List

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from ingestion_pipeline.data_loader import load_all_documents

LOCAL_MODEL_PATH = "./models"


class EmbeddingPipeline:
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.model = self._load_model()
        print(f"[INFO] Loaded embedding model: {embedding_model}")

    def _load_model(self):
        if not os.path.exists(f"{LOCAL_MODEL_PATH}/{self.embedding_model}"):
            print(f"Downloading '{self.embedding_model}'...")
            model = SentenceTransformer(self.embedding_model)
            os.makedirs(f"{LOCAL_MODEL_PATH}/{self.embedding_model}", exist_ok=True)
            model.save(f"{LOCAL_MODEL_PATH}/{self.embedding_model}")
            print(f"Saved to: {LOCAL_MODEL_PATH}")

        return SentenceTransformer(f"{LOCAL_MODEL_PATH}/{self.embedding_model}")

    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_documents(documents)
        print(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks.")
        return chunks

    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
        texts = [chunk.page_content for chunk in chunks]
        print(f"[INFO] Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"[INFO] Embeddings shape: {embeddings.shape}")
        return embeddings


# Example usage
if __name__ == "__main__":

    docs = load_all_documents("data")
    emb_pipe = EmbeddingPipeline()
    print(emb_pipe)
    chunks = emb_pipe.chunk_documents(docs)
    embeddings = emb_pipe.embed_chunks(chunks)
    print("[INFO] Example embedding:", embeddings[0] if len(embeddings) > 0 else None)
