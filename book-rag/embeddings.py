import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any, Tuple


def get_embeddings(LOCAL_MODEL_PATH: str, HF_MODEL_NAME: str):

    if not os.path.exists(LOCAL_MODEL_PATH):
        print(f"Downloading '{HF_MODEL_NAME}'...")
        model = SentenceTransformer(HF_MODEL_NAME)
        os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)
        model.save(LOCAL_MODEL_PATH)
        print(f"Saved to: {LOCAL_MODEL_PATH}")

    return HuggingFaceEmbeddings(
        model_name=LOCAL_MODEL_PATH,  # ← point to local path
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


class EmbeddingManager:
    """Handles document embedding generation using SentenceTransformers"""

    def __init__(
        self,
        local_model_path: str = "./models",
        model_name: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize the embedding model

        Args:
            model_name: HuggingFace model name for sentence embedding.
        """

        self.model_name = model_name
        self.model = None
        self.local_model_path = os.path.join(local_model_path, model_name.lower())
        self._load_model()

    def _load_model(self):
        print(f"Loading embedding model {self.model_name}")
        try:
            if not os.path.exists(self.local_model_path):
                print(f"Downloading '{self.model_name}'...")

                model = SentenceTransformer(self.model_name)
                os.makedirs(self.local_model_path, exist_ok=True)
                model.save(self.local_model_path)

                print(f"Saved to: {self.local_model_path}")

            self.model = SentenceTransformer(self.local_model_path)

        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")

    def generate_embedding(self, texts: List[str]) -> np.ndarray:
        """
         Generate embedding for list of text

        Args:
            text: List of text strings to embed

         Returns:
            numpy array of embeddings with shape (len(texts), embedding_dim)
        """

        if not self.model:
            raise ValueError("Model Not Found")

        print(f"Generating embeddings for {len(texts)} text....")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"Generated embedding with shape {embeddings.shape}")
        return embeddings
