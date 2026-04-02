import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List


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
