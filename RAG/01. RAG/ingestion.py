import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec

load_dotenv()


if __name__ == "__main__":

    # ── STEP 1: Load ──────────────────────
    print("Ingesting....")
    loader = TextLoader("D:/langchain/rag/mediumblog1.txt")
    document = loader.load()

    # ── STEP 2: Split ─────────────────────
    print("splitting....")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(document)
    print(f"Split into {len(texts)} chunks")

    # ── STEP 3: Embed + Store in Pinecone ─

    # embeddings =GoogleGenerativeAIEmbeddings(
    #    model="models/embedding-001",
    #    google_api_key = os.environ["GOOGLE_API_KEY"]
    # )

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",  # ✅ 768 dims
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # ── STEP 4: Setup Pinecone Index ───────
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]

    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=768,  # ✅ matches BAAI/bge-base-en-v1.5
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print(f"✅ Index created: {INDEX_NAME}")
    else:
        print(f"ℹ️  Using existing index: {INDEX_NAME}")

    # ── STEP 5: Embed + Store ──────────────
    print("Storing in Pinecone....")
    vectorstore = PineconeVectorStore.from_documents(
        documents=texts,
        embedding=embeddings,
        index_name=os.environ["PINECONE_INDEX_NAME"],
    )

    print(f" Done! {len(texts)} chunks stored in Pinecone.")
