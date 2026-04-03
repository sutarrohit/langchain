from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from embeddings import get_embeddings

load_dotenv()

# ── Local model path ───────────────────────
LOCAL_MODEL_PATH = "D:/langchain/book-rag/models/bge-base-en-v1.5"
HF_MODEL_NAME = "BAAI/bge-base-en-v1.5"
CHROMA_DB_PATH = "D:/langchain/book-rag/chroma_db"  # ← persisted here
COLLECTION_NAME = "book_tsz"


if __name__ == "__main__":

    # ── STEP 1: Load ──────────────────────────────────
    print("Loading PDF...")
    loader = PyPDFLoader(r"D:/langchain/book-rag/book/tsz.pdf")
    document = loader.load()
    print(f"Loaded {len(document)} pages")

    # Sanity check — print first 3 pages
    for page in document[:3]:
        print(f"--- Page {page.metadata['page']} ---")
        print(page.page_content[:500])
        print()

    # ── STEP 2: Split ─────────────────────────────────
    print("Splitting...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=150, separators=["\n\n", "\n", " ", ""]
    )
    texts = splitter.split_documents(document)
    print(f"Split into {len(texts)} chunks")

    # Extract just the text content
    plain_texts = [doc.page_content for doc in texts]

    # ── STEP 3: Embed + Store in Chroma ───────────────
    print("Embedding & storing in Chroma...")
    embeddings = get_embeddings(LOCAL_MODEL_PATH, HF_MODEL_NAME)

    vector_store = Chroma.from_texts(
        texts=plain_texts,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DB_PATH,  # ← saves to disk automatically
    )

    print(
        f"Done! {vector_store._collection.count()} vectors stored at: {CHROMA_DB_PATH}"
    )
