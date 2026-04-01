import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from embeddings import get_embeddings

load_dotenv()

# ── Local model path ───────────────────────
LOCAL_MODEL_PATH = "D:/langchain/book-rag/models/bge-base-en-v1.5"
HF_MODEL_NAME = "BAAI/bge-base-en-v1.5"
CHROMA_DB_PATH = "D:/langchain/book-rag/chroma_db"  # ← persisted here
COLLECTION_NAME = "book_tsz"

print("Initializing Components...........")

# ── STEP 1: LLM ───────────────────────────────────
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)

# ── STEP 2: Embedding ──────────────────────────────
embeddings = get_embeddings(LOCAL_MODEL_PATH, HF_MODEL_NAME)

# ── STEP 3: Vector Store ──────────────────────────
vector_store = Chroma(
    collection_name="book_tsz",
    embedding_function=embeddings,
    persist_directory="D:/langchain/book-rag/chroma_db",
)

# ── STEP 4: vector retriever ───────────────────────
retriever = vector_store.as_retriever(
    search_type="mmr", search_kwargs={"k": 10, "fetch_k": 20, "lambda_mult": 0.7}
)


# ── STEP 5: Prompt Template ───────────────────────
prompt_template = ChatPromptTemplate.from_template("""
   Answer the question based only on the following context:

   {context}

   Question: {question}

   Provide a detailed answer:
""")


def format_docs(docs):
    """Format retrieved documents in to single string"""
    return "\n\n".join(doc.page_content for doc in docs)


# ============================================================================
# IMPLEMENTATION 1: Without LCEL (Simple Function-Based Approach)
# ============================================================================
def retrieval_chain_without_lcel(query: str):
    """
    Simple retrieval chain without Langchain Expression Language
    Manually retrieves documents, format them and generates a response.

    Limitations:
      - Manual step-by-step execution
      - No built-in streaming support
      - No async support without additional code
      - Harder to compose with other chains
      - More verbose and error-prone
    """

    # Step 1: Retrieve relevant documents
    docs = retriever.invoke(query)

    # Step 2: Format documents into context string
    context = format_docs(docs)

    # Step 3: Format the prompt with context and question
    messages = prompt_template.format_messages(context=context, question=query)

    # Step 4: Invoke LLM with the formatted messages
    response = llm.invoke(messages)

    # Step 5: Return the content
    return response.content


if __name__ == "__main__":

    print("=" * 70)
    print("RAG Chat — Type 'exit' or 'quit' to stop")
    print("=" * 70)

    while True:
        # ── Get user input ─────────────────────────
        try:
            query = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break

        # ── Handle exit commands ───────────────────
        if not query:
            print("(empty input, try again)")
            continue

        if query.lower() in {"exit", "quit", "q", "bye"}:
            print("Goodbye!")
            break

        # ── Run retrieval chain ────────────────────
        print("\nThinking...", end="\r")
        try:
            result = retrieval_chain_without_lcel(query)
            print("\n")
            print(f"Answer: {result}")
        except Exception as e:
            print(f"\n[Error] {e}")
