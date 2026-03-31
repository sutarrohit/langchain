import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

load_dotenv()

# ── Local model path ───────────────────────
LOCAL_MODEL_PATH = "D:/langchain/rag/models/bge-base-en-v1.5"
HF_MODEL_NAME = "BAAI/bge-base-en-v1.5"

print("Initializing Components...........")

# ── STEP 1: Download model if not exists locally ───
if os.path.exists(LOCAL_MODEL_PATH):
    print(f"Loading embedding model from local path: {LOCAL_MODEL_PATH}")
else:
    print(f"Model not found locally. Downloading '{HF_MODEL_NAME}'...")
    model = SentenceTransformer(HF_MODEL_NAME)
    os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)  # create folder if not exists
    model.save(LOCAL_MODEL_PATH)
    print(f"Model downloaded and saved to: {LOCAL_MODEL_PATH}")

# ── STEP 2: Load embeddings from local path ────────
embeddings = HuggingFaceEmbeddings(
    model_name=LOCAL_MODEL_PATH,  # Always uses local path
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# ── STEP 3: LLM ───────────────────────────────────
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)

# ── STEP 4: Vector Store ──────────────────────────
vector_store = PineconeVectorStore(
    embedding=embeddings,
    index_name=os.environ["PINECONE_INDEX_NAME"],
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

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


# ============================================================================
# IMPLEMENTATION 2: With LCEL (LangChain Expression Language) - BETTER APPROACH
# ============================================================================


def create_retrieval_chain_with_lcel():
    """
    Create a retrieval chain using LCEL (LangChain Expression Language).
    Returns a chain that can be invoked with {"question": "..."}

    Advantages over non-LCEL approach:
    - Declarative and composable: Easy to chain operations with pipe operator (|)
    - Built-in streaming: chain.stream() works out of the box
    - Built-in async: chain.ainvoke() and chain.astream() available
    - Batch processing: chain.batch() for multiple inputs
    - Type safety: Better integration with LangChain's type system
    - Less code: More concise and readable
    - Reusable: Chain can be saved, shared, and composed with other chains
    - Better debugging: LangChain provides better observability tools
    """
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return retrieval_chain


if __name__ == "__main__":

    print("=" * 70)
    query = "what is pinecone in machine learning"
    print("\n\n" + "query:")
    print(query)

    # ========================================================================
    # Option 1: Use implementation WITHOUT LCEL
    # ========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 1: Without LCEL")
    print("=" * 70)
    result_without_lcel = retrieval_chain_without_lcel(query)
    print("\nAnswer ==============> :")
    print(result_without_lcel)

    # ========================================================================
    # Option 2: Use implementation WITH LCEL (Better Approach)
    # ========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 2: With LCEL - Better Approach")
    print("=" * 70)
    print("Why LCEL is better:")
    print("- More concise and declarative")
    print("- Built-in streaming: chain.stream()")
    print("- Built-in async: chain.ainvoke()")
    print("- Easy to compose with other chains")
    print("- Better for production use")
    print("=" * 70)

    chain_with_lcel = create_retrieval_chain_with_lcel()
    result_with_lcel = chain_with_lcel.invoke({"question": query})
    print("\nAnswer:")
    print(result_with_lcel)
