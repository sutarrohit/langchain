import os
from dotenv import load_dotenv
from src.vector_store import ChromaDBVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


prompt_template = ChatPromptTemplate.from_template("""
   Answer the question based only on the following context:
   
   {context}
   
   Question: {question}
   
   Provide a detailed answer:
""")



class RAGSearch:
    def __init__(
        self,
        collection_name: str = "pdf_documents",
        persist_dir: str = "chromaDB_store",
        embedding_model: str = "all-mpnet-base-v2",
        llm_model: str = "gemini-2.5-flash-lite",
    ):
        self.vectorstore = ChromaDBVectorStore(
            collection_name, persist_dir, embedding_model
        )
        
        self.llm = ChatGoogleGenerativeAI(model=llm_model)
        print(f"[INFO] Google Gemini LLM initialized: {llm_model}")

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        
        results = self.vectorstore.query(query, top_k=top_k)
        texts = [r.get("text", r.get("metadata", {}).get("text", "")) for r in results]
        context = "\n\n".join(texts)
        if not context:
            return "No relevant documents found."
        
        print("context ===================", context)
        print("\n" + "=" * 70)
        messages = prompt_template.format_messages(context=context, question=query)
        response = self.llm.invoke(messages)
        return response.content


if __name__ == "__main__":
    
    print("=" * 70)
    print("RAG Chat — Type 'exit' or 'quit' to stop")
    print("=" * 70)
    
    rag_search = RAGSearch()
        
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
            summary = rag_search.search_and_summarize(query, top_k=6)
            print("\n")
            print(f"Answer: {summary}")
        except Exception as e:
            print(f"\n[Error] {e}")

