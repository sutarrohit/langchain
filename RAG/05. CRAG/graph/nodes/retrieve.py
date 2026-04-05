from typing import Any, Dict
from langchain_core.documents import Document

from ingestion_pipeline.vector_store import ChromaDBVectorStore

from graph.state import GraphState



def retrieve(state: GraphState):
    print("---- RETRIEVE ----")
    question = state["question"]

    vector_store = ChromaDBVectorStore(
        collection_name="corrective-rag",
        chunk_size=500,
        chunk_overlap=100,
    )

    results = vector_store.query(question)
    documents = [Document(page_content=r["text"], metadata=r.get("metadata", {})) for r in results]
    return {"documents": documents, "question": question}
