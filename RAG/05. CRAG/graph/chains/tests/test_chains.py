from pprint import pprint

from dotenv import load_dotenv

load_dotenv()


from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from ingestion import vector_store
from graph.chains.generation import generation_chain


def test_retrival_grader_answer_yes() -> None:
    question = "agent memory"
    docs = vector_store.query(question)
    doc_txt = docs[1]["text"]

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_txt}
    )

    assert res.binary_score == "yes"


def test_retrival_grader_answer_no() -> None:
    question = "agent memory"
    docs = vector_store.query(question)
    doc_txt = docs[1]["text"]

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": "how to make pizaa", "document": doc_txt}
    )

    assert res.binary_score == "no"


def test_generation_chain() -> None:
    question = "agent memory"
    docs = vector_store.query(question)
    generation = generation_chain.invoke({"context": docs, "question": question})
    pprint(generation)
