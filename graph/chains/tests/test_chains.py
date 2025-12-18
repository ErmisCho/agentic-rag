from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from graph.chains.hallucination_grader import hallucination_grader, GradeHallucinations
from ingestion import retriever
from dotenv import load_dotenv
from graph.chains.generation import generation_chain
from pprint import pprint
from graph.chains.router import question_router, RouteQuery

load_dotenv()


def test_retrieval_grader_answer_yes() -> None:
    question = 'agent memory'
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "documents": f"[0] {doc_txt}"}
    )

    assert res.grades[0].index == 0
    assert res.grades[0].relevant is True


def test_retrieval_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": "how to make pizza", "documents": f"[0] {doc_txt}"}
    )

    assert res.grades[0].index == 0
    assert res.grades[0].relevant is False


def test_generation_chain() -> None:
    question = 'agent memory'
    docs = retriever.invoke(question)
    generation = generation_chain.invoke(
        {"context": docs, "question": question})
    pprint(generation)


def test_hallucination_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)

    generation = generation_chain.invoke(
        {"context": docs, "question": question})
    res: GradeHallucinations = hallucination_grader.invoke(
        {"documents": docs, "generation": generation}
    )
    assert res.binary_score


def test_hallucination_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)

    res: GradeHallucinations = hallucination_grader.invoke(
        {
            "documents": docs,
            "generation": "You can't make an omelette without breaking eggs",
        }
    )
    assert not res.binary_score


def test_router_to_vectorstore() -> None:
    question = "agent_memory"
    res: RouteQuery = question_router.invoke({"question": question})
    assert res.datasource == "vectorstore"


def test_router_to_websearch() -> None:
    question = "how to make pizza"

    res: RouteQuery = question_router.invoke({"question": question})
    assert res.datasource == "websearch"
