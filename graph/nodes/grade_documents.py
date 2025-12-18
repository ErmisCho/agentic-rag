from typing import Any, Dict

from graph.chains.retrieval_grader import retrieval_grader
from graph.state import GraphState

MAX_DOCS_TO_KEEP = 4
MAX_CHARS_PER_DOC = 2500  # avoids huge prompts / token blowups


def _format_docs_for_grading(documents) -> str:
    parts = []
    for i, d in enumerate(documents):
        content = getattr(d, "page_content", str(d)).strip()
        if len(content) > MAX_CHARS_PER_DOC:
            content = content[:MAX_CHARS_PER_DOC] + "…"
        parts.append(f"[{i}] {content}")
    return "\n\n".join(parts)


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Grades ALL retrieved documents in one LLM call.
    Filters out irrelevant documents and sets web_search if none remain.
    """
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION (BATCH)---")
    question = state["question"]
    documents = state["documents"]

    if not documents:
        return {"documents": [], "web_search": True}

    docs_blob = _format_docs_for_grading(documents)

    result = retrieval_grader.invoke(
        {"question": question, "documents": docs_blob})

    # Map: index -> relevant
    relevant_map = {g.index: bool(g.relevant) for g in result.grades}

    filtered_docs = []
    for i, d in enumerate(documents):
        if relevant_map.get(i, False):
            print(f"---DOC {i}: RELEVANT---")
            filtered_docs.append(d)
        else:
            print(f"---DOC {i}: NOT RELEVANT---")

    filtered_docs = filtered_docs[:MAX_DOCS_TO_KEEP]

    web_search = len(filtered_docs) == 0
    return {"documents": filtered_docs, "web_search": web_search}
