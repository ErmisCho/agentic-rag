from typing import Any, Dict, List
from langchain_core.documents import Document

from graph.chains.generation import generation_chain
from graph.state import GraphState

MAX_DOCS = 4
MAX_CHARS_PER_DOC = 1500  # keep prompts small


def _docs_to_context(docs: List[Document]) -> str:
    cleaned: List[str] = []
    for d in docs[:MAX_DOCS]:
        if not isinstance(d, Document):
            continue  # ignore poisoned entries
        src = d.metadata.get("source") or d.metadata.get("url") or ""
        text = (d.page_content or "").strip()
        if not text:
            continue
        text = text[:MAX_CHARS_PER_DOC]
        cleaned.append(f"SOURCE: {src}\n{text}")
    return "\n\n".join(cleaned)


def generate(state: GraphState) -> Dict[str, Any]:
    print("---GENERATE---")

    question = state["question"]
    documents = state.get("documents", [])

    context = _docs_to_context(documents)

    retry_count = state.get("retry_count", 0)
    print("First generation attempt" if retry_count ==
          0 else f"Retry attempt #{retry_count}")

    generation = generation_chain.invoke(
        {
            "context": context,     # IMPORTANT: string, not list[Document]
            "question": question,
            "retry": retry_count > 0,
        }
    )

    return {
        "generation": generation,
        # do NOT re-return question/documents unless you truly need to update them
        # returning them increases chance of concurrent-update and poisoning
    }
