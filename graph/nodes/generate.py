from typing import Any, Dict
from graph.chains.generation import generation_chain
from graph.state import GraphState


def generate(state: GraphState) -> Dict[str, Any]:
    print("---GENERATE---")

    question = state["question"]
    documents = state["documents"]

    retry_count = state.get("retry_count", 0)

    if retry_count > 0:
        print(f"Retry attempt #{retry_count}")
        generation = generation_chain.invoke(
            {
                "context": documents,
                "question": question,
                "retry": True,  # pass signal to the chain
            }
        )
    else:
        print("First generation attempt")
        generation = generation_chain.invoke(
            {
                "context": documents,
                "question": question,
                "retry": False,
            }
        )

    return {
        "documents": documents,
        "question": question,
        "generation": generation,
        "retry_count": retry_count,
    }
