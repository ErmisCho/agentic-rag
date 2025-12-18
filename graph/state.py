from typing import List, TypedDict, Annotated
from langchain_core.documents import Document
import operator


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """

    question: str
    generation: str
    web_search: bool
    documents: Annotated[List[Document], operator.add]
    retry_count: int
