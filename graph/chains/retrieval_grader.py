from graph.llm import get_chat_llm
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
from typing import List

load_dotenv()

llm = get_chat_llm(temperature=0.0, max_output_tokens=200)


class DocGrade(BaseModel):
    """Grade for a single document."""
    index: int = Field(
        description="0-based index of the document in the input list.")
    relevant: bool = Field(
        description="True if relevant to the user question, else False.")


class GradeDocuments(BaseModel):
    """Grades for all documents."""
    grades: List[DocGrade] = Field(
        description="A grade for every provided document index.")


structured_llm_grader = llm.with_structured_output(GradeDocuments)

system = """You are a strict grader assessing relevance of multiple retrieved documents to a user question.

Rules:
- You MUST return a grade for EVERY document index you receive.
- relevant=True if the document contains keywords OR semantic meaning that helps answer the question.
- relevant=False otherwise.
- Do not skip any indices.
"""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "User question:\n{question}\n\n"
            "Retrieved documents (each has an index):\n{documents}\n\n"
            "Return grades for ALL indices."
        ),
    ]
)

# Keep the same export name so you don't need to change imports elsewhere.
retrieval_grader = grade_prompt | structured_llm_grader
