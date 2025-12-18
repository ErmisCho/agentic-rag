from graph.llm import get_chat_llm
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv

load_dotenv()


class GradeAnswer(BaseModel):
    grounded: bool = Field(
        description="True if the answer is supported by the provided documents."
    )
    answers_question: bool = Field(
        description="True if the answer addresses/resolves the user question."
    )
    verdict: str = Field(
        description="One of: useful, not_useful, not_supported"
    )
    reason: str = Field(
        description="Short reason (1-2 sentences)."
    )


llm = get_chat_llm(temperature=0.0, max_output_tokens=200)

structured_llm_grader = llm.with_structured_output(GradeAnswer)

system = """You are a strict grader.

You receive:
1) A user question
2) Retrieved documents (facts)
3) A generated answer

Tasks:
A) Decide if the answer is grounded in the documents (no unsupported claims).
B) Decide if the answer answers the question.

Return:
- grounded (bool)
- answers_question (bool)
- verdict:
  - "useful" if grounded=True AND answers_question=True
  - "not_useful" if grounded=True AND answers_question=False
  - "not_supported" if grounded=False
- reason: short justification
"""

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "User question:\n{question}\n\n"
            "Retrieved documents:\n{documents}\n\n"
            "LLM generation:\n{generation}"
        ),
    ]
)

answer_grader: RunnableSequence = answer_prompt | structured_llm_grader
