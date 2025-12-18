from graph.llm import get_chat_llm
import os
from langchain_classic import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = get_chat_llm(temperature=0.0, max_output_tokens=200)
base_prompt = hub.pull("rlm/rag-prompt")

# Wrap the original prompt output with hard constraints
short_wrapper = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a concise RAG assistant.\n"
            "Rules:\n"
            "- Answer in 2-4 sentences.\n"
            "- Max 80 words.\n"
            "- No bullet points, no numbered lists.\n"
            "- No preamble (no 'Sure', no 'Here is').\n"
            "- If the context does not support an answer, say: "
            "'I don't know based on the provided context.'\n",
        ),
        ("human", "{input}"),
    ]
)

generation_chain = base_prompt | short_wrapper | llm | StrOutputParser()
