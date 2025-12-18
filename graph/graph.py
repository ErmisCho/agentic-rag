import os
from dotenv import load_dotenv
from pathlib import Path

from langgraph.graph import END, StateGraph
from langchain_core.runnables.graph_mermaid import draw_mermaid_png

from graph.consts import RETRIEVE, GRADE_DOCUMENTS, GENERATE, WEBSEARCH
from graph.nodes import generate, grade_documents, retrieve, web_search
from graph.state import GraphState
from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.router import question_router, RouteQuery
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
import re
import time

load_dotenv()


def decide_to_generate(state: GraphState) -> str:
    print("---ASSESS GRADED DOCUMENTS---")
    if state["web_search"]:
        print("---DECISION: INCLUDE WEB SEARCH---")
        return WEBSEARCH
    print("---DECISION: GENERATE---")
    return GENERATE


def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    print("---CHECK GENERATION QUALITY---")
    question = state["question"]
    documents = state.get("documents", [])
    generation = state["generation"]

    # Convert docs to text for the grader (works whether they are Document objects or strings)
    docs_text = "\n\n".join(
        getattr(d, "page_content", str(d)) for d in documents
    )

    try:
        score = answer_grader.invoke(
            {"question": question, "documents": docs_text, "generation": generation}
        )

    except ChatGoogleGenerativeAIError as e:
        msg = str(e)
        if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
            m = re.search(r"Please retry in ([0-9.]+)s", msg)
            wait_s = float(m.group(1)) + 0.5 if m else 12.5
            print(
                f"---SLEEPING FOR {wait_s} SECONDS DUE TO RESOURCE EXHAUSTION---")
            time.sleep(wait_s)
            score = answer_grader.invoke(
                {"question": question, "documents": docs_text,
                    "generation": generation}
            )
        else:
            raise

    verdict = score.verdict

    if verdict == "useful":
        print("---DECISION: USEFUL---")
        return "useful"

    if verdict == "not_useful":
        print("---DECISION: NOT_USEFUL---")
        return "not_useful"

    # verdict == "not_supported" -> retry
    retry_count = state.get("retry_count", 0) + 1
    state["retry_count"] = retry_count
    print(f"---DECISION: NOT_SUPPORTED (RETRY #{retry_count})---")

    if retry_count >= 2:
        print("---MAX RETRIES REACHED: FALLBACK TO WEBSEARCH---")
        return "not_useful"

    return "not_supported"


def route_question(state: GraphState) -> str:
    print("---ROUTE QUESTION---")
    question = state["question"]
    source: RouteQuery = question_router.invoke({"question": question})
    if source.datasource == WEBSEARCH:
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return WEBSEARCH
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return RETRIEVE


RETRY_GENERATE = "retry_generate"

workflow = StateGraph(GraphState)
workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(RETRY_GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)

workflow.set_conditional_entry_point(route_question,
                                     {
                                         WEBSEARCH: WEBSEARCH,
                                         RETRIEVE: RETRIEVE,
                                     },)

workflow.set_entry_point(RETRIEVE)
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)

workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {WEBSEARCH: WEBSEARCH, GENERATE: GENERATE},
)

workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {
        "useful": END,
        "not_useful": WEBSEARCH,
        "not_supported": RETRY_GENERATE,
    },
)

workflow.add_edge(WEBSEARCH, GENERATE)
# <-- this creates the loop you want
workflow.add_edge(RETRY_GENERATE, GENERATE)

app = workflow.compile()


# Pretty PNG rendering via mermaid.ink (strip HTML that breaks it)
raw = app.get_graph().draw_mermaid()
clean = raw.replace("&nbsp;", " ").replace("<p>", "").replace("</p>", "")
clean = clean.replace("graph TD;", "graph LR;", 1)

# remove existing frontmatter if present
tmp = clean.lstrip()
if tmp.startswith("---"):
    _, _, rest = clean.split("---", 2)
    clean = rest.lstrip("\n")

pretty_header = """---
config:
  flowchart:
    curve: basis
    nodeSpacing: 60
    rankSpacing: 70
---\n"""

mmd = pretty_header + clean
Path("graph_pretty.mmd").write_text(mmd, encoding="utf-8")

draw_mermaid_png(mermaid_syntax=mmd, output_file_path="graph.png")
