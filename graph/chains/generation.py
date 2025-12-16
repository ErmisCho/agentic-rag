import os
from langchain_classic import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    google_api_key=os.environ["GEMINI_API_KEY"], model="gemini-2.5-flash", temperature=0)
prompt = hub.pull('rlm/rag-prompt')

generation_chain = prompt | llm | StrOutputParser()
