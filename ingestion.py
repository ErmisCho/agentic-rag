import asyncio
import os
from pathlib import Path
import ssl
from typing import Any, Dict, List
import certifi
from dotenv import load_dotenv
# from langchain_chroma import Chroma
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader

load_dotenv()

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]


docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)

doc_splits = text_splitter.split_documents(docs_list)

embedding_function = GoogleGenerativeAIEmbeddings(
    google_api_key=os.environ["GEMINI_API_KEY"],
    model=os.environ.get("EMBEDDING_MODEL", "text-embedding-004"),
    chunk_size=50,
    retry_min_seconds=10)

vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=embedding_function,
    persist_directory="./.chroma",
)

retriever = Chroma(
    collection_name="rag-chroma",
    persist_directory="./.chroma",
    embedding_function=embedding_function,
).as_retriever()
