import os
import re
import time
from dotenv import load_dotenv

load_dotenv()


def _retry_sleep_from_msg(msg: str, default: float = 12.5) -> float:
    m = re.search(r"Please retry in ([0-9.]+)s", msg)
    if m:
        return float(m.group(1)) + 0.5
    return default


def invoke_with_429_retry(chain, payload, max_retries: int = 2):
    # Generic wrapper for chain.invoke(...)
    for attempt in range(max_retries + 1):
        try:
            return chain.invoke(payload)
        except Exception as e:
            msg = str(e)
            if "RESOURCE_EXHAUSTED" not in msg and "429" not in msg:
                raise
            if attempt >= max_retries:
                raise
            time.sleep(_retry_sleep_from_msg(msg))


def get_chat_llm(temperature: float = 0.0, max_output_tokens: int | None = None):
    """
    Centralized LLM factory.
    Choose provider via env:
      LLM_PROVIDER=gemini|ollama
    """

    provider = os.getenv("LLM_PROVIDER", "gemini").lower()

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        kwargs = dict(
            google_api_key=os.environ["GEMINI_API_KEY"],
            model=model,
            temperature=temperature,
        )
        if max_output_tokens is not None:
            kwargs["max_output_tokens"] = max_output_tokens
        return ChatGoogleGenerativeAI(**kwargs)

    if provider == "ollama":
        # Requires: pip install langchain-ollama
        from langchain_ollama import ChatOllama

        model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        kwargs = dict(model=model, temperature=temperature)
        # ChatOllama uses num_predict rather than max_output_tokens
        if max_output_tokens is not None:
            kwargs["num_predict"] = max_output_tokens
        return ChatOllama(**kwargs)

    raise ValueError(f"Unknown LLM_PROVIDER: {provider}")
