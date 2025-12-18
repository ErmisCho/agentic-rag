from graph.graph import app
from dotenv import load_dotenv

load_dotenv()


if __name__ == "__main__":
    print("Hello Advanced RAG")
    result = app.invoke({"question": "how to make pizza?", "retry_count": 0})
    # result = app.invoke(
    #     {"question": "What is agent memory?", "retry_count": 0})
    print("\nANSWER:\n", result.get("generation"))
