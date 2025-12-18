from graph.graph import app
from dotenv import load_dotenv

load_dotenv()


if __name__ == "__main__":
    print("Hello Advanced RAG")
    print(app.invoke(input={"question": "how to make pizza?"}))
