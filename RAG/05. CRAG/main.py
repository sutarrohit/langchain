from dotenv import load_dotenv

load_dotenv()

from graph.graph import app


def main():
    print("Hello Advanced RAG")
    print(app.invoke(input={"question": "agent memory?"}))


if __name__ == "__main__":
    main()
