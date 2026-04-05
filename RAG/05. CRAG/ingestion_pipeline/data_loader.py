from langchain_community.document_loaders import WebBaseLoader

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]


def load_all_documents(urls: list[str]):
    """
    Load web pages and convert to LangChain document structure.
    """
    documents = []

    print(f"[DEBUG] Found {len(urls)} URLs")

    for url in urls:
        print(f"[DEBUG] Loading URL: {url}")
        try:
            loader = WebBaseLoader(url)
            loaded = loader.load()

            print(f"[DEBUG] Loaded {len(loaded)} docs from {url}")
            documents.extend(loaded)

        except Exception as e:
            print(f"[ERROR] Failed to load URL: {url}, Error: {e}")

    return documents


# Example usage
if __name__ == "__main__":
    docs = load_all_documents(urls)

    print(f"Loaded {len(docs)} documents.")

    if docs:
        print("\nExample document content:\n", docs[0].page_content[:500])
        print("\nExample document metadata:\n", docs[0].metadata)
