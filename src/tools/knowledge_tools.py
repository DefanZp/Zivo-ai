import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# Locate the main ai directory 
PROJECT_DIR = Path(__file__).resolve().parents[2]

VECTOR_STORE_PATH = PROJECT_DIR / "rag_learning" / "chroma_db"

load_dotenv(PROJECT_DIR / ".env")


if not VECTOR_STORE_PATH.exists():
    raise FileNotFoundError(
        f"Chroma vector store tidak ditemukan: {VECTOR_STORE_PATH}"
    )

embeddings = OllamaEmbeddings(
    model=os.getenv("OLLAMA_EMBEDDINGS_MODEL"),
    base_url=os.getenv("OLLAMA_BASE_URL"),
    client_kwargs={
        "headers": {
            "Authorization": f"Bearer {os.getenv('OLLAMA_API_KEY')}"
        }
    },
)

vector_store = Chroma(
    collection_name="zivo_returns_policy",
    embedding_function=embeddings,
    persist_directory=str(VECTOR_STORE_PATH),
)


@tool
def search_knowledge(query: str) -> str:
    """
    Search Zivo knowledge documents for information about
    return policies and other documented store policies.

    Use this tool for policy questions, not for current
    product prices, stock, availability, or cart contents.
    Those must come from the relevant Laravel API tools.
    """

    documents = vector_store.similarity_search(
        query=query,
        k=2,
    )

    if not documents:
        return "No knowledge documents were found."

    results = []

    for document in documents:
        source = document.metadata.get("source", "Unknown")

        result = (
            f"Source: {source}\n"
            f"Content:\n{document.page_content}"
        )

        results.append(result)

    return "\n\n".join(results)