import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# Project Configuration
PROJECT_DIR = Path(__file__).resolve().parents[2]
RAG_DIR = PROJECT_DIR / "rag"
VECTOR_STORE_PATH = RAG_DIR / "chroma_db"

COLLECTION_NAME = "zivo_product_catalog"

PRODUCT_SEARCH_INSTRUCTION = (
    "Given a shopping query, retrieve product catalog entries "
    "that best match the requested product name and attributes."
)

load_dotenv(PROJECT_DIR / ".env")

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
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=str(VECTOR_STORE_PATH),
)

@tool 
def search_product_knowledge(query: str) -> str:
    """
    Search product knowledge for shopping recommendations.

    Use this tool to find products based on their documented
    characteristics, intended use, style, material, or other
    relevant product information.

    This tool does not provide authoritative current prices
    or stock quantities. Use Laravel product tools to verify
    those details.
    """

    formatted_query = (
        f"Instruct: {PRODUCT_SEARCH_INSTRUCTION}\n"
        f"Query: {query}"
    )

    documents = vector_store.similarity_search(
        query=formatted_query,
        k=3,
    )

    if not documents:
        return "No relevant product knowledge was found."

    results = []

    for document in documents:
        product_id = document.metadata.get("product_id")

        results.append(
            f"Product ID: {product_id}\n"
            f"{document.page_content}"
        )

    return "\n\n---\n\n".join(results)