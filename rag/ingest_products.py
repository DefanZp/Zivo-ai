# Project Configuration
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
import requests


RAG_DIR = Path(__file__).resolve().parent
PROJECT_DIR = RAG_DIR.parent

VECTOR_STORE_PATH = RAG_DIR / "chroma_db"
COLLECTION_NAME = "zivo_product_catalog"

load_dotenv(PROJECT_DIR / ".env")

BACKEND_URL = os.getenv("BACKEND_URL")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
EMBEDDINGS_MODEL = os.getenv("OLLAMA_EMBEDDINGS_MODEL")


def validate_config() -> None:
    """Check required environment variables."""
    if not BACKEND_URL:
        raise ValueError("BACKEND_URL has not been configured.")

    if not OLLAMA_BASE_URL:
        raise ValueError("OLLAMA_BASE_URL has not been configured.")

    if not EMBEDDINGS_MODEL:
        raise ValueError(
            "OLLAMA_EMBEDDINGS_MODEL has not been configured."
        )

def fetch_all_product() -> list[dict[str, Any]]:
    """Fetch the complete product catalog without pagination."""
    response = requests.get(
        f"{BACKEND_URL}/api/ai/products",
        timeout=30,
    )
    response.raise_for_status()

    payload = response.json()
    products = payload["data"] if isinstance(payload, dict) else None

    if not isinstance(products, list):
        raise ValueError(
            "The AI API response format in Laravel is incorrect."
        )

    if not products:
        raise ValueError(
            "The catalog is empty. Ingestion has been canceled to prevent "
            "existing vector data from being accidentally deleted."
        )

    return products

def format_specification_value(value: Any) -> str:
    """Convert specification values into readable text."""
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)

    if value is None:
        return ""

    if isinstance(value, bool):
        return "Yes" if value else "No"

    return str(value).strip()


def create_product_document(product: dict[str, Any]) -> Document:
    """Convert a product into a searchable document."""
    product_id = product.get("id")
    name = str(product.get("name") or "").strip()

    if product_id is None or not name:
        raise ValueError(
            "The product has an invalid ID or name."
        )

    category = str(product.get("category") or "Uncategorized")
    description = str(product.get("description") or "").strip()
    specifications = product.get("specifications") or {}

    if not isinstance(specifications, dict):
        raise ValueError(
            f"Specifications produk {product_id} harus berupa objek."
        )

    content = [
        f"Product: {name}",
        f"Category: {category}",
        f"Description: {description}",
    ]

    for key, value in specifications.items():
        formatted_value = format_specification_value(value)

        if formatted_value:
            label = str(key).replace("_", " ")
            content.append(f"{label}: {formatted_value}")

    return Document(
        page_content="\n".join(content),
        metadata={
            "product_id": int(product_id),
            "product_name": name,
            "category": category,
        },
    )

def create_vector_store() -> Chroma:
    """Connect to the product vector collection."""

    embeddings = OllamaEmbeddings(
        model=EMBEDDINGS_MODEL,
        base_url=OLLAMA_BASE_URL,
        client_kwargs={
            "headers": {
                "Authorization": f"Bearer {os.getenv("OLLAMA_API_KEY")}"
            }
        },
    )

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(VECTOR_STORE_PATH),
    )

def ingest_products() -> None:
    """Fetch products and synchronize the Chroma collection."""
    validate_config()

    print("Retrieving the product catalog from Laravel...")
    products = fetch_all_product()

    documents = [
        create_product_document(product)
        for product in products
    ]

    document_ids = [
        f"product-{document.metadata['product_id']}"
        for document in documents
    ]

    vector_store = create_vector_store()

    # Add new documents and update existing product IDs.
    vector_store.add_documents(
        documents=documents,
        ids=document_ids,
    )

    # Remove products that no longer exist in the API catalog.
    current_ids = vector_store.get()['ids']
    valid_ids = set(document_ids)

    stale_ids = [
        product_id
        for product_id in current_ids
        if product_id not in valid_ids
    ]

    if stale_ids:
        vector_store.delete(ids=stale_ids)

    print(f"Number of indexed products:: {len(documents)}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Database: {VECTOR_STORE_PATH}")
    print("Product ingestion completed.")

if __name__ == "__main__":
    ingest_products()