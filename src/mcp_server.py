import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from mcp.server import MCPServer
import requests

# Folder utama project
PROJECT_DIR = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_DIR / ".env")

backend_url = os.getenv("BACKEND_URL")

mcp = MCPServer("Zivo Product MCP")


# Konfiurasi rag
RAG_DIR = PROJECT_DIR / "rag"
VECTOR_STORE_PATH = RAG_DIR / "chroma_db"

PRODUCT_COLLECTION_NAME = "zivo_product_catalog"
POLICY_COLLECTION_NAME = "zivo_returns_policy"

PRODUCT_SEARCH_INSTRUCTION = (
    "Given a shopping query, retrieve product catalog entries "
    "that best match the requested product name and attributes."
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

product_vector_store = Chroma(
    collection_name=PRODUCT_COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=str(VECTOR_STORE_PATH),
)

policy_vector_store = Chroma(
    collection_name=POLICY_COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=str(VECTOR_STORE_PATH)
)

def cart_request(
    method: str,
    path: str,
    cart_api_token: str,
    **kwargs,
) -> str:
    """Send an authenticated request to the cart API."""

    if not cart_api_token.strip():
        return json.dumps({
            "success": False,
            "code": "AUTH_REQUIRED",
            "message": "Please log in first.",
        })

    try:
        response = requests.request(
            method=method,
            url=f"{backend_url.rstrip('/')}{path}",
            headers={
                "Authorization": f"Bearer {cart_api_token}",
                "Accept": "application/json",
            },
            timeout=15,
            **kwargs,
        )

        if response.status_code == 401:
            return json.dumps({
                "success": False,
                "code": "AUTH_REQUIRED",
                "message": (
                    "Authentication failed. The user may need to log in again."
                ),
            })

        if response.status_code == 403:
            return json.dumps({
                "success": False,
                "code": "FORBIDDEN",
                "message": (
                    "The user is not allowed to perform this action."
                ),
            })

        if response.status_code == 404:
            return json.dumps({
                "success": False,
                "message": (
                    "The requested cart item or resource was not found."
                ),
            })

        if response.status_code == 422:
            return json.dumps({
                "success": False,
                "code": "VALIDATION_ERROR",
                "message": (
                    "The request data was rejected by the backend."
                ),
            })

    except requests.Timeout:
        return json.dumps({
            "success": False,
            "code": "RESULT_UNKNOWN",
            "message": "Request timed out. Check the cart before retrying.",
        })

    except requests.RequestException:
        return json.dumps({
            "success": False,
            "message": "The cart service is temporarily unavailable.",
        })

    if not response.ok:
        return json.dumps({
            "success": False,
            "message": "The backend could not process the request.",
        })

    try:
        data = response.json()

        return json.dumps(data, ensure_ascii=False)

    except ValueError:
        return json.dumps({
            "success": False,
            "message": "The backend returned an invalid response.",
        })
    

@mcp.tool()
def search_products(query: str, limit: int = 5) -> str:
    """
    Search the product catalog by product name, category, or attributes.

    The catalog primarily uses English product names and descriptions.
    Convert the user's product search query into concise English
    search terms before calling this tool.

    Preserve relevant attributes such as color, material, and style.
    Return matching catalog products; never invent product results.
    """

    if not query.strip():
        return json.dumps({
            "success": False,
            "message": "Please provide a product search query."
        })

    if not 1 <= limit <= 10:
        return json.dumps({
            "success": False,
            "message": "Please provide a limit between 1 and 10."
        })

    if not backend_url:
        return json.dumps({
            "success": False,
            "message": "BACKEND_URL is not configured.",
        })

    try:
        response = requests.get(
            f"{backend_url}/api/ai/products",
            params={"search": query},
            timeout=15,
        )

        response.raise_for_status()

    except requests.Timeout:
        return json.dumps({
            "success": False,
            "message": "Product search timed out. Please try again.",
        })

    except requests.RequestException:
        return json.dumps({
            "success": False,
            "message": "The product service is temporarily unavailable.",
        })

    try:
        data = response.json()["data"]

        if not isinstance(data, list):
            raise ValueError("Invalid product list")

    except (ValueError, KeyError, TypeError):
        return json.dumps({
            "success": False,
            "message": "The product service returned an invalid response.",
        })

    # Batasi jumlah produk yang dikembalikan oleh MCP tool.
    products = data[:limit]

    return json.dumps(products, ensure_ascii=False)

@mcp.tool()
def get_product(product_id: int) -> str:
    """Get current product details from Zivo by product ID."""

    if product_id < 1:
        return json.dumps({
            "success": False,
            "message": "Product ID must be a positive integer.",
        })

    if not backend_url:
        return json.dumps({
            "success": False,
            "message": "BACKEND_URL is not configured."
        })

    try:
        response = requests.get(
            url=f"{backend_url}/api/products/{product_id}",
            timeout=15,
        )

        response.raise_for_status()

    except requests.Timeout:
            return json.dumps({
                "success": False,
                "message": "Product search timed out. Please try again.",
            })
    
    except requests.RequestException:
        return json.dumps({
            "success": False,
            "message": "The product service is temporarily unavailable.",
        })

    try:
        data = response.json()['data']

        if not isinstance(data, dict):
            raise ValueError("Invalid product data")

    except (ValueError, KeyError, TypeError):
            return json.dumps({
                "success": False,
                "message": "The product service returned an invalid response.",
            })

    return json.dumps(data, ensure_ascii=False)

@mcp.tool()
def search_product_knowledge(query: str) ->str:
    """
    Search product knowledge by attributes, style, material,
    features, and intended use.
    """

    formatted_query = (
        f"Instruct: {PRODUCT_SEARCH_INSTRUCTION}\n"
        f"Query: {query}"
    )

    documents = product_vector_store.similarity_search(
        query=formatted_query,
        k=3
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

@mcp.tool()
def search_policy_knowledge(query: str) -> str:
    """
    Search Zivo knowledge documents for return policies
    and other documented store policies.

    Do not use this tool for product prices, stock,
    availability, or cart contents.
    """

    if not query:
        return "Please provide a knowledge search query."
    
    documents = policy_vector_store.similarity_search(
        query=query,
        k=2,
    )

    if not documents:
        return "No knowledge documents were found."

    results = []

    for document in documents:
        source = document.metadata.get("source", "Unknown")

        results.append(
            f"Source: {source}\n"
            f"{document.page_content}"
        )

    return "\n\n".join(results)

@mcp.tool()
def get_cart(cart_api_token: str) ->str:
    """Get the authenticated user's current shopping cart."""

    return cart_request(
        method="GET",
        path="/api/cart",
        cart_api_token=cart_api_token,
    )


@mcp.tool()
def add_to_cart(
    product_id: int,
    quantity: int,
    cart_api_token: str,
) -> str:
    """Add a product to the user's cart."""

    if product_id < 1:
        return json.dumps({
            "success": False,
            "message": "Invalid product ID.",
        })

    if quantity < 1:
        return json.dumps({
            "success": False,
            "message": "Quantity must be a positive integer.",
        })

    if not backend_url:
        return json.dumps({
            "success": False,
            "message": "BACKEND_URL is not configured.",
        })

    return cart_request(
        method="POST",
        path="/api/cart/items",
        cart_api_token=cart_api_token,
        json={
            "product_id": product_id,
            "quantity": quantity,
        },
    )

@mcp.tool()
def increase_quantity(
    product_id: int, 
    cart_api_token: str
) -> str:
    """Increase a cart item's quantity by exactly one."""

    if isinstance(product_id, bool) or product_id < 1:
        return json.dumps({
            "success": False,
            "message": "Invalid product ID.",
        })

    if not backend_url:
        return json.dumps({
            "success": False,
            "message": "BACKEND_URL is not configured.",
        })

    return cart_request(
        method="POST",
        path=f"/api/cart/items/{product_id}/increase",
        cart_api_token=cart_api_token,
    )

@mcp.tool()
def decrease_quantity(
    product_id: int,
    cart_api_token: str,
) -> str:
    """Decrease a cart item's quantity by exactly one."""

    if isinstance(product_id, bool) or product_id < 1:
        return json.dumps({
            "success": False,
            "message": "Invalid product ID.",
        })

    if not backend_url:
        return json.dumps({
            "success": False,
            "message": "BACKEND_URL is not configured.",
        })

    return cart_request(
        method="POST",
        path=f"/api/cart/items/{product_id}/decrease",
        cart_api_token=cart_api_token,
    )

@mcp.tool()
def remove_from_cart(
    product_id: int,
    cart_api_token: str,
) -> str:
    """Remove a specific product from the user's cart."""

    if isinstance(product_id, bool) or product_id < 1:
        return json.dumps({
            "success": False,
            "message": "Invalid product ID.",
        })

    return cart_request(
        method="DELETE",
        path=f"/api/cart/items/{product_id}",
        cart_api_token=cart_api_token,
    )


@mcp.tool()
def clear_cart(cart_api_token: str) -> str:
    """Remove all items from the user's cart."""

    return cart_request(
        method="DELETE",
        path="/api/cart",
        cart_api_token=cart_api_token,
    )

        
if __name__ == "__main__":
    mcp.run()