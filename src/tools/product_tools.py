import json
import os 

import requests
from dotenv import load_dotenv

from langchain.tools import tool, ToolRuntime
from langchain_core.messages import ToolMessage
from langgraph.types import Command

load_dotenv()

backend_url = os.getenv("BACKEND_URL")

def cart_request(
    method: str,
    path: str,
    runtime: ToolRuntime,
    **kwargs,
):
    context = runtime.context or {}
    cart_api_token = context.get("cart_api_token")

    if (
        not isinstance(cart_api_token, str)
        or not cart_api_token.strip()
    ):  
        return {
            "success": False,
            "code": "AUTH_REQUIRED",
            "message": (
                "The user must log in before using cart features."
            ),
        }

    headers = {
        "Authorization": f"Bearer {cart_api_token}",
        "Accept": "application/json",
    }

    try:
        response = requests.request(
            method=method,
            url=f"{backend_url.rstrip('/')}{path}",
            headers=headers,
            timeout=15,
            **kwargs,
        )

        if response.status_code == 401:
            return {
                "success": False,
                "message": "Authentication failed. The user may need to log in again.",
            }

        if response.status_code == 403:
            return {
                "success": False,
                "code": "FORBIDDEN",
                "message": "The user is not allowed to perform this action.",
            }

        if response.status_code == 404:
            return {
                "success": False,
                "message": "The requested cart item or resource was not found.",
            }

        if response.status_code == 422:
            return {
                "success": False,
                "message": "The request data was rejected by the backend.",
            }

        response.raise_for_status()

    except requests.Timeout:
        return {
            "success": False,
            "message": "The cart request timed out. Its result may need to be checked before retrying.",
        }

    except requests.RequestException:
        return {
            "success": False,
            "message": "The cart service is temporarily unavailable.",
        }

    if not response.content:
        return {
            "success": True,
            "message": "The request completed successfully.",
        }

    try:
        return response.json()

    except ValueError:
        return {
            "success": False,
            "message": "The cart service returned an invalid response.",
        }

@tool
def search_products(
    search: str,
    runtime: ToolRuntime
)->Command:
    """
    Search zivo product by name or keyword.
    
    Normalize user terms to match the Zivo product catalog.
    Example: "hoodie hitam" should be normalized to "black hoodie".

    Every new product search clears the currently selected product
    before the search result is returned.
    """

    try:
        response = requests.get(
            f"{backend_url}/api/ai/products",
            params={"search": search},
            timeout=15,
        )

        response.raise_for_status()

        try:
            data = response.json()["data"]

            if not isinstance(data, list):
                raise ValueError("Invalid product list")

        except (ValueError, KeyError, TypeError):
            data = {
                "success": False,
                "message": "The product service returned an invalid response.",
            }

    except requests.Timeout:
        data = {
            "success": False,
            "message": "Product search timed out. Please try again.",
        }

    except requests.RequestException:
        data = {
            "success": False,
            "message": "The product service is temporarily unavailable.",
        }

    return Command(
        update={
            "selected_product_id": None,
            "messages": [
                ToolMessage(
                    content=json.dumps(data),
                    tool_call_id=runtime.tool_call_id
                )
            ]
        }
    )

@tool
def get_product(product_id: int):
    """
    Get the latest details of a Zivo product by product ID.
    Use the exact product ID returned by search_products.
    """

    try:
        response = requests.get(
            f"{backend_url}/api/products/{product_id}",
            timeout=15,
        )

        response.raise_for_status()

    except requests.Timeout:
        return {
            "success": False,
            "message": "Getting product details timed out. Please try again.",
        }

    except requests.HTTPError as error:
        if error.response is not None and error.response.status_code == 404:
            return {
                "success": False,
                "message": "The requested product was not found.",
            }

        return {
            "success": False,
            "message": "The backend could not retrieve product details.",
        }

    except requests.RequestException:
        return {
            "success": False,
            "message": "The product service is temporarily unavailable.",
        }

    try:
        data = response.json()["data"]

    except (ValueError, KeyError, TypeError):
        return {
            "success": False,
            "message": "The product service returned an invalid response.",
        }

    return data

@tool
def add_to_cart(quantity: int, runtime: ToolRuntime):
    """
    Add a product to the user's cart with the requested quantity.

    Use the selected_product_id from the agent state.
    Use this for adding a new quantity of the selected product.
    Do not use this for increasing an existing cart item by exactly 1.
    """

    product_id = runtime.state.get("selected_product_id")

    if product_id is None:
        return {
            "success": False,
            "message": "No product is currently selected.",
        }

    if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity < 1:
        return {
            "success": False,
            "message": "Quantity must be a positive integer.",
        }

    return cart_request(
        method="POST",
        path="/api/cart/items",
        runtime=runtime,
        json={
            "product_id": product_id,
            "quantity": quantity,
        },
    )

@tool
def get_cart(runtime: ToolRuntime):
    """
    Get the current user's cart contents and quantities.

    Use this when the user asks about the current cart,
    including which products are in the cart or their quantities.
    """

    return cart_request(
        method="GET",
        path="/api/cart",
        runtime=runtime,
    )

@tool 
def increase_quantity(runtime: ToolRuntime):
    """
    Increase the quantity of an existing cart item by exactly 1.

    Use the selected_product_id from the agent state.
    """
    product_id = runtime.state.get("selected_product_id")

    if product_id is None:
        return {
            "success": False,
            "message": "No product is currently selected.",
        }

    return cart_request(
        method="POST",
        path=f"/api/cart/items/{product_id}/increase",
        runtime=runtime,
    )

@tool 
def decrease_quantity(runtime: ToolRuntime):
    """
    Decrease the quantity of an existing cart item by exactly 1.

    Use the selected_product_id from the agent state.
    """

    product_id = runtime.state.get("selected_product_id")

    if product_id is None:
        return {
            "success": False,
            "message": "No product is currently selected.",
        }

    return cart_request(
        method="POST",
        path=f"/api/cart/items/{product_id}/decrease",
        runtime=runtime,
    )

@tool
def remove_from_cart(runtime: ToolRuntime):
    """
    Remove the currently selected product completely from the user's cart.

    This tool accepts no arguments.
    Do not provide a product_id.
    Read the product ID from selected_product_id in the agent state.
    """

    product_id = runtime.state.get("selected_product_id")

    if product_id is None:
        return {
            "success": False,
            "message": "No product is currently selected.",
        }

    return cart_request(
        method="DELETE",
        path=f"/api/cart/items/{product_id}",
        runtime=runtime,
    )

@tool
def clear_cart(runtime: ToolRuntime):
    """
    Remove all items from the user's cart.

    Use this when the user explicitly wants to clear or empty the entire cart.
    """

    return cart_request(
        method="DELETE",
        path="/api/cart",
        runtime=runtime,
    )
@tool
def set_current_product(
    product_id: int,
    runtime: ToolRuntime
)->Command:
    """
    Set the current selected product in the agent state.
    """

    return Command(
        update={
            "selected_product_id": product_id,
            "messages": [
                ToolMessage(
                    content=f"Product {product_id} is now selected.",
                    tool_call_id=runtime.tool_call_id
                )
            ]
        }
    )

@tool 
def clear_current_product(runtime: ToolRuntime)->Command:
    """
    Clear the currently selected product from the agent state.
    This tool accepts no arguments.
    """

    return Command(
        update={
            "selected_product_id": None,
            "messages": [
                ToolMessage(
                    content="The current selected product has been cleared.",
                    tool_call_id=runtime.tool_call_id
                )
            ]
        }
    )