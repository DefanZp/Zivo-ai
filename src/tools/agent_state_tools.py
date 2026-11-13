from langchain.tools import tool, ToolRuntime
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from state import ZivoAgentContext


@tool
def set_current_product(
    product_id: int,
    runtime: ToolRuntime[ZivoAgentContext],
) -> Command:
    """Save a verified product as the current selection."""

    if (
        isinstance(product_id, bool)
        or not isinstance(product_id, int)
        or product_id < 1
    ):
        raise ValueError("Product ID must be a positive integer.")

    return Command(
        update={
            "selected_product_id": product_id,
            "messages": [
                ToolMessage(
                    content=f"Product {product_id} is now selected.",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool
def clear_current_product(
    runtime: ToolRuntime[ZivoAgentContext],
) -> Command:
    """
    Clear the current product selection when the user
    explicitly cancels or discards the selected product.

    New product searches clear the selection automatically.
    """

    return Command(
        update={
            "selected_product_id": None,
            "messages": [
                ToolMessage(
                    content="The current selected product has been cleared.",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )