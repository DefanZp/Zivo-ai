
import json
import sys

from pathlib import Path
from types import SimpleNamespace

import pytest

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command


# ========================================
# ADD SRC DIRECTORY TO PYTHON PATH
# ========================================

SRC_DIR = Path(__file__).resolve().parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ========================================
# PROJECT IMPORTS
# ========================================

import agents.cart_agent as cart_agent_module

from agents.cart_agent import (
    create_cart_agent_tool,
    create_mcp_add_to_cart_tool,
    create_mcp_get_cart_tool,
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def approve_cart_action(monkeypatch):
    """Simulate user approval for cart mutation tools."""
    monkeypatch.setattr(
        cart_agent_module,
        "interrupt",
        lambda approval: {"decision": "approve"},
    )


class FakeCartAgent:

    def __init__(self, result):
        self.result = result
        self.received_input = None
        self.received_context = None

    async def ainvoke(self, input, context=None):
        self.received_input = input
        self.received_context = context
        return self.result


class FakeMCPTool:

    def __init__(self, result):
        self.result = result
        self.received_input = None

    async def ainvoke(self, input):
        self.received_input = input
        return self.result


def create_runtime(
    selected_product_id=None,
    cart_api_token="test-token",
):
    return SimpleNamespace(
        state={
            "selected_product_id": selected_product_id,
        },
        context={
            "cart_api_token": cart_api_token,
        },
        tool_call_id="test-cart-call",
    )


# ========================================
# CART AGENT WRAPPER TESTS
# ========================================

@pytest.mark.anyio
async def test_cart_agent_receives_state_and_context():

    cart_agent = FakeCartAgent({
        "messages": [
            AIMessage(content="Product added successfully.")
        ],
        "selected_product_id": 134,
    })

    cart_tool = create_cart_agent_tool(cart_agent)

    runtime = create_runtime(selected_product_id=134)

    result = await cart_tool.coroutine(
        task="Add the selected product to the cart.",
        runtime=runtime,
    )

    assert (
        cart_agent.received_input["selected_product_id"] == 134
    )

    assert cart_agent.received_context == runtime.context

    assert isinstance(result, Command)


@pytest.mark.anyio
async def test_cart_agent_returns_response_to_supervisor():

    cart_agent = FakeCartAgent({
        "messages": [
            AIMessage(content="Product added successfully.")
        ],
        "selected_product_id": 134,
    })

    cart_tool = create_cart_agent_tool(cart_agent)

    result = await cart_tool.coroutine(
        task="Add the selected product.",
        runtime=create_runtime(selected_product_id=134),
    )

    assert result.update["selected_product_id"] == 134

    message = result.update["messages"][0]

    assert isinstance(message, ToolMessage)
    assert message.content == "Product added successfully."
    assert message.tool_call_id == "test-cart-call"


# ========================================
# ADD TO CART TESTS
# ========================================

@pytest.mark.anyio
async def test_add_to_cart_passes_product_id_and_token(
    approve_cart_action,
):

    mcp_tool = FakeMCPTool([
        {
            "type": "text",
            "text": '{"success": true, "message": "Product added."}',
        }
    ])

    add_tool = create_mcp_add_to_cart_tool(mcp_tool)

    result = await add_tool.coroutine(
        quantity=2,
        runtime=create_runtime(
            selected_product_id=134,
            cart_api_token="test-token",
        ),
    )

    assert mcp_tool.received_input == {
        "product_id": 134,
        "quantity": 2,
        "cart_api_token": "test-token",
    }

    assert '"success": true' in result


@pytest.mark.anyio
async def test_add_to_cart_rejects_missing_product():

    mcp_tool = FakeMCPTool([])
    add_tool = create_mcp_add_to_cart_tool(mcp_tool)

    result = await add_tool.coroutine(
        quantity=1,
        runtime=create_runtime(selected_product_id=None),
    )

    data = json.loads(result)

    assert data["success"] is False
    assert data["message"] == "No product is currently selected."
    assert mcp_tool.received_input is None


@pytest.mark.anyio
async def test_add_to_cart_requires_authentication():

    mcp_tool = FakeMCPTool([])
    add_tool = create_mcp_add_to_cart_tool(mcp_tool)

    result = await add_tool.coroutine(
        quantity=1,
        runtime=create_runtime(
            selected_product_id=134,
            cart_api_token=None,
        ),
    )

    data = json.loads(result)

    assert data["success"] is False
    assert data["code"] == "AUTH_REQUIRED"
    assert mcp_tool.received_input is None


# ========================================
# GET CART TESTS
# ========================================

@pytest.mark.anyio
async def test_get_cart_requires_authentication():

    mcp_tool = FakeMCPTool([])
    get_cart_tool = create_mcp_get_cart_tool(mcp_tool)

    result = await get_cart_tool.coroutine(
        runtime=create_runtime(cart_api_token=None),
    )

    data = json.loads(result)

    assert data["success"] is False
    assert data["code"] == "AUTH_REQUIRED"
    assert mcp_tool.received_input is None
