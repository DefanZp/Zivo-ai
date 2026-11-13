
from types import SimpleNamespace

import pytest

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

from agents.product_agent import (
    create_product_agent_tool,
    create_mcp_search_products_tool,
)

@pytest.fixture
def anyio_backend():
    return "asyncio"


# ========================================
# MOCK OBJECTS
# ========================================

class FakeProductAgent:

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


def create_runtime(selected_product_id=None):
    return SimpleNamespace(
        state={
            "selected_product_id": selected_product_id,
        },
        context={
            "cart_api_token": None,
        },
        tool_call_id="test-call-123",
    )


# ========================================
# PRODUCT AGENT TOOL TESTS
# ========================================

@pytest.mark.anyio
async def test_product_agent_receives_supervisor_state():

    product_agent = FakeProductAgent({
        "messages": [
            AIMessage(content="Found the requested product.")
        ],
        "selected_product_id": 42,
    })

    product_tool = create_product_agent_tool(product_agent)

    runtime = create_runtime(selected_product_id=10)

    result = await product_tool.coroutine(
        task="Find a black hoodie.",
        runtime=runtime,
    )

    # Product Agent receives current Supervisor state.
    assert (
        product_agent.received_input["selected_product_id"] == 10
    )

    # Product Agent does not receive unnecessary cart context.
    assert product_agent.received_context is None

    assert isinstance(result, Command)


@pytest.mark.anyio
async def test_product_agent_returns_updated_product_id():

    product_agent = FakeProductAgent({
        "messages": [
            AIMessage(content="Product 42 has been selected.")
        ],
        "selected_product_id": 42,
    })

    product_tool = create_product_agent_tool(product_agent)

    result = await product_tool.coroutine(
        task="Select product 42.",
        runtime=create_runtime(),
    )

    assert result.update["selected_product_id"] == 42

    message = result.update["messages"][0]

    assert isinstance(message, ToolMessage)
    assert message.content == "Product 42 has been selected."
    assert message.tool_call_id == "test-call-123"


@pytest.mark.anyio
async def test_product_agent_can_clear_selected_product():

    product_agent = FakeProductAgent({
        "messages": [
            AIMessage(content="The product selection has been cleared.")
        ],
        "selected_product_id": None,
    })

    product_tool = create_product_agent_tool(product_agent)

    result = await product_tool.coroutine(
        task="Clear the selected product.",
        runtime=create_runtime(selected_product_id=42),
    )

    assert result.update["selected_product_id"] is None


# ========================================
# MCP SEARCH TOOL TESTS
# ========================================

@pytest.mark.anyio
async def test_search_products_clears_previous_selection():

    mcp_tool = FakeMCPTool([
        {
            "type": "text",
            "text": "Found Oversized Black Hoodie.",
        }
    ])

    search_tool = create_mcp_search_products_tool(mcp_tool)

    result = await search_tool.coroutine(
        search="oversized black hoodie",
        runtime=create_runtime(selected_product_id=42),
    )

    assert mcp_tool.received_input == {
        "query": "oversized black hoodie",
        "limit": 5,
    }

    assert isinstance(result, Command)
    assert result.update["selected_product_id"] is None

    message = result.update["messages"][0]

    assert isinstance(message, ToolMessage)
    assert message.content == "Found Oversized Black Hoodie."
