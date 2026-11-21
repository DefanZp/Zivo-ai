"""Focused unit tests for cart mutation approval and rejection behavior.

MCP calls are mocked. These tests do not require Laravel, the MCP server,
Gemini, PostgreSQL, or a real LangGraph checkpoint/resume cycle.

Run from the AI repository root:

    py -m pytest src/test/test_cart_agent_approval.py -v
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest


SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


import agents.cart_agent as cart_agent_module

from agents.cart_agent import (
    create_cart_agent_tool,
    create_mcp_add_to_cart_tool,
    create_mcp_clear_cart_tool,
    create_mcp_decrease_quantity_tool,
    create_mcp_get_cart_tool,
    create_mcp_increase_quantity_tool,
    create_mcp_remove_from_cart_tool,
)
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command


class FakeMCPTool:
    """MCP tool stand-in that records whether a mutation was invoked."""

    def __init__(self, result: str = '{"success": true}'):
        self.ainvoke = AsyncMock(return_value=result)


class FakeCartAgent:
    """Cart Agent stand-in for testing its Supervisor wrapper."""

    def __init__(self, result):
        self.result = result
        self.received_input = None
        self.received_context = None

    async def ainvoke(self, input, context=None):
        self.received_input = input
        self.received_context = context
        return self.result


def make_runtime(*, product_id=134, token="test-cart-token"):
    return SimpleNamespace(
        state={"selected_product_id": product_id},
        context={"cart_api_token": token},
        tool_call_id="supervisor-tool-call-001",
    )


def call_tool(tool, **kwargs):
    """Call the async function wrapped by a LangChain @tool."""
    assert tool.coroutine is not None
    return asyncio.run(tool.coroutine(**kwargs))


@pytest.mark.parametrize(
    ("factory", "tool_kwargs", "expected_approval"),
    [
        pytest.param(
            create_mcp_add_to_cart_tool,
            {
                "quantity": 2,
                "runtime": make_runtime(product_id=134),
            },
            {
                "type": "cart_action_approval",
                "message": "Review this action before adding the product.",
                "action": "add_to_cart",
                "arguments": {"product_id": 134, "quantity": 2},
            },
            id="add-to-cart",
        ),
        pytest.param(
            create_mcp_increase_quantity_tool,
            {"runtime": make_runtime(product_id=134)},
            {
                "type": "cart_action_approval",
                "message": "Review this action before increasing the product quantity.",
                "action": "increase_quantity",
                "arguments": {"product_id": 134, "quantity": 1},
            },
            id="increase-quantity",
        ),
        pytest.param(
            create_mcp_decrease_quantity_tool,
            {"runtime": make_runtime(product_id=134)},
            {
                "type": "cart_action_approval",
                "message": "Review this action before decreasing the product quantity.",
                "action": "decrease_quantity",
                "arguments": {"product_id": 134, "quantity": 1},
            },
            id="decrease-quantity",
        ),
        pytest.param(
            create_mcp_remove_from_cart_tool,
            {"runtime": make_runtime(product_id=134)},
            {
                "type": "cart_action_approval",
                "message": "Review this action before removing the product from your cart.",
                "action": "remove_from_cart",
                "arguments": {"product_id": 134},
            },
            id="remove-from-cart",
        ),
        pytest.param(
            create_mcp_clear_cart_tool,
            {"runtime": make_runtime(product_id=None)},
            {
                "type": "cart_action_approval",
                "message": "Review this action before clearing your entire cart.",
                "action": "clear_cart",
            },
            id="clear-cart",
        ),
    ],
)
def test_rejected_cart_mutation_does_not_call_mcp(
    factory,
    tool_kwargs,
    expected_approval,
    monkeypatch,
):
    """Every cart mutation must stop after a user rejects approval."""
    interrupt_mock = Mock(return_value={"decision": "reject"})
    monkeypatch.setattr(
        cart_agent_module,
        "interrupt",
        interrupt_mock,
    )

    mcp = FakeMCPTool()
    tool = factory(mcp)

    result = call_tool(tool, **tool_kwargs)
    payload = json.loads(result)

    assert payload["success"] is False
    assert payload["code"] == "ACTION_REJECTED"
    assert "No cart change was made" in payload["message"]

    interrupt_mock.assert_called_once_with(expected_approval)
    mcp.ainvoke.assert_not_awaited()


def test_get_cart_is_read_only_and_does_not_request_approval(monkeypatch):
    interrupt_mock = Mock(side_effect=AssertionError(
        "get_cart must not request mutation approval"
    ))
    monkeypatch.setattr(
        cart_agent_module,
        "interrupt",
        interrupt_mock,
    )

    mcp = FakeMCPTool(result='{"items": []}')
    tool = create_mcp_get_cart_tool(mcp)

    result = call_tool(tool, runtime=make_runtime())

    assert result == '{"items": []}'
    mcp.ainvoke.assert_awaited_once_with({
        "cart_api_token": "test-cart-token",
    })
    interrupt_mock.assert_not_called()


def test_cart_agent_marks_rejection_for_supervisor_and_disables_retry():
    rejected_tool_message = ToolMessage(
        content=json.dumps({
            "success": False,
            "code": "ACTION_REJECTED",
            "message": "The user rejected this action.",
        }),
        tool_call_id="cart-agent-tool-call-001",
    )

    cart_agent = FakeCartAgent({
        "messages": [
            rejected_tool_message,
            AIMessage(content="The operation was cancelled."),
        ],
        "selected_product_id": 134,
    })

    cart_agent_tool = create_cart_agent_tool(cart_agent)
    runtime = make_runtime(product_id=134)

    result = call_tool(
        cart_agent_tool,
        task="Add the selected product to the cart.",
        runtime=runtime,
    )

    assert isinstance(result, Command)
    assert cart_agent.received_input["selected_product_id"] == 134
    assert cart_agent.received_context == runtime.context

    supervisor_message = result.update["messages"][0]
    assert isinstance(supervisor_message, ToolMessage)
    assert "ACTION_REJECTED" in supervisor_message.content
    assert "No cart change was made" in supervisor_message.content
    assert "Do not retry this operation" in supervisor_message.content
    assert supervisor_message.tool_call_id == runtime.tool_call_id
