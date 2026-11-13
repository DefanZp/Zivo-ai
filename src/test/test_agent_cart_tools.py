"""Unit tests for MCP wrappers used by Zivo agents.

MCP calls are mocked, so no Laravel API, MCP server,
Gemini API, or PostgreSQL connection is required.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


# The project keeps agent.py inside src/. Add that directory so this test can
# be executed from the repository root or from another working directory.
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if not SRC_DIR.is_dir():
    # When this file is copied to src/test/, parents[1] is the src directory.
    SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agents.cart_agent import (
    create_mcp_add_to_cart_tool,
    create_mcp_clear_cart_tool,
    create_mcp_decrease_quantity_tool,
    create_mcp_get_cart_tool,
    create_mcp_increase_quantity_tool,
    create_mcp_remove_from_cart_tool,
)

from agents.product_agent import (
    create_mcp_search_products_tool,
)


class FakeMCPTool:
    """Small stand-in for a discovered MCP tool."""

    def __init__(self, result=None):
        self.ainvoke = AsyncMock(return_value=result)


def make_runtime(
    *,
    product_id=42,
    token="test-cart-token",
    tool_call_id="test-tool-call-001",
):
    """Create the subset of ToolRuntime used by the wrapper functions."""
    state = {"selected_product_id": product_id} if product_id is not ... else {}
    context = {"cart_api_token": token} if token is not ... else {}

    return SimpleNamespace(
        state=state,
        context=context,
        tool_call_id=tool_call_id,
    )


def call_tool(tool, **kwargs):
    """Execute the underlying async function created by @tool."""
    assert tool.coroutine is not None, "Expected an asynchronous LangChain tool"
    return asyncio.run(tool.coroutine(**kwargs))


def decode_result(result):
    """Decode JSON string results while leaving ordinary text untouched."""
    if isinstance(result, str):
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return result
    return result


# ---------------------------------------------------------------------------
# increase_quantity / decrease_quantity
# ---------------------------------------------------------------------------


def test_increase_quantity_uses_selected_product_and_awaits_mcp():
    mcp = FakeMCPTool(result='{"success": true}')
    tool = create_mcp_increase_quantity_tool(mcp)

    result = call_tool(tool, runtime=make_runtime(product_id=42))

    assert result == '{"success": true}'
    mcp.ainvoke.assert_awaited_once_with({
        "product_id": 42,
        "cart_api_token": "test-cart-token",
    })
    assert "product_id" not in tool.args
    assert "runtime" not in tool.args


@pytest.mark.parametrize("product_id", [None, 0, -1, True, "42"])
def test_increase_quantity_rejects_invalid_selected_product(product_id):
    mcp = FakeMCPTool(result="should not be called")
    tool = create_mcp_increase_quantity_tool(mcp)

    result = call_tool(tool, runtime=make_runtime(product_id=product_id))
    payload = decode_result(result)

    assert payload["success"] is False
    assert "selected" in payload["message"].lower()
    mcp.ainvoke.assert_not_awaited()


def test_increase_quantity_requires_authentication():
    mcp = FakeMCPTool(result="should not be called")
    tool = create_mcp_increase_quantity_tool(mcp)

    result = call_tool(tool, runtime=make_runtime(token=None))
    payload = decode_result(result)

    assert payload["success"] is False
    assert payload["code"] == "AUTH_REQUIRED"
    mcp.ainvoke.assert_not_awaited()


def test_decrease_quantity_uses_selected_product_and_awaits_mcp():
    mcp = FakeMCPTool(result='{"success": true}')
    tool = create_mcp_decrease_quantity_tool(mcp)

    result = call_tool(tool, runtime=make_runtime(product_id=73))

    assert result == '{"success": true}'
    mcp.ainvoke.assert_awaited_once_with({
        "product_id": 73,
        "cart_api_token": "test-cart-token",
    })
    assert "product_id" not in tool.args
    assert "runtime" not in tool.args


@pytest.mark.parametrize("product_id", [None, 0, -1, True, "73"])
def test_decrease_quantity_rejects_invalid_selected_product(product_id):
    mcp = FakeMCPTool(result="should not be called")
    tool = create_mcp_decrease_quantity_tool(mcp)

    result = call_tool(tool, runtime=make_runtime(product_id=product_id))
    payload = decode_result(result)

    assert payload["success"] is False
    assert "selected" in payload["message"].lower()
    mcp.ainvoke.assert_not_awaited()


@pytest.mark.parametrize(
    "factory",
    [
        create_mcp_increase_quantity_tool,
        create_mcp_decrease_quantity_tool,
    ],
)
def test_quantity_change_requires_non_empty_token(factory):
    mcp = FakeMCPTool(result="should not be called")
    tool = factory(mcp)

    result = call_tool(tool, runtime=make_runtime(token="   "))
    payload = decode_result(result)

    assert payload["success"] is False
    assert payload["code"] == "AUTH_REQUIRED"
    mcp.ainvoke.assert_not_awaited()


# ---------------------------------------------------------------------------
# add_to_cart
# ---------------------------------------------------------------------------


def test_add_to_cart_uses_selected_product_quantity_and_token():
    mcp = FakeMCPTool(result='{"success": true}')
    tool = create_mcp_add_to_cart_tool(mcp)

    result = call_tool(
        tool,
        quantity=3,
        runtime=make_runtime(product_id=19),
    )

    assert result == '{"success": true}'
    mcp.ainvoke.assert_awaited_once_with({
        "product_id": 19,
        "quantity": 3,
        "cart_api_token": "test-cart-token",
    })
    assert set(tool.args) == {"quantity"}
    assert "runtime" not in tool.args


@pytest.mark.parametrize("quantity", [0, -1, True, "2", 2.5, None])
def test_add_to_cart_rejects_invalid_quantity(quantity):
    mcp = FakeMCPTool(result="should not be called")
    tool = create_mcp_add_to_cart_tool(mcp)

    result = call_tool(
        tool,
        quantity=quantity,
        runtime=make_runtime(product_id=19),
    )
    payload = decode_result(result)

    assert payload["success"] is False
    assert "quantity" in payload["message"].lower()
    mcp.ainvoke.assert_not_awaited()


@pytest.mark.parametrize("product_id", [None, 0, -1, True, "19"])
def test_add_to_cart_rejects_invalid_selected_product(product_id):
    mcp = FakeMCPTool(result="should not be called")
    tool = create_mcp_add_to_cart_tool(mcp)

    result = call_tool(
        tool,
        quantity=1,
        runtime=make_runtime(product_id=product_id),
    )
    payload = decode_result(result)

    assert payload["success"] is False
    mcp.ainvoke.assert_not_awaited()


def test_add_to_cart_requires_authentication():
    mcp = FakeMCPTool(result="should not be called")
    tool = create_mcp_add_to_cart_tool(mcp)

    result = call_tool(
        tool,
        quantity=1,
        runtime=make_runtime(token=""),
    )
    payload = decode_result(result)

    assert payload["success"] is False
    assert payload["code"] == "AUTH_REQUIRED"
    mcp.ainvoke.assert_not_awaited()


# ---------------------------------------------------------------------------
# remove_from_cart / get_cart / clear_cart
# ---------------------------------------------------------------------------


def test_remove_from_cart_uses_selected_product_and_token():
    mcp = FakeMCPTool(result='{"success": true}')
    tool = create_mcp_remove_from_cart_tool(mcp)

    result = call_tool(tool, runtime=make_runtime(product_id=88))

    assert result == '{"success": true}'
    mcp.ainvoke.assert_awaited_once_with({
        "product_id": 88,
        "cart_api_token": "test-cart-token",
    })
    assert "product_id" not in tool.args


def test_get_cart_requires_authentication_and_does_not_call_mcp():
    mcp = FakeMCPTool(result="should not be called")
    tool = create_mcp_get_cart_tool(mcp)

    result = call_tool(tool, runtime=make_runtime(token=None))
    payload = decode_result(result)

    assert payload["success"] is False
    assert payload["code"] == "AUTH_REQUIRED"
    mcp.ainvoke.assert_not_awaited()


def test_get_cart_calls_mcp_and_returns_text_result():
    mcp = FakeMCPTool(result=[
        {"type": "text", "text": '{"items": []}'},
    ])
    tool = create_mcp_get_cart_tool(mcp)

    result = call_tool(tool, runtime=make_runtime())

    assert result == '{"items": []}'
    mcp.ainvoke.assert_awaited_once_with({
        "cart_api_token": "test-cart-token",
    })


def test_clear_cart_calls_mcp_with_token():
    mcp = FakeMCPTool(result='{"success": true}')
    tool = create_mcp_clear_cart_tool(mcp)

    result = call_tool(tool, runtime=make_runtime())

    assert result == '{"success": true}'
    mcp.ainvoke.assert_awaited_once_with({
        "cart_api_token": "test-cart-token",
    })


def test_clear_cart_requires_authentication():
    mcp = FakeMCPTool(result="should not be called")
    tool = create_mcp_clear_cart_tool(mcp)

    result = call_tool(tool, runtime=make_runtime(token=None))
    payload = decode_result(result)

    assert payload["success"] is False
    assert payload["code"] == "AUTH_REQUIRED"
    mcp.ainvoke.assert_not_awaited()


# ---------------------------------------------------------------------------
# MCP response handling and failure behavior
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mcp_result", "expected"),
    [
        ('{"success": true}', '{"success": true}'),
        ([{"type": "text", "text": "first"}, {"type": "text", "text": "second"}], "first\nsecond"),
        ({"success": True}, '{"success": true}'),
    ],
)
def test_increase_quantity_normalizes_mcp_result(mcp_result, expected):
    mcp = FakeMCPTool(result=mcp_result)
    tool = create_mcp_increase_quantity_tool(mcp)

    result = call_tool(tool, runtime=make_runtime())

    assert result == expected
    mcp.ainvoke.assert_awaited_once()


def test_mcp_exception_is_not_silently_reported_as_success():
    mcp = FakeMCPTool()
    mcp.ainvoke.side_effect = RuntimeError("MCP connection failed")
    tool = create_mcp_increase_quantity_tool(mcp)

    with pytest.raises(RuntimeError, match="MCP connection failed"):
        call_tool(tool, runtime=make_runtime())

    mcp.ainvoke.assert_awaited_once()


# ---------------------------------------------------------------------------
# search_products wrapper
# ---------------------------------------------------------------------------


def test_search_products_clears_selected_product_and_returns_tool_message():
    mcp = FakeMCPTool(result=[
        {"type": "text", "text": '[{"id": 12, "name": "Black Hoodie"}]'},
    ])
    tool = create_mcp_search_products_tool(mcp)
    runtime = make_runtime(product_id=99)

    result = call_tool(tool, search="black hoodie", runtime=runtime)

    assert result.update["selected_product_id"] is None
    assert result.update["messages"][0].content == '[{"id": 12, "name": "Black Hoodie"}]'
    assert result.update["messages"][0].tool_call_id == runtime.tool_call_id
    mcp.ainvoke.assert_awaited_once_with({
        "query": "black hoodie",
        "limit": 5,
    })
