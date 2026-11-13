"""Integration smoke test for the Zivo Cart Agent.

Run from src/:
    py cart_agent_smoke.py

Requires an empty, dedicated testing cart and CART_SMOKE_TEST_TOKEN.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.mcp import MCPAdapter

load_dotenv()

from agent import model
from agents.cart_agent import create_cart_agent


TEST_PRODUCT_ID = int(os.getenv("SMOKE_TEST_PRODUCT_ID", "134"))


def extract_text(content: object) -> str:
    """Convert LangChain/MCP text content into a plain string."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(getattr(item, "text", None), str):
                parts.append(item.text)
        return "\n".join(parts)

    return ""


def decode_json(value: object) -> object:
    """Parse JSON text when possible; otherwise keep the original value."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def get_called_tools(messages: list) -> list[str]:
    names = []
    for message in messages:
        for call in getattr(message, "tool_calls", []):
            name = call.get("name")
            if isinstance(name, str):
                names.append(name)
    return names


def assert_no_tool_failure(messages: list) -> None:
    """Fail when a tool explicitly returns {success: false}."""
    for message in messages:
        if getattr(message, "type", None) != "tool":
            continue

        payload = decode_json(extract_text(message.content))
        if isinstance(payload, dict) and payload.get("success") is False:
            raise RuntimeError(
                f"Cart tool failed: {payload.get('message', payload)}"
            )


def parse_cart_response(raw_result: object) -> list[dict]:
    response_text = extract_text(raw_result)
    payload = decode_json(response_text)

    if not isinstance(payload, dict):
        raise RuntimeError("MCP get_cart returned an invalid response.")
    if payload.get("success") is not True:
        raise RuntimeError(
            f"Unable to retrieve cart: {payload.get('message', 'Unknown error')}"
        )

    data = payload.get("data")
    items = data.get("items") if isinstance(data, dict) else None
    if not isinstance(items, list):
        raise RuntimeError("Cart response does not contain a valid items list.")

    return items


async def get_cart_items(get_cart_tool, token: str) -> list[dict]:
    raw_result = await get_cart_tool.ainvoke({"cart_api_token": token})
    return parse_cart_response(raw_result)


def find_product_items(items: list[dict], product_id: int) -> list[dict]:
    return [
        item for item in items
        if isinstance(item, dict) and item.get("product_id") == product_id
    ]


def assert_product_quantity(
    items: list[dict],
    product_id: int,
    expected_quantity: int,
) -> None:
    matches = find_product_items(items, product_id)
    if len(matches) != 1:
        raise AssertionError(
            f"Expected one cart item for product {product_id}; found {len(matches)}."
        )

    actual_quantity = matches[0].get("quantity")
    if actual_quantity != expected_quantity:
        raise AssertionError(
            f"Expected quantity {expected_quantity} for product {product_id}; "
            f"got {actual_quantity}."
        )


def assert_cart_empty(items: list[dict], stage: str) -> None:
    if items:
        raise AssertionError(
            f"Cart must be empty {stage}; found {len(items)} item(s)."
        )


async def run_cart_step(
    cart_agent,
    *,
    prompt: str,
    expected_tool: str,
    token: str,
) -> None:
    result = await cart_agent.ainvoke(
        {
            "messages": [{"role": "user", "content": prompt}],
            "selected_product_id": TEST_PRODUCT_ID,
        },
        context={"cart_api_token": token},
    )

    messages = result.get("messages", [])
    if not messages:
        raise RuntimeError("Cart Agent returned no messages.")

    called_tools = get_called_tools(messages)
    print(f"\nExpected tool: {expected_tool}")
    print("Called tools:", called_tools)

    if expected_tool not in called_tools:
        raise AssertionError(
            f"Expected tool '{expected_tool}' was not called. "
            f"Actual calls: {called_tools}"
        )

    assert_no_tool_failure(messages)

    response_text = extract_text(messages[-1].content)
    if not response_text.strip():
        raise RuntimeError("Cart Agent returned an empty response.")

    print("Agent:", response_text)
    print(f"{expected_tool}: PASSED")


async def cleanup_test_product(remove_tool, get_cart_tool, token: str) -> None:
    """Remove only the test product; never clear the entire cart."""
    items = await get_cart_items(get_cart_tool, token)
    if find_product_items(items, TEST_PRODUCT_ID):
        raw_result = await remove_tool.ainvoke(
            {
                "product_id": TEST_PRODUCT_ID,
                "cart_api_token": token,
            }
        )
        payload = decode_json(extract_text(raw_result))
        if isinstance(payload, dict) and payload.get("success") is False:
            raise RuntimeError(
                f"Cleanup failed: {payload.get('message', payload)}"
            )

    remaining_items = await get_cart_items(get_cart_tool, token)
    if find_product_items(remaining_items, TEST_PRODUCT_ID):
        raise RuntimeError("Cleanup failed: test product remains in the cart.")

    if remaining_items:
        raise RuntimeError(
            "Cleanup finished, but unexpected cart items remain. "
            "Review the testing cart manually."
        )


async def main() -> None:
    token = os.getenv("CART_SMOKE_TEST_TOKEN")
    if not token:
        raise RuntimeError("CART_SMOKE_TEST_TOKEN is missing from .env.")

    server_path = Path(__file__).resolve().with_name("mcp_server.py")

    async with MCPAdapter(server_path) as adapter:
        mcp_tools = await adapter.list_tools()
        tools_by_name = {tool.name: tool for tool in mcp_tools}

        required_tools = ("get_cart", "remove_from_cart")
        missing_tools = [name for name in required_tools if name not in tools_by_name]
        if missing_tools:
            raise RuntimeError(f"Missing MCP tools: {missing_tools}")

        get_cart_tool = tools_by_name["get_cart"]
        remove_tool = tools_by_name["remove_from_cart"]
        cart_agent = create_cart_agent(model, mcp_tools)

        initial_items = await get_cart_items(get_cart_tool, token)
        assert_cart_empty(initial_items, "before the test starts")
        print("Initial cart check: PASSED (empty)")

        confirmation = input(
            "This test changes the cart. Confirm this is a dedicated test account. "
            "Type LANJUT to continue: "
        )
        if confirmation.strip().upper() != "LANJUT":
            print("Smoke test cancelled.")
            return

        try:
            await run_cart_step(
                cart_agent,
                prompt="Tambahkan produk yang sedang dipilih ke keranjang dengan quantity tepat 1.",
                expected_tool="add_to_cart",
                token=token,
            )
            assert_product_quantity(
                await get_cart_items(get_cart_tool, token), TEST_PRODUCT_ID, 1
            )
            print("Backend add verification: PASSED")

            await run_cart_step(
                cart_agent,
                prompt="Naikkan quantity produk yang sedang dipilih tepat satu.",
                expected_tool="increase_quantity",
                token=token,
            )
            assert_product_quantity(
                await get_cart_items(get_cart_tool, token), TEST_PRODUCT_ID, 2
            )
            print("Backend increase verification: PASSED")

            await run_cart_step(
                cart_agent,
                prompt="Turunkan quantity produk yang sedang dipilih tepat satu.",
                expected_tool="decrease_quantity",
                token=token,
            )
            assert_product_quantity(
                await get_cart_items(get_cart_tool, token), TEST_PRODUCT_ID, 1
            )
            print("Backend decrease verification: PASSED")

            await run_cart_step(
                cart_agent,
                prompt="Tampilkan isi keranjang saya saat ini tanpa mengubahnya.",
                expected_tool="get_cart",
                token=token,
            )

            await run_cart_step(
                cart_agent,
                prompt="Hapus hanya produk yang sedang dipilih dari keranjang.",
                expected_tool="remove_from_cart",
                token=token,
            )
            assert_cart_empty(
                await get_cart_items(get_cart_tool, token),
                "after removing the test product",
            )
            print("Backend remove verification: PASSED")

            print("\nCART AGENT SMOKE TEST: PASSED")
        finally:
            # Also runs when a step fails, so the test item is not left behind.
            await cleanup_test_product(remove_tool, get_cart_tool, token)
            print("Cart cleanup: PASSED")


if __name__ == "__main__":
    asyncio.run(main())
