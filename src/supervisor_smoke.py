"""End-to-end smoke test for Supervisor routing and state handoff.

Run from src/:
    py supervisor_smoke.py

Requires LANGGRAPH_DB_URI, CART_SMOKE_TEST_TOKEN, and a dedicated empty
cart belonging to a testing account. The script removes only TEST_PRODUCT_ID.
"""

from __future__ import annotations

import asyncio
import json
import os
import selectors
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from langchain.mcp import MCPAdapter
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

load_dotenv()

from agent import create_zivo_agent


TEST_PRODUCT_ID = int(os.getenv("SMOKE_TEST_PRODUCT_ID", "134"))


def extract_text(content: object) -> str:
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
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def parse_cart_response(raw_result: object) -> list[dict]:
    payload = decode_json(extract_text(raw_result))
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
    items: list[dict], product_id: int, expected_quantity: int
) -> None:
    matches = find_product_items(items, product_id)
    if len(matches) != 1:
        raise AssertionError(
            f"Expected one item for product {product_id}; found {len(matches)}."
        )
    actual_quantity = matches[0].get("quantity")
    if actual_quantity != expected_quantity:
        raise AssertionError(
            f"Expected quantity {expected_quantity}; got {actual_quantity}."
        )


def get_called_agents(messages: list) -> list[str]:
    called_agents = []
    for message in messages:
        for call in getattr(message, "tool_calls", []):
            name = call.get("name")
            if isinstance(name, str):
                called_agents.append(name)
    return called_agents


async def run_turn(
    supervisor,
    *,
    config: dict,
    context: dict,
    prompt: str,
    expected_agent: str,
    previous_message_count: int,
    initial_state: bool = False,
) -> tuple[dict, int, str, list[str]]:
    agent_input = {
        "messages": [{"role": "user", "content": prompt}],
    }
    if initial_state:
        agent_input["selected_product_id"] = None

    result = await supervisor.ainvoke(agent_input, config, context=context)
    messages = result.get("messages", [])
    if not messages:
        raise RuntimeError("Supervisor returned no messages.")

    new_messages = messages[previous_message_count:]
    called_agents = get_called_agents(new_messages)
    print("\nSupervisor routing:", called_agents)

    if expected_agent not in called_agents:
        raise AssertionError(
            f"Expected {expected_agent}; actual calls: {called_agents}"
        )

    response_text = extract_text(messages[-1].content)
    if not response_text.strip():
        raise RuntimeError("Supervisor returned an empty response.")

    print("Supervisor:", response_text)
    print(f"{expected_agent}: PASSED")
    return result, len(messages), response_text, called_agents


async def cleanup_test_product(remove_tool, get_cart_tool, token: str) -> None:
    """Remove only the known test product and verify the cart is empty."""
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
                f"Cart cleanup failed: {payload.get('message', payload)}"
            )

    remaining_items = await get_cart_items(get_cart_tool, token)
    if remaining_items:
        raise RuntimeError(
            "The cart is not empty after cleanup. "
            "Review the dedicated testing account manually."
        )


async def main() -> None:
    db_uri = os.getenv("LANGGRAPH_DB_URI")
    token = os.getenv("CART_SMOKE_TEST_TOKEN")
    if not db_uri:
        raise RuntimeError("LANGGRAPH_DB_URI is missing from .env.")
    if not token:
        raise RuntimeError("CART_SMOKE_TEST_TOKEN is missing from .env.")

    server_path = Path(__file__).resolve().with_name("mcp_server.py")
    thread_id = f"supervisor-smoke-{uuid4().hex}"
    config = {"configurable": {"thread_id": thread_id}}
    context = {"cart_api_token": token}

    async with AsyncPostgresSaver.from_conn_string(db_uri) as checkpointer:
        await checkpointer.setup()

        async with MCPAdapter(server_path) as adapter:
            mcp_tools = await adapter.list_tools()
            tools_by_name = {tool.name: tool for tool in mcp_tools}

            required_tools = {"get_cart", "remove_from_cart"}
            missing_tools = required_tools - tools_by_name.keys()
            if missing_tools:
                raise RuntimeError(f"Missing MCP tools: {sorted(missing_tools)}")

            get_cart_tool = tools_by_name["get_cart"]
            remove_tool = tools_by_name["remove_from_cart"]
            supervisor = create_zivo_agent(checkpointer, mcp_tools)

            # Never mutate a cart that was not empty at the start.
            initial_result, message_count, _, _ = await run_turn(
                supervisor,
                config=config,
                context=context,
                prompt=(
                    "Tampilkan isi keranjang belanja saya saat ini. "
                    "Jangan mengubah isinya."
                ),
                expected_agent="cart_agent",
                previous_message_count=0,
                initial_state=True,
            )

            initial_items = await get_cart_items(get_cart_tool, token)
            if initial_items:
                raise RuntimeError(
                    "Testing cart is not empty. Empty the dedicated test cart "
                    "before running this smoke test."
                )
            print("Initial cart verification: PASSED")

            confirmation = input(
                "Pastikan akun ini khusus testing. Ketik LANJUT untuk meneruskan: "
            )
            if confirmation.strip().upper() != "LANJUT":
                print("Smoke test cancelled.")
                return

            try:
                _, message_count, _, _ = await run_turn(
                    supervisor,
                    config=config,
                    context=context,
                    prompt=(
                        "Apa batas waktu pengembalian produk dan bagaimana "
                        "kondisi produk yang memenuhi syarat menurut "
                        "kebijakan Zivo?"
                    ),
                    expected_agent="knowledge_agent",
                    previous_message_count=message_count,
                )

                result, message_count, _, _ = await run_turn(
                    supervisor,
                    config=config,
                    context=context,
                    prompt=(
                        "Cari produk dengan nama persis 'Oversized Black Hoodie'. "
                        "Jika ditemukan satu produk yang cocok, pilih produk "
                        "tersebut dan verifikasi detailnya melalui backend."
                    ),
                    expected_agent="product_agent",
                    previous_message_count=message_count,
                )

                selected_product_id = result.get("selected_product_id")
                if selected_product_id != TEST_PRODUCT_ID:
                    raise AssertionError(
                        f"Expected selected_product_id={TEST_PRODUCT_ID}; "
                        f"got {selected_product_id}. Cart test stopped."
                    )
                print("Product selection state: PASSED")

                result, message_count, _, called_agents = await run_turn(
                    supervisor,
                    config=config,
                    context=context,
                    prompt=(
                        "Tambahkan produk yang baru saja saya pilih ke keranjang "
                        "dengan quantity tepat 1. Jangan mencari produk lagi."
                    ),
                    expected_agent="cart_agent",
                    previous_message_count=message_count,
                )

                if "product_agent" in called_agents:
                    raise AssertionError(
                        "Supervisor called Product Agent again during cart add."
                    )
                if result.get("selected_product_id") != TEST_PRODUCT_ID:
                    raise AssertionError(
                        "selected_product_id was not preserved after Cart Agent."
                    )

                assert_product_quantity(
                    await get_cart_items(get_cart_tool, token),
                    TEST_PRODUCT_ID,
                    1,
                )
                print("Product-to-Cart handoff and backend add: PASSED")

                _, message_count, _, _ = await run_turn(
                    supervisor,
                    config=config,
                    context=context,
                    prompt=(
                        "Periksa isi keranjang saya. Pastikan Oversized Black "
                        "Hoodie ada dengan quantity 1. Jangan mengubah isinya."
                    ),
                    expected_agent="cart_agent",
                    previous_message_count=message_count,
                )

                _, message_count, _, _ = await run_turn(
                    supervisor,
                    config=config,
                    context=context,
                    prompt=(
                        "Hapus hanya Oversized Black Hoodie yang baru saja "
                        "ditambahkan dari keranjang. Jangan hapus produk lain."
                    ),
                    expected_agent="cart_agent",
                    previous_message_count=message_count,
                )

                final_items = await get_cart_items(get_cart_tool, token)
                if final_items:
                    raise AssertionError(
                        f"Expected an empty cart after removal; "
                        f"found {len(final_items)} item(s)."
                    )
                print("Final cart verification: PASSED")

                print("\nSUPERVISOR SMOKE TEST: PASSED")
                print("Thread ID:", thread_id)
            finally:
                # This is a final safety net if an agent step fails midway.
                await cleanup_test_product(remove_tool, get_cart_tool, token)
                print("Cart cleanup: PASSED")


if __name__ == "__main__":
    # SelectorEventLoop avoids the ProactorEventLoop issue seen with
    # Psycopg's async connection on Windows.
    asyncio.run(
        main(),
        loop_factory=lambda: asyncio.SelectorEventLoop(
            selectors.SelectSelector()
        ),
    )
