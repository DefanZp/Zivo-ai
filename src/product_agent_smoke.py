"""Integration smoke test for direct product-name search.

Run from src/:
    py product_agent_smoke.py

SMOKE_TEST_PRODUCT_ID can override the expected product ID (default: 134).
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.mcp import MCPAdapter

load_dotenv()

from agent import model
from agents.product_agent import create_product_agent


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


def get_called_tools(messages: list) -> list[str]:
    names = []
    for message in messages:
        for call in getattr(message, "tool_calls", []):
            name = call.get("name")
            if isinstance(name, str):
                names.append(name)
    return names


async def main() -> None:
    server_path = Path(__file__).resolve().with_name("mcp_server.py")

    async with MCPAdapter(server_path) as adapter:
        mcp_tools = await adapter.list_tools()
        available_names = {tool.name for tool in mcp_tools}

        required_tools = {"search_products", "get_product"}
        missing_tools = required_tools - available_names
        if missing_tools:
            raise RuntimeError(f"Missing MCP tools: {sorted(missing_tools)}")

        agent = create_product_agent(model, mcp_tools)
        result = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Cari produk dengan nama persis "
                            "'Oversized Black Hoodie'. Jika hanya ada satu "
                            "produk yang cocok, pilih produk tersebut dan "
                            "verifikasi detailnya melalui backend."
                        ),
                    }
                ],
                "selected_product_id": None,
            },
            context={"cart_api_token": None},
        )

    messages = result.get("messages", [])
    if not messages:
        raise RuntimeError("Product Agent returned no messages.")

    called_tools = get_called_tools(messages)
    print("Called tools:", called_tools)

    for expected_tool in ("search_products", "get_product"):
        if expected_tool not in called_tools:
            raise AssertionError(
                f"Product Agent did not call {expected_tool}. "
                f"Actual calls: {called_tools}"
            )

    selected_product_id = result.get("selected_product_id")
    if selected_product_id != TEST_PRODUCT_ID:
        raise AssertionError(
            f"Expected selected_product_id={TEST_PRODUCT_ID}, "
            f"got {selected_product_id}."
        )

    response_text = extract_text(messages[-1].content)
    if not response_text.strip():
        raise RuntimeError("Product Agent returned an empty response.")

    print("\nSelected product ID:", selected_product_id)
    print("\nProduct Agent response:")
    print(response_text)
    print("\nPRODUCT AGENT SMOKE TEST: PASSED")


if __name__ == "__main__":
    asyncio.run(main())
