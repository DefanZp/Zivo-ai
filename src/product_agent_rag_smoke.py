"""Integration smoke test for semantic product discovery with RAG.

Run from src/:
    py product_agent_rag_smoke.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from dotenv import load_dotenv
from langchain.mcp import MCPAdapter

load_dotenv()

from agent import model
from agents.product_agent import create_product_agent


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

        required_tools = {"search_product_knowledge", "get_product"}
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
                            "Saya membutuhkan pakaian untuk jogging yang "
                            "ringan, breathable, dan nyaman dipakai bergerak. "
                            "Temukan kandidat yang paling sesuai dari "
                            "karakteristik tersebut. Jika ada satu kandidat "
                            "yang jelas paling cocok, pilih dan verifikasi "
                            "detail terbarunya melalui backend."
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

    if "search_product_knowledge" not in called_tools:
        raise AssertionError(
            "Semantic discovery did not use search_product_knowledge."
        )

    selected_product_id = result.get("selected_product_id")
    if selected_product_id is not None and "get_product" not in called_tools:
        raise AssertionError(
            "A product was selected but get_product was not called "
            "to verify current backend details."
        )

    response_text = extract_text(messages[-1].content)
    if not response_text.strip():
        raise RuntimeError("Product Agent returned an empty response.")

    print("\nSelected product ID:", selected_product_id)
    print("\nProduct Agent response:")
    print(response_text)
    print("\nPRODUCT AGENT RAG SMOKE TEST: PASSED")


if __name__ == "__main__":
    asyncio.run(main())
