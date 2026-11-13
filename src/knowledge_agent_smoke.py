
import asyncio
from pathlib import Path

from langchain.mcp import MCPAdapter

from agent import model
from agents.knowledge_agent import create_knowledge_agent


async def main():

    # Locate the existing MCP server.
    server_path = Path(__file__).resolve().with_name(
        "mcp_server.py"
    )

    # Open MCP connection.
    async with MCPAdapter(server_path) as adapter:

        mcp_tools = await adapter.list_tools()

        print("MCP tools loaded successfully.")

        # Create Knowledge Agent.
        knowledge_agent = create_knowledge_agent(
            model,
            mcp_tools,
        )

        print("Knowledge Agent initialized successfully.")

        # Ask a store policy question.
        result = await knowledge_agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Apa kebijakan pengembalian produk "
                            "di Zivo? Jelaskan persyaratan dan "
                            "batas waktu yang tercantum dalam "
                            "dokumen kebijakan."
                        ),
                    }
                ]
            }
        )

        messages = result.get("messages", [])

        if not messages:
            raise RuntimeError(
                "Knowledge Agent returned no messages."
            )

        # Collect tools called by the agent.
        called_tools = []

        for message in messages:
            for tool_call in getattr(
                message,
                "tool_calls",
                [],
            ):
                called_tools.append(tool_call["name"])

        print("\nTools called:")

        for tool_name in called_tools:
            print(f"- {tool_name}")

        # Verify policy retrieval.
        if "search_policy_knowledge" not in called_tools:
            raise AssertionError(
                "Knowledge Agent did not call "
                "search_policy_knowledge."
            )

        # Normalize final response.
        final_message = messages[-1]
        content = final_message.content

        if isinstance(content, str):
            response_text = content

        elif isinstance(content, list):
            response_text = "\n".join(
                item["text"]
                for item in content
                if isinstance(item, dict)
                and item.get("type") == "text"
                and isinstance(item.get("text"), str)
            )

        else:
            response_text = ""

        if not response_text.strip():
            raise RuntimeError(
                "Knowledge Agent returned empty response."
            )

        print("\nKnowledge Agent response:")
        print(response_text)

        print("\nKnowledge Agent smoke test passed.")


if __name__ == "__main__":
    asyncio.run(main())
