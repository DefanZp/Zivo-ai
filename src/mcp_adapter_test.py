import asyncio
from pathlib import Path

from langchain.mcp import MCPAdapter


# Menentukan lokasi MCP server.
server_path = Path(__file__).with_name("mcp_server.py")


async def main():
    # Menghubungkan adapter ke MCP server melalui stdio.
    async with MCPAdapter(server_path) as adapter:

        # Mengambil daftar tool dari MCP server.
        tools = await adapter.list_tools()

        print("Daftar MCP Tools:")

        for tool in tools:
            print(f"- {tool.name}: {tool.description}")

        # Mencari tool yang ingin kita uji.
        search_tool = next(
            (
                tool
                for tool in tools
                if tool.name == "search_products"
            ),
            None,
        )

        if search_tool is None:
            print("Tool search_products tidak ditemukan.")
            return

        # Memanggil MCP tool seperti LangChain tool biasa.
        print("\nMemanggil search_products...")

        result = await search_tool.ainvoke(
            {
                "query": "hoodie",
                "limit": 5,
            }
        )

        print("\nHasil pencarian:")
        print(result)


if __name__ == "__main__":
    asyncio.run(main())