import asyncio
import sys
from pathlib import Path

from mcp import Client, StdioServerParameters


# Menentukan lokasi file MCP server.
server_path = Path(__file__).with_name("mcp_server.py")


async def main():
    # Menentukan cara client menjalankan MCP server.
    server = StdioServerParameters(
        command=sys.executable,
        args=[str(server_path)],
    )

    # Membuka koneksi ke MCP server.
    async with Client(server) as client:
        # Meminta daftar tool yang tersedia.
        tools_result = await client.list_tools()

        print("Daftar MCP Tools:")

        for tool in tools_result.tools:
            print(f"- {tool.name}: {tool.description}")

        # Memanggil tool pencarian produk.
        result = await client.call_tool(
            "search_products",
            {
                "query": "hoodie",
                "limit": 5,
            },
        )

        print("\nHasil pencarian produk:")

        for item in result.content:
            if hasattr(item, "text"):
                print(item.text)
            else:
                print(item)


if __name__ == "__main__":
    asyncio.run(main())