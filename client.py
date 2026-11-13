from pathlib import Path

import anyio
from mcp import Client, StdioServerParameters


async def main() -> None:
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "server.py"],
        cwd=Path(__file__).resolve().parent,
    )

    async with Client(server_params) as client:
        print("=== TOOLS/LIST ===")
        tools = await client.list_tools()
        for tool in tools.tools:
            print("name:", tool.name)
            print("description:", tool.description)
            print("input_schema:", tool.input_schema)
            print("output_schema:", tool.output_schema)

        for label, query in (("F1 QUERY", "F1"), ("TOOL ERROR CASE", "")):
            print(f"\n=== {label} (query={query!r}) ===", flush=True)
            result = await client.call_tool("search_experiments", {"query": query})
            print("result_class:", type(result).__name__)
            print("is_error:", result.is_error)
            print("content:", result.content)
            print("structured_content:", result.structured_content)


if __name__ == "__main__":
    anyio.run(main)
