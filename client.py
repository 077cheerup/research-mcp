from pathlib import Path

import anyio
from mcp import Client, StdioServerParameters


async def main() -> None:
    server_params = StdioServerParameters(  # 怎样启动 stdio Server？
        command="uv",
        args=["run", "server.py"],
        cwd=Path(__file__).resolve().parent,
    )

    async with Client(server_params) as client:  # 创建一个能够连接这个 MCP Server 的 Client。
        print("=== TOOLS/LIST ===")
        tools = await client.list_tools()  # Client 向 Server 发起 Tool Discovery。
        for tool in tools.tools:
            print("name:", tool.name)
            print("description:", tool.description)
            print("input_schema:", tool.input_schema)
            print("output_schema:", tool.output_schema)
            annotations = tool.annotations
            print("annotations:", annotations)
            if annotations is not None:
                print("read_only_hint:", annotations.read_only_hint)
                print("destructive_hint:", annotations.destructive_hint)
                print("idempotent_hint:", annotations.idempotent_hint)
                print("open_world_hint:", annotations.open_world_hint)

        for label, query in (("F1 QUERY", "F1"), ("TOOL ERROR CASE", "")):
            print(f"\n=== {label} (query={query!r}) ===", flush=True)
            result = await client.call_tool("search_experiments", {"query": query})
            print("result_class:", type(result).__name__)
            print("is_error:", result.is_error)
            print("content:", result.content)
            print("structured_content:", result.structured_content)


if __name__ == "__main__":
    anyio.run(main)
