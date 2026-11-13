from pathlib import Path

import anyio
from mcp import Client, MCPError, StdioServerParameters
from mcp.types import TextResourceContents


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

        print("\n=== RESOURCES/LIST ===")
        resources = await client.list_resources()
        for resource in resources.resources:
            print("name:", resource.name)
            print("uri:", resource.uri)
            print("description:", resource.description)
            print("mime_type:", resource.mime_type)

        print("\n=== RESOURCES/TEMPLATES/LIST ===")
        templates = await client.list_resource_templates()
        for template in templates.resource_templates:
            print("name:", template.name)
            print("uri_template:", template.uri_template)
            print("description:", template.description)
            print("mime_type:", template.mime_type)

        for uri in ("research://project/overview", "research://experiments/exp002"):
            print(f"\n=== RESOURCES/READ {uri} ===", flush=True)
            resource_result = await client.read_resource(uri)
            print("result_class:", type(resource_result).__name__)
            print("contents:", resource_result.contents)
            for content in resource_result.contents:
                print("content_class:", type(content).__name__)
                print("uri:", content.uri)
                print("mime_type:", content.mime_type)
                if isinstance(content, TextResourceContents):
                    print("text:")
                    print(content.text)
                else:
                    print("blob:", content.blob)

        missing_uri = "research://experiments/not-exist"
        print(f"\n=== RESOURCES/READ {missing_uri} ===", flush=True)
        try:
            await client.read_resource(missing_uri)
        except MCPError as error:
            print("exception_class:", type(error).__name__)
            print("code:", error.code)
            print("message:", error.message)
            print("data:", error.data)


if __name__ == "__main__":
    anyio.run(main)
