from pathlib import Path

import anyio
from mcp import Client, MCPError, StdioServerParameters
from mcp.client import ClientRequestContext
from mcp.types import (
    ElicitRequestFormParams,
    ElicitRequestParams,
    ElicitResult,
    TextContent,
    TextResourceContents,
)


async def handle_elicitation(
    _context: ClientRequestContext,
    params: ElicitRequestParams,
) -> ElicitResult:
    assert isinstance(params, ElicitRequestFormParams)
    print("=== ELICITATION CALLBACK ===")
    print("message:", params.message)
    print("requested_schema:", params.requested_schema, flush=True)
    return ElicitResult(action="accept", content={"focus": "convergence"})


async def main() -> None:
    server_params = StdioServerParameters(  # 怎样启动 stdio Server？
        command="uv",
        args=["run", "server.py"],
        cwd=Path(__file__).resolve().parent,
    )

    async with Client(server_params, elicitation_callback=handle_elicitation) as client:
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

        print("\n=== PROMPTS/LIST ===", flush=True)
        prompts = await client.list_prompts()
        for prompt in prompts.prompts:
            print("name:", prompt.name)
            print("description:", prompt.description)
            print("arguments:", prompt.arguments)

        for arguments in (
            {"run_id": "exp002", "focus": "convergence"},
            {"run_id": "exp003"},
        ):
            print(f"\n=== PROMPTS/GET analyze_experiment {arguments} ===", flush=True)
            prompt_result = await client.get_prompt("analyze_experiment", arguments=arguments)
            print("result_class:", type(prompt_result).__name__)
            print("description:", prompt_result.description)
            print("messages:", prompt_result.messages)
            for message in prompt_result.messages:
                print("role:", message.role)
                print("content_class:", type(message.content).__name__)
                if isinstance(message.content, TextContent):
                    print("text:")
                    print(message.content.text)

        print("\n=== PROMPTS/GET analyze_experiment (missing run_id) ===", flush=True)
        try:
            await client.get_prompt("analyze_experiment", arguments={"focus": "convergence"})
        except MCPError as error:
            print("exception_class:", type(error).__name__)
            print("code:", error.code)
            print("message:", error.message)
            print("data:", error.data)

        print("\n=== ELICITATION / MRTR CASE ===", flush=True)
        print("protocol_version:", client.protocol_version, flush=True)
        result = await client.call_tool("create_analysis_request", {"run_id": "exp002"})
        print("result_class:", type(result).__name__)
        print("is_error:", result.is_error)
        print("content:", result.content)
        print("structured_content:", result.structured_content)

    print("\n=== WITHOUT ELICITATION CALLBACK ===", flush=True)
    async with Client(server_params) as client:
        try:
            await client.call_tool("create_analysis_request", {"run_id": "exp002"})
        except MCPError as error:
            print("exception_class:", type(error).__name__)
            print("code:", error.code)
            print("message:", error.message)
            print("data:", error.data)


if __name__ == "__main__":
    anyio.run(main)
