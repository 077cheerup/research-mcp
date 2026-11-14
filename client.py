from pathlib import Path

import anyio
from mcp import Client, MCPError, StdioServerParameters
from mcp.client import ClientRequestContext
from mcp.types import (
    CallToolResult,
    ElicitRequest,
    ElicitRequestFormParams,
    ElicitRequestParams,
    ElicitResult,
    InputRequest,
    InputRequiredResult,
    InputResponse,
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


def fulfil_input_request(request: InputRequest) -> InputResponse:
    print("=== MANUAL FULFIL ===")
    print("request_class:", type(request).__name__)
    print("request.method:", request.method)
    print("request.params:", request.params)
    if not isinstance(request, ElicitRequest):
        raise NotImplementedError(f"Unsupported input request: {request.method}")
    if not isinstance(request.params, ElicitRequestFormParams):
        raise NotImplementedError("Only form elicitation is supported in this demo.")
    print("message:", request.params.message)
    print("requested_schema:", request.params.requested_schema)
    return ElicitResult(action="accept", content={"focus": "convergence"})


async def demonstrate_manual_mrtr(client: Client) -> None:
    tool_name = "create_analysis_request"
    arguments = {"run_id": "exp002"}
    print("\n=== MANUAL MRTR ROUND 1 ===", flush=True)
    result = await client.session.call_tool(tool_name, arguments, allow_input_required=True)
    print("result_class:", type(result).__name__)
    if not isinstance(result, InputRequiredResult):
        print("actual_result:", result)
        return

    round_number = 1
    while isinstance(result, InputRequiredResult):
        if round_number >= 5:
            raise RuntimeError("Manual MRTR did not finish within 5 rounds.")

        requests = result.input_requests or {}
        request_state = result.request_state
        print("input_requests:", requests)
        print("request_state is None:", request_state is None)
        print("request_state type:", type(request_state).__name__)
        if isinstance(request_state, str):
            print("request_state length:", len(request_state))

        responses = {}
        for key, request in requests.items():
            print("input_request key:", key)
            responses[key] = fulfil_input_request(request)
        print("input_responses keys:", list(responses))
        for response in responses.values():
            print("response_class:", type(response).__name__)
            if isinstance(response, ElicitResult):
                print("action:", response.action)
                print("content:", response.content)

        round_number += 1
        print(f"\n=== MANUAL MRTR ROUND {round_number} ===", flush=True)
        result = await client.session.call_tool(
            tool_name,
            arguments,
            input_responses=responses,
            request_state=request_state,
            allow_input_required=True,
        )
        print("result_class:", type(result).__name__)

    if isinstance(result, CallToolResult):
        print("is_error:", result.is_error)
        print("content:", result.content)
        print("structured_content:", result.structured_content)


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

        await demonstrate_manual_mrtr(client)

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
