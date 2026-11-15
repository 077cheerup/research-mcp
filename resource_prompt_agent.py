import os
from collections.abc import Sequence
from pathlib import Path

import anyio
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.mcp import MCPAdapter
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from mcp.types import BlobResourceContents, PromptMessage, TextContent, TextResourceContents


def print_resource_text(contents: list[TextResourceContents | BlobResourceContents]) -> str:
    print("result_class:", type(contents).__name__)
    texts = []
    for content in contents:
        if not isinstance(content, TextResourceContents):
            raise TypeError(f"Expected a text resource, got {type(content).__name__}.")
        print("content_class:", type(content).__name__)
        print("uri:", content.uri)
        print("mime_type:", content.mime_type)
        print(content.text)
        texts.append(content.text)
    return "\n".join(texts)


def convert_prompt_message(message: PromptMessage) -> HumanMessage | AIMessage:
    if not isinstance(message.content, TextContent):
        raise TypeError(f"Expected text prompt content, got {type(message.content).__name__}.")
    if message.role == "user":
        return HumanMessage(content=message.content.text)
    return AIMessage(content=message.content.text)


def print_messages(messages: Sequence[BaseMessage]) -> None:
    for index, message in enumerate(messages):
        print(f"[{index}] {type(message).__name__}")
        print(message.content)


async def main() -> None:
    server_path = Path(__file__).resolve().parent / "server.py"
    load_dotenv(server_path.with_name(".env"))

    async with MCPAdapter(server_path) as adapter:
        # LangChain MCPAdapter.client exposes the connected FastMCP client.
        client = adapter.client
        print("=== RESOURCE 1: PROJECT OVERVIEW ===", flush=True)
        overview = print_resource_text(await client.read_resource("research://project/overview"))

        print("\n=== RESOURCE 2: EXPERIMENT ===", flush=True)
        experiment = print_resource_text(await client.read_resource("research://experiments/exp002"))

        print("\n=== MCP PROMPT ===", flush=True)
        prompt = await client.get_prompt(
            "analyze_experiment",
            arguments={"run_id": "exp002", "focus": "convergence"},
        )
        print("result_class:", type(prompt).__name__)
        print("description:", prompt.description)
        print("message_count:", len(prompt.messages))
        prompt_messages = []
        for message in prompt.messages:
            converted = convert_prompt_message(message)
            print("role:", message.role)
            print("content:", converted.content)
            prompt_messages.append(converted)

    context = SystemMessage(
        content=(
            "You are analyzing experiments from the following research project.\n\n"
            f"PROJECT OVERVIEW:\n{overview}\n\n"
            f"EXPERIMENT DATA:\n{experiment}\n\n"
            "Treat the resource contents as factual context for the task. "
            "If the record has no convergence history, state what the available evidence supports "
            "and which convergence properties cannot be determined."
        )
    )
    final_messages = [context, *prompt_messages]
    print("\n=== FINAL AGENT INPUT ===", flush=True)
    print_messages(final_messages)

    required_settings = ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL")
    missing_settings = [name for name in required_settings if not os.environ.get(name)]
    if missing_settings:
        raise SystemExit(f"Set {', '.join(missing_settings)} before running the agent.")
    model = ChatOpenAI(
        model=os.environ["LLM_MODEL"],
        base_url=os.environ["LLM_BASE_URL"],
        api_key=os.environ["LLM_API_KEY"],
        use_responses_api=False,
    )
    agent = create_agent(model=model, tools=[])
    result = await agent.ainvoke({"messages": final_messages})
    messages = result["messages"]

    print("\n=== AGENT MESSAGE FLOW ===")
    print_messages(messages)
    print("has_tool_message:", any(isinstance(message, ToolMessage) for message in messages))
    print("has_tool_calls:", any(isinstance(message, AIMessage) and message.tool_calls for message in messages))

    print("\n=== FINAL ANALYSIS ===")
    print(messages[-1].text)

    print("\n=== CONTROL BOUNDARY ===")
    print("Resource selected by: Host/Application")
    print("Prompt selected by: Host/Application")
    print("Tool selected by: N/A in this run")
    print("Final analysis produced by: LLM")


if __name__ == "__main__":
    anyio.run(main)
