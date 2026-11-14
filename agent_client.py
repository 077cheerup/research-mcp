import os
import sys
from collections.abc import Sequence
from pathlib import Path

import anyio
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.mcp import MCPAdapter
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI


QUESTIONS = (
    ("RESEARCH ONLY", "Find all F1 experiment records and tell me which one has the lowest final_fitness."),
    ("MATH ONLY", "Use the math add tool to calculate 12.5 + 7.25."),
    (
        "CROSS SERVER",
        "Find the F1 experiment records. Then use the math percentage improvement tool, "
        "not mental arithmetic, to calculate how much exp002 improves over the Baseline. "
        "Lower final_fitness is better.",
    ),
)


def print_message_flow(messages: Sequence[BaseMessage]) -> None:
    tool_requests: list[tuple[int, AIMessage]] = []
    final_ai: tuple[int, AIMessage] | None = None

    print("\n=== MESSAGE FLOW ===")
    for index, message in enumerate(messages):
        print(f"[{index}]")
        print("message_class:", type(message).__name__)
        if isinstance(message, HumanMessage):
            print("content:", message.content)
        elif isinstance(message, AIMessage):
            final_ai = (index, message)
            if message.tool_calls:
                tool_requests.append((index, message))
            print("content:", message.content)
            print("tool_calls_count:", len(message.tool_calls))
            for call in message.tool_calls:
                print("tool_call id:", call.get("id"))
                print("tool_call name:", call.get("name"))
                print("tool_call args:", call.get("args"))
                print("tool_call type:", call.get("type"))
        elif isinstance(message, ToolMessage):
            print("name:", getattr(message, "name", None))
            print("tool_call_id:", getattr(message, "tool_call_id", None))
            print("status:", getattr(message, "status", None))
            print("content:", message.content)

    print("\n=== TOOL CALL PAIRING ===")
    if not tool_requests:
        print("No tool calls.")
    for request_index, request_message in tool_requests:
        responses_by_id = {
            getattr(message, "tool_call_id", None): message
            for message in messages[request_index + 1 :]
            if isinstance(message, ToolMessage)
        }
        for call in request_message.tool_calls:
            call_id = call.get("id")
            response = responses_by_id.get(call_id) if call_id is not None else None
            print("tool_call_id:", call_id)
            print("requested_tool:", call.get("name"))
            print("matched_tool_message:", response is not None)
            response_name = getattr(response, "name", None)
            if response_name is not None:
                print("response_tool:", response_name)

    print("\n=== AI MESSAGE ROLES IN LOOP ===")
    if not tool_requests:
        print("tool_request_ai_message_index:", None)
    else:
        request_index, request_message = tool_requests[0]
        print("tool_request_ai_message_index:", request_index)
        print("tool_calls_count:", len(request_message.tool_calls))
    if final_ai is not None:
        final_index, final_message = final_ai
        print("final_ai_message_index:", final_index)
        print("final_tool_calls_count:", len(final_message.tool_calls))
        print("final_text:", final_message.text)


async def main() -> None:
    project_dir = Path(__file__).resolve().parent
    load_dotenv(project_dir / ".env")
    config = {
        "mcpServers": {
            "research": {
                "command": sys.executable,
                "args": [str(project_dir / "server.py")],
                "transport": "stdio",
            },
            "math": {
                "command": sys.executable,
                "args": [str(project_dir / "math_server.py")],
                "transport": "stdio",
            },
        }
    }
    async with MCPAdapter(config) as adapter:
        tools = await adapter.list_tools()
        print("=== MULTI-SERVER LANGCHAIN TOOLS ===")
        for tool in tools:
            print("tool_class:", type(tool).__name__)
            print("name:", tool.name)
            print("description:", tool.description)
            print("args_schema:", tool.args_schema)
            print("metadata:", tool.metadata)

        print("\n=== TOOL SOURCES / TOOL ROUTING SUMMARY ===")
        for tool in tools:
            mcp_metadata = (tool.metadata or {}).get("mcp", {})
            print("LangChain tool:", tool.name)
            print("MCP server:", mcp_metadata.get("server", {}).get("name"))
            print("Tool annotations:", mcp_metadata.get("tool", {}).get("annotations"))

        allowed_tools = {"search_experiments", "add", "percentage_improvement"}
        agent_tools = [
            tool for tool in tools if tool.name.partition("_")[2] in allowed_tools
        ]
        assert len(agent_tools) == 3, "Expected three namespaced tools for the agent."

        required_settings = ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL")
        missing_settings = [name for name in required_settings if not os.environ.get(name)]
        if missing_settings:
            raise SystemExit(f"Set {', '.join(missing_settings)} before running the agent questions.")
        model = ChatOpenAI(
            model=os.environ["LLM_MODEL"],
            base_url=os.environ["LLM_BASE_URL"],
            api_key=os.environ["LLM_API_KEY"],
            use_responses_api=False,
        )
        agent = create_agent(model=model, tools=agent_tools)

        for label, question in QUESTIONS:
            print(f"\n=== {label} ===")
            print("\n=== AGENT QUESTION ===")
            print(question, flush=True)
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": question}]}
            )
            messages = result["messages"]
            final_message = messages[-1]
            print("message_count:", len(messages))
            print("has_tool_message:", any(isinstance(message, ToolMessage) for message in messages))
            print("final_message_class:", type(final_message).__name__)
            print("=== FINAL ANSWER ===")
            print(final_message.text, flush=True)
            print_message_flow(messages)


if __name__ == "__main__":
    anyio.run(main)
