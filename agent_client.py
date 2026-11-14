import os
from collections.abc import Sequence
from pathlib import Path

import anyio
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.mcp import MCPAdapter
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI


QUESTION = "Find all F1 experiment records and tell me which one has the lowest final_fitness."
CONTROL_QUESTION = "What is 2 + 2?"


def print_message_flow(messages: Sequence[BaseMessage]) -> None:
    first_tool_request: tuple[int, AIMessage] | None = None
    final_ai: tuple[int, AIMessage] | None = None

    print("\n=== MESSAGE FLOW ===")
    for index, message in enumerate(messages):
        print(f"[{index}]")
        print("message_class:", type(message).__name__)
        if isinstance(message, HumanMessage):
            print("content:", message.content)
        elif isinstance(message, AIMessage):
            final_ai = (index, message)
            if message.tool_calls and first_tool_request is None:
                first_tool_request = (index, message)
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
    if first_tool_request is None:
        print("No tool calls.")
    else:
        request_index, request_message = first_tool_request
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
    if first_tool_request is None:
        print("tool_request_ai_message_index:", None)
    else:
        request_index, request_message = first_tool_request
        print("tool_request_ai_message_index:", request_index)
        print("tool_calls_count:", len(request_message.tool_calls))
    if final_ai is not None:
        final_index, final_message = final_ai
        print("final_ai_message_index:", final_index)
        print("final_tool_calls_count:", len(final_message.tool_calls))
        print("final_text:", final_message.text)


async def main() -> None:
    server_path = Path(__file__).resolve().parent / "server.py"
    load_dotenv(server_path.with_name(".env"))
    async with MCPAdapter(server_path) as adapter:
        tools = await adapter.list_tools()
        print("=== LANGCHAIN TOOLS FROM MCP ===")
        for tool in tools:
            print("tool_class:", type(tool).__name__)
            print("name:", tool.name)
            print("description:", tool.description)
            print("args_schema:", tool.args_schema)
            print("metadata:", tool.metadata)

        agent_tools = [tool for tool in tools if tool.name == "search_experiments"]
        assert len(agent_tools) == 1, "Expected exactly one search_experiments tool."

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

        for question in (QUESTION, CONTROL_QUESTION):
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
