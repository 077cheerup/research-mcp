import os
from pathlib import Path

import anyio
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.mcp import MCPAdapter
from langchain_core.messages import ToolMessage


QUESTION = "Find all F1 experiment records and tell me which one has the lowest final_fitness."
CONTROL_QUESTION = "What is 2 + 2?"


async def main() -> None:
    server_path = Path(__file__).resolve().parent / "server.py"
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

        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("Set OPENAI_API_KEY before running the agent questions.")
        model = init_chat_model("openai:gpt-4.1-mini")
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


if __name__ == "__main__":
    anyio.run(main)
