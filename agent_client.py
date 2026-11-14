import os
from pathlib import Path

import anyio
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.mcp import MCPAdapter
from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI


QUESTION = "Find all F1 experiment records and tell me which one has the lowest final_fitness."
CONTROL_QUESTION = "What is 2 + 2?"


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


if __name__ == "__main__":
    anyio.run(main)
