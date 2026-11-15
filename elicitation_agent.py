import os
from pathlib import Path

import anyio
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.mcp import MCPAdapter
from langchain.mcp.elicitation import (
    ELICITATION_INTERRUPT_TYPE,
    MCPElicitationInterrupt,
    MCPElicitationResume,
)
from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, Interrupt

from agent_client import print_message_flow


QUESTION = "Create an analysis request for experiment exp002."
THREAD_ID = "phase-6e-exp002"
TOOL_NAME = "create_analysis_request"


async def main() -> None:
    server_path = Path(__file__).resolve().parent / "server.py"
    load_dotenv(server_path.with_name(".env"))

    async with MCPAdapter(server_path) as adapter:
        tools = await adapter.list_tools()
        agent_tools = [tool for tool in tools if tool.name == TOOL_NAME]
        if len(agent_tools) != 1:
            raise RuntimeError(f"Expected exactly one {TOOL_NAME} tool.")
        tool = agent_tools[0]
        schema = tool.args_schema
        print("=== AGENT TOOL ===")
        print("name:", tool.name)
        print("args_schema:", schema)
        if not isinstance(schema, dict) or set(schema["properties"]) != {"run_id"}:
            raise RuntimeError("Expected run_id to be the only public tool argument.")

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
        checkpointer = InMemorySaver()
        agent = create_agent(model=model, tools=agent_tools, checkpointer=checkpointer)
        config = {"configurable": {"thread_id": THREAD_ID}}
        print("\n=== LANGGRAPH THREAD ===")
        print("thread_id:", THREAD_ID, flush=True)

        paused = await agent.ainvoke(
            {"messages": [{"role": "user", "content": QUESTION}]},
            config,
        )
        paused_messages = paused.get("messages", [])
        print("\n=== BEFORE RESUME: MESSAGE FLOW ===")
        print_message_flow(paused_messages)
        print("has_tool_message:", any(isinstance(message, ToolMessage) for message in paused_messages))

        interrupts = paused.get("__interrupt__", ())
        print("\n=== LANGGRAPH INTERRUPT ===")
        print("interrupt_count:", len(interrupts))
        if len(interrupts) != 1:
            raise RuntimeError("Expected one MCP elicitation interrupt.")
        pending = interrupts[0]
        print("interrupt_class:", type(pending).__name__)
        print("interrupt.value:", pending.value)
        payload: MCPElicitationInterrupt = pending.value
        if payload.get("type") != ELICITATION_INTERRUPT_TYPE:
            raise RuntimeError("Expected an MCP elicitation interrupt.")
        print("This is an MCP elicitation interrupt.")

        resume: MCPElicitationResume = {"responses": {}}
        for request in payload["requests"]:
            if request["mode"] != "form":
                raise RuntimeError("This demo expects form elicitation.")
            key = request["key"]
            print("\n=== ELICITATION REQUEST ===")
            print("request_key:", key)
            print("mode:", request["mode"])
            print("message:", request["message"])
            print("requested_schema:", request["requested_schema"])
            user_focus = input("Enter analysis focus: ").strip()
            if not user_focus:
                raise SystemExit("Analysis focus cannot be empty.")
            resume["responses"][key] = {"action": "accept", "content": {"focus": user_focus}}

        print("resume:", resume, flush=True)
        resumed = await agent.ainvoke(Command(resume=resume), config)
        if resumed.get("__interrupt__"):
            raise RuntimeError("The agent still has an unresolved interrupt after resume.")
        messages = resumed["messages"]
        print("\n=== AFTER RESUME: MESSAGE FLOW ===")
        print_message_flow(messages)

        print("\n=== INTERRUPT VS MESSAGE ===")
        print("Interrupt appeared in __interrupt__:", bool(interrupts))
        print("Interrupt appears as chat message:", any(
            isinstance(message, Interrupt) for message in (*paused_messages, *messages)
        ))

        print("\n=== TWO RECOVERY LAYERS ===")
        print("LangGraph: checkpointer + thread_id resume the paused agent graph.")
        print("MCP: inputResponses + requestState resume the MCP round trip.")
        print("requestState visible to application: False (managed internally by MCPAdapter)")


if __name__ == "__main__":
    anyio.run(main)
