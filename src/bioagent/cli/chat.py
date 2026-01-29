"""
Terminal chat with the biomedical agent.

Uses LangGraph + LangChain with persistent checkpoints (SQLite) and per-conversation
research outputs (.md files under research_outputs/<thread_id>/).

Run via: python -m bioagent.cli.chat [options] or biomedagent-db chat [options].
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

import dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from bioagent.agent import get_chat_model
from bioagent.agent.tools import get_summarized_tools, think
from bioagent.agent.tools.summarizing import DEFAULT_OUTPUT_DIR

dotenv.load_dotenv()

SYSTEM_PROMPT = """\
You are a biomedical agent. Your task is to answer users' questions by planning steps,
finding relevant information in biomedical databases, and synthesizing the results.

When tool outputs exceed ~4000 characters, they are automatically summarized.
The full output is stored in files and can be retrieved:
- Use list_research_outputs() to browse stored outputs with one-line descriptions.
- Use retrieve_full_output(ref_id) to fetch full content when needed.
Only retrieve full output if the summary indicates critical details are missing.
"""


def _extract_last_response(messages: list[Any]) -> str:
    for msg in reversed(messages):
        content = getattr(msg, "content", None)
        if content:
            return str(content)
    return ""


def _extract_ref_ids(text: str) -> list[str]:
    """Extract [ref: xxx] ref IDs from agent reply."""
    return re.findall(r"\[ref:\s*([^\]]+)\]", text)


def _list_research_outputs_for_thread(thread_id: str, tool_filter: str | None = None) -> str:
    """List research output ref IDs for a thread (reads filesystem, no agent)."""
    output_dir = DEFAULT_OUTPUT_DIR / thread_id
    if not output_dir.exists():
        return "No research outputs for this thread yet."
    files = list(output_dir.glob("*.md"))
    if not files:
        return "No research outputs for this thread yet."
    files = sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)
    lines = ["**Stored Research Outputs (current thread):**\n"]
    for fp in files[:25]:
        ref_id = fp.stem
        tool_name = ref_id.rsplit("_", 1)[0] if "_" in ref_id else "unknown"
        if tool_filter and tool_filter.lower() not in tool_name.lower():
            continue
        size = fp.stat().st_size
        lines.append(f"  - **{ref_id}** ({size:,} chars)")
    lines.append("\nUse retrieve_full_output(ref_id) in a message to fetch full content.")
    return "\n".join(lines)


def build_agent(
    thread_id: str,
    checkpointer: AsyncSqliteSaver,
    model_name: str = "google/gemini-2.5-flash",
    provider: str = "openrouter",
    debug: bool = False,
):
    """Build agent with tools scoped to thread_id (research outputs under research_outputs/<thread_id>/)."""
    model = get_chat_model(model_name, provider, model_parameters={"temperature": 0.5})
    summarizer_llm = get_chat_model(
        model_name,
        provider,
        model_parameters={"temperature": 0.2},
    )
    tools = get_summarized_tools(summarizer_llm, session_id=thread_id)
    agent = create_agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        tools=[think] + tools,
        checkpointer=checkpointer,
        debug=debug,
    )
    return agent


async def run_repl(
    *,
    initial_thread_id: str | None = None,
    checkpoint_path: str | Path,
    model_name: str,
    provider: str,
    debug: bool = False,
) -> None:
    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    conn_string = str(checkpoint_path)

    async with AsyncSqliteSaver.from_conn_string(conn_string) as checkpointer:
        thread_id = initial_thread_id or str(uuid4())
        agent = build_agent(thread_id, checkpointer, model_name=model_name, provider=provider, debug=debug)
        config = {"configurable": {"thread_id": thread_id}}

        print("Biomedical Agent — terminal chat")
        print("Commands: /new, /thread [id], /list, /history, /quit")
        print(f"Thread: {thread_id}")
        print()

        while True:
            try:
                user_input = await asyncio.get_event_loop().run_in_executor(None, lambda: input("You> ").strip())
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break
            if not user_input:
                continue

            if user_input.startswith("/"):
                parts = user_input.split(maxsplit=1)
                cmd = parts[0].lower()
                arg = parts[1].strip() if len(parts) > 1 else ""

                if cmd in ("/quit", "/exit"):
                    print("Bye.")
                    break
                if cmd == "/new":
                    thread_id = str(uuid4())
                    agent = build_agent(thread_id, checkpointer, model_name=model_name, provider=provider, debug=debug)
                    config = {"configurable": {"thread_id": thread_id}}
                    print(f"New thread: {thread_id}")
                    continue
                if cmd == "/thread":
                    if arg:
                        thread_id = arg
                        agent = build_agent(
                            thread_id, checkpointer, model_name=model_name, provider=provider, debug=debug
                        )
                        config = {"configurable": {"thread_id": thread_id}}
                        print(f"Switched to thread: {thread_id}")
                    else:
                        print(f"Current thread: {thread_id}")
                    continue
                if cmd == "/list":
                    print(_list_research_outputs_for_thread(thread_id, arg or None))
                    continue
                if cmd == "/history":
                    try:
                        state = agent.get_state(config)
                        messages = state.values.get("messages", []) if hasattr(state, "values") else []
                        n = min(int(arg) if arg else 10, len(messages))
                        for msg in messages[-n:]:
                            role = getattr(msg, "type", "message")
                            content = (getattr(msg, "content", None) or "")[:200]
                            print(f"  [{role}] {content}...")
                    except Exception as e:
                        print(f"Could not get history: {e}")
                    continue
                print(f"Unknown command: {cmd}. Use /new, /thread, /list, /history, /quit")
                continue

            try:
                response = await agent.ainvoke(
                    {"messages": [HumanMessage(content=user_input)]},
                    config,
                )
                messages = response.get("messages", []) if isinstance(response, dict) else []
                reply = _extract_last_response(messages)
                print("Agent>", reply)
                refs = _extract_ref_ids(reply)
                if refs:
                    print("  [Refs:", ", ".join(refs), "]")
            except Exception as e:
                print(f"Agent error: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Terminal chat with the biomedical agent (LangGraph + persistent SQLite checkpoints)."
    )
    parser.add_argument(
        "--thread-id",
        default=os.getenv("BIOAGENT_THREAD_ID"),
        help="Thread ID to continue (default: new UUID).",
    )
    parser.add_argument(
        "--checkpoint-db",
        default=os.getenv("BIOAGENT_CHECKPOINT_DB", ".bioagent/checkpoints.db"),
        help="Path to SQLite checkpoint DB (default: .bioagent/checkpoints.db).",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("BIOAGENT_MODEL", "google/gemini-2.5-flash"),
        help="Model name (default: google/gemini-2.5-flash).",
    )
    parser.add_argument(
        "--provider",
        default=os.getenv("BIOAGENT_PROVIDER", "openrouter"),
        help="Model provider (default: openrouter).",
    )
    parser.add_argument("--debug", action="store_true", help="Enable agent debug output.")
    args = parser.parse_args()

    if not os.getenv("OPENROUTER_API_KEY") and args.provider == "openrouter":
        print(
            "Warning: OPENROUTER_API_KEY not set. Set it or use --provider openai with OPENAI_API_KEY.",
            file=sys.stderr,
        )

    asyncio.run(
        run_repl(
            initial_thread_id=args.thread_id,
            checkpoint_path=args.checkpoint_db,
            model_name=args.model,
            provider=args.provider,
            debug=args.debug,
        )
    )
