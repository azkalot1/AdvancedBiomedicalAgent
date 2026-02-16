from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from langchain.agents import create_agent

from bioagent.agent import get_chat_model
from bioagent.agent.tools import get_summarized_tools, think, with_tool_status_streaming

SYSTEM_PROMPT = """You are a biomedical co-scientist.

Your task is to answer users' questions by planning steps, using available tools,
and synthesizing evidence into clear, actionable outputs.

When tool outputs are long, they may be summarized and stored as reports.
Use:
- `list_research_outputs()` to browse what is already available.
- `retrieve_full_output(reference_id=...)` when full details are needed.
"""


def _build_checkpointer() -> Any | None:
    checkpoint_db = Path(os.getenv("BIOAGENT_CHECKPOINT_DB", ".bioagent/checkpoints.db"))
    checkpoint_db.parent.mkdir(parents=True, exist_ok=True)
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver

        return SqliteSaver.from_conn_string(str(checkpoint_db))
    except Exception:
        return None


def _build_middleware() -> list[Any]:
    try:
        from langchain.agents.middleware import ModelRetryMiddleware

        return [
            ModelRetryMiddleware(
                max_retries=3,
                backoff_factor=2.0,
                initial_delay=1.0,
            )
        ]
    except Exception:
        return []


def _build_agent() -> Any:
    model_name = os.getenv("BIOAGENT_MODEL", "google/gemini-3-flash-preview")
    summarizer_model_name = os.getenv("BIOAGENT_SUMMARIZER_MODEL", "google/gemini-3-flash-preview")
    provider = os.getenv("BIOAGENT_PROVIDER", "openrouter")

    model = get_chat_model(
        model_name,
        provider,
        model_parameters={"temperature": float(os.getenv("BIOAGENT_TEMPERATURE", "0.5"))},
    )
    summarizer = get_chat_model(
        summarizer_model_name,
        provider,
        model_parameters={"temperature": float(os.getenv("BIOAGENT_SUMMARIZER_TEMPERATURE", "0.2"))},
    )

    tools = [with_tool_status_streaming(think)] + get_summarized_tools(summarizer)
    kwargs: dict[str, Any] = {
        "model": model,
        "system_prompt": SYSTEM_PROMPT,
        "tools": tools,
        "debug": os.getenv("BIOAGENT_AGENT_DEBUG", "false").lower() == "true",
        "middleware": _build_middleware(),
    }

    checkpointer = _build_checkpointer()
    if checkpointer is not None:
        kwargs["checkpointer"] = checkpointer

    return create_agent(**kwargs)


graph = _build_agent()
