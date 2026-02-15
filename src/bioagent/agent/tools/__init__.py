from __future__ import annotations

import functools
import os
import uuid
from datetime import datetime, timezone
from time import perf_counter
from typing import Any

from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool, StructuredTool

from .dbsearch import DBSEARCH_TOOLS
from .summarizing import (
    DEFAULT_OUTPUT_DIR,
    _make_list_tool,
    _make_retrieve_tool,
    make_summarizing_tool,
)
from .target_search import TARGET_SEARCH_TOOLS
from .thinking import think
from .web_search import WEB_SEARCH_TOOLS


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y", "t"}


def _preview_value(value: Any, *, max_text: int = 240, depth: int = 0) -> Any:
    if depth > 2:
        return "<max_depth>"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= max_text else value[:max_text] + "... <truncated>"
    if isinstance(value, dict):
        preview: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= 8:
                preview["..."] = f"+{len(value) - 8} more keys"
                break
            preview[str(key)] = _preview_value(item, max_text=max_text, depth=depth + 1)
        return preview
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        preview = [_preview_value(item, max_text=max_text, depth=depth + 1) for item in items[:8]]
        if len(items) > 8:
            preview.append(f"... +{len(items) - 8} more items")
        return preview
    text = repr(value)
    return text if len(text) <= max_text else text[:max_text] + "... <truncated>"


def _runtime_stream_settings() -> tuple[bool, str | None]:
    stream_tool_args_default = _truthy(os.getenv("BIOAGENT_STREAM_TOOL_ARGS", "false"))
    tool_call_id: str | None = None
    stream_tool_args = stream_tool_args_default
    try:
        from langgraph.config import get_config

        cfg = get_config() or {}
        configurable = cfg.get("configurable", {}) if isinstance(cfg, dict) else {}
        if isinstance(configurable, dict):
            if "stream_tool_args" in configurable:
                stream_tool_args = _truthy(configurable.get("stream_tool_args"))
            raw_call_id = configurable.get("tool_call_id")
            if raw_call_id is not None:
                tool_call_id = str(raw_call_id)
    except Exception:
        pass
    return stream_tool_args, tool_call_id


def _emit_tool_status(
    *,
    status: str,
    tool_name: str,
    invocation_id: str | None = None,
    args: dict[str, Any] | None = None,
    duration_ms: int | None = None,
    error: Exception | None = None,
) -> None:
    try:
        from langgraph.config import get_stream_writer

        writer = get_stream_writer()
    except Exception:
        return

    stream_tool_args, tool_call_id = _runtime_stream_settings()
    payload: dict[str, Any] = {
        "type": "tool_status",
        "status": status,
        "tool_name": tool_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if invocation_id:
        payload["invocation_id"] = invocation_id
    if tool_call_id:
        payload["tool_call_id"] = tool_call_id
    if duration_ms is not None:
        payload["duration_ms"] = int(duration_ms)
    if args and stream_tool_args and status in {"queued", "running"}:
        payload["args_preview"] = _preview_value(args)
    if error is not None:
        payload["error_type"] = type(error).__name__
        payload["error"] = _preview_value(str(error), max_text=400)

    try:
        writer(payload)
    except Exception:
        return


def with_tool_status_streaming(tool: BaseTool) -> BaseTool:
    """Wrap a tool to emit lifecycle events over LangGraph custom stream."""
    original_func = tool.coroutine or tool.func
    if original_func is None:
        return tool

    @functools.wraps(original_func)
    async def wrapped_func(**kwargs) -> Any:
        invocation_id = f"{tool.name}_{uuid.uuid4().hex[:10]}"
        _emit_tool_status(
            status="running",
            tool_name=tool.name,
            invocation_id=invocation_id,
            args=kwargs,
        )
        started = perf_counter()
        try:
            result = await tool.ainvoke(kwargs)
        except Exception as exc:
            elapsed = int((perf_counter() - started) * 1000)
            _emit_tool_status(
                status="error",
                tool_name=tool.name,
                invocation_id=invocation_id,
                args=kwargs,
                duration_ms=elapsed,
                error=exc,
            )
            raise
        elapsed = int((perf_counter() - started) * 1000)
        _emit_tool_status(
            status="success",
            tool_name=tool.name,
            invocation_id=invocation_id,
            args=kwargs,
            duration_ms=elapsed,
        )
        return result

    return StructuredTool.from_function(
        coroutine=wrapped_func,
        name=tool.name,
        description=tool.description or "",
        args_schema=tool.args_schema,
        return_direct=tool.return_direct,
    )


def get_summarized_tools(
    summarizer_llm: Runnable,
    session_id: str | None = None,
) -> list[BaseTool]:
    """Wrap all tools with summarization and add retrieval tools."""
    
    # Create agent-specific output directory
    if session_id:
        output_dir = DEFAULT_OUTPUT_DIR / session_id
    else:
        output_dir = DEFAULT_OUTPUT_DIR / str(uuid.uuid4())
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_tools = DBSEARCH_TOOLS + TARGET_SEARCH_TOOLS + WEB_SEARCH_TOOLS
    wrapped_tools = [
        with_tool_status_streaming(make_summarizing_tool(tool, summarizer_llm, output_dir=output_dir))
        for tool in all_tools
    ]
    
    # Create agent-specific retrieval tools
    wrapped_tools.extend([
        with_tool_status_streaming(_make_retrieve_tool(output_dir)),
        with_tool_status_streaming(_make_list_tool(output_dir)),
    ])
    return wrapped_tools


__all__ = [
    "think",
    "DBSEARCH_TOOLS",
    "TARGET_SEARCH_TOOLS",
    "WEB_SEARCH_TOOLS",
    "make_summarizing_tool",
    "get_summarized_tools",
    "with_tool_status_streaming",
]
