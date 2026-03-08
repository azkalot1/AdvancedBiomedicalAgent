from __future__ import annotations

from collections.abc import Sequence

from langchain_core.tools import BaseTool, tool
from langgraph.types import Command

from .tool_utils import get_tool_summary


def _dedupe_tool_order(tools: Sequence[BaseTool]) -> list[BaseTool]:
    seen: set[str] = set()
    ordered: list[BaseTool] = []
    for item in tools:
        if item.name in seen:
            continue
        seen.add(item.name)
        ordered.append(item)
    return ordered


def _format_tool_lines(tools: Sequence[BaseTool]) -> list[str]:
    return [f"- {tool.name}: {get_tool_summary(tool)}" for tool in _dedupe_tool_order(tools)]


def create_discovery_tools(
    *,
    db_tools: Sequence[BaseTool],
    target_tools: Sequence[BaseTool],
    web_tools: Sequence[BaseTool],
    utility_tools: Sequence[BaseTool],
) -> tuple[BaseTool, BaseTool]:
    """Create tool-discovery tools bound to a concrete tool inventory."""

    available_tools: list[BaseTool] = _dedupe_tool_order([*db_tools, *target_tools, *web_tools, *utility_tools])
    allowed_names = {tool.name for tool in available_tools}

    @tool("list_available_tools", return_direct=False)
    async def list_available_tools() -> str:
        """List available tools and short summaries for discovery."""

        sections = [
            "Available tools (name: summary):",
            "",
            "Clinical/Drug DB:",
            *_format_tool_lines(db_tools),
            "",
            "Pharmacology:",
            *_format_tool_lines(target_tools),
            "",
            "Web:",
            *_format_tool_lines(web_tools),
            "",
            "Utility:",
            *_format_tool_lines(utility_tools),
            "- list_available_tools: List available tools and short summaries for discovery.",
            '- request_tools: Request tools to keep available for the next step ("tool_names").',
            "",
            "Use request_tools(tool_names=[...]) to keep specific tools available for the next step.",
        ]
        return "\n".join(sections)

    @tool("request_tools", return_direct=False)
    async def request_tools(tool_names: list[str]) -> Command:
        """Request tools to keep available for the next step by tool name."""

        requested: list[str] = []
        for raw_name in tool_names or []:
            name = str(raw_name).strip()
            if not name:
                continue
            if name in allowed_names and name not in requested:
                requested.append(name)

        return Command(update={"requested_tools": requested})

    return list_available_tools, request_tools
