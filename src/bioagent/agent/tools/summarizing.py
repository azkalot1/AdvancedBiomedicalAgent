from __future__ import annotations

import asyncio
import functools
import inspect
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool, StructuredTool, tool
from bioagent.persistence import (
    get_report_content,
    list_thread_tool_outputs,
    persist_tool_output_report,
    resolve_scope,
    thread_tool_outputs_dir,
)

DEFAULT_OUTPUT_DIR = Path(os.getenv("BIOAGENT_RESEARCH_OUTPUT_DIR", "./research_outputs"))

SUMMARIZATION_PROMPT = PromptTemplate.from_template(
    """
You are a research assistant extracting relevant information.

**Agent's Query Context:**
{agent_query}

**Tool Output (raw, may be truncated):**
{tool_output}

**Task:**
Extract and summarize ONLY the information relevant to the query.
- Preserve key facts, data points, statistics, and citations
- Keep structure (lists, tables) when useful
- Be concise but complete for decision-making
- Note if important details might be in the full output
"""
)

ONE_LINE_PROMPT = PromptTemplate.from_template(
    """
Summarize in ONE line (max 60 chars) what this output contains:
Tool: {tool_name} | Query: {query}
Preview: {preview}

One-line:
"""
)


def _escape_frontmatter(value: str) -> str:
    sanitized = value.replace("\n", " ").replace("\r", " ")
    return sanitized.replace('"', '\\"').strip()


def _split_output_sections(raw_output: str) -> tuple[str, str]:
    if "[AGENT_SIGNALS]" in raw_output:
        results_part, signals_part = raw_output.split("[AGENT_SIGNALS]", 1)
        results_part = results_part.rstrip()
        signals_part = "[AGENT_SIGNALS]" + signals_part.lstrip()
    else:
        results_part = raw_output.rstrip()
        signals_part = "[AGENT_SIGNALS]\n---\nRelated searches:\n  -> None"
    return results_part, signals_part


def _runtime_scope() -> tuple[Any | None, str | None, str | None]:
    """Best-effort access to current store + configurable scope without runtime args."""
    store: Any | None = None
    user_id: str | None = None
    thread_id: str | None = None

    try:
        from langgraph.config import get_store

        store = get_store()
    except Exception:
        store = None

    try:
        from langgraph.config import get_config

        cfg = get_config() or {}
        configurable = cfg.get("configurable", {}) if isinstance(cfg, dict) else {}
        if isinstance(configurable, dict):
            raw_user = configurable.get("user_id")
            raw_thread = configurable.get("thread_id")
            user_id = str(raw_user) if raw_user is not None else None
            thread_id = str(raw_thread) if raw_thread is not None else None
    except Exception:
        pass

    return store, user_id, thread_id


def _path_exists(path: Path) -> bool:
    return path.exists()


def _read_text(path: Path) -> str:
    return path.read_text()


def _list_report_files(path: Path) -> list[Path]:
    return list(path.glob("*.md"))


def _sorted_by_mtime_desc(paths: list[Path]) -> list[Path]:
    return sorted(paths, key=lambda item: item.stat().st_mtime, reverse=True)


async def _generate_one_line_summary(
    summarizer_llm: Runnable,
    tool_name: str,
    params: dict[str, Any],
    content_preview: str,
) -> str:
    one_line_chain = ONE_LINE_PROMPT | summarizer_llm | StrOutputParser()
    one_line = await one_line_chain.ainvoke(
        {
            "tool_name": tool_name,
            "query": json.dumps(params, default=str)[:200],
            "preview": content_preview[:500],
        }
    )
    return one_line.strip()[:80]


async def _generate_summary(
    summarizer_llm: Runnable,
    params: dict[str, Any],
    content: str,
) -> str:
    summary_chain = SUMMARIZATION_PROMPT | summarizer_llm | StrOutputParser()
    return await summary_chain.ainvoke(
        {
            "agent_query": json.dumps(params, default=str),
            "tool_output": content[:20000],
        }
    )


def make_summarizing_tool(
    original_tool: BaseTool,
    summarizer_llm: Runnable,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    max_output_length: int = 4000,
) -> BaseTool:
    """
    Factory that wraps any tool with summarization capability while preserving
    the original tool name, description, and args schema.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    original_func = original_tool.coroutine or original_tool.func
    original_is_async = inspect.iscoroutinefunction(original_func)

    @functools.wraps(original_func)
    async def wrapped_func(**kwargs) -> str:
        if original_is_async:
            raw_output = await original_tool.ainvoke(kwargs)
        else:
            raw_output = original_tool.invoke(kwargs)

        if not isinstance(raw_output, str) or len(raw_output) <= max_output_length:
            return raw_output

        results_part, signals_part = _split_output_sections(raw_output)

        ref_id = f"{original_tool.name}_{uuid4().hex[:8]}"
        one_line = ""
        try:
            one_line = await _generate_one_line_summary(
                summarizer_llm=summarizer_llm,
                tool_name=original_tool.name,
                params=kwargs,
                content_preview=raw_output,
            )
        except Exception:
            one_line = f"{original_tool.name} output ({len(raw_output):,} chars)"

        escaped_one_line = _escape_frontmatter(one_line)
        stored_content = (
            "---\n"
            f"tool: {original_tool.name}\n"
            f"ref_id: {ref_id}\n"
            f"timestamp: {datetime.now().isoformat()}\n"
            f"query_params: {json.dumps(kwargs, default=str)}\n"
            f"size_chars: {len(raw_output)}\n"
            f"one_line: \"{escaped_one_line}\"\n"
            "---\n\n"
            f"{raw_output}\n"
        )

        report_metadata: dict[str, Any] | None = None
        store, scoped_user, scoped_thread = _runtime_scope()
        user_id, thread_id = resolve_scope(user_id=scoped_user, thread_id=scoped_thread)
        try:
            report_metadata = await persist_tool_output_report(
                content=raw_output,
                tool_name=original_tool.name,
                report_id=ref_id,
                one_line=one_line,
                store=store,
                user_id=user_id,
                thread_id=thread_id,
            )
        except Exception:
            try:
                # Ensure filesystem + index are still updated even if store fails.
                report_metadata = await persist_tool_output_report(
                    content=raw_output,
                    tool_name=original_tool.name,
                    report_id=ref_id,
                    one_line=one_line,
                    store=None,
                    user_id=user_id,
                    thread_id=thread_id,
                )
            except Exception:
                scoped_dir = thread_tool_outputs_dir(user_id=user_id, thread_id=thread_id)
                await asyncio.to_thread(scoped_dir.mkdir, parents=True, exist_ok=True)
                filepath = scoped_dir / f"{ref_id}.md"
                await asyncio.to_thread(filepath.write_text, stored_content)

        try:
            summary = await _generate_summary(
                summarizer_llm=summarizer_llm,
                params=kwargs,
                content=results_part,
            )
        except Exception as e:
            summary = (
                f"[Summarization failed: {e}]\n\n"
                f"First 2000 chars:\n{raw_output[:2000]}"
            )

        if report_metadata:
            try:
                from langgraph.config import get_stream_writer

                writer = get_stream_writer()
                writer(
                    {
                        "type": "report_generated",
                        "report": report_metadata,
                    }
                )
            except Exception:
                pass

        return (
            f"**[Summary | {len(raw_output):,} chars | ref: {ref_id}]**\n\n"
            f"{summary}\n\n"
            f"{signals_part}\n\n"
            "---\n"
            f'*Full output stored. To retrieve:* `retrieve_full_output("{ref_id}")`\n'
        )

    enhanced_description = original_tool.description + (
        "\n\n**Note:** Long outputs (>4000 chars) are automatically summarized. "
        "Use `retrieve_full_output(ref_id)` to access the complete data. "
        "Use `list_research_outputs()` to browse stored outputs."
    )

    return StructuredTool.from_function(
        coroutine=wrapped_func,
        name=original_tool.name,
        description=enhanced_description,
        args_schema=original_tool.args_schema,
        return_direct=original_tool.return_direct,
    )


def _make_retrieve_tool(output_dir: Path) -> BaseTool:
    @tool("retrieve_full_output", return_direct=False)
    async def retrieve_full_output(
        reference_id: str,
        max_chars: int | None = None,
    ) -> str:
        """
        Retrieve the full, unsummarized output from a previous tool call.

        Use this when you've received a summary and need the complete data
        for deeper analysis. The reference_id appears in summaries as "[ref: xxx]".

        Args:
            reference_id: The ref ID from a previous summarized output (e.g., "pubmed_search_a1b2c3d4")
            max_chars: Optional limit on returned characters. Use if you only need a portion.

        Returns:
            The complete, raw tool output without summarization.
        """
 
        content: str | None = None
        store, scoped_user, scoped_thread = _runtime_scope()
        user_id, thread_id = resolve_scope(user_id=scoped_user, thread_id=scoped_thread)

        try:
            content = await get_report_content(
                user_id=user_id,
                report_id=reference_id,
                store=store,
            )
        except Exception:
            content = None

        if content is None:
            scoped_output_dir = thread_tool_outputs_dir(user_id=user_id, thread_id=thread_id)
            filepath = scoped_output_dir / f"{reference_id}.md"

            if not await asyncio.to_thread(_path_exists, filepath):
                try:
                    available_reports = await list_thread_tool_outputs(
                        user_id=user_id,
                        thread_id=thread_id,
                        store=store,
                    )
                except Exception:
                    available_reports = []

                available = [
                    str(item.get("id") or item.get("ref_id"))
                    for item in available_reports
                    if item.get("id") or item.get("ref_id")
                ][:10]
                if not available:
                    if not await asyncio.to_thread(_path_exists, scoped_output_dir):
                        return "No research outputs stored yet."
                    local_files = await asyncio.to_thread(_list_report_files, scoped_output_dir)
                    available = [f.stem for f in local_files][:10]
                    if not available:
                        return "No research outputs stored yet."
                return (
                    f"Reference '{reference_id}' not found.\n\n"
                    "Available references:\n" + "\n".join(f"  - {r}" for r in available)
                )

            content = await asyncio.to_thread(_read_text, filepath)

            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    content = parts[2].strip()

        if max_chars and len(content) > max_chars:
            content = (
                content[:max_chars]
                + f"\n\n[... truncated at {max_chars:,} chars. Full size: {len(content):,} chars]"
            )

        return f"[Full output for {reference_id}]\n\n{content}"
    
    return retrieve_full_output


def _make_list_tool(output_dir: Path) -> BaseTool:
    @tool("list_research_outputs", return_direct=False)
    async def list_research_outputs(
        tool_filter: str | None = None,
    ) -> str:
        """
        List all stored research outputs from this session.

        Use this to review what data you've already gathered before making
        new tool calls. Helps avoid redundant searches.

        Args:
            tool_filter: Optional filter by tool name (e.g., "pubmed", "uniprot").
                         Case-insensitive partial match.

        Returns:
            A list showing reference IDs, sizes, timestamps, and one-line summaries.
            Use the reference_id with retrieve_full_output() to access complete data.
        """
        store, scoped_user, scoped_thread = _runtime_scope()
        user_id, thread_id = resolve_scope(user_id=scoped_user, thread_id=scoped_thread)

        records: list[dict[str, Any]] = []
        try:
            records = await list_thread_tool_outputs(
                user_id=user_id,
                thread_id=thread_id,
                store=store,
            )
        except Exception:
            records = []

        lines = ["**Stored Research Outputs:**\n"]
        if records:
            for record in records[:25]:
                ref_id = str(record.get("id") or record.get("ref_id") or "")
                if not ref_id:
                    continue
                tool_name = str(record.get("tool_name") or (ref_id.rsplit("_", 1)[0] if "_" in ref_id else "unknown"))
                if tool_filter and tool_filter.lower() not in tool_name.lower():
                    continue

                one_line = str(record.get("one_line") or "").strip()
                size_chars = int(record.get("size_chars") or 0)
                timestamp = str(record.get("created_at") or record.get("timestamp") or "")[:16]

                line = f"  - **{ref_id}** ({size_chars:,} chars)" if size_chars else f"  - **{ref_id}**"
                if timestamp:
                    line += f" @ {timestamp}"
                if one_line:
                    line += f"\n    _{one_line}_"
                lines.append(line)

        if len(lines) == 1:
            scoped_output_dir = thread_tool_outputs_dir(user_id=user_id, thread_id=thread_id)
            if not await asyncio.to_thread(_path_exists, scoped_output_dir):
                return "No research outputs stored yet."

            files = await asyncio.to_thread(_list_report_files, scoped_output_dir)
            if not files:
                return "No research outputs stored yet."

            files = await asyncio.to_thread(_sorted_by_mtime_desc, files)
            for file_path in files[:25]:
                ref_id = file_path.stem
                tool_name = ref_id.rsplit("_", 1)[0] if "_" in ref_id else "unknown"

                if tool_filter and tool_filter.lower() not in tool_name.lower():
                    continue

                content = await asyncio.to_thread(_read_text, file_path)
                one_line = ""
                size_chars = (await asyncio.to_thread(file_path.stat)).st_size
                timestamp = ""

                if content.startswith("---"):
                    try:
                        parts = content.split("---", 2)
                        if len(parts) >= 2:
                            metadata = yaml.safe_load(parts[1]) or {}
                            one_line = metadata.get("one_line", "")
                            size_chars = metadata.get("size_chars", size_chars)
                            ts = metadata.get("timestamp", "")
                            if ts:
                                timestamp = ts.strftime("%Y-%m-%d %H:%M") if isinstance(ts, datetime) else str(ts)[:16]
                            else:
                                timestamp = ""
                    except yaml.YAMLError:
                        pass

                line = f"  - **{ref_id}** ({size_chars:,} chars)"
                if timestamp:
                    line += f" @ {timestamp}"
                if one_line:
                    line += f"\n    _{one_line}_"
                lines.append(line)

        if len(lines) == 1:
            return f"No outputs found for filter '{tool_filter}'." if tool_filter else "No outputs found."

        lines.append("")
        lines.append("*Use `retrieve_full_output(reference_id='...')` to get full content.*")

        return "\n".join(lines)
    
    return list_research_outputs
