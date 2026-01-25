from __future__ import annotations

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

        filepath = output_dir / f"{ref_id}.md"
        filepath.write_text(stored_content)

        try:
            summary = await _generate_summary(
                summarizer_llm=summarizer_llm,
                params=kwargs,
                content=raw_output,
            )
        except Exception as e:
            summary = (
                f"[Summarization failed: {e}]\n\n"
                f"First 2000 chars:\n{raw_output[:2000]}"
            )

        return (
            f"**[Summary | {len(raw_output):,} chars | ref: {ref_id}]**\n\n"
            f"{summary}\n\n"
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
 
        filepath = output_dir / f"{reference_id}.md"

        if not filepath.exists():
            if not output_dir.exists():
                return "No research outputs stored yet."
            available = [f.stem for f in output_dir.glob("*.md")][:10]
            if not available:
                return "No research outputs stored yet."
            return (
                f"Reference '{reference_id}' not found.\n\n"
                "Available references:\n" + "\n".join(f"  - {r}" for r in available)
            )

        content = filepath.read_text()

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
        if not output_dir.exists():
            return "No research outputs stored yet."

        files = list(output_dir.glob("*.md"))
        if not files:
            return "No research outputs stored yet."

        files = sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)

        lines = ["**Stored Research Outputs:**\n"]
        for file_path in files[:25]:
            ref_id = file_path.stem
            tool_name = ref_id.rsplit("_", 1)[0] if "_" in ref_id else "unknown"

            if tool_filter and tool_filter.lower() not in tool_name.lower():
                continue

            content = file_path.read_text()
            one_line = ""
            size_chars = file_path.stat().st_size
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