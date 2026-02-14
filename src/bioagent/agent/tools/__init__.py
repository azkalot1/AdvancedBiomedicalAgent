from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
import uuid
from .dbsearch import DBSEARCH_TOOLS
from .summarizing import _make_list_tool, make_summarizing_tool, _make_retrieve_tool, DEFAULT_OUTPUT_DIR
from .target_search import TARGET_SEARCH_TOOLS
from .thinking import think
from .web_search import WEB_SEARCH_TOOLS


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
    wrapped_tools = [make_summarizing_tool(tool, summarizer_llm, output_dir=output_dir) for tool in all_tools]
    
    # Create agent-specific retrieval tools
    wrapped_tools.extend([
        _make_retrieve_tool(output_dir),
        _make_list_tool(output_dir),
    ])
    return wrapped_tools


__all__ = [
    "think",
    "DBSEARCH_TOOLS",
    "TARGET_SEARCH_TOOLS",
    "WEB_SEARCH_TOOLS",
    "make_summarizing_tool",
    "get_summarized_tools",
]