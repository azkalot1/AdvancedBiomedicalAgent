from __future__ import annotations

import os
from typing import Any

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import tool
from tavily import AsyncTavilyClient

from .tool_utils import robust_unwrap_llm_inputs

# Initialize clients once for reuse
ddg_search = DuckDuckGoSearchResults(max_results=5, output_format="list")
tavily_client = AsyncTavilyClient(os.getenv("TAVILY_API_KEY"))


@tool("web_search", return_direct=False)
@robust_unwrap_llm_inputs
async def web_search(query: str) -> str:
    """
    Performs a web search using DuckDuckGo to find general information.

    Use this for broad queries, recent information, or topics outside
    the specialized biomedical databases.

    Args:
        query: The search query (question or topic).

    Returns:
        A formatted string with each result on a new line including title,
        link, and snippet.
    """
    try:
        search_results = await ddg_search.ainvoke(query)

        if not search_results:
            return "No web search results found for the query."

        formatted_lines = []
        for i, result in enumerate(search_results, start=1):
            title = result.get("title", "No Title Found")
            link = result.get("link", "#")
            snippet = result.get("snippet", "No snippet available.")
            formatted_lines.append(f"{i}. [{title}]({link})\n   Snippet: {snippet}")

        formatted_output = "\n\n".join(formatted_lines)
        return f"RESEARCH OUTPUT:\n{formatted_output}"

    except Exception as e:
        return f"An error occurred during the web search: {e}"


@tool("scrape_url_content", return_direct=False)
@robust_unwrap_llm_inputs
async def scrape_url_content(urls: str | Any | dict[str, Any] | list[str]) -> str:
    """
    Fetches and parses the main content from one or more URLs.

    This tool is designed to bypass common anti-scraping measures. It returns
    cleaned text extracted from the page(s).

    Args:
        urls: A URL or list of URLs to scrape.

    Returns:
        A string containing the scraped content.
    """
    if isinstance(urls, str):
        urls = [urls]
    try:
        scrape_result = await tavily_client.extract(urls=urls, format="text")
        content = scrape_result.get("results")
        if content is None:
            return "Scraping succeeded but no content was found on the page."

        results_str = ""
        for result in content:
            results_str += result.get("raw_content", "") + "\n"
        return f"RESEARCH OUTPUT:\n{results_str}"

    except Exception as e:
        return f"An error occurred in the tool: {e}"


@tool("tavily_search", return_direct=False)
@robust_unwrap_llm_inputs
async def tavily_search(
    query: str | Any | dict,
    search_depth: str | Any = "basic",
    include_answer: bool | str | dict = True,
    max_results: int | Any | dict = 3,
    include_raw_content: bool | str | dict = False,
    include_images: bool | str = False,
) -> str:
    """
    Performs a web search using the Tavily API to find information.

    This tool is ideal for broad web searches and provides an optional
    direct answer.

    Args:
        query: The search query.
        search_depth: "basic" (fast) or "advanced" (more thorough).
        include_answer: If True, include a direct answer.
        max_results: Maximum number of results to return.
        include_raw_content: If True, include raw scraped content.
        include_images: If True, include image URLs.

    Returns:
        A formatted string of the Tavily response.
    """
    try:
        response = await tavily_client.search(
            query=query,
            search_depth=search_depth,
            include_answer=include_answer,
            max_results=max_results,
            include_raw_content=include_raw_content,
            include_images=include_images,
        )
        tavily_search_result = ""
        tavily_search_result += f"Query: {response.get('query', '')}\n"
        tavily_search_result += f"Answer: {response.get('answer', '')}\n"
        for r in response.get("results", []):
            tavily_search_result += f"Title: {r.get('title', '')}\n"
            tavily_search_result += f"Content: {r.get('content', '')}\n"
        return f"RESEARCH OUTPUT:\n{tavily_search_result}"

    except Exception as e:
        return f"An error occurred while searching with Tavily: {e}"


WEB_SEARCH_TOOLS = [
    web_search,
    tavily_search,
    scrape_url_content,
]
