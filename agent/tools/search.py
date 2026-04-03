"""Web search tool using DuckDuckGo (no API key required)."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field


class WebSearchInput(BaseModel):
    query: str = Field(description="The search query to look up on the web.")
    max_results: int = Field(default=5, description="Maximum number of results to return.")


class WebSearchTool(BaseTool):
    """LangChain tool that performs web searches via DuckDuckGo."""

    name: str = "web_search"
    description: str = (
        "Search the web for current information. "
        "Use this when you need up-to-date facts, news, documentation, or any information "
        "that might not be in your training data. "
        "Input should be a clear, concise search query."
    )
    args_schema: type[BaseModel] = WebSearchInput
    max_results: int = 5

    def _run(self, query: str, max_results: int = 5, **kwargs: Any) -> str:
        try:
            from duckduckgo_search import DDGS
        except ImportError as exc:
            raise ImportError(
                "duckduckgo-search is required. Install it with: pip install duckduckgo-search"
            ) from exc

        results_limit = max_results or self.max_results
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=results_limit))
        except Exception as exc:
            return f"Search failed: {exc}"

        if not results:
            return f"No results found for: '{query}'"

        formatted = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "No title")
            href = r.get("href", "")
            body = r.get("body", "")
            formatted.append(f"{i}. **{title}**\n   URL: {href}\n   {body}")

        return "\n\n".join(formatted)

    async def _arun(self, query: str, max_results: int = 5, **kwargs: Any) -> str:
        return self._run(query, max_results=max_results)


def create_search_tool(max_results: int = 5) -> WebSearchTool:
    """Factory function to create a configured WebSearchTool."""
    return WebSearchTool(max_results=max_results)
