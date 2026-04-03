"""GitHub integration tools for MyEvolvingAgent.

Provides tools for searching repositories, reading files, exploring trending
projects, and finding code examples directly from GitHub.
"""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class GitHubSearchReposInput(BaseModel):
    query: str = Field(description="Search query for GitHub repositories.")
    language: Optional[str] = Field(default=None, description="Filter by programming language.")
    max_results: int = Field(default=5, description="Maximum number of repositories to return.")
    sort: str = Field(default="stars", description="Sort by: stars, forks, updated, help-wanted-issues.")


class GitHubSearchCodeInput(BaseModel):
    query: str = Field(description="Search query for code on GitHub.")
    language: Optional[str] = Field(default=None, description="Filter by programming language.")
    max_results: int = Field(default=5, description="Maximum number of results to return.")


class GitHubReadFileInput(BaseModel):
    repo: str = Field(description="Repository in 'owner/repo' format, e.g. 'langchain-ai/langchain'.")
    path: str = Field(description="Path to the file within the repository, e.g. 'README.md'.")
    ref: Optional[str] = Field(default=None, description="Branch, tag, or commit SHA (defaults to default branch).")


class GitHubTool:
    """Container for GitHub-related LangChain tools."""

    def __init__(self, github_token: Optional[str] = None) -> None:
        self._token = github_token
        self._gh: Any = None  # lazy initialisation

    def _get_github(self) -> Any:
        if self._gh is None:
            try:
                from github import Github
            except ImportError as exc:
                raise ImportError(
                    "PyGithub is required. Install it with: pip install PyGithub"
                ) from exc
            self._gh = Github(self._token) if self._token else Github()
        return self._gh

    def search_repositories(self, query: str, language: Optional[str] = None, max_results: int = 5, sort: str = "stars") -> str:
        """Search GitHub repositories and return formatted results."""
        gh = self._get_github()
        search_query = query
        if language:
            search_query += f" language:{language}"

        try:
            repos = gh.search_repositories(query=search_query, sort=sort)
        except Exception as exc:
            return f"GitHub search failed: {exc}"

        results = []
        for i, repo in enumerate(repos[:max_results]):
            results.append(
                f"{i + 1}. **{repo.full_name}** ⭐{repo.stargazers_count}\n"
                f"   {repo.description or 'No description'}\n"
                f"   URL: {repo.html_url}\n"
                f"   Language: {repo.language or 'N/A'} | Forks: {repo.forks_count}"
            )

        if not results:
            return f"No repositories found for: '{query}'"

        return "\n\n".join(results)

    def search_code(self, query: str, language: Optional[str] = None, max_results: int = 5) -> str:
        """Search code across GitHub and return formatted snippets."""
        gh = self._get_github()
        search_query = query
        if language:
            search_query += f" language:{language}"

        try:
            code_results = gh.search_code(query=search_query)
        except Exception as exc:
            return f"GitHub code search failed: {exc}"

        results = []
        for i, item in enumerate(code_results[:max_results]):
            results.append(
                f"{i + 1}. **{item.repository.full_name}** / `{item.path}`\n"
                f"   URL: {item.html_url}"
            )

        if not results:
            return f"No code found for: '{query}'"

        return "\n\n".join(results)

    def read_file(self, repo: str, path: str, ref: Optional[str] = None) -> str:
        """Read a file from a GitHub repository."""
        gh = self._get_github()
        try:
            repository = gh.get_repo(repo)
            kwargs: dict[str, Any] = {}
            if ref:
                kwargs["ref"] = ref
            content_file = repository.get_contents(path, **kwargs)
            if isinstance(content_file, list):
                # It's a directory – list its contents
                items = "\n".join(f"  - {f.path}" for f in content_file)
                return f"Directory listing for `{repo}/{path}`:\n{items}"
            decoded = content_file.decoded_content.decode("utf-8", errors="replace")
            # Truncate very large files
            if len(decoded) > 5000:
                decoded = decoded[:5000] + "\n\n... [truncated – file is too large] ..."
            return f"File: `{repo}/{path}`\n\n```\n{decoded}\n```"
        except Exception as exc:
            return f"Failed to read file '{path}' from '{repo}': {exc}"

    def get_trending(self, language: Optional[str] = None, since: str = "daily") -> str:
        """Get trending repositories (approximated via GitHub search)."""
        query = "stars:>100"
        if language:
            query += f" language:{language}"
        return self.search_repositories(query, sort="stars", max_results=10)


def _make_search_repos_tool(github_tool: GitHubTool) -> BaseTool:
    from langchain_core.tools import StructuredTool

    return StructuredTool.from_function(
        func=github_tool.search_repositories,
        name="github_search_repositories",
        description=(
            "Search GitHub for repositories matching a query. "
            "Use this to find open-source libraries, projects, or examples. "
            "Optionally filter by programming language."
        ),
        args_schema=GitHubSearchReposInput,
    )


def _make_search_code_tool(github_tool: GitHubTool) -> BaseTool:
    from langchain_core.tools import StructuredTool

    return StructuredTool.from_function(
        func=github_tool.search_code,
        name="github_search_code",
        description=(
            "Search for code snippets on GitHub. "
            "Use this to find real-world examples of how specific functions, "
            "patterns, or libraries are used."
        ),
        args_schema=GitHubSearchCodeInput,
    )


def _make_read_file_tool(github_tool: GitHubTool) -> BaseTool:
    from langchain_core.tools import StructuredTool

    return StructuredTool.from_function(
        func=github_tool.read_file,
        name="github_read_file",
        description=(
            "Read the contents of a specific file in a GitHub repository. "
            "Use this to inspect source code, README files, or configuration files."
        ),
        args_schema=GitHubReadFileInput,
    )


def create_github_tools(github_token: Optional[str] = None) -> list[BaseTool]:
    """Create all GitHub-related tools configured with the provided token."""
    tool_obj = GitHubTool(github_token=github_token)
    return [
        _make_search_repos_tool(tool_obj),
        _make_search_code_tool(tool_obj),
        _make_read_file_tool(tool_obj),
    ]
