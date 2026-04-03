"""Tests for tool implementations."""

import pytest

from agent.tools.code_tool import CodeAnalysisTool, create_code_tools
from agent.tools.github_tool import GitHubTool, create_github_tools
from agent.tools.search import WebSearchTool, create_search_tool


class TestCodeAnalysisTool:
    def test_analyse_valid_python(self):
        code = """
class Foo:
    def bar(self):
        pass

def standalone():
    pass
"""
        result = CodeAnalysisTool.analyse_python(code)
        assert "Foo" in result
        assert "bar" in result
        assert "standalone" in result
        assert "Lines of code" in result

    def test_analyse_syntax_error(self):
        result = CodeAnalysisTool.analyse_python("def broken(")
        assert "Syntax error" in result

    def test_count_lines(self):
        code = "# comment\nx = 1\n\ny = 2\n"
        result = CodeAnalysisTool.count_lines(code)
        assert "Total" in result
        assert "Comments" in result
        assert "Blank" in result

    def test_extract_docstrings(self):
        code = '''
def greet():
    """Say hello."""
    pass
'''
        result = CodeAnalysisTool.extract_docstrings(code)
        assert "Say hello." in result

    def test_extract_docstrings_none(self):
        result = CodeAnalysisTool.extract_docstrings("x = 1")
        assert "No docstrings found" in result

    def test_create_code_tools_returns_list(self):
        tools = create_code_tools()
        assert len(tools) == 3
        names = {t.name for t in tools}
        assert "analyse_python_code" in names
        assert "count_code_lines" in names
        assert "extract_docstrings" in names


class TestWebSearchTool:
    def test_instantiation(self):
        tool = create_search_tool(max_results=3)
        assert tool.name == "web_search"
        assert tool.max_results == 3

    def test_missing_duckduckgo(self, monkeypatch):
        """When duckduckgo_search is unavailable, _run raises ImportError."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "duckduckgo_search":
                raise ImportError("not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        tool = WebSearchTool()
        with pytest.raises(ImportError, match="duckduckgo-search"):
            tool._run("test query")


class TestGitHubTool:
    def test_create_github_tools_no_token(self):
        tools = create_github_tools(github_token=None)
        assert len(tools) == 3
        names = {t.name for t in tools}
        assert "github_search_repositories" in names
        assert "github_search_code" in names
        assert "github_read_file" in names

    def test_github_tool_init(self):
        tool = GitHubTool(github_token="fake-token")
        assert tool._token == "fake-token"
        assert tool._gh is None  # lazy init

    def test_github_search_repos_api_error(self, monkeypatch):
        """When PyGithub raises an error, a friendly message is returned."""
        tool = GitHubTool(github_token=None)

        class FakeGh:
            def search_repositories(self, *args, **kwargs):
                raise RuntimeError("API limit exceeded")

        tool._gh = FakeGh()
        result = tool.search_repositories("langchain")
        assert "failed" in result.lower()

    def test_github_read_file_api_error(self, monkeypatch):
        tool = GitHubTool(github_token=None)

        class FakeGh:
            def get_repo(self, repo):
                raise RuntimeError("Not found")

        tool._gh = FakeGh()
        result = tool.read_file("owner/repo", "README.md")
        assert "Failed" in result or "failed" in result
