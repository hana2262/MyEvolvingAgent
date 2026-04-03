"""Code analysis and utility tools for MyEvolvingAgent."""

from __future__ import annotations

import ast
import textwrap
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field


class CodeAnalysisInput(BaseModel):
    code: str = Field(description="The Python source code to analyse.")


class CodeFormatInput(BaseModel):
    code: str = Field(description="The Python code to format and clean up.")


class CodeComplexityInput(BaseModel):
    code: str = Field(description="Python code to measure cyclomatic complexity for.")


class CodeAnalysisTool:
    """Collection of code-analysis utilities (Python-focused, no external deps)."""

    @staticmethod
    def analyse_python(code: str) -> str:
        """Parse and report on Python code structure."""
        code = textwrap.dedent(code)
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return f"❌ Syntax error: {exc}"

        classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        functions = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        async_fns = [n.name for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef)]
        imports = []
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                imports.extend(alias.name for alias in n.names)
            elif isinstance(n, ast.ImportFrom):
                module = n.module or ""
                imports.extend(f"{module}.{alias.name}" for alias in n.names)

        lines = code.splitlines()
        report = [
            "## Python Code Analysis",
            f"- **Lines of code**: {len(lines)}",
            f"- **Classes**: {len(classes)}" + (f" ({', '.join(classes)})" if classes else ""),
            f"- **Functions**: {len(functions)}" + (f" ({', '.join(functions)})" if functions else ""),
            f"- **Async functions**: {len(async_fns)}" + (f" ({', '.join(async_fns)})" if async_fns else ""),
            f"- **Imports**: {len(imports)}" + (f" ({', '.join(imports[:10])}{'...' if len(imports) > 10 else ''})" if imports else ""),
        ]
        return "\n".join(report)

    @staticmethod
    def count_lines(code: str) -> str:
        """Count total, code, comment, and blank lines."""
        lines = code.splitlines()
        blank = sum(1 for l in lines if not l.strip())
        comments = sum(1 for l in lines if l.strip().startswith("#"))
        code_lines = len(lines) - blank - comments
        return (
            f"Line count breakdown:\n"
            f"  Total: {len(lines)}\n"
            f"  Code:  {code_lines}\n"
            f"  Comments: {comments}\n"
            f"  Blank: {blank}"
        )

    @staticmethod
    def extract_docstrings(code: str) -> str:
        """Extract all docstrings from Python source."""
        code = textwrap.dedent(code)
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return f"Syntax error: {exc}"

        docs = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
                docstring = ast.get_docstring(node)
                if docstring:
                    name = getattr(node, "name", "<module>")
                    docs.append(f"### `{name}`\n{docstring}")

        if not docs:
            return "No docstrings found."
        return "\n\n".join(docs)


def create_code_tools() -> list[BaseTool]:
    """Create all code analysis tools."""
    analyser = CodeAnalysisTool()

    analyse_tool = StructuredTool.from_function(
        func=analyser.analyse_python,
        name="analyse_python_code",
        description=(
            "Analyse Python source code structure: count classes, functions, imports, and lines. "
            "Useful for quick code review or understanding unfamiliar code."
        ),
        args_schema=CodeAnalysisInput,
    )

    line_count_tool = StructuredTool.from_function(
        func=analyser.count_lines,
        name="count_code_lines",
        description="Count the total, code, comment, and blank lines in a piece of code.",
        args_schema=CodeFormatInput,
    )

    docstring_tool = StructuredTool.from_function(
        func=analyser.extract_docstrings,
        name="extract_docstrings",
        description="Extract all docstrings from Python source code.",
        args_schema=CodeComplexityInput,
    )

    return [analyse_tool, line_count_tool, docstring_tool]
