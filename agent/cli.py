"""CLI entry point for MyEvolvingAgent.

Usage::

    agent chat          # start an interactive REPL
    agent chat -q "..."  # single-shot question
    agent tools         # list available tools
    agent version       # show version
"""

from __future__ import annotations

import sys
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from agent import __version__
from agent.config import AgentConfig
from agent.core import EvolvingAgent

app = typer.Typer(
    name="agent",
    help="MyEvolvingAgent – your personal, continuously-evolving AI assistant.",
    add_completion=False,
)

console = Console()


def _build_agent() -> EvolvingAgent:
    config = AgentConfig()
    return EvolvingAgent(config)


@app.command()
def chat(
    question: Optional[str] = typer.Option(None, "--question", "-q", help="Ask a single question and exit."),
    no_stream: bool = typer.Option(False, "--no-stream", help="Disable streaming output."),
    save_history: Optional[str] = typer.Option(None, "--save", "-s", help="Save conversation history to a JSON file on exit."),
    load_history: Optional[str] = typer.Option(None, "--load", "-l", help="Load conversation history from a JSON file."),
) -> None:
    """Start an interactive chat session (or ask a single question)."""
    agent = _build_agent()

    if load_history:
        try:
            agent.load_memory(load_history)
            console.print(f"[dim]Loaded conversation history from {load_history}[/dim]")
        except Exception as exc:
            console.print(f"[yellow]Warning: could not load history: {exc}[/yellow]")

    if question:
        # Single-shot mode
        _ask_and_print(agent, question, stream=not no_stream)
        if save_history:
            agent.save_memory(save_history)
        return

    # Interactive REPL
    console.print(
        Panel(
            Text.from_markup(
                f"[bold cyan]MyEvolvingAgent v{__version__}[/bold cyan]\n"
                "[dim]Type your question and press Enter. "
                "Commands: /reset, /tools, /save <file>, /load <file>, /quit[/dim]"
            ),
            border_style="cyan",
        )
    )

    while True:
        try:
            user_input = Prompt.ask("\n[bold green]You[/bold green]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user_input.strip():
            continue

        # Built-in REPL commands
        if user_input.strip().lower() in ("/quit", "/exit", "/q"):
            console.print("[dim]Goodbye![/dim]")
            break
        if user_input.strip().lower() == "/reset":
            agent.reset_memory()
            console.print("[dim]Memory cleared.[/dim]")
            continue
        if user_input.strip().lower() == "/tools":
            console.print("[dim]Available tools:[/dim]")
            for name in agent.tool_names:
                console.print(f"  • {name}")
            continue
        if user_input.strip().startswith("/save "):
            path = user_input.split(maxsplit=1)[1].strip()
            agent.save_memory(path)
            console.print(f"[dim]History saved to {path}[/dim]")
            continue
        if user_input.strip().startswith("/load "):
            path = user_input.split(maxsplit=1)[1].strip()
            agent.load_memory(path)
            console.print(f"[dim]History loaded from {path}[/dim]")
            continue

        _ask_and_print(agent, user_input, stream=not no_stream)

    if save_history:
        agent.save_memory(save_history)


def _ask_and_print(agent: EvolvingAgent, question: str, stream: bool = True) -> None:
    console.print("\n[bold blue]Agent[/bold blue]")
    try:
        if stream:
            chunks = []
            for chunk in agent.stream(question):
                console.print(chunk, end="")
                chunks.append(chunk)
            console.print()  # newline after streaming
        else:
            response = agent.chat(question)
            console.print(Markdown(response))
    except Exception as exc:
        console.print(f"[red]Error: {exc}[/red]")


@app.command()
def tools() -> None:
    """List all available tools."""
    agent = _build_agent()
    console.print("[bold]Available tools:[/bold]")
    for name in agent.tool_names:
        console.print(f"  • {name}")


@app.command()
def version() -> None:
    """Show the agent version."""
    console.print(f"MyEvolvingAgent v{__version__}")


if __name__ == "__main__":
    app()
