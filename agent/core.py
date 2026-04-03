"""Core agent implementation for MyEvolvingAgent.

Uses LangChain's ``create_agent`` (LangGraph-backed) with pluggable memory backends.
"""

from __future__ import annotations

import datetime
from typing import Any, Iterator, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import BaseTool

from agent.config import AgentConfig
from agent.memory import BaseMemory, create_memory
from agent.prompts import SYSTEM_PROMPT
from agent.tools.code_tool import create_code_tools
from agent.tools.github_tool import create_github_tools
from agent.tools.search import create_search_tool


class EvolvingAgent:
    """The core personal AI agent.

    Orchestrates an LLM with a dynamic tool-set and a pluggable memory backend.
    The agent continuously improves by integrating new tools and capabilities.

    Example::

        from agent import EvolvingAgent, AgentConfig

        config = AgentConfig()
        agent  = EvolvingAgent(config)
        reply  = agent.chat("What are the top Python LLM frameworks on GitHub?")
        print(reply)
    """

    def __init__(self, config: Optional[AgentConfig] = None) -> None:
        self.config = config or AgentConfig()
        self._llm: Any = None
        self._graph: Any = None
        self.memory: BaseMemory = create_memory(
            self.config.memory_type,
            llm=None,  # will be replaced after LLM init for SummaryMemory
            max_messages=self.config.max_memory_tokens // 100,
        )
        self._tools: list[BaseTool] = []
        self._initialised = False

    # ──────────────────────────────────────────────────────────────────────────
    # Initialisation
    # ──────────────────────────────────────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        return SYSTEM_PROMPT.format(
            agent_name=self.config.agent_name,
            current_date=datetime.date.today().isoformat(),
        )

    def _init_llm(self) -> None:
        """Lazy-initialise the LLM."""
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ImportError(
                "langchain-openai is required. Install with: pip install langchain-openai"
            ) from exc

        self._llm = ChatOpenAI(
            api_key=self.config.openai_api_key,
            model=self.config.openai_model,
            temperature=self.config.openai_temperature,
            max_tokens=self.config.openai_max_tokens,
        )

        # Re-initialise SummaryMemory with the now-available LLM
        if self.config.memory_type == "summary":
            self.memory = create_memory(
                "summary",
                llm=self._llm,
                summary_threshold=self.config.max_memory_tokens // 200,
            )

    def _init_tools(self) -> None:
        """Build the default tool set."""
        self._tools = []

        # Web search (no key required)
        self._tools.append(create_search_tool(max_results=self.config.max_search_results))

        # GitHub tools
        self._tools.extend(create_github_tools(github_token=self.config.github_token))

        # Code analysis tools
        self._tools.extend(create_code_tools())

    def _init_graph(self) -> None:
        """Build the LangGraph agent."""
        from langchain.agents import create_agent

        self._graph = create_agent(
            self._llm,
            tools=self._tools,
            system_prompt=self._build_system_prompt(),
        )

    def _ensure_init(self) -> None:
        """Ensure the agent is fully initialised (lazy)."""
        if not self._initialised:
            self._init_llm()
            self._init_tools()
            self._init_graph()
            self._initialised = True

    # ──────────────────────────────────────────────────────────────────────────
    # Public interface
    # ──────────────────────────────────────────────────────────────────────────

    def add_tool(self, tool: BaseTool) -> None:
        """Register an additional tool and reinitialise the graph.

        This is the primary mechanism for the agent to *evolve* – new
        capabilities can be added at runtime without restarting.
        """
        self._ensure_init()
        # Avoid duplicate tool names
        existing_names = {t.name for t in self._tools}
        if tool.name in existing_names:
            raise ValueError(f"A tool named '{tool.name}' is already registered.")
        self._tools.append(tool)
        self._init_graph()  # rebuild graph with new tool set

    def remove_tool(self, tool_name: str) -> None:
        """Remove a tool by name and reinitialise the graph."""
        self._ensure_init()
        self._tools = [t for t in self._tools if t.name != tool_name]
        self._init_graph()

    @property
    def tool_names(self) -> list[str]:
        """Return the names of all currently registered tools."""
        self._ensure_init()
        return [t.name for t in self._tools]

    def chat(self, user_input: str) -> str:
        """Send a message to the agent and return its reply as a string.

        Args:
            user_input: The user's message.

        Returns:
            The agent's response text.
        """
        self._ensure_init()
        history = self.memory.get_messages()
        messages = list(history) + [HumanMessage(content=user_input)]
        result = self._graph.invoke({"messages": messages})
        # The last message in the result is the AI response
        ai_messages = [m for m in result.get("messages", []) if isinstance(m, AIMessage)]
        response: str = ai_messages[-1].content if ai_messages else ""
        # Update memory
        self.memory.add_user_message(user_input)
        self.memory.add_ai_message(response)
        return response

    def stream(self, user_input: str) -> Iterator[str]:
        """Stream the agent's response token by token.

        Yields string chunks as the agent produces them. Falls back to a single
        yield when streaming is not natively available.

        Args:
            user_input: The user's message.

        Yields:
            Incremental text chunks.
        """
        self._ensure_init()
        history = self.memory.get_messages()
        messages = list(history) + [HumanMessage(content=user_input)]
        full_response = ""
        last_ai_content = ""
        for chunk in self._graph.stream({"messages": messages}, stream_mode="values"):
            msgs = chunk.get("messages", [])
            for msg in msgs:
                if isinstance(msg, AIMessage) and msg.content:
                    delta = msg.content[len(last_ai_content):]
                    if delta:
                        full_response += delta
                        last_ai_content = msg.content
                        yield delta
        # Update memory after streaming completes
        self.memory.add_user_message(user_input)
        self.memory.add_ai_message(full_response)

    def reset_memory(self) -> None:
        """Clear all conversation history."""
        self.memory.clear()

    def save_memory(self, path: str) -> None:
        """Persist the current conversation to a JSON file."""
        self.memory.save(path)

    def load_memory(self, path: str) -> None:
        """Restore a conversation from a JSON file."""
        self.memory.load(path)

    def get_history(self) -> list[dict[str, Any]]:
        """Return the conversation history as a list of dicts."""
        return self.memory.to_dict()

    def __repr__(self) -> str:
        tool_names = [t.name for t in self._tools] if self._initialised else []
        return (
            f"EvolvingAgent(name={self.config.agent_name!r}, "
            f"model={self.config.openai_model!r}, "
            f"tools={tool_names})"
        )
