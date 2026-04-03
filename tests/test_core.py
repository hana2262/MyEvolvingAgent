"""Tests for the core EvolvingAgent."""

import pytest
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from agent.config import AgentConfig
from agent.core import EvolvingAgent
from agent.memory import BufferMemory


class TestEvolvingAgentInit:
    def test_default_config(self):
        agent = EvolvingAgent()
        assert agent.config is not None
        assert agent.config.agent_name == "MyEvolvingAgent"
        assert not agent._initialised

    def test_custom_config(self):
        config = AgentConfig(agent_name="TestBot")
        agent = EvolvingAgent(config)
        assert agent.config.agent_name == "TestBot"

    def test_memory_is_buffer_by_default(self):
        agent = EvolvingAgent()
        assert isinstance(agent.memory, BufferMemory)

    def test_repr(self):
        agent = EvolvingAgent()
        r = repr(agent)
        assert "EvolvingAgent" in r
        # repr before init should not trigger LLM initialisation
        assert not agent._initialised


class TestEvolvingAgentToolManagement:
    def _make_agent_with_mock_graph(self):
        """Create an agent with a mocked LLM/graph to avoid real API calls."""
        agent = EvolvingAgent()
        agent._tools = []
        agent._initialised = True
        agent._graph = MagicMock()
        agent._graph.invoke.return_value = {"messages": [AIMessage(content="mock response")]}
        agent._graph.stream.return_value = iter([{"messages": [AIMessage(content="mock")]}])
        # Patch _init_graph so it doesn't actually call LangChain
        agent._init_graph = MagicMock()
        return agent

    def test_add_tool(self):
        from langchain_core.tools import BaseTool

        agent = self._make_agent_with_mock_graph()

        class DummyTool(BaseTool):
            name: str = "dummy_tool"
            description: str = "A dummy tool."

            def _run(self, *args, **kwargs):
                return "dummy"

            async def _arun(self, *args, **kwargs):
                return "dummy"

        dummy = DummyTool()
        agent.add_tool(dummy)
        assert "dummy_tool" in agent.tool_names

    def test_add_duplicate_tool_raises(self):
        from langchain_core.tools import BaseTool

        agent = self._make_agent_with_mock_graph()

        class DummyTool(BaseTool):
            name: str = "dup_tool"
            description: str = "Duplicate."

            def _run(self, *args, **kwargs):
                return ""

            async def _arun(self, *args, **kwargs):
                return ""

        dummy = DummyTool()
        agent.add_tool(dummy)
        with pytest.raises(ValueError, match="already registered"):
            agent.add_tool(dummy)

    def test_remove_tool(self):
        from langchain_core.tools import BaseTool

        agent = self._make_agent_with_mock_graph()

        class RemovableTool(BaseTool):
            name: str = "removable"
            description: str = "Will be removed."

            def _run(self, *args, **kwargs):
                return ""

            async def _arun(self, *args, **kwargs):
                return ""

        agent.add_tool(RemovableTool())
        assert "removable" in agent.tool_names
        agent.remove_tool("removable")
        assert "removable" not in agent.tool_names


class TestEvolvingAgentMemoryIntegration:
    def _make_agent_with_mock_graph(self):
        agent = EvolvingAgent()
        agent._tools = []
        agent._initialised = True
        agent._graph = MagicMock()
        agent._graph.invoke.return_value = {"messages": [AIMessage(content="mock response")]}
        return agent

    def test_chat_updates_memory(self):
        agent = self._make_agent_with_mock_graph()
        agent.chat("Hello")
        history = agent.get_history()
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello"
        assert history[1]["role"] == "assistant"

    def test_reset_memory(self):
        agent = self._make_agent_with_mock_graph()
        agent.chat("Hello")
        agent.reset_memory()
        assert agent.get_history() == []

    def test_save_and_load_memory(self, tmp_path):
        agent = self._make_agent_with_mock_graph()
        agent.chat("Saved question")
        path = str(tmp_path / "history.json")
        agent.save_memory(path)

        agent2 = self._make_agent_with_mock_graph()
        agent2.load_memory(path)
        history = agent2.get_history()
        assert any(h["content"] == "Saved question" for h in history)
