"""Tests for AgentConfig."""

import os
import pytest
from pydantic import ValidationError

from agent.config import AgentConfig


def test_defaults():
    config = AgentConfig()
    assert config.openai_model == "gpt-4o-mini"
    assert config.openai_temperature == 0.7
    assert config.max_iterations == 10
    assert config.memory_type == "buffer"
    assert config.agent_name == "MyEvolvingAgent"
    assert config.max_search_results == 5


def test_env_override(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    monkeypatch.setenv("AGENT_NAME", "TestBot")
    monkeypatch.setenv("OPENAI_TEMPERATURE", "0.2")
    config = AgentConfig()
    assert config.openai_model == "gpt-4o"
    assert config.agent_name == "TestBot"
    assert config.openai_temperature == 0.2


def test_invalid_temperature():
    with pytest.raises(ValidationError):
        AgentConfig(openai_temperature=3.0)


def test_invalid_max_iterations():
    with pytest.raises(ValidationError):
        AgentConfig(max_iterations=0)


def test_temperature_boundary():
    cfg = AgentConfig(openai_temperature=0.0)
    assert cfg.openai_temperature == 0.0
    cfg = AgentConfig(openai_temperature=2.0)
    assert cfg.openai_temperature == 2.0
