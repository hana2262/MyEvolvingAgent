"""Tests for memory backends."""

import json
import os
import tempfile

import pytest

from agent.memory import BufferMemory, SummaryMemory, VectorMemory, create_memory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


class TestBufferMemory:
    def test_add_and_retrieve(self):
        mem = BufferMemory()
        mem.add_user_message("Hello")
        mem.add_ai_message("Hi there!")
        msgs = mem.get_messages()
        assert len(msgs) == 2
        assert isinstance(msgs[0], HumanMessage)
        assert msgs[0].content == "Hello"
        assert isinstance(msgs[1], AIMessage)
        assert msgs[1].content == "Hi there!"

    def test_clear(self):
        mem = BufferMemory()
        mem.add_user_message("Hello")
        mem.clear()
        assert mem.get_messages() == []

    def test_max_messages_trim(self):
        mem = BufferMemory(max_messages=4)
        for i in range(10):
            mem.add_user_message(f"Message {i}")
        assert len(mem.get_messages()) == 4

    def test_to_dict(self):
        mem = BufferMemory()
        mem.add_user_message("Hello")
        mem.add_ai_message("World")
        data = mem.to_dict()
        assert data == [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "World"},
        ]

    def test_save_and_load(self, tmp_path):
        mem = BufferMemory()
        mem.add_user_message("Test question")
        mem.add_ai_message("Test answer")

        path = str(tmp_path / "memory.json")
        mem.save(path)

        mem2 = BufferMemory()
        mem2.load(path)
        msgs = mem2.get_messages()
        assert len(msgs) == 2
        assert msgs[0].content == "Test question"
        assert msgs[1].content == "Test answer"

    def test_load_preserves_roles(self, tmp_path):
        path = str(tmp_path / "mem.json")
        data = [
            {"role": "user", "content": "ping"},
            {"role": "assistant", "content": "pong"},
        ]
        with open(path, "w") as f:
            json.dump(data, f)

        mem = BufferMemory()
        mem.load(path)
        msgs = mem.get_messages()
        assert isinstance(msgs[0], HumanMessage)
        assert isinstance(msgs[1], AIMessage)


class TestCreateMemory:
    def test_creates_buffer(self):
        mem = create_memory("buffer")
        assert isinstance(mem, BufferMemory)

    def test_creates_buffer_case_insensitive(self):
        mem = create_memory("Buffer")
        assert isinstance(mem, BufferMemory)

    def test_summary_requires_llm(self):
        with pytest.raises(ValueError, match="llm"):
            create_memory("summary")

    def test_unknown_type(self):
        with pytest.raises(ValueError, match="Unknown memory type"):
            create_memory("nonexistent")
