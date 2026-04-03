"""Memory management for MyEvolvingAgent.

Supports three memory backends:
- ``buffer``  – simple in-memory sliding-window (no external dependencies).
- ``summary`` – uses the LLM to periodically summarise older turns.
- ``vector``  – stores every turn in a ChromaDB vector store for semantic retrieval
               (requires the ``vector`` extra: ``pip install my-evolving-agent[vector]``).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage


class BaseMemory(ABC):
    """Abstract base class for agent memory backends."""

    @abstractmethod
    def add_user_message(self, content: str) -> None:
        """Record a user message."""

    @abstractmethod
    def add_ai_message(self, content: str) -> None:
        """Record an AI message."""

    @abstractmethod
    def get_messages(self) -> list[BaseMessage]:
        """Return the full message history as LangChain message objects."""

    @abstractmethod
    def clear(self) -> None:
        """Wipe the memory."""

    def to_dict(self) -> list[dict[str, Any]]:
        """Serialise memory to a JSON-friendly list."""
        result = []
        for msg in self.get_messages():
            if isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            elif isinstance(msg, SystemMessage):
                role = "system"
            else:
                role = "unknown"
            result.append({"role": role, "content": msg.content})
        return result

    def save(self, path: str) -> None:
        """Persist memory to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        """Restore memory from a JSON file (clears current memory first)."""
        self.clear()
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            role = item.get("role", "user")
            content = item.get("content", "")
            if role == "user":
                self.add_user_message(content)
            elif role == "assistant":
                self.add_ai_message(content)


class BufferMemory(BaseMemory):
    """Simple sliding-window conversation buffer.

    Keeps the most recent ``max_messages`` message pairs (user + AI).
    """

    def __init__(self, max_messages: int = 20) -> None:
        self._messages: list[BaseMessage] = []
        self.max_messages = max_messages

    def add_user_message(self, content: str) -> None:
        self._messages.append(HumanMessage(content=content))
        self._trim()

    def add_ai_message(self, content: str) -> None:
        self._messages.append(AIMessage(content=content))
        self._trim()

    def _trim(self) -> None:
        # Keep at most ``max_messages`` messages (pairs of user+ai = 2 each).
        if len(self._messages) > self.max_messages:
            self._messages = self._messages[-self.max_messages :]

    def get_messages(self) -> list[BaseMessage]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()


class SummaryMemory(BaseMemory):
    """Memory that summarises older portions of the conversation using an LLM.

    When the buffer exceeds ``summary_threshold`` messages, the oldest half is
    summarised and replaced with a single SystemMessage containing the summary.
    """

    def __init__(self, llm: Any, summary_threshold: int = 10) -> None:
        self._messages: list[BaseMessage] = []
        self._llm = llm
        self.summary_threshold = summary_threshold

    def add_user_message(self, content: str) -> None:
        self._messages.append(HumanMessage(content=content))
        self._maybe_summarise()

    def add_ai_message(self, content: str) -> None:
        self._messages.append(AIMessage(content=content))
        self._maybe_summarise()

    def _maybe_summarise(self) -> None:
        if len(self._messages) <= self.summary_threshold:
            return

        half = len(self._messages) // 2
        to_summarise = self._messages[:half]
        self._messages = self._messages[half:]

        text = "\n".join(
            f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
            for m in to_summarise
        )
        prompt = (
            "Please provide a concise summary of the following conversation excerpt:\n\n"
            f"{text}\n\nSummary:"
        )
        summary = self._llm.invoke(prompt).content
        self._messages.insert(0, SystemMessage(content=f"[Conversation summary]: {summary}"))

    def get_messages(self) -> list[BaseMessage]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()


class VectorMemory(BaseMemory):
    """Semantic memory backed by ChromaDB.

    Each message is embedded and stored; retrieval returns the most relevant
    past messages for a given query rather than the most recent ones.

    Requires: ``pip install my-evolving-agent[vector]``
    """

    def __init__(self, persist_directory: str = "./chroma_db", k: int = 5) -> None:
        try:
            import chromadb
            from chromadb.utils import embedding_functions
        except ImportError as exc:
            raise ImportError(
                "VectorMemory requires chromadb. "
                "Install it with: pip install my-evolving-agent[vector]"
            ) from exc

        self._client = chromadb.PersistentClient(path=persist_directory)
        ef = embedding_functions.DefaultEmbeddingFunction()
        self._collection = self._client.get_or_create_collection(
            name="agent_memory", embedding_function=ef
        )
        self._k = k
        self._buffer: list[BaseMessage] = []  # recent messages kept in RAM too

    def add_user_message(self, content: str) -> None:
        self._store("user", content)
        self._buffer.append(HumanMessage(content=content))

    def add_ai_message(self, content: str) -> None:
        self._store("assistant", content)
        self._buffer.append(AIMessage(content=content))

    def _store(self, role: str, content: str) -> None:
        import uuid

        doc_id = str(uuid.uuid4())
        self._collection.add(
            documents=[content],
            metadatas=[{"role": role}],
            ids=[doc_id],
        )

    def search(self, query: str) -> list[BaseMessage]:
        """Return the *k* most semantically relevant past messages."""
        if self._collection.count() == 0:
            return []
        results = self._collection.query(query_texts=[query], n_results=min(self._k, self._collection.count()))
        messages: list[BaseMessage] = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            if meta["role"] == "user":
                messages.append(HumanMessage(content=doc))
            else:
                messages.append(AIMessage(content=doc))
        return messages

    def get_messages(self) -> list[BaseMessage]:
        return list(self._buffer)

    def clear(self) -> None:
        self._buffer.clear()
        self._client.delete_collection("agent_memory")


def create_memory(memory_type: str, llm: Any = None, **kwargs: Any) -> BaseMemory:
    """Factory function – create a memory instance by type name."""
    memory_type = memory_type.lower()
    if memory_type == "buffer":
        return BufferMemory(**kwargs)
    if memory_type == "summary":
        if llm is None:
            raise ValueError("SummaryMemory requires an 'llm' argument.")
        return SummaryMemory(llm=llm, **kwargs)
    if memory_type == "vector":
        return VectorMemory(**kwargs)
    raise ValueError(f"Unknown memory type: '{memory_type}'. Choose from: buffer, summary, vector.")
