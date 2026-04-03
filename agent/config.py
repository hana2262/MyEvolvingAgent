"""Configuration management for MyEvolvingAgent."""

from __future__ import annotations

from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentConfig(BaseSettings):
    """Centralised configuration loaded from environment variables or a .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # ── LLM ──────────────────────────────────────────────────────────────────
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    openai_temperature: float = Field(default=0.7, alias="OPENAI_TEMPERATURE")
    openai_max_tokens: int = Field(default=4096, alias="OPENAI_MAX_TOKENS")

    # ── GitHub ────────────────────────────────────────────────────────────────
    github_token: Optional[str] = Field(default=None, alias="GITHUB_TOKEN")

    # ── Memory ────────────────────────────────────────────────────────────────
    memory_type: str = Field(default="buffer", alias="MEMORY_TYPE")
    # buffer | summary | vector
    max_memory_tokens: int = Field(default=4000, alias="MAX_MEMORY_TOKENS")
    chroma_persist_dir: str = Field(default="./chroma_db", alias="CHROMA_PERSIST_DIR")

    # ── Agent behaviour ───────────────────────────────────────────────────────
    agent_name: str = Field(default="MyEvolvingAgent", alias="AGENT_NAME")
    verbose: bool = Field(default=False, alias="AGENT_VERBOSE")
    max_iterations: int = Field(default=10, alias="AGENT_MAX_ITERATIONS")

    # ── Search ────────────────────────────────────────────────────────────────
    max_search_results: int = Field(default=5, alias="MAX_SEARCH_RESULTS")

    @field_validator("openai_temperature")
    @classmethod
    def _validate_temperature(cls, v: float) -> float:
        if not 0.0 <= v <= 2.0:
            raise ValueError("openai_temperature must be between 0.0 and 2.0")
        return v

    @field_validator("max_iterations")
    @classmethod
    def _validate_max_iterations(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_iterations must be at least 1")
        return v
