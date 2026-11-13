"""Single configuration surface: every tunable comes from env vars (or a local .env file)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["anthropic", "ollama"]
Effort = Literal["low", "medium", "high", "xhigh", "max"]
LogLevel = Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"]


class Settings(BaseSettings):
    """Application settings. Field names map 1:1 to upper-case env vars (e.g. LLM_PROVIDER)."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM provider selection -------------------------------------------------
    llm_provider: LLMProvider = "anthropic"

    # Cloud mode (Claude). The key is optional here: if unset, the Anthropic SDK resolves
    # credentials from its own env vars / `ant auth login` profile.
    anthropic_api_key: SecretStr | None = None
    anthropic_model: str = "claude-opus-5-5"
    anthropic_effort: Effort = "low"
    anthropic_max_tokens: int = Field(default=16000, ge=256)

    # Local mode (Ollama).
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    ollama_num_ctx: int = Field(default=8192, ge=2048)
    ollama_temperature: float = Field(default=0.1, ge=0.0, le=2.0)

    # Resilience / resource limits. Timeout and concurrency cap apply to both providers;
    # retries apply to the Claude SDK only (Ollama is local, so it is not retried).
    llm_timeout_seconds: float = Field(default=60.0, gt=0)
    llm_max_retries: int = Field(default=2, ge=0, le=10)
    max_concurrent_llm_calls: int = Field(default=8, ge=1)

    # --- Retrieval ---------------------------------------------------------------
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_cache_dir: Path | None = None
    chunk_size: int = Field(default=800, ge=100, description="Max characters per chunk")
    chunk_overlap: int = Field(default=120, ge=0)
    top_k: int = Field(default=4, ge=1, le=10)
    min_relevance_score: float = Field(default=0.62, ge=-1.0, le=1.0)

    # --- Paths & lifecycle ---------------------------------------------------------
    docs_dir: Path = Path("data/docs")
    index_dir: Path = Path("data/index")
    ingest_on_startup: bool = False

    log_level: LogLevel = "INFO"

    @field_validator("log_level", mode="before")
    @classmethod
    def _upper_log_level(cls, value: object) -> object:
        return value.upper() if isinstance(value, str) else value

    @model_validator(mode="after")
    def _check_chunking(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size // 2:
            raise ValueError("CHUNK_OVERLAP must be smaller than half of CHUNK_SIZE")
        return self

    @property
    def active_llm_model(self) -> str:
        return self.anthropic_model if self.llm_provider == "anthropic" else self.ollama_model

