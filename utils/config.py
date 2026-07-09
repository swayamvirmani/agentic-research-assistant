"""
Configuration management using Pydantic Settings.
Loads from environment variables / .env file.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM ────────────────────────────────────────────────
    openai_api_key: str = Field(..., description="OpenAI API key")
    openai_model: str = Field("gpt-4o", description="Chat model name")
    openai_embedding_model: str = Field(
        "text-embedding-3-small", description="Embedding model"
    )
    cohere_api_key: str | None = Field(None, description="Cohere key for reranking")

    # ── Vector Store ────────────────────────────────────────
    faiss_index_path: str = Field("./data/faiss_index")
    vector_store_type: Literal["faiss", "chroma", "pinecone"] = "faiss"

    # ── RAG ────────────────────────────────────────────────
    chunk_size: int = Field(512, ge=64, le=4096)
    chunk_overlap: int = Field(64, ge=0, le=512)
    top_k_retrieval: int = Field(10, ge=1, le=50)
    top_k_rerank: int = Field(4, ge=1, le=20)
    hybrid_alpha: float = Field(0.5, ge=0.0, le=1.0)

    # ── Agents ─────────────────────────────────────────────
    max_iterations: int = Field(5, ge=1, le=20)
    reflection_threshold: float = Field(0.75, ge=0.0, le=1.0)
    agent_timeout: int = Field(60, ge=5, le=300)

    # ── API ────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:8080"]

    # ── Observability ───────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    enable_tracing: bool = True
    prometheus_port: int = 9090

    # ── Evaluation ─────────────────────────────────────────
    eval_dataset_path: str = "./data/eval_dataset.json"

    @field_validator("hybrid_alpha")
    @classmethod
    def validate_alpha(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("hybrid_alpha must be between 0 and 1")
        return v

    @property
    def has_cohere(self) -> bool:
        return self.cohere_api_key is not None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings singleton."""
    return Settings()


# Convenience alias
settings = get_settings()
