"""Application configuration via environment variables."""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized application settings.

    All secrets are read from environment variables / .env file.
    No API keys are hardcoded.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── App ──────────────────────────────────────────────
    APP_NAME: str = "RAG Agent Platform"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    API_PREFIX: str = "/api"
    LITE_MODE: bool = False  # True: SQLite + in-memory vector/keyword stores (no Docker needed)

    # ── Auth ─────────────────────────────────────────────
    SECRET_KEY: str = "rag-agent-platform-dev-secret-change-in-production"
    JWT_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days

    # ── CORS ─────────────────────────────────────────────
    CORS_ORIGINS: str = "http://localhost:5173,http://localhost:3000"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    # ── LLM (OpenAI-compatible) ──────────────────────────
    LLM_PROVIDER: str = "openai"
    LLM_MODEL: str = "qwen3.8-flash"
    LLM_BASE_URL: str = "https://ws-yd9ijsp7q2ieflqf.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
    OPENAI_API_KEY: str = ""
    DASHSCOPE_API_KEY: str = ""
    LLM_TEMPERATURE: float = 0.3
    LLM_MAX_TOKENS: int = 2048
    LLM_TIMEOUT: int = 60

    @property
    def effective_api_key(self) -> str:
        return self.OPENAI_API_KEY or self.DASHSCOPE_API_KEY

    # ── Embedding ────────────────────────────────────────
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    EMBEDDING_DEVICE: str = "auto"  # auto | cpu | cuda
    EMBEDDING_BATCH_SIZE: int = 32
    EMBEDDING_CACHE_DIR: str = "./data/model_cache"

    # ── Reranker ─────────────────────────────────────────
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    RERANKER_DEVICE: str = "auto"
    RERANKER_TOP_N: int = 5

    # ── Milvus ───────────────────────────────────────────
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    MILVUS_COLLECTION_PREFIX: str = "rag_kb_"

    # ── Elasticsearch ────────────────────────────────────
    ELASTICSEARCH_URL: str = "http://localhost:9200"
    ELASTICSEARCH_INDEX_PREFIX: str = "rag-kb-"

    # ── PostgreSQL ───────────────────────────────────────
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "rag_platform"
    POSTGRES_USER: str = "rag_user"
    POSTGRES_PASSWORD: str = "rag_password"
    DATABASE_URL: str = ""  # Optional override (e.g. sqlite:///./data/rag.db)

    @property
    def database_url(self) -> str:
        if self.DATABASE_URL:
            return self.DATABASE_URL
        if self.LITE_MODE:
            return "sqlite:///./data/rag_lite.db"
        return (
            f"postgresql+psycopg2://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def async_database_url(self) -> str:
        if self.DATABASE_URL:
            # Convert sync URL to async for sqlite
            if self.DATABASE_URL.startswith("sqlite"):
                return self.DATABASE_URL.replace("sqlite://", "sqlite+aiosqlite://", 1)
            return self.DATABASE_URL
        if self.LITE_MODE:
            return "sqlite+aiosqlite:///./data/rag_lite.db"
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    # ── Redis ────────────────────────────────────────────
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str = ""

    @property
    def redis_url(self) -> str:
        auth = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
        return f"redis://{auth}{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    # ── File Upload ──────────────────────────────────────
    UPLOAD_DIR: str = "./data/uploads"
    MAX_FILE_SIZE_MB: int = 50
    ALLOWED_EXTENSIONS: str = ".pdf,.docx,.txt,.md,.csv,.xlsx"

    @property
    def allowed_extension_set(self) -> set[str]:
        return {e.strip().lower() for e in self.ALLOWED_EXTENSIONS.split(",") if e.strip()}

    # ── Chunking ─────────────────────────────────────────
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64

    # ── Retrieval ────────────────────────────────────────
    RETRIEVAL_TOP_K: int = 10
    RRF_K: int = 60  # RRF constant
    HYBRID_WEIGHT_VECTOR: float = 0.7
    HYBRID_WEIGHT_BM25: float = 0.3

    # ── Agent ────────────────────────────────────────────
    AGENT_MAX_ITERATIONS: int = 5
    AGENT_TIMEOUT_SECONDS: int = 120

    # ── Memory ───────────────────────────────────────────
    MEMORY_MAX_TURNS: int = 20
    MEMORY_SUMMARY_THRESHOLD: int = 15

    # ── Logging ──────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    LOG_DIR: str = "./data/logs"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
