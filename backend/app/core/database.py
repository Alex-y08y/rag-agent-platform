"""Database engine and session management."""
from __future__ import annotations

from collections.abc import AsyncGenerator, Generator

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings

# Sync engine (for migrations, scripts)
engine = create_engine(
    settings.database_url,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    echo=settings.DEBUG,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

# Async engine (for FastAPI)
try:
    async_engine = create_async_engine(
        settings.async_database_url,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        echo=settings.DEBUG,
    )
    AsyncSessionLocal = async_sessionmaker(
        bind=async_engine, class_=AsyncSession, expire_on_commit=False
    )
except Exception:
    async_engine = None
    AsyncSessionLocal = None


def get_db() -> Generator[Session, None, None]:
    """Sync DB dependency."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """Async DB dependency."""
    if AsyncSessionLocal is None:
        raise RuntimeError("Async database engine is not available")
    async with AsyncSessionLocal() as session:
        yield session


def init_db() -> None:
    """Create all tables (used in dev / scripts)."""
    from app.models.models import Base

    Base.metadata.create_all(bind=engine)
