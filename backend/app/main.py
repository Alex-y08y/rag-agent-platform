"""FastAPI application entry point for RAG Agent Platform."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import agent, auth, chat, conversations, documents, evaluation, knowledge_bases, retrieval, settings as settings_router
from app.core.config import settings
from app.core.database import init_db
from app.core.exceptions import AppError, to_http_error
from app.core.logging import get_logger, setup_logging

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown events."""
    setup_logging()
    logger.info("Starting %s v%s", settings.APP_NAME, settings.APP_VERSION)
    try:
        init_db()
        logger.info("Database initialized")
    except Exception as exc:
        logger.warning("Database init failed (will retry on first use): %s", exc)
    yield
    logger.info("Shutting down %s", settings.APP_NAME)


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Enterprise RAG + Agent Knowledge Base Platform",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handler
@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError):
    return to_http_error(exc)


# Health check
@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    services = {}

    # Check database
    try:
        from app.core.database import SessionLocal
        from sqlalchemy import text
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        services["postgres"] = "healthy"
    except Exception as exc:
        services["postgres"] = f"unhealthy: {exc}"

    # Check Redis
    try:
        from app.memory.redis_memory import get_memory
        memory = get_memory()
        client = memory._get_client()
        client.ping()
        services["redis"] = "healthy"
    except Exception as exc:
        services["redis"] = f"unhealthy: {exc}"

    return {
        "status": "healthy" if all(v == "healthy" for v in services.values()) else "degraded",
        "version": settings.APP_VERSION,
        "services": services,
    }


# Register routers
app.include_router(auth.router, prefix=settings.API_PREFIX)
app.include_router(chat.router, prefix=settings.API_PREFIX)
app.include_router(conversations.router, prefix=settings.API_PREFIX)
app.include_router(documents.router, prefix=settings.API_PREFIX)
app.include_router(knowledge_bases.router, prefix=settings.API_PREFIX)
app.include_router(retrieval.router, prefix=settings.API_PREFIX)
app.include_router(evaluation.router, prefix=settings.API_PREFIX)
app.include_router(agent.router, prefix=settings.API_PREFIX)
app.include_router(settings_router.router, prefix=settings.API_PREFIX)


@app.get("/")
async def root():
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "api_prefix": settings.API_PREFIX,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
    )
