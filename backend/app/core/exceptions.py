"""Custom application exceptions."""
from __future__ import annotations

from fastapi import HTTPException, status


class AppError(Exception):
    """Base application error."""

    def __init__(self, message: str, code: str = "APP_ERROR") -> None:
        super().__init__(message)
        self.message = message
        self.code = code


class NotFoundError(AppError):
    def __init__(self, resource: str, resource_id: str | int) -> None:
        super().__init__(f"{resource} not found: {resource_id}", "NOT_FOUND")


class ValidationError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(message, "VALIDATION_ERROR")


class FileUploadError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(message, "FILE_UPLOAD_ERROR")


class EmbeddingError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(message, "EMBEDDING_ERROR")


class RetrievalError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(message, "RETRIEVAL_ERROR")


class LLMError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(message, "LLM_ERROR")


class AgentError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(message, "AGENT_ERROR")


class SQLError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(message, "SQL_ERROR")


class SQLSafetyError(SQLError):
    def __init__(self, message: str) -> None:
        super().__init__(f"SQL safety check failed: {message}")


class EvaluationError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(message, "EVALUATION_ERROR")


def to_http_error(exc: AppError) -> HTTPException:
    """Convert AppError to FastAPI HTTPException."""
    status_map = {
        "NOT_FOUND": status.HTTP_404_NOT_FOUND,
        "VALIDATION_ERROR": status.HTTP_400_BAD_REQUEST,
        "FILE_UPLOAD_ERROR": status.HTTP_400_BAD_REQUEST,
        "EMBEDDING_ERROR": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "RETRIEVAL_ERROR": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "LLM_ERROR": status.HTTP_502_BAD_GATEWAY,
        "AGENT_ERROR": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "SQL_ERROR": status.HTTP_400_BAD_REQUEST,
        "EVALUATION_ERROR": status.HTTP_500_INTERNAL_SERVER_ERROR,
    }
    return HTTPException(
        status_code=status_map.get(exc.code, status.HTTP_500_INTERNAL_SERVER_ERROR),
        detail={"code": exc.code, "message": exc.message},
    )
