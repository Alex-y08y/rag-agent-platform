"""FastAPI dependencies for authentication."""
from __future__ import annotations

from fastapi import Depends, Header, HTTPException, status

from app.core.database import SessionLocal
from app.core.security import decode_jwt
from app.models.models import User


def get_current_user(authorization: str | None = Header(None)) -> User:
    """Extract and verify the current user from JWT token."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="未登录或登录已过期",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = authorization[7:]
    payload = decode_jwt(token)
    if not payload or "user_id" not in payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="登录已过期，请重新登录",
            headers={"WWW-Authenticate": "Bearer"},
        )
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == payload["user_id"]).first()
        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="用户不存在或已被禁用",
            )
        return user
    finally:
        db.close()


def get_current_user_optional(authorization: str | None = Header(None)) -> User | None:
    """Optional auth - returns user if token present, None otherwise."""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization[7:]
    payload = decode_jwt(token)
    if not payload or "user_id" not in payload:
        return None
    db = SessionLocal()
    try:
        return db.query(User).filter(User.id == payload["user_id"]).first()
    finally:
        db.close()
