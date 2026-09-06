"""Authentication API routes: register, login, me."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.core.database import SessionLocal
from app.core.deps import get_current_user
from app.core.security import create_jwt, hash_password, verify_password
from app.models.models import User

router = APIRouter(prefix="/auth", tags=["auth"])


# ── Schemas ───────────────────────────────────────────────
class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=64)
    email: str = Field(..., pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    password: str = Field(..., min_length=6, max_length=128)


class LoginRequest(BaseModel):
    username: str
    password: str


class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    role: str

    model_config = {"from_attributes": True}


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


# ── Routes ────────────────────────────────────────────────
@router.post("/register", response_model=AuthResponse)
def register(req: RegisterRequest):
    """Register a new user."""
    db = SessionLocal()
    try:
        existing = (
            db.query(User)
            .filter((User.username == req.username) | (User.email == req.email))
            .first()
        )
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="用户名或邮箱已被注册",
            )
        user = User(
            username=req.username,
            email=req.email,
            hashed_password=hash_password(req.password),
            role="user",
        )
        db.add(user)
        db.commit()
        db.refresh(user)

        token = create_jwt({"user_id": user.id, "username": user.username})
        return AuthResponse(
            access_token=token,
            user=UserResponse.model_validate(user),
        )
    finally:
        db.close()


@router.post("/login", response_model=AuthResponse)
def login(req: LoginRequest):
    """Login with username and password."""
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == req.username).first()
        if not user or not verify_password(req.password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="用户名或密码错误",
            )
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="账号已被禁用",
            )
        token = create_jwt({"user_id": user.id, "username": user.username})
        return AuthResponse(
            access_token=token,
            user=UserResponse.model_validate(user),
        )
    finally:
        db.close()


@router.get("/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)):
    """Get current user info."""
    return UserResponse.model_validate(current_user)
