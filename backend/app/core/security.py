"""Security utilities: password hashing and JWT tokens (zero extra deps)."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any

from app.core.config import settings


# ── Password Hashing (PBKDF2-HMAC-SHA256) ─────────────────
def hash_password(password: str) -> str:
    """Hash a password using PBKDF2-HMAC-SHA256 with random salt."""
    import os
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
    return f"pbkdf2_sha256${base64.b64encode(salt).decode()}${base64.b64encode(dk).decode()}"


def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against a stored hash."""
    try:
        algo, salt_b64, hash_b64 = hashed.split("$")
        if algo != "pbkdf2_sha256":
            return False
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(hash_b64)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
        return hmac.compare_digest(dk, expected)
    except Exception:
        return False


# ── JWT (HS256, zero-dep implementation) ──────────────────
def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64decode(data: str) -> bytes:
    padding = 4 - len(data) % 4
    if padding != 4:
        data += "=" * padding
    return base64.urlsafe_b64decode(data)


def create_jwt(payload: dict[str, Any], expires_minutes: int | None = None) -> str:
    """Create a signed JWT (HS256)."""
    if expires_minutes is None:
        expires_minutes = settings.JWT_EXPIRE_MINUTES
    now = int(time.time())
    payload = {**payload, "iat": now, "exp": now + expires_minutes * 60}
    header = {"alg": "HS256", "typ": "JWT"}
    header_b64 = _b64encode(json.dumps(header, separators=(",", ":")).encode())
    payload_b64 = _b64encode(json.dumps(payload, separators=(",", ":")).encode())
    signing_input = f"{header_b64}.{payload_b64}".encode()
    signature = hmac.new(
        settings.SECRET_KEY.encode(), signing_input, hashlib.sha256
    ).digest()
    sig_b64 = _b64encode(signature)
    return f"{header_b64}.{payload_b64}.{sig_b64}"


def decode_jwt(token: str) -> dict[str, Any] | None:
    """Decode and verify a JWT. Returns payload or None if invalid."""
    try:
        header_b64, payload_b64, sig_b64 = token.split(".")
        signing_input = f"{header_b64}.{payload_b64}".encode()
        expected_sig = hmac.new(
            settings.SECRET_KEY.encode(), signing_input, hashlib.sha256
        ).digest()
        actual_sig = _b64decode(sig_b64)
        if not hmac.compare_digest(expected_sig, actual_sig):
            return None
        payload = json.loads(_b64decode(payload_b64))
        if payload.get("exp", 0) < int(time.time()):
            return None
        return payload
    except Exception:
        return None
