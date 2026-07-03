"""
NexaTrade — JWT Authentication Utilities.

All JWT creation, validation, and password hashing
is centralised here.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from jose import ExpiredSignatureError, JWTError, jwt
from passlib.context import CryptContext

from config.settings import get_settings
from utils.logger import get_logger

logger     = get_logger(__name__)
_pwd_ctx   = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    """Returns a bcrypt hash of the given password."""
    return _pwd_ctx.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    """Returns True if plain matches the bcrypt hash."""
    return _pwd_ctx.verify(plain, hashed)


def create_jwt_token(
    user_id: str,
    username: str,
    expire_minutes: Optional[int] = None,
    extra_claims: Optional[dict[str, Any]] = None,
) -> str:
    """
    Creates a signed JWT access token.

    Args:
        user_id: User UUID string.
        username: Username for the sub claim.
        expire_minutes: Token lifetime. Defaults to settings value.
        extra_claims: Additional payload claims.

    Returns:
        Signed JWT token string.
    """
    settings = get_settings()
    exp_mins = expire_minutes or settings.jwt.expire_minutes
    expire   = datetime.now(timezone.utc) + timedelta(minutes=exp_mins)

    payload: dict[str, Any] = {
        "sub":      username,
        "user_id":  user_id,
        "exp":      expire,
        "iat":      datetime.now(timezone.utc),
    }
    if extra_claims:
        payload.update(extra_claims)

    return jwt.encode(
        payload,
        settings.jwt.secret_key.get_secret_value(),
        algorithm=settings.jwt.algorithm,
    )


def decode_jwt_token(token: str) -> Optional[dict[str, Any]]:
    """
    Decodes and validates a JWT token.

    Args:
        token: JWT token string.

    Returns:
        Decoded payload dict, or None if invalid/expired.
    """
    settings = get_settings()
    try:
        return jwt.decode(
            token,
            settings.jwt.secret_key.get_secret_value(),
            algorithms=[settings.jwt.algorithm],
        )
    except ExpiredSignatureError:
        logger.debug("JWT token expired.")
        return None
    except JWTError as exc:
        logger.debug(f"JWT decode error: {exc}")
        return None