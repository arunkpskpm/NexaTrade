"""
NexaTrade — JWT Utilities.

Handles JWT token creation, decoding, and validation
for the local desktop authentication system.

Algorithm: HS256 (HMAC-SHA256)
Secret:    JWT_SECRET_KEY from .env (SecretStr)
Expiry:    JWT_ACCESS_TOKEN_EXPIRE_MINUTES from .env

Token payload schema:
    sub  → username (subject)
    iat  → issued-at timestamp
    exp  → expiry timestamp
    iss  → "nexatrade" (issuer)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import ExpiredSignatureError, JWTError, jwt
from pydantic import BaseModel

from utils.logger import get_logger

logger = get_logger(__name__)

# Issuer claim constant
_ISSUER = "nexatrade"


# ─────────────────────────────────────────────
# Custom Exceptions
# ─────────────────────────────────────────────

class TokenExpiredError(Exception):
    """Raised when a JWT token has passed its expiry time."""
    pass


class TokenInvalidError(Exception):
    """Raised when a JWT token fails signature or format validation."""
    pass


# ─────────────────────────────────────────────
# Token Payload Model
# ─────────────────────────────────────────────

class TokenPayload(BaseModel):
    """
    Structured representation of a decoded JWT payload.

    Attributes:
        sub: Subject — the authenticated username.
        iat: Issued-at timestamp.
        exp: Expiry timestamp.
        iss: Issuer — always 'nexatrade'.
    """
    sub: str
    iat: datetime
    exp: datetime
    iss: str = _ISSUER


# ─────────────────────────────────────────────
# Internal Config Loader
# ─────────────────────────────────────────────

def _get_jwt_config() -> tuple[str, str, int]:
    """
    Loads JWT configuration from settings.

    Returns:
        Tuple of (secret_key, algorithm, expire_minutes).
    """
    from config.settings import get_settings
    jwt_cfg = get_settings().jwt
    return (
        jwt_cfg.secret_key.get_secret_value(),
        jwt_cfg.algorithm,
        jwt_cfg.access_token_expire_minutes,
    )


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def create_access_token(
    subject: str,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Creates a signed JWT access token for the given subject.

    Args:
        subject: The username to embed as the token subject.
        expires_delta: Optional custom expiry duration.
                       Defaults to JWT_ACCESS_TOKEN_EXPIRE_MINUTES.

    Returns:
        Signed JWT string.

    Example:
        token = create_access_token(subject="admin")
    """
    secret_key, algorithm, expire_minutes = _get_jwt_config()

    now = datetime.now(tz=timezone.utc)
    expire = now + (
        expires_delta
        if expires_delta
        else timedelta(minutes=expire_minutes)
    )

    payload = {
        "sub": subject,
        "iat": now,
        "exp": expire,
        "iss": _ISSUER,
    }

    token = jwt.encode(payload, secret_key, algorithm=algorithm)
    logger.debug(
        f"Access token created | subject={subject} | "
        f"expires_at={expire.strftime('%Y-%m-%d %H:%M:%S UTC')}"
    )
    return token


def decode_access_token(token: str) -> TokenPayload:
    """
    Decodes and validates a JWT access token.

    Validation checks:
    - Signature verification using JWT_SECRET_KEY
    - Expiry check (exp claim)
    - Issuer check (iss == 'nexatrade')

    Args:
        token: Raw JWT string.

    Returns:
        TokenPayload with subject, iat, exp, iss.

    Raises:
        TokenExpiredError: If the token has expired.
        TokenInvalidError: If signature or format is invalid.

    Example:
        try:
            payload = decode_access_token(token)
            username = payload.sub
        except TokenExpiredError:
            # Redirect to login
            ...
    """
    secret_key, algorithm, _ = _get_jwt_config()

    try:
        raw_payload = jwt.decode(
            token,
            secret_key,
            algorithms=[algorithm],
            issuer=_ISSUER,
            options={"verify_exp": True},
        )
    except ExpiredSignatureError as exc:
        logger.debug("JWT token expired.")
        raise TokenExpiredError("Token has expired. Please log in again.") from exc
    except JWTError as exc:
        logger.warning(f"JWT validation failed | error={exc}")
        raise TokenInvalidError(
            f"Token is invalid or tampered: {exc}"
        ) from exc

    try:
        payload = TokenPayload(
            sub=raw_payload["sub"],
            iat=datetime.fromtimestamp(
                raw_payload["iat"], tz=timezone.utc
            ),
            exp=datetime.fromtimestamp(
                raw_payload["exp"], tz=timezone.utc
            ),
            iss=raw_payload.get("iss", _ISSUER),
        )
    except (KeyError, ValueError) as exc:
        raise TokenInvalidError(
            f"Token payload is malformed: {exc}"
        ) from exc

    return payload


def is_token_valid(token: str) -> bool:
    """
    Non-raising wrapper around decode_access_token.
    Returns True if the token is valid and not expired.

    Args:
        token: Raw JWT string.

    Returns:
        True if valid, False if expired or invalid.

    Example:
        if not is_token_valid(token):
            redirect_to_login()
    """
    if not token:
        return False
    try:
        decode_access_token(token)
        return True
    except (TokenExpiredError, TokenInvalidError):
        return False


def get_token_subject(token: str) -> Optional[str]:
    """
    Extracts the subject (username) from a token without raising.

    Args:
        token: Raw JWT string.

    Returns:
        Username string or None if token is invalid/expired.

    Example:
        username = get_token_subject(token)
        if username:
            load_user_session(username)
    """
    try:
        payload = decode_access_token(token)
        return payload.sub
    except (TokenExpiredError, TokenInvalidError):
        return None


def token_expires_in(token: str) -> Optional[int]:
    """
    Returns the number of seconds until the token expires.
    Returns None if the token is invalid.
    Returns 0 if the token has already expired.

    Args:
        token: Raw JWT string.

    Returns:
        Seconds until expiry, 0 if expired, None if invalid.

    Example:
        secs = token_expires_in(token)
        if secs and secs < 300:
            refresh_token()
    """
    try:
        payload = decode_access_token(token)
        now = datetime.now(tz=timezone.utc)
        remaining = (payload.exp - now).total_seconds()
        return max(0, int(remaining))
    except TokenExpiredError:
        return 0
    except TokenInvalidError:
        return None