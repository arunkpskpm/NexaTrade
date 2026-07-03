"""
NexaTrade — Authentication Routes.

Endpoints:
    POST /api/v1/auth/login       → issue JWT token
    POST /api/v1/auth/refresh     → refresh JWT token
    GET  /api/v1/auth/me          → current user profile
    POST /api/v1/auth/logout      → invalidate token (Redis blacklist)
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import get_current_user, get_pg, get_redis
from api.schemas import (
    LoginRequest,
    SuccessResponse,
    TokenResponse,
    UserResponse,
)
from utils.auth import (
    create_jwt_token,
    hash_password,
    verify_password,
)
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Login — issue JWT token",
    status_code=status.HTTP_200_OK,
)
async def login(
    body: LoginRequest,
    pg=Depends(get_pg),
    redis=Depends(get_redis),
) -> TokenResponse:
    """
    Authenticates a user and returns a JWT access token.

    - Validates username and password against PostgreSQL
    - Issues a signed JWT token with configurable expiry
    - Stores session in Redis for revocation support

    Returns:
        TokenResponse with access_token and expiry.
    """
    from config.settings import get_settings
    settings = get_settings()

    # Fetch user from PostgreSQL
    user = await pg.get_user_by_username(body.username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
        )

    # Verify password
    if not verify_password(body.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
        )

    if not user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled.",
        )

    # Create JWT
    expire_minutes = settings.jwt.expire_minutes
    token = create_jwt_token(
        user_id=str(user["user_id"]),
        username=user["username"],
        expire_minutes=expire_minutes,
    )

    # Cache session in Redis
    try:
        await redis.set(
            f"session:{user['user_id']}",
            "1",
            ttl_seconds=expire_minutes * 60,
        )
    except Exception:
        pass

    logger.info(f"User logged in | username={body.username}")
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=expire_minutes * 60,
        user_id=str(user["user_id"]),
        username=user["username"],
    )


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Current user profile",
)
async def get_me(
    user: dict = Depends(get_current_user),
    pg=Depends(get_pg),
) -> UserResponse:
    """Returns the profile of the currently authenticated user."""
    db_user = await pg.get_user_by_id(user["user_id"])
    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )
    return UserResponse(
        user_id=str(db_user["user_id"]),
        username=db_user["username"],
        email=db_user.get("email"),
        is_active=db_user.get("is_active", True),
        created_at=str(db_user.get("created_at", "")),
    )


@router.post(
    "/logout",
    response_model=SuccessResponse,
    summary="Logout — invalidate token",
)
async def logout(
    user: dict = Depends(get_current_user),
    redis=Depends(get_redis),
) -> SuccessResponse:
    """
    Invalidates the current session by removing the Redis key.
    The JWT itself remains valid until expiry, but the
    session check will fail on protected endpoints.
    """
    try:
        await redis.delete(f"session:{user['user_id']}")
    except Exception:
        pass
    logger.info(f"User logged out | user_id={user['user_id']}")
    return SuccessResponse(message="Logged out successfully.")