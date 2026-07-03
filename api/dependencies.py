"""
NexaTrade — FastAPI Dependency Injection Helpers.

All route dependencies are defined here.
Import and use with FastAPI's Depends() system.

Available dependencies:
    get_container()        → Container singleton
    get_broker_service()   → BrokerService
    get_feed_service()     → FeedService
    get_data_service()     → DataService
    get_strategy_engine()  → StrategyEngine
    get_risk_manager()     → RiskManager
    get_backtest_runner()  → BacktestRunner
    get_redis()            → RedisClient
    get_pg()               → PostgresClient
    get_current_user()     → Authenticated user (JWT)
    require_live_mode()    → Blocks if not live trading
    require_paper_mode()   → Blocks if not paper trading

Usage in routes:
    @router.get("/quote")
    async def get_quote(
        broker: BrokerService = Depends(get_broker_service),
        user: dict = Depends(get_current_user),
    ):
        ...
"""

from __future__ import annotations

from typing import Any

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config.settings import get_settings
from container import Container
from utils.auth import decode_jwt_token
from utils.logger import get_logger

logger = get_logger(__name__)
_bearer = HTTPBearer(auto_error=False)
settings = get_settings()


# ─────────────────────────────────────────────
# Container Accessor
# ─────────────────────────────────────────────

def get_container(request: Request) -> Container:
    """
    Returns the application Container from FastAPI app state.

    Args:
        request: FastAPI request object.

    Returns:
        Running Container instance.
    """
    return request.app.state.container


# ─────────────────────────────────────────────
# Service Dependencies
# ─────────────────────────────────────────────

def get_broker_service(
    container: Container = Depends(get_container),
):
    """Returns the active BrokerService."""
    return container.broker_service


def get_feed_service(
    container: Container = Depends(get_container),
):
    """Returns the FeedService."""
    return container.feed_service


def get_data_service(
    container: Container = Depends(get_container),
):
    """Returns the DataService."""
    return container.data_service


def get_strategy_engine(
    container: Container = Depends(get_container),
):
    """Returns the StrategyEngine."""
    return container.strategy_engine


def get_risk_manager(
    container: Container = Depends(get_container),
):
    """Returns the RiskManager."""
    return container.risk_manager


def get_backtest_runner(
    container: Container = Depends(get_container),
):
    """Returns the BacktestRunner."""
    return container.backtest_runner


def get_redis(
    container: Container = Depends(get_container),
):
    """Returns the RedisClient."""
    return container.redis


def get_pg(
    container: Container = Depends(get_container),
):
    """Returns the PostgresClient."""
    return container.pg


# ─────────────────────────────────────────────
# Authentication Dependencies
# ─────────────────────────────────────────────

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> dict[str, Any]:
    """
    Validates the JWT Bearer token and returns the decoded payload.

    Args:
        credentials: Bearer token from Authorization header.

    Returns:
        Decoded JWT payload dict.

    Raises:
        HTTPException 401: If token is missing, invalid, or expired.
    """
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    payload = decode_jwt_token(token)

    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return payload


async def get_optional_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> dict[str, Any] | None:
    """
    Returns the decoded JWT payload if token is present,
    or None for unauthenticated requests.
    Use for endpoints that support both auth and anonymous.
    """
    if not credentials or not credentials.credentials:
        return None
    return decode_jwt_token(credentials.credentials)


# ─────────────────────────────────────────────
# Mode Guard Dependencies
# ─────────────────────────────────────────────

def require_live_mode(
    user: dict = Depends(get_current_user),
) -> dict:
    """
    Blocks the request if the application is not in live mode.
    Use on endpoints that should only run with real broker.

    Raises:
        HTTPException 403: If trading_mode != "live".
    """
    if settings.trading_mode != "live":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "This endpoint requires TRADING_MODE=live. "
                f"Current mode: {settings.trading_mode}"
            ),
        )
    return user


def require_paper_mode(
    user: dict = Depends(get_current_user),
) -> dict:
    """
    Blocks the request if not in paper trading mode.
    Use on test/simulation endpoints.

    Raises:
        HTTPException 403: If trading_mode != "paper".
    """
    if settings.trading_mode != "paper":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "This endpoint requires TRADING_MODE=paper. "
                f"Current mode: {settings.trading_mode}"
            ),
        )
    return user