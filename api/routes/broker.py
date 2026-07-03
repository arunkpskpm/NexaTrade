"""
NexaTrade — Broker Management Routes.

Endpoints:
    GET  /api/v1/broker/info          → broker info + capabilities
    GET  /api/v1/broker/list          → all registered brokers
    POST /api/v1/broker/switch        → hot-swap active broker
    GET  /api/v1/broker/funds         → margin and cash balance
    GET  /api/v1/broker/health        → broker connectivity check
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import (
    get_broker_service,
    get_current_user,
)
from api.schemas import (
    BrokerInfoResponse,
    BrokerSwitchRequest,
    FundsResponse,
    SuccessResponse,
)
from brokers.registry import list_registered_brokers
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get(
    "/info",
    response_model=BrokerInfoResponse,
    summary="Active broker info and capabilities",
)
async def get_broker_info(
    broker=Depends(get_broker_service),
    user: dict = Depends(get_current_user),
) -> BrokerInfoResponse:
    """Returns metadata and capabilities for the active broker."""
    info = await broker.get_broker_info()
    return BrokerInfoResponse(
        name=info.name,
        display_name=info.display_name,
        version=info.version,
        supports_websocket=info.supports_websocket,
        supports_historical_data=info.supports_historical_data,
        supports_paper_trading=info.supports_paper_trading,
        supports_options=info.supports_options,
        supports_futures=info.supports_futures,
        connection_state=str(info.connection_state),
        is_authenticated=info.is_authenticated,
        active_broker=broker.active_broker_name,
        trading_mode=broker._settings.trading_mode
        if hasattr(broker, "_settings")
        else "paper",
    )


@router.get(
    "/list",
    summary="All registered broker adapters",
)
async def list_brokers(
    user: dict = Depends(get_current_user),
) -> dict:
    """Returns all registered broker names."""
    return {
        "brokers": list_registered_brokers(),
        "count":   len(list_registered_brokers()),
    }


@router.post(
    "/switch",
    response_model=SuccessResponse,
    summary="Hot-swap active broker",
)
async def switch_broker(
    body: BrokerSwitchRequest,
    broker=Depends(get_broker_service),
    user: dict = Depends(get_current_user),
) -> SuccessResponse:
    """
    Switches the active broker at runtime without restart.
    All registered tick and order subscribers are preserved.

    Args:
        body.broker_name: Target broker identifier.
    """
    success = await broker.switch_broker(body.broker_name)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to connect to broker: {body.broker_name}",
        )
    logger.info(
        f"Broker switched via API | "
        f"new_broker={body.broker_name} | "
        f"user={user.get('username')}"
    )
    return SuccessResponse(
        message=f"Switched to broker: {body.broker_name}"
    )


@router.get(
    "/funds",
    response_model=FundsResponse,
    summary="Available funds and margin",
)
async def get_funds(
    broker=Depends(get_broker_service),
    user: dict = Depends(get_current_user),
) -> FundsResponse:
    """Returns available cash and margin from the active broker."""
    from config.settings import get_settings
    settings = get_settings()
    funds = await broker.get_funds()
    return FundsResponse(
        available_cash=funds.get("available_cash", 0.0),
        used_margin=funds.get("used_margin", 0.0),
        total_balance=funds.get("total_balance", 0.0),
        broker_name=broker.active_broker_name,
        trading_mode=settings.trading_mode,
    )


@router.get(
    "/health",
    summary="Broker connectivity health check",
)
async def broker_health(
    broker=Depends(get_broker_service),
    user: dict = Depends(get_current_user),
) -> dict:
    """Pings the active broker to verify connectivity."""
    health = await broker.health_check()
    return {
        "broker":    broker.active_broker_name,
        "connected": health.get("broker_connected", False),
        "running":   health.get("service_running", False),
        "is_paper":  health.get("is_paper", True),
    }