"""
NexaTrade — Order Management Routes.

Endpoints:
    POST   /api/v1/orders/place          → place new order
    PUT    /api/v1/orders/{order_id}     → modify open order
    DELETE /api/v1/orders/{order_id}     → cancel open order
    GET    /api/v1/orders/{order_id}     → single order status
    GET    /api/v1/orders/history        → order history
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.dependencies import get_broker_service, get_current_user, get_pg
from api.schemas import (
    ModifyOrderRequest,
    OrderResponse,
    PlaceOrderRequest,
    SuccessResponse,
)
from brokers.models import (
    Exchange,
    OrderModifyRequest,
    OrderRequest,
    OrderType,
    ProductType,
    Segment,
    TradingMode,
    TransactionType,
)
from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()
settings = get_settings()


def _map_order_response(response) -> OrderResponse:
    """Maps broker OrderResponse to API schema."""
    return OrderResponse(
        order_id=response.order_id,
        broker_order_id=response.broker_order_id,
        status=str(response.status),
        message=response.message or "",
        rejection_reason=response.rejection_reason,
        filled_quantity=response.filled_quantity,
        average_price=response.average_price,
        placed_at=str(response.placed_at) if response.placed_at else None,
        updated_at=str(response.updated_at) if response.updated_at else None,
        broker_name=response.broker_name,
        trading_mode=str(response.trading_mode),
    )


@router.post(
    "/place",
    response_model=OrderResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Place a new order",
)
async def place_order(
    body: PlaceOrderRequest,
    broker=Depends(get_broker_service),
    pg=Depends(get_pg),
    user: dict = Depends(get_current_user),
) -> OrderResponse:
    """
    Places a new order via the active broker.
    Routes through BrokerService (not strategy engine).
    For strategy signals use the strategy endpoints.
    """
    try:
        request = OrderRequest(
            symbol=body.symbol,
            exchange=Exchange(body.exchange),
            segment=Segment.EQ,
            transaction_type=TransactionType(body.transaction_type),
            order_type=OrderType(body.order_type),
            product_type=ProductType(body.product_type),
            quantity=body.quantity,
            price=body.price,
            trigger_price=body.trigger_price,
            trading_mode=(
                TradingMode.LIVE
                if settings.trading_mode == "live"
                else TradingMode.PAPER
            ),
        )
        response = await broker.place_order(request)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Order validation failed: {exc}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Order placement failed: {exc}",
        )

    logger.info(
        f"Manual order placed | "
        f"symbol={body.symbol} | "
        f"side={body.transaction_type} | "
        f"qty={body.quantity} | "
        f"user={user.get('username')}"
    )
    return _map_order_response(response)


@router.put(
    "/{order_id}",
    response_model=OrderResponse,
    summary="Modify an open order",
)
async def modify_order(
    order_id: str,
    body: ModifyOrderRequest,
    broker=Depends(get_broker_service),
    user: dict = Depends(get_current_user),
) -> OrderResponse:
    """Modifies an existing open or partial order."""
    try:
        request = OrderModifyRequest(
            order_id=order_id,
            broker_order_id=body.broker_order_id,
            quantity=body.quantity,
            price=body.price,
            trigger_price=body.trigger_price,
        )
        response = await broker.broker.modify_order(request)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Order modification failed: {exc}",
        )
    return _map_order_response(response)


@router.delete(
    "/{order_id}",
    response_model=OrderResponse,
    summary="Cancel an open order",
)
async def cancel_order(
    order_id: str,
    broker_order_id: str = Query(...),
    broker=Depends(get_broker_service),
    user: dict = Depends(get_current_user),
) -> OrderResponse:
    """Cancels an open order by NexaTrade order ID."""
    try:
        response = await broker.broker.cancel_order(
            order_id, broker_order_id
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Order cancellation failed: {exc}",
        )
    logger.info(
        f"Order cancelled | order_id={order_id} | "
        f"user={user.get('username')}"
    )
    return _map_order_response(response)


@router.get(
    "/history",
    summary="Order history",
)
async def get_order_history(
    from_date: str = Query(default=None),
    to_date:   str = Query(default=None),
    broker=Depends(get_broker_service),
    user: dict = Depends(get_current_user),
) -> dict:
    """Returns order history for a date range."""
    try:
        orders = await broker.broker.get_order_history(
            from_date=from_date, to_date=to_date
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Order history fetch failed: {exc}",
        )
    return {
        "orders": [_map_order_response(o).model_dump() for o in orders],
        "count":  len(orders),
    }


@router.get(
    "/{order_id}",
    response_model=OrderResponse,
    summary="Single order status",
)
async def get_order_status(
    order_id: str,
    broker_order_id: str = Query(...),
    broker=Depends(get_broker_service),
    user: dict = Depends(get_current_user),
) -> OrderResponse:
    """Returns the current status of a specific order."""
    try:
        response = await broker.broker.get_order_status(
            order_id, broker_order_id
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Order status fetch failed: {exc}",
        )
    return _map_order_response(response)