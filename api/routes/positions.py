"""
NexaTrade — Positions & Holdings Routes.

Endpoints:
    GET /api/v1/positions/open       → all open positions
    GET /api/v1/positions/holdings   → delivery holdings
    GET /api/v1/positions/summary    → portfolio P&L summary
"""

from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import get_broker_service, get_current_user
from api.schemas import PortfolioSummaryResponse, PositionResponse
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


def _map_position(pos, broker_name: str, mode: str) -> PositionResponse:
    """Maps broker Position model to API schema."""
    total_pnl = (
        pos.unrealized_pnl + pos.realized_pnl
    )
    market_value = abs(pos.quantity) * pos.last_price
    return PositionResponse(
        symbol=pos.symbol,
        exchange=str(pos.exchange),
        segment=str(pos.segment),
        quantity=pos.quantity,
        average_price=pos.average_price,
        last_price=pos.last_price,
        unrealized_pnl=pos.unrealized_pnl,
        realized_pnl=pos.realized_pnl,
        total_pnl=total_pnl,
        market_value=market_value,
        broker_name=broker_name,
        trading_mode=mode,
        is_long=pos.quantity > 0,
        is_short=pos.quantity < 0,
    )


@router.get(
    "/open",
    response_model=list[PositionResponse],
    summary="All open positions",
)
async def get_positions(
    broker=Depends(get_broker_service),
    user: dict = Depends(get_current_user),
) -> list[PositionResponse]:
    """Returns all non-zero positions from the active broker."""
    from config.settings import get_settings
    settings = get_settings()
    try:
        positions = await broker.get_positions()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Positions fetch failed: {exc}",
        )
    return [
        _map_position(p, broker.active_broker_name, settings.trading_mode)
        for p in positions
        if p.quantity != 0
    ]


@router.get(
    "/holdings",
    response_model=list[PositionResponse],
    summary="Delivery holdings",
)
async def get_holdings(
    broker=Depends(get_broker_service),
    user: dict = Depends(get_current_user),
) -> list[PositionResponse]:
    """Returns delivery holdings from the active broker."""
    from config.settings import get_settings
    settings = get_settings()
    try:
        holdings = await broker.broker.get_holdings()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Holdings fetch failed: {exc}",
        )
    return [
        _map_position(h, broker.active_broker_name, settings.trading_mode)
        for h in holdings
    ]


@router.get(
    "/summary",
    response_model=PortfolioSummaryResponse,
    summary="Portfolio P&L summary",
)
async def get_portfolio_summary(
    broker=Depends(get_broker_service),
    user: dict = Depends(get_current_user),
) -> PortfolioSummaryResponse:
    """Returns a rolled-up portfolio P&L summary."""
    from config.settings import get_settings
    settings = get_settings()
    try:
        positions = await broker.get_positions()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Portfolio summary failed: {exc}",
        )

    open_pos  = [p for p in positions if p.quantity != 0]
    long_pos  = [p for p in open_pos if p.quantity > 0]
    short_pos = [p for p in open_pos if p.quantity < 0]
    total_unrealised = sum(p.unrealized_pnl for p in open_pos)
    total_realised   = sum(p.realized_pnl   for p in open_pos)

    return PortfolioSummaryResponse(
        total_positions=len(open_pos),
        long_positions=len(long_pos),
        short_positions=len(short_pos),
        total_unrealized_pnl=round(total_unrealised, 2),
        total_realized_pnl=round(total_realised, 2),
        total_pnl=round(total_unrealised + total_realised, 2),
        positions=[
            _map_position(
                p,
                broker.active_broker_name,
                settings.trading_mode,
            )
            for p in open_pos
        ],
    )