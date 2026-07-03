"""
NexaTrade — Live Feed Routes.

Endpoints:
    POST /api/v1/feed/subscribe       → subscribe to live feed
    POST /api/v1/feed/unsubscribe     → unsubscribe from feed
    GET  /api/v1/feed/candles         → latest candles from aggregator
    GET  /api/v1/feed/stats           → feed service statistics
    GET  /api/v1/feed/last-price      → last known price for a symbol
"""

from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.dependencies import get_current_user, get_feed_service
from api.schemas import FeedStatsResponse, SubscribeRequest, SuccessResponse
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post(
    "/subscribe",
    response_model=SuccessResponse,
    summary="Subscribe to live market feed",
)
async def subscribe(
    body: SubscribeRequest,
    feed=Depends(get_feed_service),
    user: dict = Depends(get_current_user),
) -> SuccessResponse:
    """
    Subscribes to live tick and candle feed for an instrument.
    Uses ref-counting — safe to call multiple times.
    """
    consumer_id = await feed.subscribe(
        symbol=body.symbol,
        exchange=body.exchange,
        interval=body.interval,
        consumer_id=f"api:{user.get('user_id', 'anon')}",
        seed_history=True,
    )
    return SuccessResponse(
        message=(
            f"Subscribed to {body.symbol}:{body.exchange} "
            f"({body.interval})"
        ),
        data={"consumer_id": consumer_id},
    )


@router.post(
    "/unsubscribe",
    response_model=SuccessResponse,
    summary="Unsubscribe from live market feed",
)
async def unsubscribe(
    body: SubscribeRequest,
    feed=Depends(get_feed_service),
    user: dict = Depends(get_current_user),
) -> SuccessResponse:
    """Unsubscribes from the live feed for an instrument."""
    consumer_id = f"api:{user.get('user_id', 'anon')}"
    await feed.unsubscribe(
        symbol=body.symbol,
        exchange=body.exchange,
        consumer_id=consumer_id,
        interval=body.interval,
    )
    return SuccessResponse(
        message=f"Unsubscribed from {body.symbol}:{body.exchange}"
    )


@router.get(
    "/candles",
    summary="Latest N candles from live aggregator",
)
async def get_feed_candles(
    symbol:   str = Query(...),
    exchange: str = Query(default="NSE"),
    interval: str = Query(default="5minute"),
    n:        int = Query(default=100, ge=1, le=500),
    feed=Depends(get_feed_service),
    user: dict = Depends(get_current_user),
) -> dict:
    """
    Returns candles from the in-memory live aggregator buffer.
    Faster than the /data/historical endpoint (no I/O).
    """
    candles = feed.get_candles(
        symbol.upper(), exchange.upper(), interval, n=n
    )
    return {
        "symbol":   symbol.upper(),
        "exchange": exchange.upper(),
        "interval": interval,
        "bars":     len(candles),
        "data": [
            {
                "datetime": str(c.datetime),
                "open":     c.open,
                "high":     c.high,
                "low":      c.low,
                "close":    c.close,
                "volume":   c.volume,
            }
            for c in candles
        ],
    }


@router.get(
    "/last-price",
    summary="Last known price for a symbol",
)
async def get_last_price(
    symbol:   str = Query(...),
    exchange: str = Query(default="NSE"),
    feed=Depends(get_feed_service),
    user: dict = Depends(get_current_user),
) -> dict:
    """Returns the most recent close price from the aggregator."""
    price = feed.get_last_price(symbol.upper(), exchange.upper())
    if price is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"{symbol} is not subscribed. "
                "Call /feed/subscribe first."
            ),
        )
    return {
        "symbol":     symbol.upper(),
        "exchange":   exchange.upper(),
        "last_price": price,
    }


@router.get(
    "/stats",
    response_model=FeedStatsResponse,
    summary="Feed service statistics",
)
async def get_feed_stats(
    feed=Depends(get_feed_service),
    user: dict = Depends(get_current_user),
) -> FeedStatsResponse:
    """Returns live feed subscription and aggregator stats."""
    stats = feed.get_feed_stats()
    return FeedStatsResponse(
        is_running=stats["is_running"],
        active_broker=stats["active_broker"],
        subscribed_symbols=stats["subscribed_symbols"],
        active_consumers=stats["active_consumers"],
        active_aggregators=stats["active_aggregators"],
        symbol_keys=stats["symbol_keys"],
    )