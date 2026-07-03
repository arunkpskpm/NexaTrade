"""
NexaTrade — Market Data Routes.

Endpoints:
    GET  /api/v1/data/quote              → single live quote
    POST /api/v1/data/quotes             → multiple live quotes
    POST /api/v1/data/historical         → OHLCV historical data
    GET  /api/v1/data/candles/latest     → latest N candles
    GET  /api/v1/data/intraday           → today's intraday candles
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.dependencies import (
    get_broker_service,
    get_data_service,
    get_current_user,
)
from api.schemas import (
    HistoricalDataRequest,
    HistoricalDataResponse,
    OHLCVResponse,
    QuoteResponse,
    SuccessResponse,
)
from utils.logger import get_logger
from utils.time_utils import now_ist

logger = get_logger(__name__)
router = APIRouter()


@router.get(
    "/quote",
    response_model=QuoteResponse,
    summary="Live quote for a single instrument",
)
async def get_quote(
    symbol: str   = Query(..., description="Instrument symbol"),
    exchange: str = Query(default="NSE", description="Exchange code"),
    broker=Depends(get_broker_service),
    user: dict = Depends(get_current_user),
) -> QuoteResponse:
    """Fetches a live market quote for the given symbol."""
    try:
        quote = await broker.get_quote(
            symbol.upper(), exchange.upper()
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Quote fetch failed: {exc}",
        )
    return QuoteResponse(
        symbol=quote.symbol,
        exchange=quote.exchange,
        last_price=quote.last_price,
        open=quote.open,
        high=quote.high,
        low=quote.low,
        close=quote.close,
        bid=quote.bid,
        ask=quote.ask,
        volume=quote.volume,
        change=quote.change,
        change_pct=quote.change_pct,
        timestamp=str(quote.timestamp) if quote.timestamp else None,
        broker_name=quote.broker_name,
    )


@router.post(
    "/quotes",
    summary="Live quotes for multiple instruments",
)
async def get_quotes(
    instruments: list[dict],
    broker=Depends(get_broker_service),
    user: dict = Depends(get_current_user),
) -> dict:
    """
    Fetches live quotes for multiple instruments in one call.

    Request body: [{"symbol": "RELIANCE", "exchange": "NSE"}, ...]
    """
    try:
        quotes = await broker.broker.get_quotes(instruments)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Quotes fetch failed: {exc}",
        )
    return {
        "quotes": [
            {
                "symbol":     q.symbol,
                "exchange":   q.exchange,
                "last_price": q.last_price,
                "change_pct": q.change_pct,
                "volume":     q.volume,
                "broker_name": q.broker_name,
            }
            for q in quotes
        ],
        "count": len(quotes),
    }


@router.post(
    "/historical",
    response_model=HistoricalDataResponse,
    summary="OHLCV historical data",
)
async def get_historical(
    body: HistoricalDataRequest,
    data_svc=Depends(get_data_service),
    user: dict = Depends(get_current_user),
) -> HistoricalDataResponse:
    """
    Returns OHLCV historical candle data as a list.
    Uses two-tier cache (InfluxDB → broker API fallback).
    """
    try:
        df = await data_svc.get_ohlcv(
            symbol=body.symbol,
            exchange=body.exchange,
            interval=body.interval,
            from_date=body.from_date,
            to_date=body.to_date,
            broker_name=body.broker_name,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Historical data fetch failed: {exc}",
        )

    if df.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No data found for {body.symbol} "
                f"{body.interval} "
                f"{body.from_date}→{body.to_date}"
            ),
        )

    ohlcv_list = [
        OHLCVResponse(
            datetime=str(idx),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
        )
        for idx, row in df.iterrows()
    ]

    return HistoricalDataResponse(
        symbol=body.symbol,
        exchange=body.exchange,
        interval=body.interval,
        from_date=body.from_date,
        to_date=body.to_date,
        bars=len(ohlcv_list),
        data=ohlcv_list,
    )


@router.get(
    "/candles/latest",
    summary="Latest N candles for an instrument",
)
async def get_latest_candles(
    symbol:   str = Query(...),
    exchange: str = Query(default="NSE"),
    interval: str = Query(default="5minute"),
    n:        int = Query(default=100, ge=1, le=1000),
    data_svc=Depends(get_data_service),
    user: dict = Depends(get_current_user),
) -> dict:
    """Returns the latest N candles from cache or broker."""
    try:
        df = await data_svc.get_latest_candles(
            symbol=symbol.upper(),
            exchange=exchange.upper(),
            interval=interval,
            n=n,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Candle fetch failed: {exc}",
        )

    return {
        "symbol":   symbol.upper(),
        "exchange": exchange.upper(),
        "interval": interval,
        "bars":     len(df),
        "data": [
            {
                "datetime": str(idx),
                "open":     float(row["open"]),
                "high":     float(row["high"]),
                "low":      float(row["low"]),
                "close":    float(row["close"]),
                "volume":   float(row["volume"]),
            }
            for idx, row in df.iterrows()
        ],
    }


@router.get(
    "/intraday",
    summary="Today's intraday candles",
)
async def get_intraday(
    symbol:   str = Query(...),
    exchange: str = Query(default="NSE"),
    interval: str = Query(default="5minute"),
    date:     Optional[str] = Query(
        default=None,
        description="Date YYYY-MM-DD (default: today)"
    ),
    data_svc=Depends(get_data_service),
    user: dict = Depends(get_current_user),
) -> dict:
    """Returns all intraday OHLCV candles for a trading day."""
    try:
        df = await data_svc.get_intraday(
            symbol=symbol.upper(),
            exchange=exchange.upper(),
            interval=interval,
            date=date,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Intraday data fetch failed: {exc}",
        )

    return {
        "symbol":   symbol.upper(),
        "exchange": exchange.upper(),
        "interval": interval,
        "date":     date or now_ist().strftime("%Y-%m-%d"),
        "bars":     len(df),
        "data": [
            {
                "datetime": str(idx),
                "open":     float(row["open"]),
                "high":     float(row["high"]),
                "low":      float(row["low"]),
                "close":    float(row["close"]),
                "volume":   float(row["volume"]),
            }
            for idx, row in df.iterrows()
        ],
    }