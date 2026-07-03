"""
NexaTrade — WebSocket Routes.

Endpoints:
    WS /api/v1/ws/ticks/{symbol}     → live tick stream
    WS /api/v1/ws/orders             → live order update stream
    WS /api/v1/ws/candles/{symbol}   → live candle close stream

Connection lifecycle:
    1. Client connects with ?token=<jwt>
    2. Server validates JWT
    3. Server subscribes to feed on client's behalf
    4. Server streams ticks/candles to client as JSON
    5. On disconnect: client unsubscribed, resources freed

Message format (ticks):
    {
        "type":       "tick",
        "symbol":     "RELIANCE",
        "exchange":   "NSE",
        "last_price": 2450.50,
        "bid":        2450.40,
        "ask":        2450.60,
        "volume":     124500,
        "timestamp":  "2024-01-15T10:32:45+05:30"
    }

Message format (candles):
    {
        "type":     "candle",
        "symbol":   "RELIANCE",
        "exchange": "NSE",
        "interval": "5minute",
        "datetime": "2024-01-15T10:30:00+05:30",
        "open":     2445.00,
        "high":     2452.00,
        "low":      2444.50,
        "close":    2450.50,
        "volume":   31200.0
    }
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect, status
from starlette.websockets import WebSocketState

from brokers.models import OHLCV, TickData
from utils.auth import decode_jwt_token
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


# ─────────────────────────────────────────────
# WebSocket Connection Manager
# ─────────────────────────────────────────────

class WebSocketManager:
    """
    Manages active WebSocket connections.
    Tracks connections per symbol for efficient fan-out.
    """

    def __init__(self) -> None:
        # {symbol_key: [WebSocket, ...]}
        self._connections: dict[str, list[WebSocket]] = {}

    async def connect(
        self, ws: WebSocket, key: str
    ) -> None:
        """Accepts and registers a WebSocket connection."""
        await ws.accept()
        self._connections.setdefault(key, []).append(ws)
        logger.debug(
            f"WS connected | key={key} | "
            f"total={len(self._connections.get(key, []))}"
        )

    def disconnect(self, ws: WebSocket, key: str) -> None:
        """Removes a WebSocket from the connection pool."""
        conns = self._connections.get(key, [])
        if ws in conns:
            conns.remove(ws)
        if not conns:
            self._connections.pop(key, None)

    async def broadcast(
        self, key: str, message: dict[str, Any]
    ) -> None:
        """
        Sends a JSON message to all connections on a key.
        Dead connections are removed silently.
        """
        conns = self._connections.get(key, [])
        dead  = []
        for ws in conns:
            if ws.client_state == WebSocketState.CONNECTED:
                try:
                    await ws.send_text(json.dumps(message))
                except Exception:
                    dead.append(ws)
            else:
                dead.append(ws)

        for ws in dead:
            self.disconnect(ws, key)

    def connection_count(self, key: str) -> int:
        """Returns the number of active connections for a key."""
        return len(self._connections.get(key, []))


# Module-level manager (shared across all WS routes)
_ws_manager = WebSocketManager()


# ─────────────────────────────────────────────
# Authentication Helper
# ─────────────────────────────────────────────

async def _authenticate_ws(
    ws: WebSocket, token: str | None
) -> dict | None:
    """
    Validates the JWT token for a WebSocket connection.
    Closes the connection with 4001 if auth fails.

    Args:
        ws: WebSocket connection.
        token: JWT token string from query param.

    Returns:
        Decoded JWT payload or None if auth failed.
    """
    if not token:
        await ws.close(
            code=4001, reason="Authentication required."
        )
        return None
    payload = decode_jwt_token(token)
    if not payload:
        await ws.close(
            code=4001, reason="Invalid or expired token."
        )
        return None
    return payload


# ─────────────────────────────────────────────
# WebSocket Endpoints
# ─────────────────────────────────────────────

@router.websocket("/ticks/{symbol}")
async def ws_ticks(
    ws: WebSocket,
    symbol: str,
    exchange: str = Query(default="NSE"),
    token:    str = Query(default=None),
):
    """
    WebSocket — live tick stream for a symbol.

    Connect: ws://host/api/v1/ws/ticks/RELIANCE?token=<jwt>

    The server subscribes to the FeedService on behalf of this
    WebSocket client. Ticks arrive as JSON messages.
    On disconnect, the feed subscription is released.
    """
    user = await _authenticate_ws(ws, token)
    if not user:
        return

    sym_key      = f"{symbol.upper()}:{exchange.upper()}"
    consumer_id  = f"ws_tick:{user['user_id']}:{sym_key}"

    # Get feed service from app state
    feed = ws.app.state.container.feed_service

    await _ws_manager.connect(ws, sym_key)
    logger.info(
        f"WS tick connection | "
        f"symbol={symbol.upper()} | "
        f"user={user.get('username')}"
    )

    # Tick callback — broadcast to this WS client
    async def on_tick(tick: TickData) -> None:
        if tick.symbol.upper() == symbol.upper():
            await _ws_manager.broadcast(
                sym_key,
                {
                    "type":       "tick",
                    "symbol":     tick.symbol,
                    "exchange":   tick.exchange,
                    "last_price": tick.last_price,
                    "bid":        tick.bid,
                    "ask":        tick.ask,
                    "volume":     tick.volume,
                    "change":     tick.change,
                    "change_pct": tick.change_pct,
                    "timestamp":  str(tick.timestamp),
                },
            )

    # Subscribe to feed
    await feed.subscribe(
        symbol=symbol.upper(),
        exchange=exchange.upper(),
        consumer_id=consumer_id,
        tick_callback=on_tick,
    )

    try:
        while True:
            # Keep connection alive — wait for client ping or close
            data = await ws.receive_text()
            if data == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        logger.info(
            f"WS tick disconnected | "
            f"symbol={symbol.upper()} | "
            f"user={user.get('username')}"
        )
    finally:
        _ws_manager.disconnect(ws, sym_key)
        await feed.unsubscribe(
            symbol=symbol.upper(),
            exchange=exchange.upper(),
            consumer_id=consumer_id,
        )


@router.websocket("/candles/{symbol}")
async def ws_candles(
    ws: WebSocket,
    symbol:   str,
    exchange: str = Query(default="NSE"),
    interval: str = Query(default="5minute"),
    token:    str = Query(default=None),
):
    """
    WebSocket — live candle close stream for a symbol.

    Connect: ws://host/api/v1/ws/candles/RELIANCE?interval=5minute&token=<jwt>

    Emits a JSON message each time a candle closes.
    """
    user = await _authenticate_ws(ws, token)
    if not user:
        return

    sym_key     = f"{symbol.upper()}:{exchange.upper()}"
    agg_key     = f"{sym_key}:{interval}"
    consumer_id = f"ws_candle:{user['user_id']}:{agg_key}"

    feed = ws.app.state.container.feed_service

    await _ws_manager.connect(ws, agg_key)
    logger.info(
        f"WS candle connection | "
        f"symbol={symbol.upper()} | "
        f"interval={interval} | "
        f"user={user.get('username')}"
    )

    # Candle close callback
    async def on_candle(candle: OHLCV) -> None:
        if candle.symbol.upper() == symbol.upper():
            await _ws_manager.broadcast(
                agg_key,
                {
                    "type":     "candle",
                    "symbol":   candle.symbol,
                    "exchange": candle.exchange,
                    "interval": candle.interval,
                    "datetime": str(candle.datetime),
                    "open":     candle.open,
                    "high":     candle.high,
                    "low":      candle.low,
                    "close":    candle.close,
                    "volume":   candle.volume,
                },
            )

    # Subscribe
    await feed.subscribe(
        symbol=symbol.upper(),
        exchange=exchange.upper(),
        interval=interval,
        consumer_id=consumer_id,
        candle_callback=on_candle,
        seed_history=True,
    )

    try:
        while True:
            data = await ws.receive_text()
            if data == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        logger.info(
            f"WS candle disconnected | "
            f"symbol={symbol.upper()} | "
            f"user={user.get('username')}"
        )
    finally:
        _ws_manager.disconnect(ws, agg_key)
        await feed.unsubscribe(
            symbol=symbol.upper(),
            exchange=exchange.upper(),
            consumer_id=consumer_id,
            interval=interval,
        )


@router.websocket("/orders")
async def ws_orders(
    ws: WebSocket,
    token: str = Query(default=None),
):
    """
    WebSocket — live order update stream.

    Connect: ws://host/api/v1/ws/orders?token=<jwt>

    Receives real-time order status updates
    for orders placed in this session.
    """
    user = await _authenticate_ws(ws, token)
    if not user:
        return

    user_key    = f"orders:{user['user_id']}"
    broker_svc  = ws.app.state.container.broker_service

    await _ws_manager.connect(ws, user_key)
    logger.info(
        f"WS orders connection | "
        f"user={user.get('username')}"
    )

    async def on_order_update(response) -> None:
        await _ws_manager.broadcast(
            user_key,
            {
                "type":            "order_update",
                "order_id":        response.order_id,
                "broker_order_id": response.broker_order_id,
                "status":          str(response.status),
                "filled_quantity": response.filled_quantity,
                "average_price":   response.average_price,
                "message":         response.message,
                "updated_at":      str(response.updated_at),
            },
        )

    broker_svc.register_order_subscriber(on_order_update)

    try:
        while True:
            data = await ws.receive_text()
            if data == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        logger.info(
            f"WS orders disconnected | "
            f"user={user.get('username')}"
        )
    finally:
        _ws_manager.disconnect(ws, user_key)
        broker_svc.unregister_order_subscriber(on_order_update)