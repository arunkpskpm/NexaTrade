"""
NexaTrade — Async Redis Client.

Wraps aioredis with a clean interface for all NexaTrade
Redis operations — quote caching, kill switches, P&L,
signal deduplication, pub/sub, and session management.

Key namespacing:
    quote:{broker}:{symbol}          → latest quote JSON (TTL 60s)
    signal:{strategy}:{symbol}       → latest signal JSON
    dedup:{strategy}:{symbol}:{dir}  → signal dedup sentinel
    position:{broker}:{symbol}       → position snapshot JSON
    pnl:daily:{broker}               → daily realised P&L float
    ks:{broker}                      → kill switch flag (1/0)
    ks:global                        → global kill switch flag
    session:{user_id}                → session validity flag

All keys use colon-separated namespacing for clarity.
All TTL values are explicit — no indefinite keys in production.

Usage:
    redis = RedisClient()
    await redis.initialise()

    await redis.set_quote("breeze", "RELIANCE", {...}, ttl_seconds=60)
    quote = await redis.get_quote("breeze", "RELIANCE")
    await redis.set_kill_switch("breeze", active=True)
    await redis.shutdown()
"""

from __future__ import annotations

import json
from typing import Any, Optional

import aioredis

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)


class RedisClient:
    """
    NexaTrade Async Redis Client.

    All methods are async coroutines.
    Connection is a single aioredis client (not pool —
    aioredis 2.x handles multiplexing internally).
    """

    def __init__(self) -> None:
        self._redis: Optional[aioredis.Redis] = None
        self._settings = get_settings()

    # ─────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────

    async def initialise(self) -> None:
        """
        Connects to Redis and verifies with PING.

        Raises:
            RuntimeError: If Redis connection fails.
        """
        url = self._settings.redis.url
        try:
            self._redis = await aioredis.from_url(
                url,
                encoding="utf-8",
                decode_responses=True,
                socket_keepalive=True,
                socket_connect_timeout=5,
                retry_on_timeout=True,
            )
            await self._redis.ping()
            logger.info(
                f"Redis connected | "
                f"host={self._settings.redis.host}:"
                f"{self._settings.redis.port}"
            )
        except Exception as exc:
            raise RuntimeError(
                f"Redis init failed: {exc}"
            ) from exc

    async def shutdown(self) -> None:
        """Closes the Redis connection."""
        if self._redis:
            await self._redis.close()
            logger.info("Redis connection closed.")

    async def health_check(self) -> bool:
        """Returns True if PING succeeds."""
        try:
            return await self._redis.ping() == True
        except Exception:
            return False

    def _require(self) -> None:
        """Raises if Redis is not initialised."""
        if not self._redis:
            raise RuntimeError(
                "RedisClient not initialised. "
                "Call await redis.initialise() first."
            )

    # ─────────────────────────────────────────
    # Generic Key–Value Operations
    # ─────────────────────────────────────────

    async def set(
        self,
        key: str,
        value: str,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """
        Sets a raw string value.

        Args:
            key: Redis key.
            value: String value.
            ttl_seconds: Expiry in seconds. No expiry if None.
        """
        self._require()
        if ttl_seconds:
            await self._redis.setex(key, ttl_seconds, value)
        else:
            await self._redis.set(key, value)

    async def get(self, key: str) -> Optional[str]:
        """Returns the raw string value for a key, or None."""
        self._require()
        return await self._redis.get(key)

    async def delete(self, key: str) -> None:
        """Deletes a key."""
        self._require()
        await self._redis.delete(key)

    async def exists(self, key: str) -> bool:
        """Returns True if the key exists."""
        self._require()
        return bool(await self._redis.exists(key))

    async def set_json(
        self,
        key: str,
        value: dict[str, Any],
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """Serialises a dict to JSON and stores it."""
        self._require()
        serialised = json.dumps(value)
        if ttl_seconds:
            await self._redis.setex(key, ttl_seconds, serialised)
        else:
            await self._redis.set(key, serialised)

    async def get_json(
        self, key: str
    ) -> Optional[dict[str, Any]]:
        """Retrieves and deserialises a JSON-stored dict."""
        self._require()
        raw = await self._redis.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    async def increment_float(
        self, key: str, amount: float
    ) -> float:
        """Atomically increments a float value by amount."""
        self._require()
        result = await self._redis.incrbyfloat(key, amount)
        return float(result)

    # ─────────────────────────────────────────
    # Quote Cache
    # ─────────────────────────────────────────

    async def set_quote(
        self,
        broker_name: str,
        symbol: str,
        quote: dict[str, Any],
        ttl_seconds: int = 60,
    ) -> None:
        """
        Caches a live market quote.

        Args:
            broker_name: Source broker.
            symbol: Instrument symbol.
            quote: Quote data dict.
            ttl_seconds: Cache TTL (default 60s).
        """
        key = f"quote:{broker_name}:{symbol.upper()}"
        await self.set_json(key, quote, ttl_seconds=ttl_seconds)

    async def get_quote(
        self, broker_name: str, symbol: str
    ) -> Optional[dict[str, Any]]:
        """
        Returns the cached quote for a symbol.

        Args:
            broker_name: Source broker.
            symbol: Instrument symbol.

        Returns:
            Quote dict or None if cache miss / expired.
        """
        key = f"quote:{broker_name}:{symbol.upper()}"
        return await self.get_json(key)

    async def get_all_quotes(
        self, broker_name: str
    ) -> dict[str, dict[str, Any]]:
        """
        Returns all cached quotes for a broker.

        Args:
            broker_name: Source broker.

        Returns:
            Dict mapping symbol → quote dict.
        """
        self._require()
        pattern = f"quote:{broker_name}:*"
        keys    = await self._redis.keys(pattern)
        if not keys:
            return {}

        values  = await self._redis.mget(*keys)
        result  = {}
        for key, val in zip(keys, values):
            if val:
                symbol = key.split(":")[-1]
                try:
                    result[symbol] = json.loads(val)
                except json.JSONDecodeError:
                    pass
        return result

    # ─────────────────────────────────────────
    # Signal Cache
    # ─────────────────────────────────────────

    async def set_signal(
        self,
        strategy_name: str,
        symbol: str,
        signal: dict[str, Any],
        ttl_seconds: int = 300,
    ) -> None:
        """
        Caches the latest signal for a strategy+symbol pair.

        Args:
            strategy_name: Originating strategy.
            symbol: Target symbol.
            signal: Signal data dict.
            ttl_seconds: Cache TTL (default 300s = 5 min).
        """
        key = f"signal:{strategy_name}:{symbol.upper()}"
        await self.set_json(key, signal, ttl_seconds=ttl_seconds)

    async def get_signal(
        self, strategy_name: str, symbol: str
    ) -> Optional[dict[str, Any]]:
        """Returns the cached signal for a strategy+symbol pair."""
        key = f"signal:{strategy_name}:{symbol.upper()}"
        return await self.get_json(key)

    # ─────────────────────────────────────────
    # Position Cache
    # ─────────────────────────────────────────

    async def set_position(
        self,
        broker_name: str,
        symbol: str,
        position: dict[str, Any],
        ttl_seconds: int = 3600,
    ) -> None:
        """
        Caches a position snapshot for fast RiskManager reads.

        Args:
            broker_name: Broker identifier.
            symbol: Instrument symbol.
            position: Position data dict.
            ttl_seconds: Cache TTL (default 1h).
        """
        key = f"position:{broker_name}:{symbol.upper()}"
        await self.set_json(
            key, position, ttl_seconds=ttl_seconds
        )

    async def get_position(
        self, broker_name: str, symbol: str
    ) -> Optional[dict[str, Any]]:
        """Returns the cached position for a broker+symbol."""
        key = f"position:{broker_name}:{symbol.upper()}"
        return await self.get_json(key)

    async def delete_position(
        self, broker_name: str, symbol: str
    ) -> None:
        """Removes a position from cache (on position close)."""
        key = f"position:{broker_name}:{symbol.upper()}"
        await self.delete(key)

    # ─────────────────────────────────────────
    # Daily P&L Tracker
    # ─────────────────────────────────────────

    async def get_daily_pnl(self, broker_name: str) -> float:
        """
        Returns today's accumulated realised P&L for a broker.

        Initialises to 0.0 if key does not exist.

        Args:
            broker_name: Broker identifier.

        Returns:
            Realised P&L float (negative = loss).
        """
        key = f"pnl:daily:{broker_name}"
        raw = await self.get(key)
        return float(raw) if raw else 0.0

    async def add_to_daily_pnl(
        self, broker_name: str, amount: float
    ) -> float:
        """
        Atomically adds a P&L amount to today's total.

        Sets a midnight expiry on the key if it's new,
        so the counter resets automatically each day.

        Args:
            broker_name: Broker identifier.
            amount: P&L delta (positive = profit, negative = loss).

        Returns:
            New cumulative daily P&L.
        """
        key = f"pnl:daily:{broker_name}"
        self._require()
        # Atomically increment
        result = await self._redis.incrbyfloat(key, amount)
        # Set TTL to end of day if key was just created
        ttl = await self._redis.ttl(key)
        if ttl == -1:
            from datetime import datetime, time
            import pytz
            IST = pytz.timezone("Asia/Kolkata")
            now = datetime.now(IST)
            midnight = datetime.combine(
                now.date(), time(23, 59, 59)
            ).replace(tzinfo=IST)
            seconds_left = int((midnight - now).total_seconds())
            await self._redis.expire(key, max(seconds_left, 60))
        return float(result)

    async def reset_daily_pnl(self, broker_name: str) -> None:
        """Resets the daily P&L counter to zero."""
        key = f"pnl:daily:{broker_name}"
        await self.delete(key)

    # ─────────────────────────────────────────
    # Kill Switch
    # ─────────────────────────────────────────

    async def set_kill_switch(
        self, broker_name: str, active: bool
    ) -> None:
        """
        Arms or disarms the kill switch for a broker.

        Args:
            broker_name: Target broker.
            active: True to arm, False to disarm.
        """
        key = f"ks:{broker_name}"
        await self.set(key, "1" if active else "0")
        logger.info(
            f"Kill switch {'ARMED' if active else 'DISARMED'} "
            f"| broker={broker_name}"
        )

    async def is_kill_switch_active(
        self, broker_name: str
    ) -> bool:
        """Returns True if the kill switch is armed for a broker."""
        key = f"ks:{broker_name}"
        val = await self.get(key)
        return val == "1"

    async def set_global_kill_switch(self, active: bool) -> None:
        """Arms or disarms the global kill switch."""
        await self.set("ks:global", "1" if active else "0")
        logger.info(
            f"Global kill switch "
            f"{'ARMED' if active else 'DISARMED'}"
        )

    async def is_global_kill_switch_active(self) -> bool:
        """Returns True if the global kill switch is armed."""
        val = await self.get("ks:global")
        return val == "1"

    # ─────────────────────────────────────────
    # Pub/Sub Publisher
    # ─────────────────────────────────────────

    async def publish(
        self, channel: str, message: dict[str, Any]
    ) -> int:
        """
        Publishes a JSON message to a Redis pub/sub channel.

        Channels used by NexaTrade:
            ticks    → live tick events
            orders   → order placed/updated events
            candles  → candle close events
            risk     → risk events (kill switch, signal blocked)

        Args:
            channel: Channel name.
            message: Dict to serialise and publish.

        Returns:
            Number of subscribers that received the message.
        """
        self._require()
        try:
            payload   = json.dumps(message)
            receivers = await self._redis.publish(
                channel, payload
            )
            return receivers
        except Exception as exc:
            logger.debug(
                f"Redis publish failed | "
                f"channel={channel} | error={exc}"
            )
            return 0

    async def subscribe(
        self, *channels: str
    ) -> aioredis.client.PubSub:
        """
        Returns a pub/sub handle subscribed to the given channels.

        Args:
            *channels: One or more channel names.

        Returns:
            aioredis PubSub handle — iterate with async for.

        Example:
            pubsub = await redis.subscribe("ticks", "orders")
            async for msg in pubsub.listen():
                if msg["type"] == "message":
                    data = json.loads(msg["data"])
        """
        self._require()
        pubsub = self._redis.pubsub()
        await pubsub.subscribe(*channels)
        return pubsub

    # ─────────────────────────────────────────
    # Rate Limiting
    # ─────────────────────────────────────────

    async def check_rate_limit(
        self,
        key: str,
        max_calls: int,
        window_seconds: int,
    ) -> tuple[bool, int]:
        """
        Token-bucket rate limiter using Redis INCR + EXPIRE.

        Args:
            key: Rate limit key (e.g. "rate:orders:breeze").
            max_calls: Maximum calls per window.
            window_seconds: Window duration in seconds.

        Returns:
            Tuple of (allowed: bool, current_count: int).

        Example:
            allowed, count = await redis.check_rate_limit(
                "rate:orders:breeze", max_calls=10, window_seconds=60
            )
            if not allowed:
                raise RateLimitError()
        """
        self._require()
        pipe   = self._redis.pipeline()
        pipe.incr(key)
        pipe.ttl(key)
        results = await pipe.execute()

        current = int(results[0])
        ttl     = int(results[1])

        if ttl == -1:
            # First call — set the window expiry
            await self._redis.expire(key, window_seconds)

        return current <= max_calls, current

    # ─────────────────────────────────────────
    # Flush (Test / Dev Only)
    # ─────────────────────────────────────────

    async def flush_all(self, confirm: bool = False) -> None:
        """
        Flushes the entire Redis database.
        ONLY for test/development environments.

        Args:
            confirm: Must be True to execute.

        Raises:
            PermissionError: If called in production.
        """
        from config.settings import get_settings
        settings = get_settings()
        if settings.is_production:
            raise PermissionError(
                "flush_all() is forbidden in production."
            )
        if not confirm:
            raise ValueError(
                "Pass confirm=True to flush Redis."
            )
        self._require()
        await self._redis.flushdb()
        logger.warning(
            "⚠️  Redis database flushed (dev/test only)."
        )