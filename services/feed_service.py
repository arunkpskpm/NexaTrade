"""
NexaTrade — Feed Service.

The FeedService is the single owner of all live market
data subscriptions. It sits between the broker adapter
and all consumers (strategy engine, UI, Redis cache).

Architecture:
    BrokerService
        └── FeedService (subscriber)
                ├── Redis  (quote cache writer)
                ├── InfluxDB (tick persistence)
                ├── CandleAggregator (tick → OHLCV)
                └── Registered consumers (strategies, UI)

Responsibilities:
  - Subscribe / unsubscribe instruments on the active broker
  - Receive raw TickData from BrokerService callback
  - Cache latest quote in Redis (per broker, per symbol)
  - Persist ticks to InfluxDB (configurable)
  - Aggregate ticks into OHLCV candles in-memory
  - Fan-out normalised ticks to all registered consumers
  - Track subscription counts (ref-counting) so symbols are
    only unsubscribed when NO consumer needs them
  - Gracefully handle broker feed reconnections

Key design decisions:
  - All state is in-process (no shared DB for tick routing)
  - Redis is write-through cache only — not the tick bus
  - Candle aggregation is pure in-memory ring-buffer
  - Subscription ref-counting prevents race conditions when
    multiple strategies subscribe to the same symbol

Usage:
    feed = FeedService(broker_svc, redis_client, influx_client)
    await feed.start()

    # Subscribe a symbol
    await feed.subscribe("RELIANCE", "NSE", interval="1minute")

    # Register a consumer callback
    feed.register_consumer("strategy_a", on_tick_callback)

    # Unsubscribe when done
    await feed.unsubscribe("RELIANCE", "NSE", consumer="strategy_a")

    await feed.stop()
"""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, Callable, Optional

from brokers.models import OHLCV, TickData
from data.storage.influx_client import InfluxClient
from data.storage.redis_client import RedisClient
from services.broker_service import BrokerService
from utils.logger import get_logger
from utils.time_utils import now_ist, floor_to_minute, to_utc

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# Candle Aggregator
# ─────────────────────────────────────────────

class CandleAggregator:
    """
    Aggregates live ticks into OHLCV candles in memory.

    One aggregator instance per (symbol, interval) pair.
    Maintains a ring buffer of completed candles plus
    the current (in-progress) candle.

    Design:
        - open:   first tick price in the interval
        - high:   running max price in the interval
        - low:    running min price in the interval
        - close:  latest tick price in the interval
        - volume: running cumulative volume in the interval

    When a new candle boundary is crossed:
        - Current candle is finalised → emitted to listeners
        - New candle starts from the crossing tick
    """

    def __init__(
        self,
        symbol: str,
        exchange: str,
        interval: str,
        interval_minutes: int,
        broker_name: str,
        buffer_size: int = 500,
    ) -> None:
        self.symbol = symbol.upper()
        self.exchange = exchange.upper()
        self.interval = interval
        self.interval_minutes = interval_minutes
        self.broker_name = broker_name

        # Ring buffer of completed candles
        self._buffer: deque[OHLCV] = deque(maxlen=buffer_size)

        # Current in-progress candle state
        self._current: Optional[dict[str, Any]] = None
        self._current_boundary: Optional[datetime] = None

        # Listeners notified on candle close
        self._on_candle_close: list[Callable] = []

    # ─────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────

    @property
    def candle_count(self) -> int:
        """Number of completed candles in buffer."""
        return len(self._buffer)

    @property
    def latest_candle(self) -> Optional[OHLCV]:
        """The most recently completed candle."""
        return self._buffer[-1] if self._buffer else None

    @property
    def current_candle(self) -> Optional[dict[str, Any]]:
        """The in-progress (not yet closed) candle state."""
        return dict(self._current) if self._current else None

    def get_candles(self, n: Optional[int] = None) -> list[OHLCV]:
        """
        Returns completed candles from the ring buffer.

        Args:
            n: Number of recent candles to return.
               Returns all if None.

        Returns:
            List of OHLCV candles (oldest first).
        """
        candles = list(self._buffer)
        return candles[-n:] if n else candles

    # ─────────────────────────────────────────
    # Callback Registration
    # ─────────────────────────────────────────

    def on_candle_close(self, callback: Callable) -> None:
        """
        Registers a callback invoked when a candle closes.

        Args:
            callback: Async function accepting an OHLCV argument.
        """
        if callback not in self._on_candle_close:
            self._on_candle_close.append(callback)

    def remove_candle_close_callback(self, callback: Callable) -> None:
        """Removes a candle close callback."""
        if callback in self._on_candle_close:
            self._on_candle_close.remove(callback)

    # ─────────────────────────────────────────
    # Tick Ingestion
    # ─────────────────────────────────────────

    async def process_tick(self, tick: TickData) -> None:
        """
        Ingests a tick and updates the current in-progress candle.
        Closes the current candle and starts a new one when the
        interval boundary is crossed.

        Args:
            tick: Normalised TickData from the feed.
        """
        ts = tick.timestamp or now_ist()
        boundary = floor_to_minute(ts, self.interval_minutes)

        if self._current is None:
            # First tick — start the first candle
            self._start_new_candle(tick, boundary)
            return

        if boundary > self._current_boundary:
            # Interval boundary crossed — close current candle
            closed = self._close_current_candle()
            await self._emit_candle_close(closed)
            # Start new candle from this tick
            self._start_new_candle(tick, boundary)
        else:
            # Same interval — update running candle
            self._update_current_candle(tick)

    def _start_new_candle(
        self, tick: TickData, boundary: datetime
    ) -> None:
        """Initialises a new in-progress candle from a tick."""
        self._current = {
            "datetime": boundary,
            "open":   tick.last_price,
            "high":   tick.last_price,
            "low":    tick.last_price,
            "close":  tick.last_price,
            "volume": float(tick.volume),
        }
        self._current_boundary = boundary

    def _update_current_candle(self, tick: TickData) -> None:
        """Updates the running OHLCV values from a new tick."""
        if not self._current:
            return
        price = tick.last_price
        self._current["high"]   = max(self._current["high"], price)
        self._current["low"]    = min(self._current["low"], price)
        self._current["close"]  = price
        self._current["volume"] += float(tick.volume)

    def _close_current_candle(self) -> OHLCV:
        """
        Finalises the current candle and adds it to the buffer.

        Returns:
            The completed OHLCV candle.
        """
        candle = OHLCV(
            datetime=self._current["datetime"],
            open=self._current["open"],
            high=self._current["high"],
            low=self._current["low"],
            close=self._current["close"],
            volume=self._current["volume"],
            symbol=self.symbol,
            exchange=self.exchange,
            interval=self.interval,
            broker_name=self.broker_name,
        )
        self._buffer.append(candle)
        return candle

    async def _emit_candle_close(self, candle: OHLCV) -> None:
        """Emits a closed candle to all registered callbacks."""
        for callback in self._on_candle_close:
            try:
                await callback(candle)
            except Exception as exc:
                logger.error(
                    f"Candle close callback error | "
                    f"symbol={self.symbol} | "
                    f"fn={callback.__name__} | "
                    f"error={exc}"
                )

    def force_close_current(self) -> Optional[OHLCV]:
        """
        Force-closes the current in-progress candle.
        Called on feed shutdown to flush the partial candle.

        Returns:
            The partially completed OHLCV candle, or None.
        """
        if self._current:
            return self._close_current_candle()
        return None

    def seed_from_history(self, candles: list[OHLCV]) -> None:
        """
        Pre-fills the ring buffer with historical candles.
        Called during strategy warmup to provide indicator history.

        Args:
            candles: Historical OHLCV list (oldest first).
        """
        for candle in candles:
            self._buffer.append(candle)
        logger.debug(
            f"Aggregator seeded | "
            f"symbol={self.symbol} | "
            f"interval={self.interval} | "
            f"candles={len(candles)}"
        )


# ─────────────────────────────────────────────
# Interval → Minutes Mapping
# ─────────────────────────────────────────────
INTERVAL_MINUTES: dict[str, int] = {
    "1second":  0,      # Sub-minute — use 0 as sentinel
    "1minute":  1,
    "5minute":  5,
    "15minute": 15,
    "30minute": 30,
    "1hour":    60,
    "1day":     1440,
}


# ─────────────────────────────────────────────
# Feed Service
# ─────────────────────────────────────────────

class FeedService:
    """
    NexaTrade live market data feed manager.

    Owns:
        - All active WebSocket subscriptions
        - Subscription ref-counts per symbol
        - CandleAggregator instances per (symbol, interval)
        - Consumer callback registry
        - Redis quote cache writes
        - InfluxDB tick persistence

    Thread safety:
        All state mutations happen in the asyncio event loop.
        No locks required.
    """

    def __init__(
        self,
        broker_service: BrokerService,
        redis_client: RedisClient,
        influx_client: InfluxClient,
    ) -> None:
        self._broker_svc = broker_service
        self._redis = redis_client
        self._influx = influx_client

        # ── Subscription state ───────────────
        # {symbol_key: ref_count}
        # symbol_key = "RELIANCE:NSE"
        self._subscription_refs: defaultdict[str, int] = (
            defaultdict(int)
        )

        # ── Consumer registry ────────────────
        # {consumer_id: {symbol_key: callback}}
        self._consumers: dict[str, dict[str, Callable]] = {}

        # ── Candle aggregators ────────────────
        # {agg_key: CandleAggregator}
        # agg_key = "RELIANCE:NSE:5minute"
        self._aggregators: dict[str, CandleAggregator] = {}

        # ── Config ───────────────────────────
        self._persist_ticks: bool = False
        self._default_interval: str = "1minute"
        self._warmup_bars: int = 200
        self._is_running: bool = False

    # ─────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────

    async def start(self) -> None:
        """
        Starts the FeedService and registers as a tick
        subscriber on the BrokerService.

        Reads feed config from app_config.yaml.
        """
        from config.settings import get_settings
        cfg = get_settings().feed_config
        self._default_interval = cfg.get(
            "default_interval", "1minute"
        )
        self._warmup_bars = int(
            cfg.get("candle_warmup_bars", 200)
        )
        self._persist_ticks = False  # Enable via config if needed

        # Register as tick subscriber on BrokerService
        self._broker_svc.register_tick_subscriber(
            self._on_tick
        )
        self._is_running = True
        logger.info(
            f"FeedService started | "
            f"broker={self._broker_svc.active_broker_name} | "
            f"default_interval={self._default_interval}"
        )

    async def stop(self) -> None:
        """
        Stops the FeedService.
        Flushes all in-progress candles and clears state.
        """
        self._is_running = False
        self._broker_svc.unregister_tick_subscriber(self._on_tick)

        # Force-close all in-progress candles
        for agg in self._aggregators.values():
            partial = agg.force_close_current()
            if partial:
                logger.debug(
                    f"Partial candle flushed on shutdown | "
                    f"symbol={agg.symbol} | interval={agg.interval}"
                )

        self._aggregators.clear()
        self._subscription_refs.clear()
        self._consumers.clear()
        logger.info("FeedService stopped.")

    # ─────────────────────────────────────────
    # Subscribe / Unsubscribe
    # ─────────────────────────────────────────

    async def subscribe(
        self,
        symbol: str,
        exchange: str,
        interval: Optional[str] = None,
        consumer_id: Optional[str] = None,
        tick_callback: Optional[Callable] = None,
        candle_callback: Optional[Callable] = None,
        seed_history: bool = True,
    ) -> str:
        """
        Subscribes to live feed for an instrument.

        If the symbol is already subscribed (by another consumer),
        the broker subscription is NOT duplicated — only the
        ref-count is incremented and callbacks are added.

        Args:
            symbol: Instrument symbol (e.g. "RELIANCE").
            exchange: Exchange code (e.g. "NSE").
            interval: Candle aggregation interval.
                      Defaults to feed.default_interval config.
            consumer_id: Unique consumer identifier string.
                         Defaults to a generated UUID.
            tick_callback: Async callback for raw TickData.
            candle_callback: Async callback for closed OHLCV candles.
            seed_history: If True, pre-seeds the aggregator with
                          historical candles from InfluxDB for
                          indicator warmup.

        Returns:
            consumer_id string (generated or provided).

        Example:
            cid = await feed.subscribe(
                "RELIANCE", "NSE",
                interval="5minute",
                consumer_id="strategy_ema",
                tick_callback=on_tick,
                candle_callback=on_candle,
            )
        """
        import uuid
        symbol = symbol.upper()
        exchange = exchange.upper()
        interval = interval or self._default_interval
        consumer_id = consumer_id or str(uuid.uuid4())[:8]
        sym_key = f"{symbol}:{exchange}"
        agg_key = f"{symbol}:{exchange}:{interval}"

        # ── Candle aggregator setup ──────────
        if agg_key not in self._aggregators:
            interval_mins = INTERVAL_MINUTES.get(interval, 1)
            aggregator = CandleAggregator(
                symbol=symbol,
                exchange=exchange,
                interval=interval,
                interval_minutes=interval_mins,
                broker_name=self._broker_svc.active_broker_name,
            )

            # Seed with historical candles for warmup
            if seed_history:
                await self._seed_aggregator(
                    aggregator, symbol, exchange, interval
                )

            self._aggregators[agg_key] = aggregator
            logger.debug(
                f"Candle aggregator created | "
                f"symbol={symbol} | interval={interval}"
            )

        # ── Register candle callback ──────────
        if candle_callback:
            self._aggregators[agg_key].on_candle_close(
                candle_callback
            )

        # ── Consumer tick callback registration ──
        if consumer_id not in self._consumers:
            self._consumers[consumer_id] = {}

        if tick_callback:
            self._consumers[consumer_id][sym_key] = tick_callback

        # ── Broker subscription (ref-counted) ──
        self._subscription_refs[sym_key] += 1
        if self._subscription_refs[sym_key] == 1:
            # First subscriber — actually subscribe to broker feed
            await self._broker_svc.broker.subscribe_ticks(
                [{"symbol": symbol, "exchange": exchange}]
            )
            logger.info(
                f"Broker feed subscribed | "
                f"symbol={symbol} | exchange={exchange} | "
                f"consumer={consumer_id}"
            )
        else:
            logger.debug(
                f"Feed ref-count incremented | "
                f"symbol={symbol} | "
                f"refs={self._subscription_refs[sym_key]} | "
                f"consumer={consumer_id}"
            )

        return consumer_id

    async def unsubscribe(
        self,
        symbol: str,
        exchange: str,
        consumer_id: str,
        interval: Optional[str] = None,
    ) -> None:
        """
        Unsubscribes a consumer from an instrument's feed.

        Decrements ref-count. Only unsubscribes from the broker
        WebSocket when ref-count reaches zero.

        Args:
            symbol: Instrument symbol.
            exchange: Exchange code.
            consumer_id: Consumer to remove.
            interval: Interval to unregister candle callback.
                      If None, removes all intervals for this symbol.
        """
        symbol = symbol.upper()
        exchange = exchange.upper()
        sym_key = f"{symbol}:{exchange}"
        interval = interval or self._default_interval
        agg_key = f"{symbol}:{exchange}:{interval}"

        # Remove consumer tick callback
        if consumer_id in self._consumers:
            self._consumers[consumer_id].pop(sym_key, None)
            if not self._consumers[consumer_id]:
                del self._consumers[consumer_id]

        # Decrement ref-count
        if self._subscription_refs[sym_key] > 0:
            self._subscription_refs[sym_key] -= 1
            logger.debug(
                f"Feed ref-count decremented | "
                f"symbol={symbol} | "
                f"refs={self._subscription_refs[sym_key]}"
            )

        # Unsubscribe from broker if no more consumers
        if self._subscription_refs[sym_key] == 0:
            del self._subscription_refs[sym_key]
            await self._broker_svc.broker.unsubscribe_ticks(
                [{"symbol": symbol, "exchange": exchange}]
            )
            # Clean up aggregator
            if agg_key in self._aggregators:
                self._aggregators[agg_key].force_close_current()
                del self._aggregators[agg_key]
            logger.info(
                f"Broker feed unsubscribed | "
                f"symbol={symbol} | exchange={exchange}"
            )

    async def subscribe_many(
        self,
        instruments: list[dict[str, str]],
        consumer_id: str,
        interval: Optional[str] = None,
        tick_callback: Optional[Callable] = None,
        candle_callback: Optional[Callable] = None,
    ) -> None:
        """
        Subscribes a consumer to multiple instruments at once.

        Args:
            instruments: List of {"symbol": ..., "exchange": ...}.
            consumer_id: Consumer identifier.
            interval: Candle interval for all instruments.
            tick_callback: Async tick callback.
            candle_callback: Async candle close callback.
        """
        for inst in instruments:
            await self.subscribe(
                symbol=inst["symbol"],
                exchange=inst.get("exchange", "NSE"),
                interval=interval,
                consumer_id=consumer_id,
                tick_callback=tick_callback,
                candle_callback=candle_callback,
            )
        logger.info(
            f"Bulk subscribe complete | "
            f"consumer={consumer_id} | "
            f"instruments={len(instruments)}"
        )

    async def unsubscribe_all(self, consumer_id: str) -> None:
        """
        Unsubscribes a consumer from all its subscriptions.
        Called when a strategy stops.

        Args:
            consumer_id: Consumer to fully unsubscribe.
        """
        if consumer_id not in self._consumers:
            return

        sym_keys = list(self._consumers[consumer_id].keys())
        for sym_key in sym_keys:
            symbol, exchange = sym_key.split(":", 1)
            await self.unsubscribe(
                symbol=symbol,
                exchange=exchange,
                consumer_id=consumer_id,
            )

        logger.info(
            f"Consumer fully unsubscribed | "
            f"consumer={consumer_id} | "
            f"symbols_removed={len(sym_keys)}"
        )

    # ─────────────────────────────────────────
    # Tick Handler (Core)
    # ─────────────────────────────────────────

    async def _on_tick(self, tick: TickData) -> None:
        """
        Master tick handler — called for every tick from broker.

        Pipeline:
            1. Cache quote in Redis
            2. Persist tick to InfluxDB (if enabled)
            3. Update all candle aggregators for this symbol
            4. Fan-out to registered consumer tick callbacks
            5. Publish to Redis pub/sub events channel

        Args:
            tick: Normalised TickData from broker.
        """
        sym_key = f"{tick.symbol}:{tick.exchange}"

        # 1 ── Redis quote cache ───────────────
        try:
            await self._redis.set_quote(
                broker_name=tick.broker_name,
                symbol=tick.symbol,
                quote={
                    "symbol":       tick.symbol,
                    "exchange":     tick.exchange,
                    "last_price":   tick.last_price,
                    "bid":          tick.bid,
                    "ask":          tick.ask,
                    "volume":       tick.volume,
                    "change":       tick.change,
                    "change_pct":   tick.change_pct,
                    "timestamp":    str(tick.timestamp),
                },
                ttl_seconds=60,
            )
        except Exception as exc:
            logger.warning(f"Redis quote cache write failed: {exc}")

        # 2 ── InfluxDB tick persistence ───────
        if self._persist_ticks:
            try:
                await self._influx.write_tick(
                    broker_name=tick.broker_name,
                    symbol=tick.symbol,
                    exchange=tick.exchange,
                    last_price=tick.last_price,
                    bid=tick.bid,
                    ask=tick.ask,
                    volume=float(tick.volume),
                    oi=float(tick.oi),
                    timestamp=tick.timestamp,
                )
            except Exception as exc:
                logger.warning(
                    f"InfluxDB tick write failed: {exc}"
                )

        # 3 ── Candle aggregators ──────────────
        for agg_key, aggregator in self._aggregators.items():
            if agg_key.startswith(sym_key):
                try:
                    await aggregator.process_tick(tick)
                except Exception as exc:
                    logger.error(
                        f"Candle aggregation error | "
                        f"key={agg_key} | error={exc}"
                    )

        # 4 ── Consumer tick fan-out ───────────
        for consumer_id, sym_callbacks in self._consumers.items():
            callback = sym_callbacks.get(sym_key)
            if callback:
                try:
                    await callback(tick)
                except Exception as exc:
                    logger.error(
                        f"Consumer tick callback error | "
                        f"consumer={consumer_id} | error={exc}"
                    )

        # 5 ── Redis pub/sub event ─────────────
        try:
            await self._redis.publish(
                "ticks",
                {
                    "symbol":     tick.symbol,
                    "exchange":   tick.exchange,
                    "last_price": tick.last_price,
                    "broker":     tick.broker_name,
                },
            )
        except Exception as exc:
            logger.debug(f"Redis tick publish failed: {exc}")

    # ─────────────────────────────────────────
    # History Seeding
    # ─────────────────────────────────────────

    async def _seed_aggregator(
        self,
        aggregator: CandleAggregator,
        symbol: str,
        exchange: str,
        interval: str,
    ) -> None:
        """
        Pre-seeds a CandleAggregator with historical candles.

        Query order:
            1. InfluxDB (fastest — local time-series store)
            2. Broker historical API (fallback if InfluxDB empty)

        After fetching, also writes broker data back to InfluxDB
        to avoid repeated API calls.

        Args:
            aggregator: Target CandleAggregator to seed.
            symbol: Instrument symbol.
            exchange: Exchange code.
            interval: Candle interval.
        """
        broker_name = self._broker_svc.active_broker_name
        from datetime import datetime
        end_dt = now_ist()
        # Calculate look-back based on interval and warmup bars
        minutes = INTERVAL_MINUTES.get(interval, 1)
        lookback_minutes = minutes * self._warmup_bars
        start_dt = end_dt - timedelta(minutes=max(lookback_minutes, 1440))

        # 1 ── Try InfluxDB first ──────────────
        try:
            df = await self._influx.get_candles(
                symbol=symbol,
                exchange=exchange,
                interval=interval,
                broker_name=broker_name,
                from_dt=start_dt,
                to_dt=end_dt,
                limit=self._warmup_bars,
            )
            if not df.empty:
                candles = [
                    OHLCV(
                        datetime=idx,
                        open=row["open"],
                        high=row["high"],
                        low=row["low"],
                        close=row["close"],
                        volume=row["volume"],
                        symbol=symbol,
                        exchange=exchange,
                        interval=interval,
                        broker_name=broker_name,
                    )
                    for idx, row in df.iterrows()
                ]
                aggregator.seed_from_history(candles)
                logger.info(
                    f"Aggregator seeded from InfluxDB | "
                    f"symbol={symbol} | interval={interval} | "
                    f"bars={len(candles)}"
                )
                return
        except Exception as exc:
            logger.warning(
                f"InfluxDB seed failed | "
                f"symbol={symbol} | error={exc}"
            )

        # 2 ── Fallback: broker historical API ──
        try:
            candles = await self._broker_svc.broker.get_historical_data(
                symbol=symbol,
                exchange=exchange,
                interval=interval,
                from_date=start_dt.strftime("%Y-%m-%d"),
                to_date=end_dt.strftime("%Y-%m-%d"),
            )
            if candles:
                aggregator.seed_from_history(candles)
                # Back-fill InfluxDB for future use
                await self._influx.write_candles(
                    candles=[c.to_dict() for c in candles],
                    broker_name=broker_name,
                    symbol=symbol,
                    exchange=exchange,
                    interval=interval,
                )
                logger.info(
                    f"Aggregator seeded from broker API | "
                    f"symbol={symbol} | interval={interval} | "
                    f"bars={len(candles)}"
                )
        except Exception as exc:
            logger.warning(
                f"Broker seed failed | "
                f"symbol={symbol} | error={exc}"
            )

    # ─────────────────────────────────────────
    # Public Accessors
    # ─────────────────────────────────────────

    def get_aggregator(
        self, symbol: str, exchange: str, interval: str
    ) -> Optional[CandleAggregator]:
        """
        Returns the CandleAggregator for a symbol+interval pair.

        Args:
            symbol: Instrument symbol.
            exchange: Exchange code.
            interval: Candle interval string.

        Returns:
            CandleAggregator or None if not subscribed.
        """
        agg_key = (
            f"{symbol.upper()}:{exchange.upper()}:{interval}"
        )
        return self._aggregators.get(agg_key)

    def get_candles(
        self,
        symbol: str,
        exchange: str,
        interval: str,
        n: Optional[int] = None,
    ) -> list[OHLCV]:
        """
        Returns completed candles from the in-memory aggregator.

        Args:
            symbol: Instrument symbol.
            exchange: Exchange code.
            interval: Candle interval.
            n: Number of recent candles. Returns all if None.

        Returns:
            List of OHLCV candles (oldest first).
        """
        aggregator = self.get_aggregator(symbol, exchange, interval)
        if not aggregator:
            return []
        return aggregator.get_candles(n)

    def get_last_price(
        self, symbol: str, exchange: str = "NSE"
    ) -> Optional[float]:
        """
        Returns the last known price for a symbol from any aggregator.

        Args:
            symbol: Instrument symbol.
            exchange: Exchange code.

        Returns:
            Last price float or None if not subscribed.
        """
        sym_key = f"{symbol.upper()}:{exchange.upper()}"
        for agg_key, agg in self._aggregators.items():
            if agg_key.startswith(sym_key):
                candle = agg.latest_candle
                if candle:
                    return candle.close
                cur = agg.current_candle
                if cur:
                    return cur.get("close")
        return None

    @property
    def subscribed_symbols(self) -> list[str]:
        """Returns list of currently subscribed symbol keys."""
        return list(self._subscription_refs.keys())

    @property
    def active_consumers(self) -> list[str]:
        """Returns list of active consumer IDs."""
        return list(self._consumers.keys())

    @property
    def is_running(self) -> bool:
        """Returns True if FeedService is active."""
        return self._is_running

    def get_feed_stats(self) -> dict[str, Any]:
        """
        Returns a summary of current feed state for monitoring.

        Returns:
            Dict with subscription counts, consumer counts,
            aggregator states.
        """
        return {
            "is_running":           self._is_running,
            "active_broker":        self._broker_svc.active_broker_name,
            "subscribed_symbols":   len(self._subscription_refs),
            "active_consumers":     len(self._consumers),
            "active_aggregators":   len(self._aggregators),
            "symbol_keys":          self.subscribed_symbols,
            "consumer_ids":         self.active_consumers,
            "aggregator_stats": {
                key: agg.candle_count
                for key, agg in self._aggregators.items()
            },
        }