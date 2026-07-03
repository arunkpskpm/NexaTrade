"""
NexaTrade — AbstractStrategy Interface.

Every user-defined strategy MUST inherit from AbstractStrategy
and implement the abstract methods. The strategy engine
discovers, loads, and manages all subclasses automatically.

Strategy lifecycle (managed by StrategyEngine):
    __init__()          → called once on load
    on_start()          → called when strategy is activated
    on_tick(tick)       → called on every live tick
    on_candle(candle)   → called on every closed OHLCV candle
    on_order_update()   → called when an order changes state
    on_stop()           → called when strategy is deactivated
    on_error(exc)       → called when an unhandled error occurs

Design rules:
    - Strategies are stateful — maintain state in self
    - Strategies NEVER call broker directly — emit signals only
    - Strategies NEVER import any broker SDK
    - All order placement goes via self.emit_signal()
    - All indicator computation uses utils.indicators
    - All time checks use utils.time_utils

Adding a new strategy:
    1. Create plugins/my_strategy.py
    2. class MyStrategy(AbstractStrategy): ...
    3. Implement all abstract methods
    4. Done — StrategyEngine auto-discovers it

Zero changes needed anywhere else.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

from brokers.models import (
    OHLCV,
    OrderResponse,
    StrategySignal,
    TickData,
    SignalDirection,
    Exchange,
    Segment,
)
from utils.logger import get_logger, get_trade_logger


class AbstractStrategy(ABC):
    """
    NexaTrade Strategy Interface.

    Every strategy plugin inherits this class.
    The strategy engine manages its lifecycle and
    routes ticks, candles, and order updates to it.

    Key attributes (set by engine before on_start()):
        self.name          → strategy identifier (snake_case)
        self.broker_name   → active broker name
        self.trading_mode  → "paper" or "live"
        self.parameters    → dict of user-configurable parameters
        self.capital       → allocated capital (INR)

    Signal emission:
        Strategies communicate intent via self.emit_signal().
        The strategy engine picks up signals and routes them
        to the order engine. Strategies never place orders directly.

    Data access:
        self._feed   → FeedService (candles, last price)
        self._data   → DataService (historical OHLCV)
        self._redis  → RedisClient (signal cache, state)
    """

    # ─────────────────────────────────────────
    # Class-level metadata
    # Override these in your strategy subclass
    # ─────────────────────────────────────────
    STRATEGY_NAME: str = "abstract_strategy"
    DISPLAY_NAME:  str = "Abstract Strategy"
    DESCRIPTION:   str = "Base strategy class — do not use directly."
    VERSION:       str = "1.0.0"
    AUTHOR:        str = "NexaTrade"

    # Default parameters — override in subclass
    # These are shown in the UI parameter editor
    DEFAULT_PARAMETERS: dict[str, Any] = {}

    # Instruments this strategy trades by default
    # Format: [{"symbol": "RELIANCE", "exchange": "NSE"}]
    DEFAULT_INSTRUMENTS: list[dict[str, str]] = []

    # Default candle interval for on_candle() events
    DEFAULT_INTERVAL: str = "5minute"

    def __init__(self) -> None:
        # Set by StrategyEngine before on_start()
        self.name:          str = self.STRATEGY_NAME
        self.broker_name:   str = "paper"
        self.trading_mode:  str = "paper"
        self.parameters:    dict[str, Any] = dict(
            self.DEFAULT_PARAMETERS
        )
        self.capital:       float = 0.0
        self.instruments:   list[dict[str, str]] = list(
            self.DEFAULT_INSTRUMENTS
        )

        # Injected by StrategyEngine before on_start()
        self._feed   = None   # FeedService
        self._data   = None   # DataService
        self._redis  = None   # RedisClient
        self._pg     = None   # PostgresClient

        # Signal callback — set by StrategyEngine
        self._signal_callback: Optional[Callable] = None

        # Internal state
        self._is_running:  bool = False
        self._tick_count:  int  = 0
        self._order_count: int  = 0
        self._signal_count: int = 0

        # Loggers
        self._logger = get_logger(
            f"strategies.{self.STRATEGY_NAME}"
        )
        self._trade_logger = get_trade_logger(
            f"strategies.{self.STRATEGY_NAME}"
        )

    # ═════════════════════════════════════════
    # Section 1 — Abstract Lifecycle Methods
    # Implement ALL of these in your strategy
    # ═════════════════════════════════════════

    @abstractmethod
    async def on_start(self) -> None:
        """
        Called once when the strategy is activated.

        Use this to:
        - Subscribe to feed symbols via self._feed.subscribe()
        - Load historical data via self._data.get_ohlcv()
        - Initialise indicator state
        - Log startup parameters

        Example:
            async def on_start(self):
                self.period = self.parameters.get("period", 20)
                await self._feed.subscribe(
                    "RELIANCE", "NSE",
                    interval=self.DEFAULT_INTERVAL,
                    consumer_id=self.name,
                    candle_callback=self.on_candle,
                    tick_callback=self.on_tick,
                )
                self._logger.info(
                    f"{self.DISPLAY_NAME} started | "
                    f"period={self.period}"
                )
        """
        ...

    @abstractmethod
    async def on_tick(self, tick: TickData) -> None:
        """
        Called on every live tick for subscribed instruments.

        Use this for:
        - Tick-level signal generation
        - Real-time stop-loss monitoring
        - Bid/ask spread analysis

        Args:
            tick: Normalised TickData from the broker feed.

        Example:
            async def on_tick(self, tick: TickData):
                self._tick_count += 1
                if tick.last_price > self._stop_loss:
                    await self.emit_signal(StrategySignal(
                        strategy_name=self.name,
                        symbol=tick.symbol,
                        exchange=Exchange.NSE,
                        segment=Segment.EQ,
                        direction=SignalDirection.EXIT,
                        reason="Stop-loss breached",
                    ))
        """
        ...

    @abstractmethod
    async def on_candle(self, candle: OHLCV) -> None:
        """
        Called on every closed OHLCV candle for subscribed instruments.

        This is the primary entry point for candle-based strategies.
        Use this for:
        - Moving average crossover signals
        - Momentum indicator signals
        - Pattern detection

        Args:
            candle: The just-closed OHLCV candle.

        Example:
            async def on_candle(self, candle: OHLCV):
                candles = self._feed.get_candles(
                    candle.symbol, candle.exchange,
                    self.DEFAULT_INTERVAL, n=50
                )
                if len(candles) < self.period:
                    return
                df = pd.DataFrame([c.to_dict() for c in candles])
                df["sma"] = sma(df["close"], self.period)
                if df["close"].iloc[-1] > df["sma"].iloc[-1]:
                    await self.emit_signal(...)
        """
        ...

    @abstractmethod
    async def on_order_update(self, response: OrderResponse) -> None:
        """
        Called when an order placed by this strategy changes state.

        Use this for:
        - Confirming fills and updating position state
        - Placing stop-loss / target orders after entry fill
        - Logging trade executions

        Args:
            response: Updated OrderResponse for a strategy order.

        Example:
            async def on_order_update(self, response):
                if response.status == OrderStatus.COMPLETE:
                    self._logger.info(
                        f"Order filled | "
                        f"avg_price={response.average_price}"
                    )
                    self._position_price = response.average_price
        """
        ...

    @abstractmethod
    async def on_stop(self) -> None:
        """
        Called when the strategy is deactivated.

        Use this to:
        - Unsubscribe from feed: await self._feed.unsubscribe_all(self.name)
        - Cancel open orders
        - Flush any cached state to Redis

        Example:
            async def on_stop(self):
                await self._feed.unsubscribe_all(self.name)
                self._logger.info(
                    f"{self.DISPLAY_NAME} stopped | "
                    f"signals={self._signal_count}"
                )
        """
        ...

    @abstractmethod
    async def on_error(self, exc: Exception) -> None:
        """
        Called when an unhandled exception occurs in any lifecycle method.

        Implement graceful degradation here.
        The strategy engine catches all exceptions and routes
        them here before deciding whether to stop the strategy.

        Args:
            exc: The exception that was raised.

        Example:
            async def on_error(self, exc: Exception):
                self._logger.error(f"Strategy error: {exc}")
                await self._feed.unsubscribe_all(self.name)
        """
        ...

    # ═════════════════════════════════════════
    # Section 2 — Signal Emission (Concrete)
    # Call this to emit a trading signal
    # ═════════════════════════════════════════

    async def emit_signal(self, signal: StrategySignal) -> None:
        """
        Emits a trading signal to the strategy engine.
        The engine routes it to the order engine for execution.

        Signals are:
        - Cached in Redis with TTL for idempotency
        - Logged to the trade audit log
        - Counted in self._signal_count

        Args:
            signal: Populated StrategySignal model.

        Example:
            await self.emit_signal(StrategySignal(
                strategy_name=self.name,
                symbol="RELIANCE",
                exchange=Exchange.NSE,
                segment=Segment.EQ,
                direction=SignalDirection.BUY,
                strength=0.8,
                suggested_quantity=100,
                stop_loss_price=2400.0,
                target_price=2550.0,
                reason="EMA crossover confirmed",
            ))
        """
        if not self._signal_callback:
            self._logger.warning(
                f"Signal emitted but no callback registered | "
                f"direction={signal.direction} | "
                f"symbol={signal.symbol}"
            )
            return

        self._signal_count += 1

        # Cache signal in Redis for idempotency
        if self._redis:
            try:
                from config.settings import get_settings
                ttl = int(
                    get_settings()
                    .app_config
                    .get("strategy", {})
                    .get("signal_ttl_seconds", 300)
                )
                await self._redis.set_signal(
                    strategy_name=self.name,
                    symbol=signal.symbol,
                    signal={
                        "signal_id":  signal.signal_id,
                        "direction":  str(signal.direction),
                        "strength":   signal.strength,
                        "reason":     signal.reason,
                        "generated_at": str(signal.generated_at),
                    },
                    ttl_seconds=ttl,
                )
            except Exception as exc:
                self._logger.warning(
                    f"Signal Redis cache failed: {exc}"
                )

        # Trade audit log
        self._trade_logger.info(
            f"SIGNAL | "
            f"strategy={self.name} | "
            f"symbol={signal.symbol} | "
            f"direction={signal.direction} | "
            f"strength={signal.strength:.2f} | "
            f"reason={signal.reason}"
        )

        # Deliver to engine callback
        try:
            await self._signal_callback(signal)
        except Exception as exc:
            self._logger.error(
                f"Signal callback failed: {exc}"
            )

    # ═════════════════════════════════════════
    # Section 3 — Parameter Helpers (Concrete)
    # ═════════════════════════════════════════

    def get_param(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        """
        Returns a strategy parameter value.
        Falls back to DEFAULT_PARAMETERS, then to default.

        Args:
            key: Parameter key string.
            default: Fallback value if key not found.

        Returns:
            Parameter value.

        Example:
            period = self.get_param("period", 20)
            multiplier = self.get_param("atr_multiplier", 3.0)
        """
        return self.parameters.get(
            key,
            self.DEFAULT_PARAMETERS.get(key, default),
        )

    def update_parameters(
        self, new_params: dict[str, Any]
    ) -> None:
        """
        Updates strategy parameters at runtime.
        Called by the UI parameter editor.

        Args:
            new_params: Dict of parameter updates.
        """
        self.parameters.update(new_params)
        self._logger.info(
            f"Parameters updated | "
            f"strategy={self.name} | "
            f"params={new_params}"
        )

    # ═════════════════════════════════════════
    # Section 4 — State Helpers (Concrete)
    # ═════════════════════════════════════════

    @property
    def is_running(self) -> bool:
        """Returns True if the strategy is active."""
        return self._is_running

    @property
    def is_paper(self) -> bool:
        """Returns True if running in paper trading mode."""
        return self.trading_mode == "paper"

    @property
    def is_live(self) -> bool:
        """Returns True if running in live trading mode."""
        return self.trading_mode == "live"

    def get_stats(self) -> dict[str, Any]:
        """
        Returns strategy runtime statistics.

        Returns:
            Dict with tick/signal/order counts and state.
        """
        return {
            "name":          self.name,
            "display_name":  self.DISPLAY_NAME,
            "is_running":    self._is_running,
            "trading_mode":  self.trading_mode,
            "broker":        self.broker_name,
            "tick_count":    self._tick_count,
            "signal_count":  self._signal_count,
            "order_count":   self._order_count,
            "capital":       self.capital,
            "instruments":   self.instruments,
            "parameters":    self.parameters,
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"running={self._is_running}, "
            f"mode={self.trading_mode!r})"
        )