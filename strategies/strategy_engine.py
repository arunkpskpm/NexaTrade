"""
NexaTrade — Strategy Engine.

The StrategyEngine is the core orchestrator that:
  - Discovers strategy plugins via importlib
  - Manages the lifecycle of all active strategies
  - Routes ticks and candles to the correct strategies
  - Intercepts strategy signals and passes them
    through the RiskManager before order execution
  - Routes order updates back to the originating strategy
  - Manages per-strategy error handling and recovery

Architecture:
    FeedService
        └── StrategyEngine
                ├── RiskManager (signal gate)
                ├── OrderEngine (signal → order)
                └── Strategies[] (AbstractStrategy instances)

Plugin discovery:
    - Scans the plugins/ directory for Python files
    - Imports each module and finds AbstractStrategy subclasses
    - Registers each discovered class in the strategy registry
    - Supports hot-reload via reload_plugins()

Strategy isolation:
    - Each strategy runs in its own try/except scope
    - Errors in one strategy never affect others
    - Errors are routed to strategy.on_error()
    - Repeated errors trigger auto-deactivation

Usage:
    engine = StrategyEngine(
        broker_service=broker_svc,
        feed_service=feed_svc,
        data_service=data_svc,
        risk_manager=risk_mgr,
        redis_client=redis,
        pg_client=pg,
    )
    await engine.start()
    await engine.activate_strategy("ema_crossover")
    await engine.stop()
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Any, Optional, Type

from brokers.models import (
    OHLCV,
    OrderResponse,
    OrderType,
    OrderRequest,
    ProductType,
    StrategySignal,
    TickData,
    TradingMode,
    Exchange,
    Segment,
    TransactionType,
    SignalDirection,
)
from strategies.abstract_strategy import AbstractStrategy
from strategies.risk_manager import RiskManager
from utils.logger import get_logger, get_trade_logger

logger = get_logger(__name__)
trade_logger = get_trade_logger(__name__)


class StrategyEngine:
    """
    NexaTrade Strategy Engine.

    Owns:
        - Strategy plugin registry (discovered classes)
        - Active strategy instances (running strategies)
        - Signal → risk → order pipeline
        - Order → strategy routing table
        - Per-strategy error counters

    Thread safety:
        All operations run in the asyncio event loop.
    """

    # Max consecutive errors before auto-deactivating a strategy
    MAX_STRATEGY_ERRORS = 5

    def __init__(
        self,
        broker_service,     # BrokerService
        feed_service,       # FeedService
        data_service,       # DataService
        risk_manager: RiskManager,
        redis_client,       # RedisClient
        pg_client,          # PostgresClient
    ) -> None:
        self._broker_svc = broker_service
        self._feed_svc   = feed_service
        self._data_svc   = data_service
        self._risk_mgr   = risk_manager
        self._redis      = redis_client
        self._pg         = pg_client

        # ── Plugin registry ───────────────────
        # {strategy_name: AbstractStrategy subclass}
        self._registry: dict[str, Type[AbstractStrategy]] = {}

        # ── Active strategy instances ──────────
        # {strategy_name: AbstractStrategy instance}
        self._active: dict[str, AbstractStrategy] = {}

        # ── Order → Strategy routing ──────────
        # {order_id: strategy_name}
        self._order_strategy_map: dict[str, str] = {}

        # ── Error counters ────────────────────
        # {strategy_name: consecutive_error_count}
        self._error_counts: dict[str, int] = {}

        # ── State ─────────────────────────────
        self._is_running: bool = False
        self._plugin_dir: Optional[Path] = None

    # ─────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────

    async def start(self) -> None:
        """
        Starts the StrategyEngine.

        1. Loads plugin directory path from config
        2. Discovers all strategy plugins
        3. Registers all strategies in PostgreSQL
        4. Registers order update callback on BrokerService
        5. Activates any strategies marked is_active in DB

        Raises:
            RuntimeError: If engine fails to start.
        """
        from config.settings import get_settings
        cfg = get_settings().app_config.get("strategy", {})
        plugin_dir = cfg.get("plugin_directory", "plugins")
        auto_discover = cfg.get("auto_discover", True)

        self._plugin_dir = (
            Path(plugin_dir)
            if Path(plugin_dir).is_absolute()
            else Path.cwd() / plugin_dir
        )
        self._plugin_dir.mkdir(parents=True, exist_ok=True)

        # Discover plugins
        if auto_discover:
            await self.discover_plugins()

        # Register order update handler on broker service
        self._broker_svc.register_order_subscriber(
            self._on_order_update
        )

        self._is_running = True
        logger.info(
            f"StrategyEngine started | "
            f"registered_strategies={len(self._registry)} | "
            f"plugin_dir={self._plugin_dir}"
        )

        # Auto-activate strategies marked active in DB
        await self._auto_activate_from_db()

    async def stop(self) -> None:
        """
        Stops all active strategies and shuts down the engine.
        Deactivates strategies in reverse activation order.
        """
        self._is_running = False
        self._broker_svc.unregister_order_subscriber(
            self._on_order_update
        )

        # Stop all active strategies
        for name in list(self._active.keys()):
            await self.deactivate_strategy(name)

        self._active.clear()
        self._order_strategy_map.clear()
        logger.info("StrategyEngine stopped.")

    # ─────────────────────────────────────────
    # Plugin Discovery
    # ─────────────────────────────────────────

    async def discover_plugins(self) -> int:
        """
        Scans the plugin directory for AbstractStrategy subclasses.
        Imports each .py file and registers discovered classes.

        Returns:
            Number of new strategy classes discovered.

        Example:
            count = await engine.discover_plugins()
            logger.info(f"Discovered {count} strategy plugins")
        """
        if not self._plugin_dir or not self._plugin_dir.exists():
            logger.warning(
                f"Plugin directory not found: {self._plugin_dir}"
            )
            return 0

        discovered = 0
        plugin_files = list(self._plugin_dir.glob("*.py"))
        plugin_files = [
            f for f in plugin_files
            if not f.name.startswith("_")
        ]

        for plugin_file in plugin_files:
            try:
                count = self._import_plugin(plugin_file)
                discovered += count
            except Exception as exc:
                logger.error(
                    f"Plugin import failed | "
                    f"file={plugin_file.name} | error={exc}"
                )

        logger.info(
            f"Plugin discovery complete | "
            f"files_scanned={len(plugin_files)} | "
            f"strategies_found={discovered}"
        )

        # Sync with PostgreSQL
        await self._sync_registry_to_db()
        return discovered

    def _import_plugin(self, plugin_file: Path) -> int:
        """
        Imports a single plugin file and registers
        all AbstractStrategy subclasses found.

        Args:
            plugin_file: Path to the plugin .py file.

        Returns:
            Number of strategy classes registered from this file.
        """
        module_name = f"plugins.{plugin_file.stem}"

        # Reload if already imported (hot-reload support)
        if module_name in sys.modules:
            module = importlib.reload(sys.modules[module_name])
        else:
            spec = importlib.util.spec_from_file_location(
                module_name, plugin_file
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

        count = 0
        for attr_name in dir(module):
            obj = getattr(module, attr_name)
            if (
                inspect.isclass(obj)
                and issubclass(obj, AbstractStrategy)
                and obj is not AbstractStrategy
                and obj.STRATEGY_NAME != "abstract_strategy"
            ):
                strategy_name = obj.STRATEGY_NAME
                if strategy_name not in self._registry:
                    self._registry[strategy_name] = obj
                    count += 1
                    logger.info(
                        f"Strategy registered | "
                        f"name={strategy_name} | "
                        f"class={obj.__name__} | "
                        f"file={plugin_file.name}"
                    )
                else:
                    # Update existing registration (hot-reload)
                    self._registry[strategy_name] = obj
                    logger.debug(
                        f"Strategy reloaded | "
                        f"name={strategy_name}"
                    )
        return count

    async def reload_plugins(self) -> int:
        """
        Hot-reloads all plugin files.
        Running strategies are NOT interrupted.
        Only the registry is updated with new class definitions.

        Returns:
            Number of strategy classes reloaded.
        """
        logger.info("Hot-reloading strategy plugins...")
        return await self.discover_plugins()

    # ─────────────────────────────────────────
    # Strategy Activation / Deactivation
    # ─────────────────────────────────────────

    async def activate_strategy(
        self,
        strategy_name: str,
        parameters: Optional[dict[str, Any]] = None,
        instruments: Optional[list[dict[str, str]]] = None,
        capital: float = 0.0,
    ) -> bool:
        """
        Activates a registered strategy.

        Steps:
        1. Look up class in registry
        2. Instantiate the strategy
        3. Inject dependencies (feed, data, redis, pg)
        4. Set broker/mode/parameter context
        5. Register signal callback
        6. Call strategy.on_start()
        7. Mark as active in PostgreSQL

        Args:
            strategy_name: Registered strategy name.
            parameters: Override parameter dict.
                        Merged with DEFAULT_PARAMETERS.
            instruments: Override instrument list.
            capital: Allocated capital in INR.

        Returns:
            True if activated successfully.

        Raises:
            ValueError: If strategy name is not registered.
        """
        if strategy_name not in self._registry:
            raise ValueError(
                f"Strategy '{strategy_name}' not in registry. "
                f"Registered: {list(self._registry.keys())}"
            )

        if strategy_name in self._active:
            logger.warning(
                f"Strategy already active | name={strategy_name}"
            )
            return True

        strategy_class = self._registry[strategy_name]

        # Instantiate
        strategy = strategy_class()

        # Inject broker/mode context
        from config.settings import get_settings
        settings = get_settings()
        strategy.broker_name  = self._broker_svc.active_broker_name
        strategy.trading_mode = settings.trading_mode

        # Apply parameters
        if parameters:
            strategy.parameters.update(parameters)

        # Apply instruments
        if instruments:
            strategy.instruments = instruments

        # Apply capital
        strategy.capital = capital

        # Inject service dependencies
        strategy._feed  = self._feed_svc
        strategy._data  = self._data_svc
        strategy._redis = self._redis
        strategy._pg    = self._pg

        # Register signal callback
        strategy._signal_callback = (
            self._on_strategy_signal_wrapper(strategy_name)
        )

        # Call on_start()
        try:
            await strategy.on_start()
            strategy._is_running = True
        except Exception as exc:
            logger.error(
                f"Strategy on_start() failed | "
                f"name={strategy_name} | error={exc}"
            )
            await self._safe_error(strategy, exc)
            return False

        self._active[strategy_name] = strategy
        self._error_counts[strategy_name] = 0

        # Update PostgreSQL
        try:
            await self._pg.set_strategy_active(
                strategy_name, is_active=True
            )
        except Exception as exc:
            logger.warning(
                f"DB strategy activation update failed: {exc}"
            )

        logger.info(
            f"Strategy activated | "
            f"name={strategy_name} | "
            f"broker={strategy.broker_name} | "
            f"mode={strategy.trading_mode} | "
            f"capital={capital}"
        )
        trade_logger.info(
            f"STRATEGY STARTED | "
            f"name={strategy_name} | "
            f"mode={strategy.trading_mode} | "
            f"broker={strategy.broker_name}"
        )
        return True

    async def deactivate_strategy(
        self,
        strategy_name: str,
        reason: str = "manual",
    ) -> bool:
        """
        Deactivates a running strategy.

        Calls strategy.on_stop(), unregisters callbacks,
        and marks as inactive in PostgreSQL.

        Args:
            strategy_name: Strategy to deactivate.
            reason: Deactivation reason for logging.

        Returns:
            True if deactivated successfully.
        """
        if strategy_name not in self._active:
            logger.warning(
                f"Strategy not active | name={strategy_name}"
            )
            return False

        strategy = self._active[strategy_name]

        # Call on_stop()
        try:
            await strategy.on_stop()
        except Exception as exc:
            logger.error(
                f"Strategy on_stop() error | "
                f"name={strategy_name} | error={exc}"
            )

        strategy._is_running = False
        del self._active[strategy_name]

        # Clean up order routing table
        orphaned = [
            oid for oid, sname
            in self._order_strategy_map.items()
            if sname == strategy_name
        ]
        for oid in orphaned:
            del self._order_strategy_map[oid]

        # Update PostgreSQL
        try:
            await self._pg.set_strategy_active(
                strategy_name, is_active=False
            )
        except Exception as exc:
            logger.warning(
                f"DB strategy deactivation update failed: {exc}"
            )

        logger.info(
            f"Strategy deactivated | "
            f"name={strategy_name} | reason={reason}"
        )
        trade_logger.info(
            f"STRATEGY STOPPED | "
            f"name={strategy_name} | reason={reason}"
        )
        return True

    async def restart_strategy(
        self,
        strategy_name: str,
    ) -> bool:
        """
        Restarts a strategy (deactivate + activate).
        Preserves parameters and instruments.

        Args:
            strategy_name: Strategy to restart.

        Returns:
            True if restart succeeded.
        """
        strategy = self._active.get(strategy_name)
        params = dict(strategy.parameters) if strategy else {}
        instruments = list(strategy.instruments) if strategy else []
        capital = strategy.capital if strategy else 0.0

        await self.deactivate_strategy(
            strategy_name, reason="restart"
        )
        await asyncio.sleep(0.5)  # Brief settle delay
        return await self.activate_strategy(
            strategy_name,
            parameters=params,
            instruments=instruments,
            capital=capital,
        )

    # ─────────────────────────────────────────
    # Signal Pipeline
    # ─────────────────────────────────────────

    def _on_strategy_signal_wrapper(
        self, strategy_name: str
    ) -> Any:
        """
        Returns a bound async callback for a specific strategy.
        The callback is what strategies call via self.emit_signal().

        Args:
            strategy_name: Strategy that will emit signals.

        Returns:
            Async callable accepting a StrategySignal.
        """
        async def _callback(signal: StrategySignal) -> None:
            await self._process_signal(signal, strategy_name)
        return _callback

    async def _process_signal(
        self,
        signal: StrategySignal,
        strategy_name: str,
    ) -> None:
        """
        Full signal pipeline:
            1. Risk manager approval
            2. Signal → OrderRequest translation
            3. Order placement via broker service
            4. Order → strategy routing registration
            5. PostgreSQL order persistence
            6. Redis pub/sub event

        Args:
            signal: StrategySignal from the strategy.
            strategy_name: Originating strategy name.
        """
        from config.settings import get_settings
        settings = get_settings()
        broker_name  = self._broker_svc.active_broker_name
        trading_mode = settings.trading_mode

        # 1 ── Risk gate ───────────────────────
        approved = await self._risk_mgr.approve_signal(
            signal=signal,
            broker_name=broker_name,
            trading_mode=trading_mode,
        )
        if not approved:
            return

        # 2 ── Signal → OrderRequest ───────────
        order_request = self._signal_to_order_request(
            signal=signal,
            strategy_name=strategy_name,
            trading_mode=trading_mode,
        )
        if not order_request:
            logger.warning(
                f"Could not translate signal to order | "
                f"direction={signal.direction}"
            )
            return

        # 3 ── Place order ─────────────────────
        try:
            response = await self._broker_svc.place_order(
                order_request
            )
        except Exception as exc:
            logger.error(
                f"Order placement failed | "
                f"strategy={strategy_name} | error={exc}"
            )
            return

        # 4 ── Register order → strategy routing ─
        if response.broker_order_id or response.order_id:
            self._order_strategy_map[response.order_id] = (
                strategy_name
            )

        # 5 ── PostgreSQL persistence ──────────
        strategy = self._active.get(strategy_name)
        try:
            await self._pg.insert_order(
                order_id=order_request.order_id,
                broker_name=broker_name,
                symbol=order_request.symbol,
                exchange=str(order_request.exchange),
                segment=str(order_request.segment),
                order_type=str(order_request.order_type),
                transaction_type=str(
                    order_request.transaction_type
                ),
                quantity=order_request.quantity,
                price=order_request.price,
                trigger_price=order_request.trigger_price,
                trading_mode=trading_mode,
                strategy_name=strategy_name,
                broker_order_id=response.broker_order_id,
            )
        except Exception as exc:
            logger.warning(
                f"Order DB persist failed: {exc}"
            )

        # 6 ── Redis pub/sub event ─────────────
        try:
            await self._redis.publish(
                "orders",
                {
                    "event":        "order_placed",
                    "order_id":     order_request.order_id,
                    "strategy":     strategy_name,
                    "symbol":       signal.symbol,
                    "direction":    str(signal.direction),
                    "status":       str(response.status),
                    "broker":       broker_name,
                    "trading_mode": trading_mode,
                },
            )
        except Exception:
            pass

        if strategy:
            strategy._order_count += 1

        logger.info(
            f"Order placed | "
            f"strategy={strategy_name} | "
            f"symbol={signal.symbol} | "
            f"direction={signal.direction} | "
            f"status={response.status} | "
            f"broker_id={response.broker_order_id}"
        )

    def _signal_to_order_request(
        self,
        signal: StrategySignal,
        strategy_name: str,
        trading_mode: str,
    ) -> Optional[OrderRequest]:
        """
        Translates a StrategySignal into an OrderRequest.

        Rules:
        - BUY → TransactionType.BUY, MARKET order
        - SELL → TransactionType.SELL, MARKET order
        - EXIT → derives direction from existing position
        - HOLD → returns None (no order)
        - NONE → returns None (no order)

        Uses suggested_quantity if provided, else derives
        quantity from signal.strength * max_position_size.

        Args:
            signal: StrategySignal to translate.
            strategy_name: Originating strategy name.
            trading_mode: "paper" or "live".

        Returns:
            OrderRequest or None.
        """
        direction = signal.direction

        if direction in (SignalDirection.HOLD, SignalDirection.NONE):
            return None

        # Determine transaction type
        if direction == SignalDirection.BUY:
            txn_type = TransactionType.BUY
        elif direction == SignalDirection.SELL:
            txn_type = TransactionType.SELL
        elif direction == SignalDirection.EXIT:
            # EXIT: determine based on position direction
            # Default to SELL (closing a long) if unknown
            txn_type = TransactionType.SELL
        else:
            return None

        # Determine quantity
        quantity = signal.suggested_quantity
        if not quantity:
            from config.settings import get_settings
            max_size = int(
                get_settings()
                .risk_params
                .get("position_limits", {})
                .get("max_position_size", 100)
            )
            quantity = max(
                1, int(max_size * signal.strength)
            )

        # Determine order type and price
        if signal.suggested_price:
            order_type = OrderType.LIMIT
            price = signal.suggested_price
        else:
            order_type = OrderType.MARKET
            price = None

        return OrderRequest(
            symbol=signal.symbol,
            exchange=signal.exchange,
            segment=signal.segment,
            transaction_type=txn_type,
            order_type=order_type,
            product_type=ProductType.INTRADAY,
            quantity=quantity,
            price=price,
            trigger_price=signal.stop_loss_price,
            strategy_name=strategy_name,
            trading_mode=(
                TradingMode.PAPER
                if trading_mode == "paper"
                else TradingMode.LIVE
            ),
            tags={
                "signal_id":    signal.signal_id,
                "signal_reason": signal.reason,
                "strength":     signal.strength,
            },
        )

    # ─────────────────────────────────────────
    # Order Update Routing
    # ─────────────────────────────────────────

    async def _on_order_update(
        self, response: OrderResponse
    ) -> None:
        """
        Receives order updates from BrokerService.
        Routes them to the originating strategy's
        on_order_update() method.

        Also updates:
        - PostgreSQL order status
        - Redis P&L cache (on COMPLETE)
        - Redis position cache (on COMPLETE)

        Args:
            response: Updated OrderResponse from broker.
        """
        order_id = response.order_id
        strategy_name = self._order_strategy_map.get(order_id)

        # Update PostgreSQL order status
        try:
            await self._pg.update_order_status(
                order_id=order_id,
                status=str(response.status),
                broker_order_id=response.broker_order_id,
                filled_quantity=response.filled_quantity,
                average_price=response.average_price,
                rejection_reason=response.rejection_reason,
            )
        except Exception as exc:
            logger.warning(
                f"Order status DB update failed: {exc}"
            )

        # Route to originating strategy
        if strategy_name and strategy_name in self._active:
            strategy = self._active[strategy_name]
            try:
                await strategy.on_order_update(response)
            except Exception as exc:
                logger.error(
                    f"Strategy on_order_update error | "
                    f"strategy={strategy_name} | error={exc}"
                )
                await self._handle_strategy_error(
                    strategy_name, exc
                )

        # Publish order update event
        try:
            await self._redis.publish(
                "orders",
                {
                    "event":            "order_updated",
                    "order_id":         order_id,
                    "status":           str(response.status),
                    "filled_quantity":  response.filled_quantity,
                    "average_price":    response.average_price,
                    "strategy":         strategy_name,
                },
            )
        except Exception:
            pass

    # ─────────────────────────────────────────
    # Error Handling
    # ─────────────────────────────────────────

    async def _safe_error(
        self,
        strategy: AbstractStrategy,
        exc: Exception,
    ) -> None:
        """
        Safely calls strategy.on_error() without propagating.

        Args:
            strategy: Target strategy instance.
            exc: Exception to report.
        """
        try:
            await strategy.on_error(exc)
        except Exception as inner_exc:
            logger.error(
                f"Strategy on_error() itself raised: {inner_exc}"
            )

    async def _handle_strategy_error(
        self,
        strategy_name: str,
        exc: Exception,
    ) -> None:
        """
        Increments error counter and auto-deactivates
        a strategy if MAX_STRATEGY_ERRORS is reached.

        Args:
            strategy_name: Affected strategy.
            exc: The exception that occurred.
        """
        self._error_counts[strategy_name] = (
            self._error_counts.get(strategy_name, 0) + 1
        )
        count = self._error_counts[strategy_name]

        logger.warning(
            f"Strategy error #{count} | "
            f"name={strategy_name} | error={exc}"
        )

        if strategy := self._active.get(strategy_name):
            await self._safe_error(strategy, exc)

        if count >= self.MAX_STRATEGY_ERRORS:
            logger.error(
                f"Strategy auto-deactivated after "
                f"{count} errors | name={strategy_name}"
            )
            await self.deactivate_strategy(
                strategy_name,
                reason=f"auto_deactivated_after_{count}_errors",
            )

    # ─────────────────────────────────────────
    # DB Sync Helpers
    # ─────────────────────────────────────────

    async def _sync_registry_to_db(self) -> None:
        """
        Upserts all registered strategy classes to PostgreSQL.
        Called after plugin discovery to keep the DB in sync.
        """
        for name, cls in self._registry.items():
            try:
                await self._pg.upsert_strategy(
                    name=name,
                    display_name=cls.DISPLAY_NAME,
                    module_path=cls.__module__,
                    class_name=cls.__name__,
                    description=cls.DESCRIPTION,
                    parameters=cls.DEFAULT_PARAMETERS,
                    is_active=(name in self._active),
                    broker_name=(
                        self._broker_svc.active_broker_name
                    ),
                )
            except Exception as exc:
                logger.warning(
                    f"Strategy DB sync failed | "
                    f"name={name} | error={exc}"
                )

    async def _auto_activate_from_db(self) -> None:
        """
        Activates strategies marked is_active=True in PostgreSQL.
        Called on engine start to restore previous session state.
        """
        try:
            active_strategies = await self._pg.get_active_strategies(
                broker_name=self._broker_svc.active_broker_name
            )
            for record in active_strategies:
                name = record.get("name")
                if name and name in self._registry:
                    params = record.get("parameters") or {}
                    await self.activate_strategy(
                        strategy_name=name,
                        parameters=params,
                    )
        except Exception as exc:
            logger.warning(
                f"Auto-activate from DB failed: {exc}"
            )

    # ─────────────────────────────────────────
    # Public Accessors
    # ─────────────────────────────────────────

    def get_registered_strategies(
        self,
    ) -> list[dict[str, Any]]:
        """
        Returns metadata for all registered strategy classes.

        Returns:
            List of strategy metadata dicts.
        """
        return [
            {
                "name":         name,
                "display_name": cls.DISPLAY_NAME,
                "description":  cls.DESCRIPTION,
                "version":      cls.VERSION,
                "author":       cls.AUTHOR,
                "parameters":   cls.DEFAULT_PARAMETERS,
                "instruments":  cls.DEFAULT_INSTRUMENTS,
                "interval":     cls.DEFAULT_INTERVAL,
                "is_active":    name in self._active,
            }
            for name, cls in self._registry.items()
        ]

    def get_active_strategies(
        self,
    ) -> list[dict[str, Any]]:
        """
        Returns runtime stats for all active strategy instances.

        Returns:
            List of strategy stats dicts from get_stats().
        """
        return [
            strategy.get_stats()
            for strategy in self._active.values()
        ]

    def get_strategy_instance(
        self, strategy_name: str
    ) -> Optional[AbstractStrategy]:
        """
        Returns the active instance of a strategy.

        Args:
            strategy_name: Strategy name.

        Returns:
            AbstractStrategy instance or None if not active.
        """
        return self._active.get(strategy_name)

    @property
    def is_running(self) -> bool:
        """Returns True if the engine is active."""
        return self._is_running

    @property
    def registered_count(self) -> int:
        """Returns number of registered strategy classes."""
        return len(self._registry)

    @property
    def active_count(self) -> int:
        """Returns number of currently active strategies."""
        return len(self._active)

    def get_engine_stats(self) -> dict[str, Any]:
        """
        Returns overall engine statistics for monitoring.

        Returns:
            Dict with counts, broker, and risk stats.
        """
        return {
            "is_running":           self._is_running,
            "active_broker":        self._broker_svc.active_broker_name,
            "registered_strategies": self.registered_count,
            "active_strategies":    self.active_count,
            "active_names":         list(self._active.keys()),
            "registered_names":     list(self._registry.keys()),
            "order_routing_entries": len(self._order_strategy_map),
            "error_counts":         dict(self._error_counts),
            "risk_stats":           self._risk_mgr.get_stats(),
        }

    def __repr__(self) -> str:
        return (
            f"StrategyEngine("
            f"registered={self.registered_count}, "
            f"active={self.active_count}, "
            f"running={self._is_running})"
        )