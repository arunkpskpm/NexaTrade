"""
NexaTrade — Risk Manager.

The RiskManager is the gatekeeper between the strategy engine
and the order engine. Every signal emitted by a strategy passes
through the RiskManager before becoming an order.

Checks performed on every signal:
    1.  Kill switch active (global or per-broker)
    2.  Market hours check
    3.  Daily loss limit breached
    4.  Max drawdown breached
    5.  Max open positions reached
    6.  Max capital per trade exceeded
    7.  Symbol blacklist check
    8.  Duplicate signal check (Redis TTL-based idempotency)
    9.  Position size validation
    10. Signal direction vs existing position conflict

All checks are read from:
    - risk_config.yaml  → risk parameters
    - Redis             → real-time state (P&L, positions)
    - PostgreSQL        → persistent position counts

When a signal is blocked:
    - Reason is logged to trade audit log
    - Event published to Redis pub/sub
    - Signal is silently dropped (no exception raised)
    - Counter incremented for monitoring

Usage:
    risk_mgr = RiskManager(redis_client, pg_client)
    approved = await risk_mgr.approve_signal(signal, broker_name)
    if approved:
        await order_engine.execute_signal(signal)
"""

from __future__ import annotations

from typing import Any, Optional

from brokers.models import SignalDirection, StrategySignal
from data.storage.postgres_client import PostgresClient
from data.storage.redis_client import RedisClient
from utils.logger import get_logger, get_trade_logger
from utils.time_utils import is_market_open, now_ist

logger = get_logger(__name__)
trade_logger = get_trade_logger(__name__)


class RiskManager:
    """
    NexaTrade signal risk gate.

    All 10 checks run sequentially on every incoming signal.
    First failing check blocks the signal immediately
    (short-circuit evaluation).

    Thread safety:
        All checks are async coroutines.
        Redis reads/writes are atomic where required.
    """

    def __init__(
        self,
        redis_client: RedisClient,
        pg_client: PostgresClient,
    ) -> None:
        self._redis = redis_client
        self._pg    = pg_client

        # Runtime counters
        self._signals_approved:  int = 0
        self._signals_blocked:   int = 0
        self._block_reasons: dict[str, int] = {}

        self._logger = get_logger("strategies.risk_manager")

    # ─────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────

    async def approve_signal(
        self,
        signal: StrategySignal,
        broker_name: str,
        trading_mode: str = "paper",
    ) -> bool:
        """
        Runs all risk checks on an incoming strategy signal.

        Args:
            signal: The StrategySignal to evaluate.
            broker_name: Active broker identifier.
            trading_mode: "paper" or "live".

        Returns:
            True if the signal passes all risk checks.
            False if any check fails (signal is blocked).

        Example:
            approved = await risk_mgr.approve_signal(
                signal, broker_name="breeze",
                trading_mode="live"
            )
        """
        cfg = self._load_risk_config()

        # ── Paper mode bypass ─────────────────
        # Paper trading skips some hard-money checks
        # but still enforces position limits and kill switch
        checks = [
            self._check_kill_switch(broker_name),
            self._check_market_hours(signal, trading_mode),
            self._check_daily_loss_limit(broker_name, cfg),
            self._check_drawdown(broker_name, cfg),
            self._check_max_positions(broker_name, cfg, trading_mode),
            self._check_capital_per_trade(signal, cfg),
            self._check_blacklist(signal, cfg),
            self._check_duplicate_signal(signal),
            self._check_position_size(signal, cfg),
            self._check_direction_conflict(signal, broker_name, trading_mode),
        ]

        for check_coro in checks:
            approved, reason = await check_coro
            if not approved:
                self._signals_blocked += 1
                self._block_reasons[reason] = (
                    self._block_reasons.get(reason, 0) + 1
                )
                trade_logger.warning(
                    f"SIGNAL BLOCKED | "
                    f"strategy={signal.strategy_name} | "
                    f"symbol={signal.symbol} | "
                    f"direction={signal.direction} | "
                    f"reason={reason}"
                )
                # Publish block event to Redis
                try:
                    await self._redis.publish(
                        "risk",
                        {
                            "event":    "signal_blocked",
                            "strategy": signal.strategy_name,
                            "symbol":   signal.symbol,
                            "reason":   reason,
                            "broker":   broker_name,
                        },
                    )
                except Exception:
                    pass
                return False

        self._signals_approved += 1
        self._logger.debug(
            f"Signal approved | "
            f"strategy={signal.strategy_name} | "
            f"symbol={signal.symbol} | "
            f"direction={signal.direction}"
        )
        return True

    # ─────────────────────────────────────────
    # Individual Risk Checks
    # Each returns (passed: bool, reason: str)
    # ─────────────────────────────────────────

    async def _check_kill_switch(
        self, broker_name: str
    ) -> tuple[bool, str]:
        """
        Check 1: Kill switch (global or per-broker).
        Blocks ALL signals if kill switch is armed.
        """
        try:
            global_ks = (
                await self._redis.is_global_kill_switch_active()
            )
            if global_ks:
                return False, "global_kill_switch_active"

            broker_ks = await self._redis.is_kill_switch_active(
                broker_name
            )
            if broker_ks:
                return False, f"broker_kill_switch_active:{broker_name}"
        except Exception as exc:
            logger.warning(f"Kill switch check failed: {exc}")
        return True, ""

    async def _check_market_hours(
        self,
        signal: StrategySignal,
        trading_mode: str,
    ) -> tuple[bool, str]:
        """
        Check 2: Market hours.
        Blocks signals outside NSE trading hours (09:15–15:30).
        Paper trading respects market hours by default.
        EXIT signals are always allowed (stop-loss exits).
        """
        # EXIT signals bypass market hours check
        if signal.direction == SignalDirection.EXIT:
            return True, ""

        if not is_market_open():
            now = now_ist()
            return (
                False,
                f"market_closed:{now.strftime('%H:%M IST')}",
            )
        return True, ""

    async def _check_daily_loss_limit(
        self,
        broker_name: str,
        cfg: dict,
    ) -> tuple[bool, str]:
        """
        Check 3: Daily loss limit.
        Reads the current daily P&L from Redis.
        Blocks new entry signals if limit is breached.
        """
        try:
            loss_limits = cfg.get("loss_limits", {})
            daily_limit = float(
                loss_limits.get("daily_loss_limit", 10000.0)
            )
            daily_pnl = await self._redis.get_daily_pnl(
                broker_name
            )
            # P&L is negative for losses
            if daily_pnl <= -abs(daily_limit):
                # Arm kill switch automatically
                await self._redis.set_kill_switch(
                    broker_name, active=True
                )
                return (
                    False,
                    f"daily_loss_limit_breached:"
                    f"pnl={daily_pnl:.2f}",
                )
        except Exception as exc:
            logger.warning(f"Daily loss check failed: {exc}")
        return True, ""

    async def _check_drawdown(
        self,
        broker_name: str,
        cfg: dict,
    ) -> tuple[bool, str]:
        """
        Check 4: Maximum drawdown.
        Reads latest risk snapshot from PostgreSQL.
        Blocks if realised drawdown exceeds threshold.
        """
        try:
            loss_limits = cfg.get("loss_limits", {})
            max_dd_pct = float(
                loss_limits.get("max_drawdown_pct", 5.0)
            )
            snapshot = await self._pg.get_latest_risk_snapshot(
                broker_name=broker_name
            )
            if snapshot:
                current_dd = float(
                    snapshot.get("max_drawdown_pct", 0.0)
                )
                if current_dd >= max_dd_pct:
                    return (
                        False,
                        f"max_drawdown_breached:"
                        f"drawdown={current_dd:.2f}%",
                    )
        except Exception as exc:
            logger.warning(f"Drawdown check failed: {exc}")
        return True, ""

    async def _check_max_positions(
        self,
        broker_name: str,
        cfg: dict,
        trading_mode: str,
    ) -> tuple[bool, str]:
        """
        Check 5: Maximum concurrent open positions.
        Counts non-zero quantity positions from PostgreSQL.
        """
        try:
            pos_limits = cfg.get("position_limits", {})
            max_positions = int(
                pos_limits.get("max_open_positions", 10)
            )
            open_positions = await self._pg.get_positions(
                broker_name=broker_name,
                trading_mode=trading_mode,
                open_only=True,
            )
            if len(open_positions) >= max_positions:
                return (
                    False,
                    f"max_positions_reached:{len(open_positions)}",
                )
        except Exception as exc:
            logger.warning(f"Max positions check failed: {exc}")
        return True, ""

    async def _check_capital_per_trade(
        self,
        signal: StrategySignal,
        cfg: dict,
    ) -> tuple[bool, str]:
        """
        Check 6: Maximum capital per trade.
        Validates that suggested_quantity * suggested_price
        does not exceed max_capital_per_trade_pct of total capital.
        """
        try:
            capital_cfg = cfg.get("capital", {})
            total_capital = float(
                capital_cfg.get("total_capital", 1_000_000.0)
            )
            max_pct = float(
                capital_cfg.get("max_capital_per_trade_pct", 10.0)
            )
            max_trade_value = total_capital * (max_pct / 100.0)

            if signal.suggested_quantity and signal.suggested_price:
                trade_value = (
                    signal.suggested_quantity * signal.suggested_price
                )
                if trade_value > max_trade_value:
                    return (
                        False,
                        f"capital_per_trade_exceeded:"
                        f"value={trade_value:.2f}:"
                        f"limit={max_trade_value:.2f}",
                    )
        except Exception as exc:
            logger.warning(f"Capital per trade check failed: {exc}")
        return True, ""

    async def _check_blacklist(
        self,
        signal: StrategySignal,
        cfg: dict,
    ) -> tuple[bool, str]:
        """
        Check 7: Symbol and exchange blacklist.
        Reads from risk_config.yaml blacklist section.
        """
        try:
            blacklist = cfg.get("blacklist", {})
            blocked_symbols = [
                s.upper() for s in blacklist.get("symbols", [])
            ]
            blocked_exchanges = [
                e.upper() for e in blacklist.get("exchanges", [])
            ]

            if signal.symbol.upper() in blocked_symbols:
                return (
                    False,
                    f"symbol_blacklisted:{signal.symbol}",
                )
            if str(signal.exchange).upper() in blocked_exchanges:
                return (
                    False,
                    f"exchange_blacklisted:{signal.exchange}",
                )
        except Exception as exc:
            logger.warning(f"Blacklist check failed: {exc}")
        return True, ""

    async def _check_duplicate_signal(
        self,
        signal: StrategySignal,
    ) -> tuple[bool, str]:
        """
        Check 8: Duplicate signal guard (Redis TTL-based).
        Prevents the same strategy from firing the same
        direction on the same symbol within the signal TTL window.
        Idempotency key: strategy + symbol + direction.
        """
        try:
            dedup_key = (
                f"dedup:{signal.strategy_name}:"
                f"{signal.symbol}:{signal.direction}"
            )
            existing = await self._redis.get(dedup_key)
            if existing:
                return (
                    False,
                    f"duplicate_signal:"
                    f"{signal.strategy_name}:{signal.symbol}:"
                    f"{signal.direction}",
                )

            # Mark signal as seen for TTL window
            from config.settings import get_settings
            ttl = int(
                get_settings()
                .app_config
                .get("strategy", {})
                .get("signal_ttl_seconds", 300)
            )
            await self._redis.set(dedup_key, "1", ttl_seconds=ttl)
        except Exception as exc:
            logger.warning(f"Duplicate signal check failed: {exc}")
        return True, ""

    async def _check_position_size(
        self,
        signal: StrategySignal,
        cfg: dict,
    ) -> tuple[bool, str]:
        """
        Check 9: Maximum position size.
        Validates suggested_quantity against position_limits config.
        """
        try:
            pos_limits = cfg.get("position_limits", {})
            max_size = int(
                pos_limits.get("max_position_size", 500)
            )
            if signal.suggested_quantity:
                if signal.suggested_quantity > max_size:
                    return (
                        False,
                        f"position_size_exceeded:"
                        f"qty={signal.suggested_quantity}:"
                        f"limit={max_size}",
                    )
        except Exception as exc:
            logger.warning(f"Position size check failed: {exc}")
        return True, ""

    async def _check_direction_conflict(
        self,
        signal: StrategySignal,
        broker_name: str,
        trading_mode: str,
    ) -> tuple[bool, str]:
        """
        Check 10: Direction conflict with existing position.
        Blocks a BUY signal if already long,
        blocks a SELL signal if already short.
        Allows EXIT regardless of direction.
        """
        try:
            if signal.direction == SignalDirection.HOLD:
                return False, "hold_signal_not_executable"

            if signal.direction == SignalDirection.EXIT:
                return True, ""

            # Read cached position from Redis
            position = await self._redis.get_position(
                broker_name=broker_name,
                symbol=signal.symbol,
            )
            if not position:
                return True, ""

            current_qty = int(position.get("quantity", 0))

            if (
                signal.direction == SignalDirection.BUY
                and current_qty > 0
            ):
                return (
                    False,
                    f"already_long:{signal.symbol}:"
                    f"qty={current_qty}",
                )
            if (
                signal.direction == SignalDirection.SELL
                and current_qty < 0
            ):
                return (
                    False,
                    f"already_short:{signal.symbol}:"
                    f"qty={current_qty}",
                )
        except Exception as exc:
            logger.warning(
                f"Direction conflict check failed: {exc}"
            )
        return True, ""

    # ─────────────────────────────────────────
    # Kill Switch Control
    # ─────────────────────────────────────────

    async def arm_kill_switch(
        self,
        broker_name: str,
        reason: str = "manual",
    ) -> None:
        """
        Arms the kill switch for a broker.
        Blocks all new signals immediately.

        Args:
            broker_name: Target broker.
            reason: Human-readable reason string.
        """
        await self._redis.set_kill_switch(
            broker_name, active=True
        )
        trade_logger.warning(
            f"KILL SWITCH ARMED | "
            f"broker={broker_name} | reason={reason}"
        )

    async def disarm_kill_switch(
        self,
        broker_name: str,
    ) -> None:
        """
        Disarms the kill switch for a broker.
        Resumes signal processing.

        Args:
            broker_name: Target broker.
        """
        await self._redis.set_kill_switch(
            broker_name, active=False
        )
        logger.info(
            f"Kill switch disarmed | broker={broker_name}"
        )

    async def arm_global_kill_switch(
        self, reason: str = "manual"
    ) -> None:
        """Arms the global kill switch across all brokers."""
        await self._redis.set_global_kill_switch(active=True)
        trade_logger.warning(
            f"GLOBAL KILL SWITCH ARMED | reason={reason}"
        )

    async def disarm_global_kill_switch(self) -> None:
        """Disarms the global kill switch."""
        await self._redis.set_global_kill_switch(active=False)
        logger.info("Global kill switch disarmed.")

    # ─────────────────────────────────────────
    # Monitoring
    # ─────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        """
        Returns risk manager runtime statistics.

        Returns:
            Dict with approval/block counts and reasons.
        """
        total = self._signals_approved + self._signals_blocked
        return {
            "signals_approved": self._signals_approved,
            "signals_blocked":  self._signals_blocked,
            "total_signals":    total,
            "approval_rate":    (
                self._signals_approved / max(total, 1) * 100
            ),
            "block_reasons":    dict(
                sorted(
                    self._block_reasons.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
            ),
        }

    def reset_stats(self) -> None:
        """Resets runtime counters. Called at start of each day."""
        self._signals_approved = 0
        self._signals_blocked  = 0
        self._block_reasons    = {}

    # ─────────────────────────────────────────
    # Internal Helpers
    # ─────────────────────────────────────────

    def _load_risk_config(self) -> dict[str, Any]:
        """Loads risk parameters from risk_config.yaml."""
        try:
            from config.settings import get_settings
            return get_settings().risk_params
        except Exception:
            return {}