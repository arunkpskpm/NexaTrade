"""
NexaTrade — EMA Crossover Strategy Plugin.

Strategy logic:
    - Buy when fast EMA crosses above slow EMA (golden cross)
    - Sell when fast EMA crosses below slow EMA (death cross)
    - ATR-based dynamic stop-loss
    - Volume confirmation filter (volume > avg_volume * multiplier)
    - Only trades during market hours (enforced by RiskManager)

Parameters (configurable via UI):
    fast_period     → Fast EMA period (default: 9)
    slow_period     → Slow EMA period (default: 21)
    atr_period      → ATR period for stop-loss (default: 14)
    atr_multiplier  → Stop-loss = close - (ATR × multiplier)
    volume_filter   → Min volume ratio to confirm signal
    quantity        → Fixed order quantity per signal

This file is a complete, production-ready strategy example.
Copy and modify it to create your own strategies.
"""

from __future__ import annotations

import pandas as pd
from typing import Any

from brokers.models import (
    OHLCV,
    Exchange,
    OrderResponse,
    OrderStatus,
    Segment,
    SignalDirection,
    StrategySignal,
    TickData,
)
from strategies.abstract_strategy import AbstractStrategy
from utils.indicators import (
    atr,
    crossover,
    crossunder,
    ema,
    volume_ratio,
)


class EMACrossoverStrategy(AbstractStrategy):
    """
    EMA Crossover Strategy with ATR stop-loss and volume filter.

    Signals:
        BUY  → fast EMA crosses above slow EMA + volume confirmed
        SELL → fast EMA crosses below slow EMA + volume confirmed
        EXIT → stop-loss breached on tick level

    Position management:
        Tracks entry price and stop-loss in instance variables.
        Stop-loss updated dynamically on each candle close.
    """

    # ── Strategy Metadata ────────────────────
    STRATEGY_NAME = "ema_crossover"
    DISPLAY_NAME  = "EMA Crossover"
    DESCRIPTION   = (
        "Dual EMA crossover with ATR stop-loss "
        "and volume confirmation filter."
    )
    VERSION       = "1.2.0"
    AUTHOR        = "NexaTrade"

    # ── Default Parameters ───────────────────
    DEFAULT_PARAMETERS: dict[str, Any] = {
        "fast_period":    9,
        "slow_period":    21,
        "atr_period":     14,
        "atr_multiplier": 2.0,
        "volume_filter":  1.5,   # Min volume/avg ratio
        "quantity":       50,    # Units per order
    }

    # ── Default Instruments ──────────────────
    DEFAULT_INSTRUMENTS = [
        {"symbol": "RELIANCE", "exchange": "NSE"},
    ]

    DEFAULT_INTERVAL = "5minute"

    def __init__(self) -> None:
        super().__init__()

        # Position tracking state
        self._position_side:   str            = "FLAT"  # FLAT/LONG/SHORT
        self._entry_price:     float          = 0.0
        self._stop_loss_price: float          = 0.0
        self._position_qty:    int            = 0

        # Loaded parameters (set in on_start)
        self._fast_period:    int   = 9
        self._slow_period:    int   = 21
        self._atr_period:     int   = 14
        self._atr_mult:       float = 2.0
        self._volume_filter:  float = 1.5
        self._quantity:       int   = 50

    # ═════════════════════════════════════════
    # Lifecycle Methods
    # ═════════════════════════════════════════

    async def on_start(self) -> None:
        """
        Strategy startup:
        - Load parameters
        - Subscribe to feed for all instruments
        - Log startup summary
        """
        # Load parameters
        self._fast_period   = int(self.get_param("fast_period",    9))
        self._slow_period   = int(self.get_param("slow_period",    21))
        self._atr_period    = int(self.get_param("atr_period",     14))
        self._atr_mult      = float(self.get_param("atr_multiplier", 2.0))
        self._volume_filter = float(self.get_param("volume_filter",  1.5))
        self._quantity      = int(self.get_param("quantity",        50))

        # Subscribe to all configured instruments
        for inst in self.instruments:
            await self._feed.subscribe(
                symbol=inst["symbol"],
                exchange=inst.get("exchange", "NSE"),
                interval=self.DEFAULT_INTERVAL,
                consumer_id=self.name,
                tick_callback=self.on_tick,
                candle_callback=self.on_candle,
                seed_history=True,
            )

        self._logger.info(
            f"{self.DISPLAY_NAME} started | "
            f"fast={self._fast_period} | slow={self._slow_period} | "
            f"atr_mult={self._atr_mult} | qty={self._quantity} | "
            f"instruments={[i['symbol'] for i in self.instruments]} | "
            f"mode={self.trading_mode}"
        )

    async def on_tick(self, tick: TickData) -> None:
        """
        Tick-level stop-loss monitoring.
        Emits EXIT signal if price breaches the stop-loss.
        """
        self._tick_count += 1

        # Only monitor stop-loss if we have an open position
        if self._position_side == "FLAT":
            return

        # Long position: exit if price drops below stop-loss
        if (
            self._position_side == "LONG"
            and self._stop_loss_price > 0
            and tick.last_price <= self._stop_loss_price
        ):
            await self.emit_signal(StrategySignal(
                strategy_name=self.name,
                symbol=tick.symbol,
                exchange=Exchange.NSE,
                segment=Segment.EQ,
                direction=SignalDirection.EXIT,
                suggested_quantity=self._position_qty,
                reason=(
                    f"Stop-loss hit | "
                    f"price={tick.last_price:.2f} | "
                    f"stop={self._stop_loss_price:.2f}"
                ),
            ))
            self._reset_position()

        # Short position: exit if price rises above stop-loss
        elif (
            self._position_side == "SHORT"
            and self._stop_loss_price > 0
            and tick.last_price >= self._stop_loss_price
        ):
            await self.emit_signal(StrategySignal(
                strategy_name=self.name,
                symbol=tick.symbol,
                exchange=Exchange.NSE,
                segment=Segment.EQ,
                direction=SignalDirection.EXIT,
                suggested_quantity=self._position_qty,
                reason=(
                    f"Stop-loss hit (short) | "
                    f"price={tick.last_price:.2f} | "
                    f"stop={self._stop_loss_price:.2f}"
                ),
            ))
            self._reset_position()

    async def on_candle(self, candle: OHLCV) -> None:
        """
        Candle-close EMA crossover signal generation.

        Pipeline:
        1. Fetch recent candles from aggregator
        2. Compute fast EMA, slow EMA, ATR, volume ratio
        3. Check for crossover / crossunder
        4. Apply volume confirmation filter
        5. Emit BUY or SELL signal
        6. Update dynamic stop-loss on each candle
        """
        symbol   = candle.symbol
        exchange = candle.exchange or "NSE"

        # Need at least slow_period + buffer bars
        min_bars = self._slow_period + self._atr_period + 5
        candles  = self._feed.get_candles(
            symbol, exchange, self.DEFAULT_INTERVAL, n=min_bars
        )
        if len(candles) < min_bars:
            return

        # Build DataFrame
        df = pd.DataFrame([c.to_dict() for c in candles])
        df.set_index("datetime", inplace=True)

        # Compute indicators
        fast_ema    = ema(df["close"], self._fast_period)
        slow_ema    = ema(df["close"], self._slow_period)
        atr_vals    = atr(df, self._atr_period)
        vol_ratio   = volume_ratio(df, period=20)
        golden_x    = crossover(fast_ema, slow_ema)
        death_x     = crossunder(fast_ema, slow_ema)

        # Latest bar values
        current_close  = df["close"].iloc[-1]
        current_atr    = atr_vals.iloc[-1]
        current_vol_r  = vol_ratio.iloc[-1]
        is_golden      = golden_x.iloc[-1]
        is_death       = death_x.iloc[-1]

        # ── Dynamic stop-loss update ──────────
        # Trailing stop updated on every candle regardless of signal
        if self._position_side == "LONG":
            new_stop = current_close - (
                self._atr_mult * current_atr
            )
            self._stop_loss_price = max(
                self._stop_loss_price, new_stop
            )

        elif self._position_side == "SHORT":
            new_stop = current_close + (
                self._atr_mult * current_atr
            )
            self._stop_loss_price = min(
                self._stop_loss_price, new_stop
            )

        # ── Volume filter ─────────────────────
        volume_confirmed = (
            current_vol_r is not None
            and current_vol_r >= self._volume_filter
        )

        # ── BUY signal: Golden cross + volume ─
        if (
            is_golden
            and volume_confirmed
            and self._position_side != "LONG"
        ):
            stop_price = current_close - (
                self._atr_mult * current_atr
            )
            target_price = current_close + (
                2 * self._atr_mult * current_atr  # 2:1 R:R
            )
            await self.emit_signal(StrategySignal(
                strategy_name=self.name,
                symbol=symbol,
                exchange=Exchange.NSE,
                segment=Segment.EQ,
                direction=SignalDirection.BUY,
                strength=min(current_vol_r / 3.0, 1.0),
                suggested_quantity=self._quantity,
                stop_loss_price=round(stop_price, 2),
                target_price=round(target_price, 2),
                reason=(
                    f"Golden cross | "
                    f"fast={fast_ema.iloc[-1]:.2f} > "
                    f"slow={slow_ema.iloc[-1]:.2f} | "
                    f"vol_ratio={current_vol_r:.2f} | "
                    f"atr={current_atr:.2f}"
                ),
                tags={
                    "fast_ema":   round(fast_ema.iloc[-1], 4),
                    "slow_ema":   round(slow_ema.iloc[-1], 4),
                    "atr":        round(current_atr, 4),
                    "vol_ratio":  round(current_vol_r, 4),
                },
            ))
            # Optimistic position tracking
            self._position_side   = "LONG"
            self._stop_loss_price = round(stop_price, 2)
            self._position_qty    = self._quantity

        # ── SELL signal: Death cross + volume ─
        elif (
            is_death
            and volume_confirmed
            and self._position_side != "SHORT"
        ):
            stop_price = current_close + (
                self._atr_mult * current_atr
            )
            target_price = current_close - (
                2 * self._atr_mult * current_atr
            )
            await self.emit_signal(StrategySignal(
                strategy_name=self.name,
                symbol=symbol,
                exchange=Exchange.NSE,
                segment=Segment.EQ,
                direction=SignalDirection.SELL,
                strength=min(current_vol_r / 3.0, 1.0),
                suggested_quantity=self._quantity,
                stop_loss_price=round(stop_price, 2),
                target_price=round(target_price, 2),
                reason=(
                    f"Death cross | "
                    f"fast={fast_ema.iloc[-1]:.2f} < "
                    f"slow={slow_ema.iloc[-1]:.2f} | "
                    f"vol_ratio={current_vol_r:.2f} | "
                    f"atr={current_atr:.2f}"
                ),
                tags={
                    "fast_ema":   round(fast_ema.iloc[-1], 4),
                    "slow_ema":   round(slow_ema.iloc[-1], 4),
                    "atr":        round(current_atr, 4),
                    "vol_ratio":  round(current_vol_r, 4),
                },
            ))
            # Optimistic position tracking
            self._position_side   = "SHORT"
            self._stop_loss_price = round(stop_price, 2)
            self._position_qty    = self._quantity

    async def on_order_update(
        self, response: OrderResponse
    ) -> None:
        """
        Updates confirmed entry price from fill response.
        Recalculates stop-loss on confirmed fill.
        """
        if response.status == OrderStatus.COMPLETE:
            fill_price = response.average_price or self._entry_price
            self._entry_price = fill_price

            # Recalculate stop-loss from confirmed fill price
            if self._position_side == "LONG":
                self._stop_loss_price = fill_price * 0.98  # 2% hard stop
            elif self._position_side == "SHORT":
                self._stop_loss_price = fill_price * 1.02

            self._logger.info(
                f"Order filled | "
                f"status={response.status} | "
                f"avg_price={fill_price:.2f} | "
                f"stop={self._stop_loss_price:.2f} | "
                f"side={self._position_side}"
            )

        elif response.status == OrderStatus.REJECTED:
            self._logger.warning(
                f"Order rejected | "
                f"reason={response.rejection_reason}"
            )
            # Reset optimistic position tracking
            self._reset_position()

    async def on_stop(self) -> None:
        """
        Unsubscribes from all feeds and logs final stats.
        """
        await self._feed.unsubscribe_all(self.name)
        self._logger.info(
            f"{self.DISPLAY_NAME} stopped | "
            f"ticks={self._tick_count} | "
            f"signals={self._signal_count} | "
            f"orders={self._order_count} | "
            f"final_side={self._position_side}"
        )

    async def on_error(self, exc: Exception) -> None:
        """
        Logs error and unsubscribes feed to prevent further errors.
        """
        self._logger.error(
            f"{self.DISPLAY_NAME} error: {exc}"
        )
        try:
            await self._feed.unsubscribe_all(self.name)
        except Exception:
            pass

    # ─────────────────────────────────────────
    # Internal Helpers
    # ─────────────────────────────────────────

    def _reset_position(self) -> None:
        """Resets position tracking state to FLAT."""
        self._position_side   = "FLAT"
        self._entry_price     = 0.0
        self._stop_loss_price = 0.0
        self._position_qty    = 0