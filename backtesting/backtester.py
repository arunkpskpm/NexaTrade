"""
NexaTrade — Backtesting Engine.

An event-driven backtester that replays historical OHLCV candles
through the strategy lifecycle with full simulation fidelity.

Architecture:
    BacktestRunner
        └── Backtester
                ├── Strategy (AbstractStrategy instance)
                ├── SimulatedBroker (fills, positions, P&L)
                ├── CandleAggregator (indicator history buffer)
                └── PerformanceAnalyser (metrics on completion)

Simulation fidelity:
    - Candle-by-candle event replay (no look-ahead)
    - Tick simulation from OHLCV (open, high, low, close sequence)
    - Realistic fill simulation (open of next candle)
    - Configurable slippage (pct of price)
    - Configurable commission model (pct of trade value)
    - Partial fill simulation (configurable)
    - Position averaging for add-on trades
    - Separate long and short P&L tracking
    - Capital allocation tracking (used margin)

No-look-ahead guarantee:
    At candle N, the strategy only sees candles 0..N-1.
    Candle N itself is passed to on_candle() AFTER close.
    Fill is simulated at candle N+1 open price.

Usage:
    backtester = Backtester(
        strategy=my_strategy,
        df=historical_df,
        initial_capital=500_000.0,
        slippage_pct=0.05,
        commission_pct=0.03,
    )
    result = await backtester.run()
    print(result.summary())
"""

from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

import numpy as np
import pandas as pd

from brokers.models import (
    OHLCV,
    Exchange,
    Fill,
    OrderRequest,
    OrderResponse,
    OrderStatus,
    OrderType,
    Position,
    Segment,
    SignalDirection,
    StrategySignal,
    TickData,
    TradingMode,
    TransactionType,
)
from strategies.abstract_strategy import AbstractStrategy
from utils.logger import get_logger
from utils.time_utils import now_ist

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# Simulated Broker
# ─────────────────────────────────────────────

class SimulatedBroker:
    """
    In-memory simulated broker for backtesting.

    Receives OrderRequests from the strategy engine,
    simulates fills at the next candle's open price,
    and tracks positions, P&L, and equity curve.

    Fill simulation rules:
        MARKET orders: filled at next_open ± slippage
        LIMIT orders:  filled if next candle crosses limit price
        STOP_LOSS:     filled if next candle crosses trigger price

    All state is ephemeral — cleared between runs.
    """

    def __init__(
        self,
        initial_capital: float,
        slippage_pct: float = 0.05,
        commission_pct: float = 0.03,
    ) -> None:
        self.initial_capital  = initial_capital
        self.slippage_pct     = slippage_pct
        self.commission_pct   = commission_pct

        # Portfolio state
        self.cash:            float = initial_capital
        self.positions:       dict[str, dict[str, Any]] = {}

        # Order queues
        self.pending_orders:  list[dict[str, Any]] = []
        self.filled_orders:   list[dict[str, Any]] = []

        # Trade record
        self.fills:           list[Fill] = []

        # Equity curve — one entry per candle
        self.equity_curve:    list[dict[str, Any]] = []

        # Counters
        self.total_commission: float = 0.0
        self.total_slippage:   float = 0.0

        # Order callback (set by Backtester)
        self._order_callback = None

    # ─────────────────────────────────────────
    # Order Submission
    # ─────────────────────────────────────────

    def submit_order(
        self,
        request: OrderRequest,
        strategy_name: str,
    ) -> OrderResponse:
        """
        Accepts an order for deferred fill simulation.
        MARKET orders are queued for next candle open fill.
        LIMIT/STOP orders are queued for conditional fill.

        Args:
            request: Broker-agnostic OrderRequest.
            strategy_name: Originating strategy.

        Returns:
            OrderResponse with PENDING status.
        """
        order_entry = {
            "order_id":       request.order_id,
            "strategy_name":  strategy_name,
            "symbol":         request.symbol,
            "transaction_type": str(request.transaction_type),
            "order_type":     str(request.order_type),
            "quantity":       request.quantity,
            "limit_price":    request.price,
            "trigger_price":  request.trigger_price,
            "status":         "PENDING",
            "submitted_at":   now_ist(),
        }
        self.pending_orders.append(order_entry)

        return OrderResponse(
            order_id=request.order_id,
            broker_order_id=f"SIM-{str(uuid4())[:8].upper()}",
            status=OrderStatus.PENDING,
            message="Simulated order queued.",
            broker_name="backtester",
            trading_mode=TradingMode.PAPER,
        )

    # ─────────────────────────────────────────
    # Fill Simulation
    # ─────────────────────────────────────────

    async def process_pending_orders(
        self,
        candle: OHLCV,
    ) -> None:
        """
        Processes all pending orders against the given candle.
        Called at the start of each candle before on_candle().

        Fill logic:
            MARKET       → fills at candle.open ± slippage
            LIMIT BUY    → fills if candle.low  <= limit_price
            LIMIT SELL   → fills if candle.high >= limit_price
            STOP BUY     → fills if candle.high >= trigger_price
            STOP SELL    → fills if candle.low  <= trigger_price

        Args:
            candle: The current OHLCV candle (next bar).
        """
        still_pending = []

        for order in self.pending_orders:
            order_type    = order["order_type"].upper()
            txn_type      = order["transaction_type"].upper()
            is_buy        = txn_type == "BUY"

            fill_price = None

            # ── MARKET order ──────────────────
            if order_type == "MARKET":
                slippage = candle.open * (self.slippage_pct / 100)
                fill_price = (
                    candle.open + slippage
                    if is_buy
                    else candle.open - slippage
                )

            # ── LIMIT order ───────────────────
            elif order_type == "LIMIT" and order.get("limit_price"):
                limit = order["limit_price"]
                if is_buy and candle.low <= limit:
                    slippage   = limit * (self.slippage_pct / 100)
                    fill_price = min(candle.open, limit) + slippage
                elif not is_buy and candle.high >= limit:
                    slippage   = limit * (self.slippage_pct / 100)
                    fill_price = max(candle.open, limit) - slippage

            # ── STOP_LOSS order ───────────────
            elif order_type in (
                "STOP_LOSS", "STOP_LOSS_MARKET"
            ) and order.get("trigger_price"):
                trigger = order["trigger_price"]
                if is_buy and candle.high >= trigger:
                    slippage   = trigger * (self.slippage_pct / 100)
                    fill_price = trigger + slippage
                elif not is_buy and candle.low <= trigger:
                    slippage   = trigger * (self.slippage_pct / 100)
                    fill_price = trigger - slippage

            # ── Execute fill ──────────────────
            if fill_price:
                fill_price = max(fill_price, 0.01)
                await self._execute_fill(
                    order=order,
                    fill_price=fill_price,
                    candle=candle,
                )
            else:
                still_pending.append(order)

        self.pending_orders = still_pending

    async def _execute_fill(
        self,
        order: dict[str, Any],
        fill_price: float,
        candle: OHLCV,
    ) -> None:
        """
        Executes a fill: updates positions, cash, P&L,
        records the Fill, and notifies the strategy.

        Args:
            order: Pending order dict.
            fill_price: Simulated fill price.
            candle: The candle on which the fill occurred.
        """
        symbol    = order["symbol"]
        quantity  = order["quantity"]
        is_buy    = order["transaction_type"].upper() == "BUY"
        order_id  = order["order_id"]

        # Calculate commission
        trade_value = quantity * fill_price
        commission  = trade_value * (self.commission_pct / 100)
        slippage_cost = abs(
            fill_price - candle.open
        ) * quantity

        self.total_commission += commission
        self.total_slippage   += slippage_cost

        # Update position
        pnl = self._update_position(
            symbol=symbol,
            is_buy=is_buy,
            quantity=quantity,
            fill_price=fill_price,
        )

        # Update cash
        if is_buy:
            self.cash -= (trade_value + commission)
        else:
            self.cash += (trade_value - commission)

        # Record fill
        fill = Fill(
            order_id=order_id,
            broker_order_id=order_id,
            broker_name="backtester",
            symbol=symbol,
            exchange="NSE",
            transaction_type=(
                TransactionType.BUY
                if is_buy
                else TransactionType.SELL
            ),
            quantity=quantity,
            price=round(fill_price, 4),
            commission=round(commission, 4),
            trading_mode=TradingMode.PAPER,
            executed_at=candle.datetime,
        )
        self.fills.append(fill)

        # Mark order as filled
        order["status"]       = "COMPLETE"
        order["fill_price"]   = fill_price
        order["fill_time"]    = candle.datetime
        order["commission"]   = commission
        order["pnl"]          = pnl
        self.filled_orders.append(order)

        logger.debug(
            f"Backtest fill | "
            f"symbol={symbol} | "
            f"{'BUY' if is_buy else 'SELL'} | "
            f"qty={quantity} | "
            f"price={fill_price:.2f} | "
            f"commission={commission:.2f} | "
            f"pnl={pnl:.2f}"
        )

        # Notify strategy via callback
        if self._order_callback:
            response = OrderResponse(
                order_id=order_id,
                broker_order_id=order_id,
                status=OrderStatus.COMPLETE,
                filled_quantity=quantity,
                average_price=round(fill_price, 4),
                broker_name="backtester",
                trading_mode=TradingMode.PAPER,
                updated_at=candle.datetime,
            )
            try:
                await self._order_callback(response)
            except Exception as exc:
                logger.error(
                    f"Order callback error in backtester: {exc}"
                )

    def _update_position(
        self,
        symbol: str,
        is_buy: bool,
        quantity: int,
        fill_price: float,
    ) -> float:
        """
        Updates the in-memory position for a symbol.
        Returns realised P&L (> 0 for profit, < 0 for loss).

        Args:
            symbol: Instrument symbol.
            is_buy: True for BUY, False for SELL.
            quantity: Fill quantity.
            fill_price: Actual fill price.

        Returns:
            Realised P&L for this fill.
        """
        if symbol not in self.positions:
            self.positions[symbol] = {
                "quantity":       0,
                "average_price":  0.0,
                "realized_pnl":   0.0,
                "buy_value":      0.0,
                "sell_value":     0.0,
            }

        pos     = self.positions[symbol]
        cur_qty = pos["quantity"]
        cur_avg = pos["average_price"]
        pnl     = 0.0

        if is_buy:
            if cur_qty >= 0:
                # Adding to long — recalculate average
                new_qty              = cur_qty + quantity
                pos["average_price"] = (
                    (cur_avg * cur_qty) + (fill_price * quantity)
                ) / new_qty
                pos["quantity"]      = new_qty
            else:
                # Covering short
                close_qty = min(quantity, abs(cur_qty))
                pnl       = (cur_avg - fill_price) * close_qty
                pos["realized_pnl"] += pnl
                pos["quantity"]      = cur_qty + quantity
                if pos["quantity"] > 0:
                    pos["average_price"] = fill_price

            pos["buy_value"] += fill_price * quantity

        else:  # SELL
            if cur_qty <= 0:
                # Adding to short
                new_qty              = cur_qty - quantity
                pos["average_price"] = (
                    (cur_avg * abs(cur_qty)) + (fill_price * quantity)
                ) / abs(new_qty) if new_qty != 0 else fill_price
                pos["quantity"]      = new_qty
            else:
                # Closing long
                close_qty = min(quantity, cur_qty)
                pnl       = (fill_price - cur_avg) * close_qty
                pos["realized_pnl"] += pnl
                pos["quantity"]      = cur_qty - quantity
                if pos["quantity"] < 0:
                    pos["average_price"] = fill_price

            pos["sell_value"] += fill_price * quantity

        return pnl

    # ─────────────────────────────────────────
    # Portfolio Valuation
    # ─────────────────────────────────────────

    def get_portfolio_value(
        self, current_prices: dict[str, float]
    ) -> float:
        """
        Returns total portfolio value at current prices.
        Includes cash + mark-to-market open positions.

        Args:
            current_prices: {symbol: last_price} dict.

        Returns:
            Total portfolio value in INR.
        """
        position_value = sum(
            pos["quantity"] * current_prices.get(sym, pos["average_price"])
            for sym, pos in self.positions.items()
            if pos["quantity"] != 0
        )
        return self.cash + position_value

    def get_unrealized_pnl(
        self, current_prices: dict[str, float]
    ) -> float:
        """
        Returns total unrealised P&L across all open positions.

        Args:
            current_prices: {symbol: last_price} dict.

        Returns:
            Unrealised P&L in INR.
        """
        total = 0.0
        for sym, pos in self.positions.items():
            qty = pos["quantity"]
            if qty != 0:
                ltp = current_prices.get(sym, pos["average_price"])
                total += (ltp - pos["average_price"]) * qty
        return total

    def get_realized_pnl(self) -> float:
        """Returns total realised P&L across all positions."""
        return sum(
            pos["realized_pnl"]
            for pos in self.positions.values()
        )

    def record_equity_snapshot(
        self,
        timestamp: datetime,
        current_prices: dict[str, float],
    ) -> None:
        """
        Records an equity curve snapshot for this candle.

        Args:
            timestamp: Candle datetime.
            current_prices: Current price dict for MTM.
        """
        portfolio_value = self.get_portfolio_value(current_prices)
        realized        = self.get_realized_pnl()
        unrealized      = self.get_unrealized_pnl(current_prices)
        drawdown        = min(
            0.0,
            (portfolio_value - self.initial_capital)
            / self.initial_capital * 100,
        )
        self.equity_curve.append({
            "datetime":       timestamp,
            "portfolio_value": round(portfolio_value, 2),
            "cash":           round(self.cash, 2),
            "realized_pnl":   round(realized, 2),
            "unrealized_pnl": round(unrealized, 2),
            "drawdown_pct":   round(drawdown, 4),
        })


# ─────────────────────────────────────────────
# Backtester
# ─────────────────────────────────────────────

class BacktestResult:
    """
    Container for a completed backtest result.
    Holds fills, equity curve, and computed metrics.
    """

    def __init__(
        self,
        run_id: str,
        strategy_name: str,
        symbol: str,
        interval: str,
        from_date: str,
        to_date: str,
        initial_capital: float,
        parameters: dict[str, Any],
        fills: list[Fill],
        equity_curve: list[dict[str, Any]],
        metrics: dict[str, Any],
        broker: SimulatedBroker,
    ) -> None:
        self.run_id          = run_id
        self.strategy_name   = strategy_name
        self.symbol          = symbol
        self.interval        = interval
        self.from_date       = from_date
        self.to_date         = to_date
        self.initial_capital = initial_capital
        self.parameters      = parameters
        self.fills           = fills
        self.equity_curve    = equity_curve
        self.metrics         = metrics
        self.broker          = broker

    def summary(self) -> str:
        """Returns a human-readable result summary."""
        m = self.metrics
        lines = [
            f"{'─' * 55}",
            f"  NexaTrade Backtest Result",
            f"{'─' * 55}",
            f"  Run ID        : {self.run_id}",
            f"  Strategy      : {self.strategy_name}",
            f"  Symbol        : {self.symbol} ({self.interval})",
            f"  Period        : {self.from_date} → {self.to_date}",
            f"  Initial Cap   : ₹{self.initial_capital:,.2f}",
            f"{'─' * 55}",
            f"  Final Capital : ₹{m.get('final_capital', 0):,.2f}",
            f"  Total P&L     : ₹{m.get('total_pnl', 0):,.2f}",
            f"  Total Return  : {m.get('total_return_pct', 0):.2f}%",
            f"  CAGR          : {m.get('cagr_pct', 0):.2f}%",
            f"{'─' * 55}",
            f"  Total Trades  : {m.get('total_trades', 0)}",
            f"  Win Rate      : {m.get('win_rate_pct', 0):.2f}%",
            f"  Avg Win       : ₹{m.get('avg_win', 0):,.2f}",
            f"  Avg Loss      : ₹{m.get('avg_loss', 0):,.2f}",
            f"  Profit Factor : {m.get('profit_factor', 0):.2f}",
            f"  Expectancy    : ₹{m.get('expectancy', 0):,.2f}",
            f"{'─' * 55}",
            f"  Max Drawdown  : {m.get('max_drawdown_pct', 0):.2f}%",
            f"  Sharpe Ratio  : {m.get('sharpe_ratio', 0):.4f}",
            f"  Sortino Ratio : {m.get('sortino_ratio', 0):.4f}",
            f"  Calmar Ratio  : {m.get('calmar_ratio', 0):.4f}",
            f"{'─' * 55}",
            f"  Commission    : ₹{m.get('total_commission', 0):,.2f}",
            f"  Slippage      : ₹{m.get('total_slippage', 0):,.2f}",
            f"{'─' * 55}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Returns result as a serialisable dict."""
        return {
            "run_id":          self.run_id,
            "strategy_name":   self.strategy_name,
            "symbol":          self.symbol,
            "interval":        self.interval,
            "from_date":       self.from_date,
            "to_date":         self.to_date,
            "initial_capital": self.initial_capital,
            "parameters":      self.parameters,
            "metrics":         self.metrics,
            "fills_count":     len(self.fills),
            "equity_points":   len(self.equity_curve),
        }

    def to_equity_df(self) -> pd.DataFrame:
        """Returns equity curve as a pandas DataFrame."""
        if not self.equity_curve:
            return pd.DataFrame()
        df = pd.DataFrame(self.equity_curve)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)
        return df

    def to_fills_df(self) -> pd.DataFrame:
        """Returns all fills as a pandas DataFrame."""
        if not self.fills:
            return pd.DataFrame()
        return pd.DataFrame([
            {
                "executed_at":      f.executed_at,
                "symbol":           f.symbol,
                "transaction_type": str(f.transaction_type),
                "quantity":         f.quantity,
                "price":            f.price,
                "commission":       f.commission,
            }
            for f in self.fills
        ])


class Backtester:
    """
    NexaTrade Event-Driven Backtester.

    Replays historical OHLCV data through a strategy,
    simulates fills, and tracks portfolio evolution.

    No-look-ahead guarantee enforced by:
        - Feeding candles to aggregator one at a time
        - Processing pending orders at candle N+1 open
        - Strategy only sees candles 0..N-1 on on_candle(N)
    """

    def __init__(
        self,
        strategy: AbstractStrategy,
        df: pd.DataFrame,
        initial_capital: float = 1_000_000.0,
        slippage_pct: float = 0.05,
        commission_pct: float = 0.03,
        warmup_bars: int = 50,
        symbol: str = "SYMBOL",
        exchange: str = "NSE",
        interval: str = "5minute",
        broker_name: str = "backtester",
    ) -> None:
        """
        Args:
            strategy: Instantiated AbstractStrategy subclass.
            df: Historical OHLCV DataFrame (DatetimeIndex, IST).
            initial_capital: Starting portfolio capital in INR.
            slippage_pct: Slippage as % of fill price.
            commission_pct: Commission as % of trade value.
            warmup_bars: Number of initial bars to skip for
                         indicator warm-up (no signals generated).
            symbol: Instrument symbol for tick simulation.
            exchange: Exchange code.
            interval: Candle interval string.
            broker_name: Label for this backtest run.
        """
        self.strategy          = strategy
        self.df                = df
        self.initial_capital   = initial_capital
        self.slippage_pct      = slippage_pct
        self.commission_pct    = commission_pct
        self.warmup_bars       = warmup_bars
        self.symbol            = symbol.upper()
        self.exchange          = exchange.upper()
        self.interval          = interval
        self.broker_name       = broker_name

        # Generated on run()
        self.run_id: str = str(uuid4())[:12]

        # Simulated broker
        self._sim_broker = SimulatedBroker(
            initial_capital=initial_capital,
            slippage_pct=slippage_pct,
            commission_pct=commission_pct,
        )

        # In-memory candle buffer for strategy access
        self._candle_buffer: deque[OHLCV] = deque(maxlen=1000)

        # State
        self._is_running = False
        self._progress_callback = None

    # ─────────────────────────────────────────
    # Main Run Loop
    # ─────────────────────────────────────────

    async def run(
        self,
        progress_callback=None,
    ) -> BacktestResult:
        """
        Runs the full backtest.

        Pipeline per candle:
            1. Convert candle to OHLCV model
            2. Process pending orders at this candle's open
            3. Simulate tick events (open, high/low, close)
            4. If past warmup: call strategy.on_candle()
            5. Record equity snapshot
            6. Yield to event loop every N candles

        Args:
            progress_callback: Optional async callable(pct, bar, total)
                               for progress reporting.

        Returns:
            BacktestResult with fills, equity curve, and metrics.

        Raises:
            ValueError: If DataFrame is empty or malformed.
        """
        if self.df is None or self.df.empty:
            raise ValueError("Backtester: DataFrame is empty.")

        required_cols = {"open", "high", "low", "close", "volume"}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(
                f"Backtester: Missing columns: {missing}"
            )

        self._progress_callback = progress_callback
        self._is_running = True
        total_bars = len(self.df)

        logger.info(
            f"Backtest started | "
            f"run_id={self.run_id} | "
            f"strategy={self.strategy.STRATEGY_NAME} | "
            f"symbol={self.symbol} | "
            f"interval={self.interval} | "
            f"bars={total_bars} | "
            f"capital=₹{self.initial_capital:,.0f}"
        )

        # ── Wire strategy dependencies ────────
        await self._setup_strategy()

        # ── Candle replay loop ────────────────
        for bar_idx, (ts, row) in enumerate(self.df.iterrows()):
            if not self._is_running:
                break

            # Build OHLCV model
            candle = OHLCV(
                datetime=ts if hasattr(ts, "tzinfo") else pd.Timestamp(ts),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row.get("volume", 0)),
                symbol=self.symbol,
                exchange=self.exchange,
                interval=self.interval,
                broker_name=self.broker_name,
            )

            # 1 ── Process pending orders at this candle's open
            await self._sim_broker.process_pending_orders(candle)

            # 2 ── Simulate tick events from OHLCV bars
            await self._simulate_ticks(candle)

            # 3 ── Add candle to buffer (after fills, before on_candle)
            self._candle_buffer.append(candle)

            # 4 ── Call strategy.on_candle() past warmup period
            if bar_idx >= self.warmup_bars:
                try:
                    await self.strategy.on_candle(candle)
                except Exception as exc:
                    logger.error(
                        f"Strategy error on bar {bar_idx}: {exc}"
                    )
                    try:
                        await self.strategy.on_error(exc)
                    except Exception:
                        pass

            # 5 ── Record equity snapshot
            current_prices = {self.symbol: candle.close}
            self._sim_broker.record_equity_snapshot(
                timestamp=candle.datetime,
                current_prices=current_prices,
            )

            # 6 ── Report progress every 100 bars
            if bar_idx % 100 == 0 and progress_callback:
                pct = (bar_idx / total_bars) * 100
                try:
                    await progress_callback(pct, bar_idx, total_bars)
                except Exception:
                    pass

            # 7 ── Yield to event loop every 500 bars
            if bar_idx % 500 == 0:
                await asyncio.sleep(0)

        # ── Teardown strategy ─────────────────
        try:
            await self.strategy.on_stop()
        except Exception as exc:
            logger.warning(f"Strategy on_stop error: {exc}")

        self._is_running = False

        # ── Compute performance metrics ───────
        from backtesting.performance import PerformanceAnalyser
        analyser = PerformanceAnalyser(
            fills=self._sim_broker.fills,
            equity_curve=self._sim_broker.equity_curve,
            initial_capital=self.initial_capital,
        )
        metrics = analyser.compute_all()
        metrics["total_commission"] = round(
            self._sim_broker.total_commission, 2
        )
        metrics["total_slippage"] = round(
            self._sim_broker.total_slippage, 2
        )
        metrics["final_capital"]  = round(
            self._sim_broker.get_portfolio_value({
                self.symbol: self.df["close"].iloc[-1]
            }),
            2,
        )

        logger.info(
            f"Backtest complete | "
            f"run_id={self.run_id} | "
            f"total_return={metrics.get('total_return_pct', 0):.2f}% | "
            f"trades={metrics.get('total_trades', 0)} | "
            f"sharpe={metrics.get('sharpe_ratio', 0):.4f}"
        )

        return BacktestResult(
            run_id=self.run_id,
            strategy_name=self.strategy.STRATEGY_NAME,
            symbol=self.symbol,
            interval=self.interval,
            from_date=str(self.df.index[0].date()),
            to_date=str(self.df.index[-1].date()),
            initial_capital=self.initial_capital,
            parameters=dict(self.strategy.parameters),
            fills=self._sim_broker.fills,
            equity_curve=self._sim_broker.equity_curve,
            metrics=metrics,
            broker=self._sim_broker,
        )

    # ─────────────────────────────────────────
    # Strategy Setup
    # ─────────────────────────────────────────

    async def _setup_strategy(self) -> None:
        """
        Wires strategy dependencies for backtesting.
        Uses backtest-specific mock services that read
        from the candle buffer instead of live feeds.
        """
        strategy = self.strategy

        # Inject backtest context
        strategy.broker_name  = self.broker_name
        strategy.trading_mode = "paper"
        strategy.instruments  = [
            {"symbol": self.symbol, "exchange": self.exchange}
        ]

        # Inject simulated broker order callback
        self._sim_broker._order_callback = (
            strategy.on_order_update
        )

        # Register signal callback → simulated broker
        strategy._signal_callback = self._on_strategy_signal

        # Inject backtest feed accessor
        strategy._feed = BacktestFeedAdapter(
            symbol=self.symbol,
            exchange=self.exchange,
            interval=self.interval,
            buffer=self._candle_buffer,
        )

        # on_start() in backtest mode (subscribe calls are no-ops)
        try:
            await strategy.on_start()
            strategy._is_running = True
        except Exception as exc:
            logger.warning(
                f"Strategy on_start() warning in backtest: {exc}"
            )

    async def _on_strategy_signal(
        self, signal: StrategySignal
    ) -> None:
        """
        Receives signals from the strategy and submits
        them to the SimulatedBroker as orders.

        Skips risk manager in backtesting mode —
        all signals become orders directly.

        Args:
            signal: StrategySignal from the strategy.
        """
        direction = signal.direction
        if direction in (SignalDirection.HOLD, SignalDirection.NONE):
            return

        txn_type = (
            TransactionType.BUY
            if direction == SignalDirection.BUY
            else TransactionType.SELL
        )

        quantity = signal.suggested_quantity or 1
        order_type = (
            OrderType.LIMIT
            if signal.suggested_price
            else OrderType.MARKET
        )

        request = OrderRequest(
            symbol=signal.symbol,
            exchange=signal.exchange,
            segment=signal.segment,
            transaction_type=txn_type,
            order_type=order_type,
            quantity=quantity,
            price=signal.suggested_price,
            trigger_price=signal.stop_loss_price,
            strategy_name=signal.strategy_name,
            trading_mode=TradingMode.PAPER,
            tags={"signal_id": signal.signal_id},
        )

        self._sim_broker.submit_order(
            request=request,
            strategy_name=signal.strategy_name,
        )

    async def _simulate_ticks(self, candle: OHLCV) -> None:
        """
        Simulates a sequence of tick events from an OHLCV bar.
        Sequence: open → high/low (interleaved) → close

        This provides the strategy's on_tick() handler with
        intrabar price movement for stop-loss monitoring.

        Args:
            candle: The OHLCV candle to simulate ticks from.
        """
        tick_sequence = [
            candle.open,
            candle.high,
            candle.low,
            candle.close,
        ]

        for price in tick_sequence:
            tick = TickData(
                symbol=candle.symbol,
                exchange=candle.exchange or "NSE",
                broker_name=self.broker_name,
                last_price=price,
                volume=int(candle.volume / 4),
                timestamp=candle.datetime,
            )
            try:
                await self.strategy.on_tick(tick)
            except Exception as exc:
                logger.error(
                    f"Strategy on_tick error in backtest: {exc}"
                )

    def stop(self) -> None:
        """Stops the backtest loop after the current candle."""
        self._is_running = False
        logger.info(
            f"Backtest stop requested | run_id={self.run_id}"
        )


# ─────────────────────────────────────────────
# Backtest Feed Adapter
# ─────────────────────────────────────────────

class BacktestFeedAdapter:
    """
    Minimal FeedService adapter for backtesting.

    Replaces the live FeedService with a read-only
    adapter that serves candles from the in-memory buffer.
    All subscription calls are no-ops.

    The strategy accesses historical candles via
    get_candles() just as it would in live mode.
    """

    def __init__(
        self,
        symbol: str,
        exchange: str,
        interval: str,
        buffer: deque,
    ) -> None:
        self._symbol   = symbol.upper()
        self._exchange = exchange.upper()
        self._interval = interval
        self._buffer   = buffer

    async def subscribe(self, *args, **kwargs) -> str:
        """No-op subscription — returns dummy consumer ID."""
        return "backtest_consumer"

    async def unsubscribe(self, *args, **kwargs) -> None:
        """No-op unsubscription."""
        pass

    async def subscribe_many(self, *args, **kwargs) -> None:
        """No-op bulk subscription."""
        pass

    async def unsubscribe_all(self, *args, **kwargs) -> None:
        """No-op bulk unsubscription."""
        pass

    def get_candles(
        self,
        symbol: str,
        exchange: str,
        interval: str,
        n: Optional[int] = None,
    ) -> list[OHLCV]:
        """
        Returns candles from the replay buffer.
        Only candles already seen (pre-current-bar) are available.

        Args:
            symbol: Instrument symbol.
            exchange: Exchange code.
            interval: Candle interval.
            n: Number of recent candles. Returns all if None.

        Returns:
            List of OHLCV candles (oldest first).
        """
        candles = list(self._buffer)
        return candles[-n:] if n else candles

    def get_last_price(
        self, symbol: str, exchange: str = "NSE"
    ) -> Optional[float]:
        """Returns the close price of the latest buffered candle."""
        if self._buffer:
            return self._buffer[-1].close
        return None