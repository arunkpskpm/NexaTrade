"""
NexaTrade — Performance Analyser.

Computes a comprehensive suite of trading performance metrics
from a list of fills and an equity curve.

All metrics are computed from first principles using numpy
and pandas — no external finance library dependencies.

Metrics computed:
    ── Returns ──────────────────────────────
    total_pnl           Total realised P&L (INR)
    total_return_pct    Total return as percentage
    cagr_pct            Compound Annual Growth Rate

    ── Trade Statistics ─────────────────────
    total_trades        Total number of completed round trips
    winning_trades      Trades with positive P&L
    losing_trades       Trades with negative P&L
    win_rate_pct        Win rate as percentage
    avg_win             Average winning trade P&L
    avg_loss            Average losing trade P&L
    largest_win         Largest single winning trade
    largest_loss        Largest single losing trade
    profit_factor       Gross profit / Gross loss
    expectancy          Expected P&L per trade
    avg_trade_duration  Average bars held per round trip

    ── Risk Metrics ─────────────────────────
    max_drawdown_pct    Maximum peak-to-trough drawdown (%)
    max_drawdown_inr    Maximum drawdown in INR
    max_drawdown_start  Drawdown start datetime
    max_drawdown_end    Drawdown end datetime
    recovery_bars       Bars to recover from max drawdown

    ── Risk-Adjusted Returns ─────────────────
    sharpe_ratio        Sharpe Ratio (annualised)
    sortino_ratio       Sortino Ratio (downside deviation)
    calmar_ratio        Calmar Ratio (CAGR / Max Drawdown)
    omega_ratio         Omega Ratio (threshold = 0)

    ── Streaks ──────────────────────────────
    max_win_streak      Longest consecutive winning streak
    max_loss_streak     Longest consecutive losing streak
    current_streak      Current win/loss streak

Usage:
    analyser = PerformanceAnalyser(
        fills=backtest_result.fills,
        equity_curve=backtest_result.equity_curve,
        initial_capital=500_000.0,
    )
    metrics = analyser.compute_all()
    sharpe  = metrics["sharpe_ratio"]
    drawdown = metrics["max_drawdown_pct"]
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd

from brokers.models import Fill
from utils.logger import get_logger

logger = get_logger(__name__)

# Annualisation constants
TRADING_DAYS_PER_YEAR  = 252
TRADING_HOURS_PER_DAY  = 6.25          # NSE: 09:15–15:30
MINUTES_PER_YEAR       = (
    TRADING_DAYS_PER_YEAR * TRADING_HOURS_PER_DAY * 60
)


class PerformanceAnalyser:
    """
    NexaTrade Trading Performance Analyser.

    Computes all standard quantitative trading metrics
    from fill records and an equity curve time series.

    Args:
        fills: List of Fill objects from the simulated broker.
        equity_curve: List of equity snapshot dicts from SimulatedBroker.
        initial_capital: Starting portfolio capital in INR.
        risk_free_rate: Annual risk-free rate (default 6.5% for India).
        interval: Candle interval — used to determine annualisation factor.
    """

    def __init__(
        self,
        fills: list[Fill],
        equity_curve: list[dict[str, Any]],
        initial_capital: float,
        risk_free_rate: float = 0.065,
        interval: str = "5minute",
    ) -> None:
        self.fills           = fills
        self.equity_curve    = equity_curve
        self.initial_capital = initial_capital
        self.risk_free_rate  = risk_free_rate
        self.interval        = interval

        # Pre-built DataFrames (lazy)
        self._trades_df:  Optional[pd.DataFrame] = None
        self._equity_df:  Optional[pd.DataFrame] = None
        self._returns_sr: Optional[pd.Series]    = None

    # ─────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────

    def compute_all(self) -> dict[str, Any]:
        """
        Computes and returns all performance metrics.

        Returns:
            Dict mapping metric name → value.
            All monetary values are in INR.
            All percentages are as float (e.g. 12.5 for 12.5%).
        """
        metrics: dict[str, Any] = {}

        try:
            metrics.update(self._compute_return_metrics())
        except Exception as exc:
            logger.warning(f"Return metrics error: {exc}")

        try:
            metrics.update(self._compute_trade_statistics())
        except Exception as exc:
            logger.warning(f"Trade statistics error: {exc}")

        try:
            metrics.update(self._compute_drawdown_metrics())
        except Exception as exc:
            logger.warning(f"Drawdown metrics error: {exc}")

        try:
            metrics.update(self._compute_risk_adjusted_returns())
        except Exception as exc:
            logger.warning(f"Risk-adjusted metrics error: {exc}")

        try:
            metrics.update(self._compute_streak_metrics())
        except Exception as exc:
            logger.warning(f"Streak metrics error: {exc}")

        return metrics

    # ─────────────────────────────────────────
    # Return Metrics
    # ─────────────────────────────────────────

    def _compute_return_metrics(self) -> dict[str, Any]:
        """Computes total P&L, return %, and CAGR."""
        eq_df = self._get_equity_df()
        if eq_df.empty:
            return {
                "total_pnl":        0.0,
                "total_return_pct": 0.0,
                "cagr_pct":         0.0,
            }

        final_value  = eq_df["portfolio_value"].iloc[-1]
        total_pnl    = final_value - self.initial_capital
        total_return = (total_pnl / self.initial_capital) * 100

        # CAGR
        start_dt = eq_df.index[0]
        end_dt   = eq_df.index[-1]
        years    = max(
            (end_dt - start_dt).days / 365.25, 1 / 365.25
        )
        cagr     = (
            ((final_value / self.initial_capital) ** (1 / years)) - 1
        ) * 100

        return {
            "total_pnl":        round(total_pnl, 2),
            "total_return_pct": round(total_return, 4),
            "cagr_pct":         round(cagr, 4),
        }

    # ─────────────────────────────────────────
    # Trade Statistics
    # ─────────────────────────────────────────

    def _compute_trade_statistics(self) -> dict[str, Any]:
        """
        Computes round-trip trade statistics.

        Pairs BUY fills with subsequent SELL fills per symbol
        to compute per-trade P&L.
        """
        trades_df = self._get_trades_df()
        if trades_df.empty:
            return {
                "total_trades":      0,
                "winning_trades":    0,
                "losing_trades":     0,
                "break_even_trades": 0,
                "win_rate_pct":      0.0,
                "avg_win":           0.0,
                "avg_loss":          0.0,
                "largest_win":       0.0,
                "largest_loss":      0.0,
                "profit_factor":     0.0,
                "expectancy":        0.0,
                "avg_trade_pnl":     0.0,
            }

        pnl_series = trades_df["pnl"]
        total      = len(pnl_series)
        winners    = pnl_series[pnl_series > 0]
        losers     = pnl_series[pnl_series < 0]
        break_even = pnl_series[pnl_series == 0]

        win_count  = len(winners)
        loss_count = len(losers)
        win_rate   = (win_count / total * 100) if total else 0.0

        avg_win    = float(winners.mean()) if win_count  else 0.0
        avg_loss   = float(losers.mean())  if loss_count else 0.0

        gross_profit = float(winners.sum()) if win_count  else 0.0
        gross_loss   = abs(float(losers.sum()))  if loss_count else 0.0
        profit_factor = (
            gross_profit / gross_loss if gross_loss > 0 else float("inf")
        )

        # Expectancy = (win_rate × avg_win) - (loss_rate × |avg_loss|)
        loss_rate  = loss_count / total if total else 0.0
        expectancy = (
            (win_rate / 100 * avg_win)
            - (loss_rate * abs(avg_loss))
        )

        return {
            "total_trades":      total,
            "winning_trades":    win_count,
            "losing_trades":     loss_count,
            "break_even_trades": len(break_even),
            "win_rate_pct":      round(win_rate, 4),
            "avg_win":           round(avg_win, 2),
            "avg_loss":          round(avg_loss, 2),
            "largest_win":       round(float(pnl_series.max()), 2),
            "largest_loss":      round(float(pnl_series.min()), 2),
            "profit_factor":     round(profit_factor, 4),
            "expectancy":        round(expectancy, 2),
            "avg_trade_pnl":     round(float(pnl_series.mean()), 2),
        }

    # ─────────────────────────────────────────
    # Drawdown Metrics
    # ─────────────────────────────────────────

    def _compute_drawdown_metrics(self) -> dict[str, Any]:
        """
        Computes maximum drawdown and recovery statistics.

        Peak-to-trough methodology:
            Running peak = max(portfolio_value seen so far)
            Drawdown     = (peak - current) / peak × 100
            Max drawdown = worst drawdown ever reached
        """
        eq_df = self._get_equity_df()
        if eq_df.empty:
            return {
                "max_drawdown_pct":   0.0,
                "max_drawdown_inr":   0.0,
                "max_drawdown_start": None,
                "max_drawdown_end":   None,
                "recovery_bars":      0,
                "avg_drawdown_pct":   0.0,
            }

        values = eq_df["portfolio_value"].values
        timestamps = eq_df.index

        # Running peak
        peak_values = np.maximum.accumulate(values)
        drawdowns   = (peak_values - values) / np.where(
            peak_values > 0, peak_values, 1
        ) * 100

        max_dd_pct     = float(drawdowns.max())
        max_dd_idx     = int(np.argmax(drawdowns))
        max_dd_inr     = float(peak_values[max_dd_idx] - values[max_dd_idx])

        # Find drawdown start (most recent peak before max drawdown)
        peak_series    = np.maximum.accumulate(values[:max_dd_idx + 1])
        dd_start_idx   = int(np.argmax(peak_series == peak_series[-1]))
        dd_start       = timestamps[dd_start_idx]
        dd_end         = timestamps[max_dd_idx]

        # Count recovery bars
        recovery_bars  = 0
        if max_dd_idx < len(values) - 1:
            peak_at_dd = peak_values[max_dd_idx]
            for i in range(max_dd_idx + 1, len(values)):
                if values[i] >= peak_at_dd:
                    recovery_bars = i - max_dd_idx
                    break
            else:
                recovery_bars = -1  # Not yet recovered

        avg_dd = float(drawdowns[drawdowns > 0].mean()) if any(
            drawdowns > 0
        ) else 0.0

        return {
            "max_drawdown_pct":   round(max_dd_pct, 4),
            "max_drawdown_inr":   round(max_dd_inr, 2),
            "max_drawdown_start": str(dd_start),
            "max_drawdown_end":   str(dd_end),
            "recovery_bars":      recovery_bars,
            "avg_drawdown_pct":   round(avg_dd, 4),
        }

    # ─────────────────────────────────────────
    # Risk-Adjusted Returns
    # ─────────────────────────────────────────

    def _compute_risk_adjusted_returns(self) -> dict[str, Any]:
        """
        Computes Sharpe, Sortino, Calmar, and Omega ratios.

        Annualisation uses the candle interval to determine
        the number of periods per year for scaling.
        """
        returns = self._get_period_returns()
        if returns is None or len(returns) < 10:
            return {
                "sharpe_ratio":  0.0,
                "sortino_ratio": 0.0,
                "calmar_ratio":  0.0,
                "omega_ratio":   0.0,
            }

        # Periods per year based on interval
        periods_per_year = self._get_periods_per_year()

        # Per-period risk-free rate
        rf_per_period = self.risk_free_rate / periods_per_year

        # ── Sharpe Ratio ──────────────────────
        excess_returns = returns - rf_per_period
        sharpe = (
            excess_returns.mean() / excess_returns.std()
            * np.sqrt(periods_per_year)
            if excess_returns.std() > 0
            else 0.0
        )

        # ── Sortino Ratio ─────────────────────
        downside_returns = returns[returns < rf_per_period]
        downside_std     = (
            downside_returns.std()
            if len(downside_returns) > 1
            else 0.0
        )
        sortino = (
            excess_returns.mean() / downside_std
            * np.sqrt(periods_per_year)
            if downside_std > 0
            else 0.0
        )

        # ── Calmar Ratio ──────────────────────
        cagr_metrics  = self._compute_return_metrics()
        cagr          = cagr_metrics.get("cagr_pct", 0.0)
        dd_metrics    = self._compute_drawdown_metrics()
        max_dd        = dd_metrics.get("max_drawdown_pct", 0.0)
        calmar        = cagr / max_dd if max_dd > 0 else 0.0

        # ── Omega Ratio ───────────────────────
        # Threshold = risk-free per period
        gains  = (returns[returns > rf_per_period] - rf_per_period).sum()
        losses = abs(
            (returns[returns < rf_per_period] - rf_per_period).sum()
        )
        omega  = gains / losses if losses > 0 else float("inf")

        return {
            "sharpe_ratio":  round(float(sharpe),  6),
            "sortino_ratio": round(float(sortino), 6),
            "calmar_ratio":  round(float(calmar),  6),
            "omega_ratio":   round(float(omega),   6),
        }

    # ─────────────────────────────────────────
    # Streak Metrics
    # ─────────────────────────────────────────

    def _compute_streak_metrics(self) -> dict[str, Any]:
        """Computes winning and losing streak statistics."""
        trades_df = self._get_trades_df()
        if trades_df.empty or len(trades_df) < 2:
            return {
                "max_win_streak":  0,
                "max_loss_streak": 0,
                "current_streak":  0,
            }

        pnl_series = trades_df["pnl"].values
        outcomes   = np.sign(pnl_series)  # 1, -1, or 0

        max_win_streak  = 0
        max_loss_streak = 0
        current_streak  = 0
        current_sign    = 0

        win_run  = 0
        loss_run = 0

        for outcome in outcomes:
            if outcome > 0:
                win_run  += 1
                loss_run  = 0
                max_win_streak = max(max_win_streak, win_run)
                current_sign   = 1
                current_streak = win_run
            elif outcome < 0:
                loss_run += 1
                win_run   = 0
                max_loss_streak = max(max_loss_streak, loss_run)
                current_sign    = -1
                current_streak  = loss_run
            else:
                win_run  = 0
                loss_run = 0

        return {
            "max_win_streak":  int(max_win_streak),
            "max_loss_streak": int(max_loss_streak),
            "current_streak":  (
                int(current_streak) * current_sign
            ),
        }

    # ─────────────────────────────────────────
    # Data Preparation Helpers
    # ─────────────────────────────────────────

    def _get_equity_df(self) -> pd.DataFrame:
        """Returns equity curve as a cached DataFrame."""
        if self._equity_df is not None:
            return self._equity_df

        if not self.equity_curve:
            self._equity_df = pd.DataFrame()
            return self._equity_df

        df = pd.DataFrame(self.equity_curve)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)
        df.sort_index(inplace=True)
        self._equity_df = df
        return self._equity_df

    def _get_trades_df(self) -> pd.DataFrame:
        """
        Builds a DataFrame of round-trip trades from fills.
        Pairs BUY fills with subsequent SELL fills per symbol.
        """
        if self._trades_df is not None:
            return self._trades_df

        if not self.fills:
            self._trades_df = pd.DataFrame()
            return self._trades_df

        # Group fills by symbol
        symbol_fills: dict[str, list[Fill]] = {}
        for fill in self.fills:
            symbol_fills.setdefault(fill.symbol, []).append(fill)

        trades = []
        for symbol, sym_fills in symbol_fills.items():
            # Separate buys and sells chronologically
            buys  = deque(
                f for f in sym_fills
                if str(f.transaction_type).upper() == "BUY"
            )
            sells = deque(
                f for f in sym_fills
                if str(f.transaction_type).upper() == "SELL"
            )

            # Match buys to sells (FIFO)
            while buys and sells:
                buy  = buys.popleft()
                sell = sells.popleft()
                pnl  = (sell.price - buy.price) * min(
                    buy.quantity, sell.quantity
                )
                pnl -= (buy.commission + sell.commission)
                trades.append({
                    "symbol":       symbol,
                    "entry_time":   buy.executed_at,
                    "exit_time":    sell.executed_at,
                    "entry_price":  buy.price,
                    "exit_price":   sell.price,
                    "quantity":     min(buy.quantity, sell.quantity),
                    "pnl":          round(pnl, 4),
                    "commission":   buy.commission + sell.commission,
                })

        if not trades:
            self._trades_df = pd.DataFrame()
        else:
            self._trades_df = pd.DataFrame(trades)
            self._trades_df.sort_values("entry_time", inplace=True)
            self._trades_df.reset_index(drop=True, inplace=True)

        return self._trades_df

    def _get_period_returns(self) -> Optional[pd.Series]:
        """Computes period-by-period returns from equity curve."""
        if self._returns_sr is not None:
            return self._returns_sr

        eq_df = self._get_equity_df()
        if eq_df.empty or len(eq_df) < 2:
            return None

        values = eq_df["portfolio_value"]
        returns = values.pct_change().dropna()
        self._returns_sr = returns
        return returns

    def _get_periods_per_year(self) -> float:
        """Returns the number of candle periods per trading year."""
        from services.feed_service import INTERVAL_MINUTES
        mins = INTERVAL_MINUTES.get(self.interval, 5)
        if mins == 0:
            mins = 1
        # NSE trading minutes per year
        trading_mins = (
            TRADING_DAYS_PER_YEAR
            * TRADING_HOURS_PER_DAY
            * 60
        )
        return trading_mins / mins

    # ─────────────────────────────────────────
    # Standalone Metric Methods (public)
    # ─────────────────────────────────────────

    def sharpe_ratio(self) -> float:
        """Returns the annualised Sharpe Ratio."""
        return self._compute_risk_adjusted_returns().get(
            "sharpe_ratio", 0.0
        )

    def sortino_ratio(self) -> float:
        """Returns the annualised Sortino Ratio."""
        return self._compute_risk_adjusted_returns().get(
            "sortino_ratio", 0.0
        )

    def max_drawdown(self) -> float:
        """Returns the maximum drawdown as a percentage."""
        return self._compute_drawdown_metrics().get(
            "max_drawdown_pct", 0.0
        )

    def win_rate(self) -> float:
        """Returns the win rate as a percentage."""
        return self._compute_trade_statistics().get(
            "win_rate_pct", 0.0
        )

    def profit_factor(self) -> float:
        """Returns the profit factor."""
        return self._compute_trade_statistics().get(
            "profit_factor", 0.0
        )

    def to_report_df(self) -> pd.DataFrame:
        """
        Returns all metrics as a single-row DataFrame.
        Useful for comparing multiple backtest runs.

        Returns:
            Single-row DataFrame of all metrics.
        """
        metrics = self.compute_all()
        return pd.DataFrame([metrics])