"""
Unit tests for backtesting/performance.py

Tests cover: PerformanceAnalyser — all metric groups.
"""

import pytest
import numpy as np

from backtesting.performance import PerformanceAnalyser


@pytest.fixture
def analyser(sample_fills, sample_equity_curve):
    """Returns a PerformanceAnalyser with sample data."""
    return PerformanceAnalyser(
        fills=sample_fills,
        equity_curve=sample_equity_curve,
        initial_capital=1_000_000.0,
    )


@pytest.fixture
def empty_analyser():
    """Returns a PerformanceAnalyser with no data."""
    return PerformanceAnalyser(
        fills=[],
        equity_curve=[],
        initial_capital=1_000_000.0,
    )


class TestPerformanceAnalyser:

    def test_compute_all_returns_dict(self, analyser):
        result = analyser.compute_all()
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_empty_returns_zeros(self, empty_analyser):
        result = empty_analyser.compute_all()
        assert result.get("total_trades", 0) == 0
        assert result.get("total_pnl", 0.0) == 0.0
        assert result.get("sharpe_ratio", 0.0) == 0.0

    def test_total_pnl_type(self, analyser):
        result = analyser.compute_all()
        assert isinstance(result.get("total_pnl"), float)

    def test_win_rate_bounded(self, analyser):
        result = analyser.compute_all()
        wr = result.get("win_rate_pct", 0.0)
        assert 0.0 <= wr <= 100.0

    def test_max_drawdown_non_negative(self, analyser):
        result = analyser.compute_all()
        dd = result.get("max_drawdown_pct", 0.0)
        assert dd >= 0.0

    def test_profit_factor_positive(self, analyser):
        result = analyser.compute_all()
        pf = result.get("profit_factor", 0.0)
        assert pf >= 0.0

    def test_sharpe_ratio_is_float(self, analyser):
        result = analyser.compute_all()
        assert isinstance(result.get("sharpe_ratio"), float)

    def test_standalone_methods(self, analyser):
        assert isinstance(analyser.sharpe_ratio(), float)
        assert isinstance(analyser.sortino_ratio(), float)
        assert isinstance(analyser.max_drawdown(), float)
        assert isinstance(analyser.win_rate(), float)
        assert isinstance(analyser.profit_factor(), float)

    def test_to_report_df(self, analyser):
        df = analyser.to_report_df()
        assert len(df) == 1
        assert "sharpe_ratio" in df.columns
        assert "total_trades" in df.columns

    def test_streak_metrics(self, analyser):
        result = analyser.compute_all()
        assert "max_win_streak"  in result
        assert "max_loss_streak" in result
        assert isinstance(result["max_win_streak"], int)

    def test_drawdown_fields(self, analyser):
        result = analyser.compute_all()
        assert "max_drawdown_pct"   in result
        assert "max_drawdown_inr"   in result
        assert "max_drawdown_start" in result
        assert "recovery_bars"      in result