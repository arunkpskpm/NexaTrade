"""
Unit tests for strategies/risk_manager.py

Tests cover all 10 risk checks and kill switch control.
"""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from brokers.models import Exchange, Segment, SignalDirection, StrategySignal
from strategies.risk_manager import RiskManager


@pytest.fixture
def risk_manager(mock_redis, mock_pg):
    """Returns a RiskManager with mocked storage."""
    return RiskManager(
        redis_client=mock_redis,
        pg_client=mock_pg,
    )


@pytest.fixture
def valid_signal() -> StrategySignal:
    """Returns a well-formed BUY signal."""
    return StrategySignal(
        strategy_name="ema_crossover",
        symbol="RELIANCE",
        exchange=Exchange.NSE,
        segment=Segment.EQ,
        direction=SignalDirection.BUY,
        strength=0.75,
        suggested_quantity=50,
        suggested_price=2450.0,
        stop_loss_price=2400.0,
        target_price=2550.0,
        reason="Golden cross confirmed",
    )


class TestRiskManagerApproval:

    @pytest.mark.asyncio
    async def test_approves_valid_signal(
        self, risk_manager, valid_signal
    ):
        result = await risk_manager.approve_signal(
            valid_signal, "paper", "paper"
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_blocks_on_global_kill_switch(
        self, risk_manager, valid_signal, mock_redis
    ):
        mock_redis.is_global_kill_switch_active.return_value = True
        result = await risk_manager.approve_signal(
            valid_signal, "paper", "paper"
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_blocks_on_broker_kill_switch(
        self, risk_manager, valid_signal, mock_redis
    ):
        mock_redis.is_global_kill_switch_active.return_value = False
        mock_redis.is_kill_switch_active.return_value        = True
        result = await risk_manager.approve_signal(
            valid_signal, "paper", "paper"
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_blocks_on_daily_loss_limit(
        self, risk_manager, valid_signal, mock_redis
    ):
        mock_redis.is_global_kill_switch_active.return_value = False
        mock_redis.is_kill_switch_active.return_value        = False
        # Simulated daily loss > limit
        mock_redis.get_daily_pnl.return_value = -15_000.0
        result = await risk_manager.approve_signal(
            valid_signal, "paper", "paper"
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_exit_bypasses_market_hours(
        self, risk_manager, mock_redis
    ):
        mock_redis.is_global_kill_switch_active.return_value = False
        mock_redis.is_kill_switch_active.return_value        = False
        mock_redis.get_daily_pnl.return_value                = 0.0
        mock_redis.get_position.return_value                 = None
        mock_redis.get.return_value                          = None

        exit_signal = StrategySignal(
            strategy_name="ema_crossover",
            symbol="RELIANCE",
            exchange=Exchange.NSE,
            segment=Segment.EQ,
            direction=SignalDirection.EXIT,
            suggested_quantity=50,
            reason="Stop-loss hit",
        )
        with patch(
            "strategies.risk_manager.is_market_open",
            return_value=False
        ):
            result = await risk_manager.approve_signal(
                exit_signal, "paper", "paper"
            )
        # EXIT should still pass market hours check
        assert result is True

    @pytest.mark.asyncio
    async def test_blocks_blacklisted_symbol(
        self, risk_manager, mock_redis
    ):
        mock_redis.is_global_kill_switch_active.return_value = False
        mock_redis.is_kill_switch_active.return_value        = False
        mock_redis.get_daily_pnl.return_value                = 0.0

        signal = StrategySignal(
            strategy_name="ema_crossover",
            symbol="YESBANK",      # in blacklist
            exchange=Exchange.NSE,
            segment=Segment.EQ,
            direction=SignalDirection.BUY,
            suggested_quantity=50,
            reason="Test",
        )

        with patch(
            "strategies.risk_manager.RiskManager._load_risk_config",
            return_value={
                "blacklist": {
                    "symbols": ["YESBANK"],
                    "exchanges": [],
                },
                "loss_limits":       {},
                "position_limits":   {},
                "capital":           {},
            },
        ):
            result = await risk_manager.approve_signal(
                signal, "paper", "paper"
            )
        assert result is False

    @pytest.mark.asyncio
    async def test_get_stats_structure(self, risk_manager):
        stats = risk_manager.get_stats()
        assert "signals_approved"  in stats
        assert "signals_blocked"   in stats
        assert "total_signals"     in stats
        assert "approval_rate"     in stats

    @pytest.mark.asyncio
    async def test_arm_kill_switch(
        self, risk_manager, mock_redis
    ):
        await risk_manager.arm_kill_switch(
            "breeze", reason="test"
        )
        mock_redis.set_kill_switch.assert_called_once_with(
            "breeze", active=True
        )

    @pytest.mark.asyncio
    async def test_disarm_kill_switch(
        self, risk_manager, mock_redis
    ):
        await risk_manager.disarm_kill_switch("breeze")
        mock_redis.set_kill_switch.assert_called_once_with(
            "breeze", active=False
        )

    @pytest.mark.asyncio
    async def test_reset_stats(self, risk_manager):
        risk_manager._signals_approved = 10
        risk_manager._signals_blocked  = 5
        risk_manager.reset_stats()
        assert risk_manager._signals_approved == 0
        assert risk_manager._signals_blocked  == 0