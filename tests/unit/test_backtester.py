"""
Unit tests for backtesting/backtester.py

Tests cover: SimulatedBroker fills, Backtester replay loop,
and BacktestResult methods.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from backtesting.backtester import SimulatedBroker, Backtester
from brokers.models import (
    OHLCV, Exchange, OrderRequest, OrderType,
    ProductType, Segment, TradingMode, TransactionType,
)
import pytz

IST = pytz.timezone("Asia/Kolkata")


@pytest.fixture
def sim_broker():
    """Returns a clean SimulatedBroker."""
    return SimulatedBroker(
        initial_capital=500_000.0,
        slippage_pct=0.05,
        commission_pct=0.03,
    )


@pytest.fixture
def sample_candle():
    """Returns a single OHLCV candle."""
    from datetime import datetime
    return OHLCV(
        datetime=datetime(2024, 1, 2, 9, 15, tzinfo=IST),
        open=2400.0,
        high=2420.0,
        low=2395.0,
        close=2415.0,
        volume=125_000.0,
        symbol="RELIANCE",
        exchange="NSE",
        interval="5minute",
        broker_name="backtester",
    )


@pytest.fixture
def buy_order_request():
    """Returns a MARKET BUY OrderRequest."""
    return OrderRequest(
        symbol="RELIANCE",
        exchange=Exchange.NSE,
        segment=Segment.EQ,
        transaction_type=TransactionType.BUY,
        order_type=OrderType.MARKET,
        product_type=ProductType.INTRADAY,
        quantity=50,
        trading_mode=TradingMode.PAPER,
    )


class TestSimulatedBroker:

    def test_initial_cash(self, sim_broker):
        assert sim_broker.cash == 500_000.0

    def test_submit_order_queues_pending(
        self, sim_broker, buy_order_request
    ):
        response = sim_broker.submit_order(
            buy_order_request, "test_strategy"
        )
        assert len(sim_broker.pending_orders) == 1
        assert response.status.name == "PENDING"

    @pytest.mark.asyncio
    async def test_market_buy_fills(
        self, sim_broker, buy_order_request, sample_candle
    ):
        sim_broker.submit_order(
            buy_order_request, "test_strategy"
        )
        await sim_broker.process_pending_orders(sample_candle)

        assert len(sim_broker.filled_orders) == 1
        assert len(sim_broker.pending_orders) == 0
        fill = sim_broker.filled_orders[0]
        assert fill["status"] == "COMPLETE"
        # Fill price ≈ candle.open + slippage
        assert fill["fill_price"] > sample_candle.open

    @pytest.mark.asyncio
    async def test_cash_reduced_after_buy(
        self, sim_broker, buy_order_request, sample_candle
    ):
        initial_cash = sim_broker.cash
        sim_broker.submit_order(
            buy_order_request, "test_strategy"
        )
        await sim_broker.process_pending_orders(sample_candle)
        assert sim_broker.cash < initial_cash

    @pytest.mark.asyncio
    async def test_position_updated_after_buy(
        self, sim_broker, buy_order_request, sample_candle
    ):
        sim_broker.submit_order(
            buy_order_request, "test_strategy"
        )
        await sim_broker.process_pending_orders(sample_candle)
        pos = sim_broker.positions.get("RELIANCE")
        assert pos is not None
        assert pos["quantity"] == 50

    @pytest.mark.asyncio
    async def test_sell_realises_pnl(
        self, sim_broker, sample_candle
    ):
        # Buy 50 shares
        buy_req = OrderRequest(
            symbol="RELIANCE",
            exchange=Exchange.NSE,
            segment=Segment.EQ,
            transaction_type=TransactionType.BUY,
            order_type=OrderType.MARKET,
            product_type=ProductType.INTRADAY,
            quantity=50,
            trading_mode=TradingMode.PAPER,
        )
        sim_broker.submit_order(buy_req, "strat")
        await sim_broker.process_pending_orders(sample_candle)

        # Sell 50 shares at higher candle
        from datetime import datetime
        higher_candle = OHLCV(
            datetime=datetime(2024, 1, 2, 9, 20, tzinfo=IST),
            open=2450.0, high=2460.0,
            low=2445.0, close=2455.0,
            volume=100_000.0,
            symbol="RELIANCE", exchange="NSE",
            interval="5minute", broker_name="backtester",
        )
        sell_req = OrderRequest(
            symbol="RELIANCE",
            exchange=Exchange.NSE,
            segment=Segment.EQ,
            transaction_type=TransactionType.SELL,
            order_type=OrderType.MARKET,
            product_type=ProductType.INTRADAY,
            quantity=50,
            trading_mode=TradingMode.PAPER,
        )
        sim_broker.submit_order(sell_req, "strat")
        await sim_broker.process_pending_orders(higher_candle)

        realized = sim_broker.get_realized_pnl()
        assert realized > 0

    def test_equity_snapshot(self, sim_broker):
        from datetime import datetime
        sim_broker.record_equity_snapshot(
            timestamp=datetime(2024, 1, 2, 9, 15, tzinfo=IST),
            current_prices={"RELIANCE": 2450.0},
        )
        assert len(sim_broker.equity_curve) == 1
        snap = sim_broker.equity_curve[0]
        assert "portfolio_value" in snap
        assert "cash" in snap
        assert "drawdown_pct" in snap


class TestBacktester:

    @pytest.mark.asyncio
    async def test_run_returns_result(self, sample_ohlcv_df):
        from plugins.ema_crossover import EMACrossoverStrategy
        strategy = EMACrossoverStrategy()
        strategy.instruments = [
            {"symbol": "RELIANCE", "exchange": "NSE"}
        ]

        backtester = Backtester(
            strategy=strategy,
            df=sample_ohlcv_df,
            initial_capital=500_000.0,
            warmup_bars=50,
            symbol="RELIANCE",
            exchange="NSE",
            interval="5minute",
        )
        result = await backtester.run()

        assert result is not None
        assert result.run_id
        assert result.strategy_name == "ema_crossover"
        assert result.initial_capital == 500_000.0
        assert isinstance(result.metrics, dict)

    @pytest.mark.asyncio
    async def test_metrics_populated(self, sample_ohlcv_df):
        from plugins.ema_crossover import EMACrossoverStrategy
        strategy = EMACrossoverStrategy()
        strategy.instruments = [
            {"symbol": "RELIANCE", "exchange": "NSE"}
        ]

        backtester = Backtester(
            strategy=strategy,
            df=sample_ohlcv_df,
            initial_capital=500_000.0,
            warmup_bars=30,
            symbol="RELIANCE",
            exchange="NSE",
            interval="5minute",
        )
        result = await backtester.run()

        required_metrics = [
            "total_return_pct",
            "sharpe_ratio",
            "max_drawdown_pct",
            "total_trades",
            "win_rate_pct",
        ]
        for m in required_metrics:
            assert m in result.metrics

    @pytest.mark.asyncio
    async def test_equity_curve_populated(self, sample_ohlcv_df):
        from plugins.ema_crossover import EMACrossoverStrategy
        strategy = EMACrossoverStrategy()
        strategy.instruments = [
            {"symbol": "RELIANCE", "exchange": "NSE"}
        ]

        backtester = Backtester(
            strategy=strategy,
            df=sample_ohlcv_df,
            initial_capital=500_000.0,
            warmup_bars=30,
            symbol="RELIANCE",
            exchange="NSE",
            interval="5minute",
        )
        result = await backtester.run()
        assert len(result.equity_curve) == len(sample_ohlcv_df)

    def test_raises_on_empty_df(self):
        import pandas as pd
        from plugins.ema_crossover import EMACrossoverStrategy
        strategy = EMACrossoverStrategy()
        bt = Backtester(
            strategy=strategy,
            df=pd.DataFrame(),
            initial_capital=500_000.0,
        )
        with pytest.raises(ValueError, match="empty"):
            asyncio.get_event_loop().run_until_complete(
                bt.run()
            )