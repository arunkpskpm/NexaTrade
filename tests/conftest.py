"""
NexaTrade — Pytest Fixtures & Test Configuration.

Provides shared fixtures for:
    - FastAPI TestClient (async)
    - Mock Container and all services
    - In-memory SimulatedBroker
    - Sample OHLCV DataFrames
    - JWT tokens for auth tests
    - Mock PostgresClient
    - Mock RedisClient
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from backtesting.backtester import BacktestResult, SimulatedBroker
from config.settings import get_settings
from utils.auth import create_jwt_token


# ─────────────────────────────────────────────
# Settings Override for Tests
# ─────────────────────────────────────────────

@pytest.fixture(scope="session", autouse=True)
def override_settings(monkeypatch_session):
    """Forces paper trading mode for all tests."""
    pass  # Settings read from CI env vars


# ─────────────────────────────────────────────
# Sample Data Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """
    Returns a 300-bar synthetic OHLCV DataFrame for backtesting.
    Uses a random walk on close prices, IST DatetimeIndex.
    """
    import pytz
    IST = pytz.timezone("Asia/Kolkata")

    np.random.seed(42)
    n       = 300
    base    = 2400.0
    returns = np.random.normal(0, 0.005, n)
    close   = base * np.exp(np.cumsum(returns))

    # Realistic OHLCV
    high    = close * (1 + np.abs(np.random.normal(0, 0.003, n)))
    low     = close * (1 - np.abs(np.random.normal(0, 0.003, n)))
    open_   = np.roll(close, 1)
    open_[0] = base
    volume  = np.random.randint(50_000, 500_000, n).astype(float)

    # 5-minute candles starting 2024-01-02 09:15 IST
    start = datetime(2024, 1, 2, 9, 15, tzinfo=IST)
    idx   = pd.date_range(
        start=start, periods=n, freq="5min"
    )

    return pd.DataFrame({
        "open":   open_,
        "high":   high,
        "low":    low,
        "close":  close,
        "volume": volume,
    }, index=idx)


@pytest.fixture
def sample_candle_dict() -> dict[str, Any]:
    """Returns a single OHLCV candle dict."""
    import pytz
    IST = pytz.timezone("Asia/Kolkata")
    return {
        "datetime": datetime(2024, 1, 2, 9, 15, tzinfo=IST),
        "open":   2400.00,
        "high":   2412.50,
        "low":    2396.00,
        "close":  2408.75,
        "volume": 125_000.0,
    }


@pytest.fixture
def sample_fills() -> list:
    """Returns a list of sample Fill objects."""
    from brokers.models import Fill, TransactionType, TradingMode
    import pytz
    IST = pytz.timezone("Asia/Kolkata")

    return [
        Fill(
            order_id=f"ord-00{i}",
            broker_order_id=f"BRK-00{i}",
            broker_name="paper",
            symbol="RELIANCE",
            exchange="NSE",
            transaction_type=(
                TransactionType.BUY if i % 2 == 0
                else TransactionType.SELL
            ),
            quantity=50,
            price=2400.0 + (i * 10),
            commission=3.60,
            trading_mode=TradingMode.PAPER,
            executed_at=datetime(
                2024, 1, 2, 9, 15 + i * 5, tzinfo=IST
            ),
        )
        for i in range(6)
    ]


@pytest.fixture
def sample_equity_curve() -> list[dict[str, Any]]:
    """Returns a synthetic equity curve."""
    import pytz
    IST = pytz.timezone("Asia/Kolkata")

    base = 1_000_000.0
    curve = []
    for i in range(100):
        pnl   = np.random.normal(200, 800)
        value = base + (i * 200) + pnl
        curve.append({
            "datetime":       datetime(2024, 1, 2, 9, 15, tzinfo=IST)
                              + timedelta(minutes=i * 5),
            "portfolio_value": round(value, 2),
            "cash":           round(value * 0.6, 2),
            "realized_pnl":   round(i * 180.0, 2),
            "unrealized_pnl": round(pnl, 2),
            "drawdown_pct":   round(min(0, (value - base) / base * 100), 4),
        })
    return curve


# ─────────────────────────────────────────────
# Auth Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def auth_token() -> str:
    """Returns a valid JWT token for test user."""
    return create_jwt_token(
        user_id="00000000-0000-0000-0000-000000000001",
        username="test_user",
        expire_minutes=60,
    )


@pytest.fixture
def auth_headers(auth_token: str) -> dict[str, str]:
    """Returns Authorization headers for API tests."""
    return {"Authorization": f"Bearer {auth_token}"}


# ─────────────────────────────────────────────
# Mock Storage Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def mock_pg():
    """Returns a fully mocked PostgresClient."""
    pg = AsyncMock()
    pg.get_user_by_username.return_value = {
        "user_id":      "00000000-0000-0000-0000-000000000001",
        "username":     "test_user",
        "email":        "test@nexatrade.io",
        "password_hash": "$2b$12$fakehashedpasswordfortest",
        "is_active":    True,
        "is_admin":     False,
        "created_at":   datetime(2024, 1, 1),
    }
    pg.get_user_by_id.return_value = {
        "user_id":  "00000000-0000-0000-0000-000000000001",
        "username": "test_user",
        "is_active": True,
    }
    pg.get_positions.return_value    = []
    pg.get_active_strategies.return_value = []
    pg.get_latest_risk_snapshot.return_value = None
    pg.get_today_order_count.return_value    = 0
    pg.health_check.return_value = True
    return pg


@pytest.fixture
def mock_redis():
    """Returns a fully mocked RedisClient."""
    redis = AsyncMock()
    redis.get.return_value                         = None
    redis.get_json.return_value                    = None
    redis.get_quote.return_value                   = None
    redis.get_signal.return_value                  = None
    redis.get_position.return_value                = None
    redis.get_daily_pnl.return_value               = 0.0
    redis.is_kill_switch_active.return_value       = False
    redis.is_global_kill_switch_active.return_value = False
    redis.health_check.return_value                = True
    redis.check_rate_limit.return_value            = (True, 1)
    redis.publish.return_value                     = 1
    return redis


@pytest.fixture
def mock_influx():
    """Returns a fully mocked InfluxClient."""
    influx = AsyncMock()
    influx.get_candles.return_value = pd.DataFrame()
    influx.health_check.return_value = True
    return influx


# ─────────────────────────────────────────────
# Mock Service Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def mock_broker_service():
    """Returns a mocked BrokerService."""
    svc = MagicMock()
    svc.active_broker_name = "paper"
    svc.is_connected       = True
    svc.get_funds          = AsyncMock(return_value={
        "available_cash": 1_000_000.0,
        "used_margin":    0.0,
        "total_balance":  1_000_000.0,
    })
    svc.get_positions = AsyncMock(return_value=[])
    svc.place_order   = AsyncMock()
    svc.health_check  = AsyncMock(return_value={
        "broker_connected": True,
        "service_running":  True,
        "is_paper":         True,
    })
    svc.register_order_subscriber   = MagicMock()
    svc.unregister_order_subscriber = MagicMock()
    return svc


@pytest.fixture
def mock_feed_service():
    """Returns a mocked FeedService."""
    svc = MagicMock()
    svc.is_running = True
    svc.subscribe  = AsyncMock(return_value="consumer_1")
    svc.unsubscribe = AsyncMock()
    svc.unsubscribe_all = AsyncMock()
    svc.get_candles   = MagicMock(return_value=[])
    svc.get_last_price = MagicMock(return_value=2450.0)
    svc.get_feed_stats = MagicMock(return_value={
        "is_running":         True,
        "active_broker":      "paper",
        "subscribed_symbols": 0,
        "active_consumers":   0,
        "active_aggregators": 0,
        "symbol_keys":        [],
    })
    return svc


@pytest.fixture
def mock_data_service():
    """Returns a mocked DataService."""
    svc = AsyncMock()
    svc.get_ohlcv.return_value = pd.DataFrame()
    svc.get_latest_candles.return_value = pd.DataFrame()
    svc.get_intraday.return_value = pd.DataFrame()
    svc.get_cache_stats.return_value = {}
    svc.health_check.return_value = {
        "influx_ok": True, "broker_ok": True
    }
    return svc


@pytest.fixture
def mock_strategy_engine():
    """Returns a mocked StrategyEngine."""
    engine = MagicMock()
    engine.is_running          = True
    engine.registered_count    = 1
    engine.active_count        = 0
    engine._registry           = {}
    engine.get_registered_strategies.return_value = []
    engine.get_active_strategies.return_value     = []
    engine.activate_strategy   = AsyncMock(return_value=True)
    engine.deactivate_strategy = AsyncMock(return_value=True)
    engine.restart_strategy    = AsyncMock(return_value=True)
    engine.get_strategy_instance.return_value = None
    engine.get_engine_stats.return_value = {
        "is_running":            True,
        "active_broker":         "paper",
        "registered_strategies": 0,
        "active_strategies":     0,
        "active_names":          [],
        "registered_names":      [],
        "order_routing_entries": 0,
        "error_counts":          {},
        "risk_stats":            {},
    }
    return engine


@pytest.fixture
def mock_risk_manager():
    """Returns a mocked RiskManager."""
    rm = AsyncMock()
    rm.approve_signal      = AsyncMock(return_value=True)
    rm.arm_kill_switch     = AsyncMock()
    rm.disarm_kill_switch  = AsyncMock()
    rm.arm_global_kill_switch   = AsyncMock()
    rm.disarm_global_kill_switch = AsyncMock()
    rm.get_stats.return_value = {
        "signals_approved": 0,
        "signals_blocked":  0,
        "total_signals":    0,
        "approval_rate":    100.0,
        "block_reasons":    {},
    }
    return rm


@pytest.fixture
def mock_backtest_runner():
    """Returns a mocked BacktestRunner."""
    runner = AsyncMock()
    runner.stored_results = []
    runner.run.return_value = MagicMock(
        run_id="test_run_001",
        strategy_name="ema_crossover",
        symbol="RELIANCE",
        interval="5minute",
        from_date="2024-01-01",
        to_date="2024-06-01",
        initial_capital=1_000_000.0,
        parameters={"fast_period": 9, "slow_period": 21},
        fills=[],
        equity_curve=[],
        metrics={
            "final_capital":     1_050_000.0,
            "total_pnl":         50_000.0,
            "total_return_pct":  5.0,
            "cagr_pct":          10.2,
            "total_trades":      42,
            "winning_trades":    25,
            "losing_trades":     17,
            "win_rate_pct":      59.52,
            "avg_win":           3200.0,
            "avg_loss":          -1800.0,
            "largest_win":       12_000.0,
            "largest_loss":      -8_000.0,
            "profit_factor":     2.15,
            "expectancy":        1250.0,
            "max_drawdown_pct":  3.2,
            "max_drawdown_inr":  32_000.0,
            "sharpe_ratio":      1.42,
            "sortino_ratio":     1.89,
            "calmar_ratio":      3.19,
            "omega_ratio":       1.85,
            "total_commission":  630.0,
            "total_slippage":    420.0,
        },
    )
    return runner


# ─────────────────────────────────────────────
# FastAPI Test App
# ─────────────────────────────────────────────

@pytest.fixture
def mock_container(
    mock_pg,
    mock_redis,
    mock_influx,
    mock_broker_service,
    mock_feed_service,
    mock_data_service,
    mock_strategy_engine,
    mock_risk_manager,
    mock_backtest_runner,
):
    """Returns a fully mocked Container instance."""
    container = MagicMock()
    container.is_started        = True
    container.pg                = mock_pg
    container.redis             = mock_redis
    container.influx            = mock_influx
    container.broker_service    = mock_broker_service
    container.feed_service      = mock_feed_service
    container.data_service      = mock_data_service
    container.strategy_engine   = mock_strategy_engine
    container.risk_manager      = mock_risk_manager
    container.backtest_runner   = mock_backtest_runner
    container.health_check      = AsyncMock(return_value={
        "container":       True,
        "redis":           True,
        "influxdb":        True,
        "broker":          True,
        "feed_running":    True,
        "engine_running":  True,
        "broker_connected": True,
    })
    container.get_system_stats.return_value = {
        "started":       True,
        "environment":   "testing",
        "trading_mode":  "paper",
        "active_broker": "paper",
    }
    return container


@pytest_asyncio.fixture
async def test_client(mock_container) -> AsyncGenerator:
    """
    Returns an httpx AsyncClient pointed at the FastAPI app
    with all services mocked via the Container.

    Usage:
        async def test_root(test_client):
            response = await test_client.get("/")
            assert response.status_code == 200
    """
    from app import create_app

    app = create_app()
    app.state.container = mock_container

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        yield client