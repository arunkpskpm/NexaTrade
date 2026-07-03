"""
Integration tests for FastAPI API routes.

All services are mocked via mock_container.
Tests verify HTTP status codes, response shapes,
and authentication guards.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock


class TestRootEndpoints:

    @pytest.mark.asyncio
    async def test_root_returns_200(self, test_client):
        response = await test_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        assert "version"      in data
        assert "trading_mode" in data

    @pytest.mark.asyncio
    async def test_health_returns_200(self, test_client):
        response = await test_client.get("/health")
        assert response.status_code in (200, 503)
        data = response.json()
        assert "status" in data
        assert "checks" in data

    @pytest.mark.asyncio
    async def test_stats_returns_200(self, test_client, auth_headers):
        response = await test_client.get(
            "/stats", headers=auth_headers
        )
        assert response.status_code == 200


class TestAuthRoutes:

    @pytest.mark.asyncio
    async def test_login_success(self, test_client, mock_container):
        from utils.auth import hash_password
        mock_container.pg.get_user_by_username.return_value = {
            "user_id":      "00000000-0000-0000-0000-000000000001",
            "username":     "test_user",
            "password_hash": hash_password("password123"),
            "is_active":    True,
        }
        response = await test_client.post(
            "/api/v1/auth/login",
            json={"username": "test_user", "password": "password123"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    @pytest.mark.asyncio
    async def test_login_wrong_password(
        self, test_client, mock_container
    ):
        from utils.auth import hash_password
        mock_container.pg.get_user_by_username.return_value = {
            "user_id":      "00000000-0000-0000-0000-000000000001",
            "username":     "test_user",
            "password_hash": hash_password("correct_password"),
            "is_active":    True,
        }
        response = await test_client.post(
            "/api/v1/auth/login",
            json={
                "username": "test_user",
                "password": "wrong_password"
            },
        )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_me_requires_auth(self, test_client):
        response = await test_client.get("/api/v1/auth/me")
        assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_me_returns_user(
        self, test_client, auth_headers
    ):
        response = await test_client.get(
            "/api/v1/auth/me", headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert "username" in data
        assert "user_id"  in data

    @pytest.mark.asyncio
    async def test_logout_success(
        self, test_client, auth_headers
    ):
        response = await test_client.post(
            "/api/v1/auth/logout", headers=auth_headers
        )
        assert response.status_code == 200


class TestBrokerRoutes:

    @pytest.mark.asyncio
    async def test_broker_info_requires_auth(self, test_client):
        response = await test_client.get("/api/v1/broker/info")
        assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_broker_info_success(
        self, test_client, auth_headers, mock_container
    ):
        broker_info = MagicMock()
        broker_info.name                      = "paper"
        broker_info.display_name              = "Paper Trading"
        broker_info.version                   = "1.0.0"
        broker_info.supports_websocket        = True
        broker_info.supports_historical_data  = True
        broker_info.supports_paper_trading    = True
        broker_info.supports_options          = False
        broker_info.supports_futures          = False
        broker_info.connection_state          = "CONNECTED"
        broker_info.is_authenticated          = True
        mock_container.broker_service.get_broker_info = AsyncMock(
            return_value=broker_info
        )
        mock_container.broker_service.active_broker_name = "paper"

        response = await test_client.get(
            "/api/v1/broker/info", headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "paper"

    @pytest.mark.asyncio
    async def test_broker_funds_success(
        self, test_client, auth_headers
    ):
        response = await test_client.get(
            "/api/v1/broker/funds", headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert "available_cash" in data
        assert "trading_mode"   in data


class TestOrderRoutes:

    @pytest.mark.asyncio
    async def test_place_order_requires_auth(self, test_client):
        response = await test_client.post(
            "/api/v1/orders/place",
            json={
                "symbol":           "RELIANCE",
                "transaction_type": "BUY",
                "quantity":         50,
            },
        )
        assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_place_order_success(
        self, test_client, auth_headers, mock_container
    ):
        from brokers.models import OrderResponse, OrderStatus, TradingMode
        mock_container.broker_service.place_order = AsyncMock(
            return_value=OrderResponse(
                order_id="ord-001",
                broker_order_id="BRK-001",
                status=OrderStatus.PENDING,
                broker_name="paper",
                trading_mode=TradingMode.PAPER,
            )
        )
        response = await test_client.post(
            "/api/v1/orders/place",
            headers=auth_headers,
            json={
                "symbol":           "RELIANCE",
                "transaction_type": "BUY",
                "quantity":         50,
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert "order_id"    in data
        assert "status"      in data
        assert "broker_name" in data

    @pytest.mark.asyncio
    async def test_place_order_invalid_side(
        self, test_client, auth_headers
    ):
        response = await test_client.post(
            "/api/v1/orders/place",
            headers=auth_headers,
            json={
                "symbol":           "RELIANCE",
                "transaction_type": "INVALID",
                "quantity":         50,
            },
        )
        assert response.status_code == 422


class TestStrategyRoutes:

    @pytest.mark.asyncio
    async def test_get_registered_requires_auth(
        self, test_client
    ):
        response = await test_client.get(
            "/api/v1/strategies/registered"
        )
        assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_get_active_strategies(
        self, test_client, auth_headers
    ):
        response = await test_client.get(
            "/api/v1/strategies/active",
            headers=auth_headers,
        )
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    @pytest.mark.asyncio
    async def test_activate_strategy_success(
        self, test_client, auth_headers, mock_container
    ):
        mock_container.strategy_engine.activate_strategy = AsyncMock(
            return_value=True
        )
        response = await test_client.post(
            "/api/v1/strategies/activate",
            headers=auth_headers,
            json={
                "strategy_name": "ema_crossover",
                "capital":        500_000.0,
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["success"] is True


class TestRiskRoutes:

    @pytest.mark.asyncio
    async def test_risk_stats_success(
        self, test_client, auth_headers
    ):
        response = await test_client.get(
            "/api/v1/risk/stats", headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert "signals_approved" in data
        assert "approval_rate"    in data

    @pytest.mark.asyncio
    async def test_arm_kill_switch(
        self, test_client, auth_headers
    ):
        response = await test_client.post(
            "/api/v1/risk/kill-switch/arm",
            headers=auth_headers,
            json={"broker_name": "paper", "reason": "test"},
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_disarm_kill_switch(
        self, test_client, auth_headers
    ):
        response = await test_client.post(
            "/api/v1/risk/kill-switch/disarm",
            headers=auth_headers,
            json={"broker_name": "paper"},
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_kill_switch_status(
        self, test_client, auth_headers
    ):
        response = await test_client.get(
            "/api/v1/risk/kill-switch/status",
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert "global_kill_switch" in data


class TestBacktestRoutes:

    @pytest.mark.asyncio
    async def test_run_backtest_strategy_not_found(
        self, test_client, auth_headers, mock_container
    ):
        mock_container.strategy_engine._registry = {}
        response = await test_client.post(
            "/api/v1/backtest/run",
            headers=auth_headers,
            json={
                "strategy_name":  "unknown_strategy",
                "symbol":         "RELIANCE",
                "exchange":       "NSE",
                "interval":       "5minute",
                "from_date":      "2024-01-01",
                "to_date":        "2024-06-01",
                "initial_capital": 1_000_000.0,
            },
        )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_run_backtest_success(
        self, test_client, auth_headers, mock_container
    ):
        from plugins.ema_crossover import EMACrossoverStrategy
        mock_container.strategy_engine._registry = {
            "ema_crossover": EMACrossoverStrategy
        }
        response = await test_client.post(
            "/api/v1/backtest/run",
            headers=auth_headers,
            json={
                "strategy_name":  "ema_crossover",
                "symbol":         "RELIANCE",
                "exchange":       "NSE",
                "interval":       "5minute",
                "from_date":      "2024-01-01",
                "to_date":        "2024-06-01",
                "initial_capital": 1_000_000.0,
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert "run_id"          in data
        assert "total_trades"    in data
        assert "sharpe_ratio"    in data
        assert "max_drawdown_pct" in data