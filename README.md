# 🚀 NexaTrade

**Algorithmic Trading Platform for Indian Markets**

NexaTrade is a production-grade, fully async Python trading platform
built on FastAPI, supporting live trading via ICICI Breeze Connect
and a high-fidelity paper trading simulator.

---

## 🏗️ Architecture

```
NexaTrade
├── api/                  FastAPI routes + schemas + dependencies
├── backtesting/          Event-driven backtester + performance analytics
├── brokers/              Broker abstraction layer (Breeze + Paper)
├── config/               YAML configs + Pydantic settings
├── data/storage/         PostgreSQL + Redis + InfluxDB clients
├── plugins/              Strategy plugins (auto-discovered)
├── services/             BrokerService + FeedService + DataService
├── strategies/           AbstractStrategy + StrategyEngine + RiskManager
├── tests/                Unit + integration test suite
└── utils/                Indicators + time utils + auth + logger
```

---

## ⚡ Quick Start

### 1. Prerequisites

- Python 3.11+
- Docker + Docker Compose
- ICICI Direct account (for live trading; optional for paper mode)

### 2. Clone & Configure

```bash
git clone https://github.com/yourorg/nexatrade.git
cd nexatrade
cp .env.example .env
# Edit .env — set SECRET_KEY and JWT_SECRET_KEY at minimum
```

### 3. Start Infrastructure

```bash
docker compose up -d postgres redis influxdb
```

### 4. Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 5. Run NexaTrade

```bash
python main.py
# or
python main.py --reload        # dev mode with hot-reload
```

**API Docs**: http://localhost:8000/docs  
**ReDoc**:    http://localhost:8000/redoc

---

## 🐳 Docker (Full Stack)

```bash
# Start everything (app + databases)
docker compose up -d

# View logs
docker compose logs -f nexatrade

# Stop all
docker compose down
```

---

## 📈 Usage Guide

### Paper Trading (Default)

No broker credentials required. Set in `.env`:

```dotenv
TRADING_MODE=paper
ACTIVE_BROKER=paper
```

### Live Trading (Breeze Connect)

1. Get API credentials from [ICICI Direct](https://api.icicidirect.com/)
2. Generate session token daily before market open
3. Set in `.env`:

```dotenv
TRADING_MODE=live
ACTIVE_BROKER=breeze
BREEZE_API_KEY=your_api_key
BREEZE_API_SECRET=your_api_secret
BREEZE_SESSION_TOKEN=your_session_token
```

---

## 🔌 API Reference

### Authentication

```bash
# Login
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "password"}'

# Use token
export TOKEN="<access_token>"
```

### Place an Order

```bash
curl -X POST http://localhost:8000/api/v1/orders/place \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "RELIANCE",
    "transaction_type": "BUY",
    "quantity": 50,
    "order_type": "MARKET"
  }'
```

### Activate a Strategy

```bash
curl -X POST http://localhost:8000/api/v1/strategies/activate \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "strategy_name": "ema_crossover",
    "capital": 500000,
    "parameters": {"fast_period": 9, "slow_period": 21}
  }'
```

### Run a Backtest

```bash
curl -X POST http://localhost:8000/api/v1/backtest/run \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "strategy_name":   "ema_crossover",
    "symbol":          "RELIANCE",
    "interval":        "5minute",
    "from_date":       "2023-01-01",
    "to_date":         "2024-01-01",
    "initial_capital": 1000000
  }'
```

### WebSocket Live Feed

```javascript
// Connect with JWT token
const ws = new WebSocket(
  "ws://localhost:8000/api/v1/ws/ticks/RELIANCE?token=<jwt>"
);
ws.onmessage = (event) => {
  const tick = JSON.parse(event.data);
  console.log(tick.last_price);
};
```

---

## 🧩 Writing a Strategy

Create `plugins/my_strategy.py`:

```python
from strategies.abstract_strategy import AbstractStrategy
from brokers.models import OHLCV, OrderResponse, SignalDirection, \
    StrategySignal, TickData, Exchange, Segment

class MyStrategy(AbstractStrategy):
    STRATEGY_NAME    = "my_strategy"
    DISPLAY_NAME     = "My First Strategy"
    DEFAULT_PARAMETERS = {"threshold": 50.0}
    DEFAULT_INSTRUMENTS = [{"symbol": "NIFTY", "exchange": "NSE"}]
    DEFAULT_INTERVAL = "5minute"

    async def on_start(self):
        self.threshold = self.get_param("threshold", 50.0)
        await self._feed.subscribe(
            "NIFTY", "NSE",
            interval=self.DEFAULT_INTERVAL,
            consumer_id=self.name,
            candle_callback=self.on_candle,
            tick_callback=self.on_tick,
        )

    async def on_tick(self, tick: TickData): pass

    async def on_candle(self, candle: OHLCV):
        if candle.close > self.threshold:
            await self.emit_signal(StrategySignal(
                strategy_name=self.name,
                symbol=candle.symbol,
                exchange=Exchange.NSE,
                segment=Segment.EQ,
                direction=SignalDirection.BUY,
                reason="Price above threshold",
            ))

    async def on_order_update(self, response: OrderResponse): pass
    async def on_stop(self):
        await self._feed.unsubscribe_all(self.name)
    async def on_error(self, exc: Exception):
        self._logger.error(f"Error: {exc}")
```

NexaTrade **auto-discovers** your plugin — no other changes needed.

---

## 🧪 Running Tests

```bash
# All tests
pytest tests/ -v

# Unit tests only (no I/O)
pytest tests/unit/ -v -m unit

# With coverage
pytest tests/ --cov=. --cov-report=html

# Specific test file
pytest tests/unit/test_indicators.py -v
```

---

## 🔒 Risk Management

NexaTrade's `RiskManager` evaluates **10 checks** on every signal:

| # | Check | Config Key |
|---|-------|------------|
| 1 | Global / broker kill switch | Redis `ks:` keys |
| 2 | Market hours (09:15–15:30 IST) | Automatic |
| 3 | Daily loss limit | `loss_limits.daily_loss_limit` |
| 4 | Max drawdown | `loss_limits.max_drawdown_pct` |
| 5 | Max open positions | `position_limits.max_open_positions` |
| 6 | Capital per trade | `capital.max_capital_per_trade_pct` |
| 7 | Symbol / exchange blacklist | `blacklist.symbols` |
| 8 | Duplicate signal (Redis TTL) | `strategy.signal_ttl_seconds` |
| 9 | Max position size | `position_limits.max_position_size` |
| 10 | Direction conflict | Redis position cache |

**Arm kill switch** (blocks all new orders immediately):

```bash
curl -X POST http://localhost:8000/api/v1/risk/kill-switch/arm \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"global_switch": true, "reason": "market_emergency"}'
```

---

## 📁 Project Structure

```
nexatrade/
├── .env.example
├── .github/workflows/ci.yml
├── Dockerfile
├── docker-compose.yml
├── pytest.ini
├── requirements.txt
├── README.md
├── app.py                        ← FastAPI app factory
├── container.py                  ← Dependency injection container
├── main.py                       ← CLI entry point
│
├── api/
│   ├── dependencies.py
│   ├── schemas.py
│   └── routes/
│       ├── auth.py
│       ├── backtest.py
│       ├── broker.py
│       ├── data.py
│       ├── feed.py
│       ├── orders.py
│       ├── positions.py
│       ├── risk.py
│       ├── strategies.py
│       └── websocket.py
│
├── backtesting/
│   ├── backtester.py
│   ├── backtest_runner.py
│   └── performance.py
│
├── brokers/
│   ├── abstract_broker.py
│   ├── models.py
│   ├── registry.py
│   ├── breeze/
│   │   └── breeze_broker.py
│   └── paper/
│       └── paper_broker.py
│
├── config/
│   ├── settings.py
│   ├── app_config.yaml
│   ├── risk_config.yaml
│   └── brokers/
│       ├── breeze.yaml
│       └── paper.yaml
│
├── data/
│   └── storage/
│       ├── influx_client.py
│       ├── postgres_client.py
│       └── redis_client.py
│
├── plugins/
│   └── ema_crossover.py
│
├── services/
│   ├── broker_service.py
│   ├── data_service.py
│   └── feed_service.py
│
├── strategies/
│   ├── abstract_strategy.py
│   ├── risk_manager.py
│   └── strategy_engine.py
│
├── tests/
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_backtester.py
│   │   ├── test_indicators.py
│   │   ├── test_performance.py
│   │   └── test_risk_manager.py
│   └── integration/
│       └── test_api_routes.py
│
└── utils/
    ├── auth.py
    ├── indicators.py
    ├── logger.py
    └── time_utils.py
```

---

## 🛡️ Security

- All secrets via environment variables — never in code
- JWT Bearer auth on all protected endpoints
- `SecretStr` for all credentials — never logged or serialised
- bcrypt password hashing (12 rounds)
- Non-root Docker user
- `flush_all()` blocked in production
- Rate limiting via Redis token bucket

---

## 📄 License

MIT License — see `LICENSE` for details.

---

**Built with ❤️ for Indian algo traders.**