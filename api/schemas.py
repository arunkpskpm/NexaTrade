"""
NexaTrade — Pydantic Request & Response Schemas.

All API request bodies and response models are defined here.
These are separate from the broker domain models —
they are the HTTP contract with API consumers.

Design rules:
    - Request schemas: validate incoming data, strict types
    - Response schemas: serialise outgoing data, safe types
    - No SecretStr in responses — credentials never leak
    - All monetary values are float (INR)
    - All timestamps are ISO-8601 strings
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


# ═════════════════════════════════════════════
# Section 1 — Auth Schemas
# ═════════════════════════════════════════════

class LoginRequest(BaseModel):
    username: str = Field(..., min_length=3)
    password: str = Field(..., min_length=6)


class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    expires_in:   int          # seconds
    user_id:      str
    username:     str


class UserResponse(BaseModel):
    user_id:    str
    username:   str
    email:      Optional[str] = None
    is_active:  bool = True
    created_at: Optional[str] = None


# ═════════════════════════════════════════════
# Section 2 — Broker Schemas
# ═════════════════════════════════════════════

class BrokerSwitchRequest(BaseModel):
    broker_name: str = Field(
        ..., description="Broker identifier e.g. breeze / paper"
    )

    @field_validator("broker_name")
    @classmethod
    def validate_broker(cls, v: str) -> str:
        return v.lower().strip()


class BrokerInfoResponse(BaseModel):
    name:                    str
    display_name:            str
    version:                 str
    supports_websocket:      bool
    supports_historical_data: bool
    supports_paper_trading:  bool
    supports_options:        bool
    supports_futures:        bool
    connection_state:        str
    is_authenticated:        bool
    active_broker:           str
    trading_mode:            str


class FundsResponse(BaseModel):
    available_cash:  float
    used_margin:     float
    total_balance:   float
    broker_name:     str
    trading_mode:    str


# ═════════════════════════════════════════════
# Section 3 — Order Schemas
# ═════════════════════════════════════════════

class PlaceOrderRequest(BaseModel):
    symbol:           str  = Field(..., min_length=1)
    exchange:         str  = Field(default="NSE")
    transaction_type: str  = Field(..., pattern="^(BUY|SELL)$")
    order_type:       str  = Field(
        default="MARKET",
        pattern="^(MARKET|LIMIT|STOP_LOSS|STOP_LOSS_MARKET)$",
    )
    product_type:     str  = Field(
        default="INTRADAY",
        pattern="^(INTRADAY|DELIVERY|FUTURES|OPTIONS)$",
    )
    quantity:         int  = Field(..., gt=0)
    price:            Optional[float] = Field(default=None, gt=0)
    trigger_price:    Optional[float] = Field(default=None, gt=0)

    @field_validator("symbol")
    @classmethod
    def normalise_symbol(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("exchange", "transaction_type",
                     "order_type", "product_type")
    @classmethod
    def normalise_str(cls, v: str) -> str:
        return v.strip().upper()


class ModifyOrderRequest(BaseModel):
    broker_order_id: str
    quantity:        Optional[int]   = Field(default=None, gt=0)
    price:           Optional[float] = Field(default=None, gt=0)
    trigger_price:   Optional[float] = Field(default=None, gt=0)


class OrderResponse(BaseModel):
    order_id:          str
    broker_order_id:   Optional[str]   = None
    status:            str
    symbol:            Optional[str]   = None
    message:           str             = ""
    rejection_reason:  Optional[str]   = None
    filled_quantity:   int             = 0
    average_price:     Optional[float] = None
    placed_at:         Optional[str]   = None
    updated_at:        Optional[str]   = None
    broker_name:       str
    trading_mode:      str


# ═════════════════════════════════════════════
# Section 4 — Market Data Schemas
# ═════════════════════════════════════════════

class QuoteResponse(BaseModel):
    symbol:       str
    exchange:     str
    last_price:   float
    open:         float  = 0.0
    high:         float  = 0.0
    low:          float  = 0.0
    close:        float  = 0.0
    bid:          float  = 0.0
    ask:          float  = 0.0
    volume:       int    = 0
    change:       float  = 0.0
    change_pct:   float  = 0.0
    timestamp:    Optional[str] = None
    broker_name:  str


class OHLCVResponse(BaseModel):
    datetime: str
    open:     float
    high:     float
    low:      float
    close:    float
    volume:   float


class HistoricalDataRequest(BaseModel):
    symbol:     str  = Field(..., min_length=1)
    exchange:   str  = Field(default="NSE")
    interval:   str  = Field(
        default="5minute",
        pattern=(
            "^(1minute|5minute|15minute|"
            "30minute|1hour|1day)$"
        ),
    )
    from_date:  str  = Field(
        ..., pattern=r"^\d{4}-\d{2}-\d{2}$"
    )
    to_date:    str  = Field(
        ..., pattern=r"^\d{4}-\d{2}-\d{2}$"
    )
    broker_name: Optional[str] = None

    @field_validator("symbol")
    @classmethod
    def normalise_symbol(cls, v: str) -> str:
        return v.strip().upper()


class HistoricalDataResponse(BaseModel):
    symbol:    str
    exchange:  str
    interval:  str
    from_date: str
    to_date:   str
    bars:      int
    data:      list[OHLCVResponse]


# ═════════════════════════════════════════════
# Section 5 — Position Schemas
# ═════════════════════════════════════════════

class PositionResponse(BaseModel):
    symbol:          str
    exchange:        str
    segment:         str
    quantity:        int
    average_price:   float
    last_price:      float
    unrealized_pnl:  float
    realized_pnl:    float
    total_pnl:       float
    market_value:    float
    broker_name:     str
    trading_mode:    str
    is_long:         bool
    is_short:        bool


class PortfolioSummaryResponse(BaseModel):
    total_positions:      int
    long_positions:       int
    short_positions:      int
    total_unrealized_pnl: float
    total_realized_pnl:   float
    total_pnl:            float
    positions:            list[PositionResponse]


# ═════════════════════════════════════════════
# Section 6 — Strategy Schemas
# ═════════════════════════════════════════════

class StrategyActivateRequest(BaseModel):
    strategy_name: str  = Field(..., min_length=1)
    parameters:    Optional[dict[str, Any]] = None
    instruments:   Optional[list[dict[str, str]]] = None
    capital:       float = Field(default=0.0, ge=0)

    @field_validator("strategy_name")
    @classmethod
    def normalise_name(cls, v: str) -> str:
        return v.strip().lower()


class StrategyUpdateParamsRequest(BaseModel):
    parameters: dict[str, Any] = Field(..., min_length=1)


class StrategyInfoResponse(BaseModel):
    name:          str
    display_name:  str
    description:   str
    version:       str
    author:        str
    parameters:    dict[str, Any]
    instruments:   list[dict[str, str]]
    interval:      str
    is_active:     bool


class StrategyStatsResponse(BaseModel):
    name:          str
    display_name:  str
    is_running:    bool
    trading_mode:  str
    broker:        str
    tick_count:    int
    signal_count:  int
    order_count:   int
    capital:       float
    instruments:   list[dict[str, str]]
    parameters:    dict[str, Any]


# ═════════════════════════════════════════════
# Section 7 — Backtest Schemas
# ═════════════════════════════════════════════

class BacktestRunRequest(BaseModel):
    strategy_name:    str   = Field(..., min_length=1)
    symbol:           str   = Field(..., min_length=1)
    exchange:         str   = Field(default="NSE")
    interval:         str   = Field(default="5minute")
    from_date:        str   = Field(
        ..., pattern=r"^\d{4}-\d{2}-\d{2}$"
    )
    to_date:          str   = Field(
        ..., pattern=r"^\d{4}-\d{2}-\d{2}$"
    )
    initial_capital:  float = Field(default=1_000_000.0, gt=0)
    slippage_pct:     float = Field(default=0.05,  ge=0, le=5)
    commission_pct:   float = Field(default=0.03,  ge=0, le=5)
    warmup_bars:      int   = Field(default=50,    ge=0)
    parameters:       Optional[dict[str, Any]] = None
    broker_name:      Optional[str] = None

    @field_validator("strategy_name")
    @classmethod
    def normalise_strategy(cls, v: str) -> str:
        return v.strip().lower()

    @field_validator("symbol")
    @classmethod
    def normalise_symbol(cls, v: str) -> str:
        return v.strip().upper()


class ParameterSweepRequest(BaseModel):
    strategy_name:   str                    = Field(..., min_length=1)
    symbol:          str                    = Field(..., min_length=1)
    exchange:        str                    = Field(default="NSE")
    interval:        str                    = Field(default="5minute")
    from_date:       str
    to_date:         str
    param_grid:      dict[str, list[Any]]   = Field(..., min_length=1)
    initial_capital: float                  = Field(
        default=1_000_000.0, gt=0
    )
    slippage_pct:    float                  = Field(default=0.05)
    commission_pct:  float                  = Field(default=0.03)
    max_concurrent:  int                    = Field(default=3, ge=1, le=10)
    rank_by:         str                    = Field(default="sharpe_ratio")


class BacktestMetricsResponse(BaseModel):
    run_id:              str
    strategy_name:       str
    symbol:              str
    interval:            str
    from_date:           str
    to_date:             str
    initial_capital:     float
    parameters:          dict[str, Any]
    # Return metrics
    final_capital:       float
    total_pnl:           float
    total_return_pct:    float
    cagr_pct:            float
    # Trade stats
    total_trades:        int
    winning_trades:      int
    losing_trades:       int
    win_rate_pct:        float
    avg_win:             float
    avg_loss:            float
    largest_win:         float
    largest_loss:        float
    profit_factor:       float
    expectancy:          float
    # Risk
    max_drawdown_pct:    float
    max_drawdown_inr:    float
    sharpe_ratio:        float
    sortino_ratio:       float
    calmar_ratio:        float
    # Costs
    total_commission:    float
    total_slippage:      float


# ═════════════════════════════════════════════
# Section 8 — Risk Schemas
# ═════════════════════════════════════════════

class KillSwitchRequest(BaseModel):
    broker_name: Optional[str] = None
    global_switch: bool         = False
    reason:        str          = "manual"


class RiskStatsResponse(BaseModel):
    signals_approved: int
    signals_blocked:  int
    total_signals:    int
    approval_rate:    float
    block_reasons:    dict[str, int]


# ═════════════════════════════════════════════
# Section 9 — Feed Schemas
# ═════════════════════════════════════════════

class SubscribeRequest(BaseModel):
    symbol:     str = Field(..., min_length=1)
    exchange:   str = Field(default="NSE")
    interval:   str = Field(default="5minute")

    @field_validator("symbol")
    @classmethod
    def normalise(cls, v: str) -> str:
        return v.strip().upper()


class FeedStatsResponse(BaseModel):
    is_running:          bool
    active_broker:       str
    subscribed_symbols:  int
    active_consumers:    int
    active_aggregators:  int
    symbol_keys:         list[str]


# ═════════════════════════════════════════════
# Section 10 — Generic Response Wrappers
# ═════════════════════════════════════════════

class SuccessResponse(BaseModel):
    success: bool   = True
    message: str    = "OK"
    data:    Any    = None


class ErrorResponse(BaseModel):
    success: bool   = False
    error:   str
    message: str
    path:    Optional[str] = None


class PaginatedResponse(BaseModel):
    total:   int
    page:    int
    limit:   int
    data:    list[Any]