"""
NexaTrade — Broker-Agnostic Domain Models.

All data flowing between NexaTrade core and any broker adapter
must use these models exclusively. No broker-specific types
ever leave the adapter boundary.

Model hierarchy:
    OrderRequest   → what the strategy sends to the order engine
    OrderResponse  → what the broker returns after placement
    Quote          → live market price snapshot
    OHLCV          → single candlestick
    Position       → open position snapshot
    Fill           → single trade execution record
    BrokerInfo     → broker metadata and capabilities
    TickData       → raw WebSocket tick from broker feed
    InstrumentInfo → instrument master record

Design rules:
    - All models are immutable (frozen=True where feasible)
    - All monetary values are plain float (INR)
    - All timestamps are timezone-aware datetime (IST)
    - All enums are str-based for JSON serialisation
    - No broker-specific fields ever appear here
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


# ═════════════════════════════════════════════
# Section 1 — Enums
# ═════════════════════════════════════════════

class TransactionType(str, Enum):
    """Direction of a trade."""
    BUY  = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Order execution type."""
    MARKET           = "MARKET"
    LIMIT            = "LIMIT"
    STOP_LOSS        = "STOP_LOSS"
    STOP_LOSS_MARKET = "STOP_LOSS_MARKET"


class OrderStatus(str, Enum):
    """Order lifecycle state."""
    PENDING   = "PENDING"
    OPEN      = "OPEN"
    COMPLETE  = "COMPLETE"
    CANCELLED = "CANCELLED"
    REJECTED  = "REJECTED"
    MODIFIED  = "MODIFIED"
    PARTIAL   = "PARTIAL"


class ProductType(str, Enum):
    """Trading product / margin type."""
    DELIVERY  = "DELIVERY"
    INTRADAY  = "INTRADAY"
    FUTURES   = "FUTURES"
    OPTIONS   = "OPTIONS"


class Exchange(str, Enum):
    """Supported exchange codes."""
    NSE = "NSE"
    BSE = "BSE"
    NFO = "NFO"
    MCX = "MCX"
    BFO = "BFO"


class Segment(str, Enum):
    """Instrument segment."""
    EQ  = "EQ"
    FUT = "FUT"
    OPT = "OPT"
    COM = "COM"


class TradingMode(str, Enum):
    """Execution mode."""
    PAPER = "paper"
    LIVE  = "live"


class SignalDirection(str, Enum):
    """Strategy signal direction."""
    BUY    = "BUY"
    SELL   = "SELL"
    HOLD   = "HOLD"
    EXIT   = "EXIT"
    NONE   = "NONE"


class BrokerConnectionState(str, Enum):
    """Broker session connection state."""
    DISCONNECTED = "DISCONNECTED"
    CONNECTING   = "CONNECTING"
    CONNECTED    = "CONNECTED"
    RECONNECTING = "RECONNECTING"
    ERROR        = "ERROR"


# ═════════════════════════════════════════════
# Section 2 — Order Models
# ═════════════════════════════════════════════

class OrderRequest(BaseModel):
    """
    Broker-agnostic order request.
    Created by the strategy or order engine and passed
    to the active broker adapter for execution.

    The broker adapter translates this into its own
    SDK-specific format internally.
    """

    # Auto-generated NexaTrade order identifier
    order_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="NexaTrade internal order UUID",
    )

    # Instrument identification
    symbol: str = Field(..., description="Instrument symbol e.g. RELIANCE")
    exchange: Exchange = Field(..., description="Exchange code")
    segment: Segment = Field(default=Segment.EQ, description="Instrument segment")

    # Order parameters
    transaction_type: TransactionType = Field(..., description="BUY or SELL")
    order_type: OrderType = Field(..., description="MARKET/LIMIT/STOP_LOSS")
    product_type: ProductType = Field(
        default=ProductType.INTRADAY,
        description="Product/margin type",
    )
    quantity: int = Field(..., gt=0, description="Order quantity (must be > 0)")
    price: Optional[float] = Field(
        default=None,
        description="Limit price (None for MARKET orders)",
    )
    trigger_price: Optional[float] = Field(
        default=None,
        description="Trigger price for STOP_LOSS orders",
    )

    # Metadata
    strategy_name: Optional[str] = Field(
        default=None,
        description="Originating strategy identifier",
    )
    trading_mode: TradingMode = Field(
        default=TradingMode.PAPER,
        description="paper or live execution mode",
    )
    tags: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata for tracking",
    )

    @field_validator("symbol")
    @classmethod
    def normalise_symbol(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("price")
    @classmethod
    def validate_price(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v <= 0:
            raise ValueError("price must be > 0 if provided")
        return v

    @field_validator("trigger_price")
    @classmethod
    def validate_trigger_price(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v <= 0:
            raise ValueError("trigger_price must be > 0 if provided")
        return v

    class Config:
        use_enum_values = True


class OrderResponse(BaseModel):
    """
    Broker-agnostic order response.
    Returned by the broker adapter after order placement.
    Contains both NexaTrade's and the broker's order identifiers.
    """

    # Identifiers
    order_id: str = Field(..., description="NexaTrade internal order ID")
    broker_order_id: Optional[str] = Field(
        default=None,
        description="Broker's own order ID (None for paper orders)",
    )

    # Status
    status: OrderStatus = Field(..., description="Current order status")
    message: str = Field(default="", description="Broker status message")
    rejection_reason: Optional[str] = Field(
        default=None,
        description="Rejection reason if status=REJECTED",
    )

    # Fill details
    filled_quantity: int = Field(default=0, description="Quantity filled so far")
    average_price: Optional[float] = Field(
        default=None,
        description="Volume-weighted average fill price",
    )

    # Timestamps
    placed_at: Optional[datetime] = Field(
        default=None,
        description="Order placement timestamp",
    )
    updated_at: Optional[datetime] = Field(
        default=None,
        description="Last status update timestamp",
    )

    # Broker name for traceability
    broker_name: str = Field(..., description="Broker that processed this order")
    trading_mode: TradingMode = Field(..., description="paper or live")

    class Config:
        use_enum_values = True


class OrderModifyRequest(BaseModel):
    """
    Parameters for modifying an existing open order.
    Only OPEN or PARTIAL orders can be modified.
    """
    order_id: str = Field(..., description="NexaTrade order ID to modify")
    broker_order_id: str = Field(..., description="Broker's order ID")
    quantity: Optional[int] = Field(default=None, gt=0)
    price: Optional[float] = Field(default=None, gt=0)
    trigger_price: Optional[float] = Field(default=None, gt=0)
    order_type: Optional[OrderType] = Field(default=None)

    class Config:
        use_enum_values = True


# ═════════════════════════════════════════════
# Section 3 — Market Data Models
# ═════════════════════════════════════════════

class Quote(BaseModel):
    """
    Live market quote snapshot for an instrument.
    Returned by broker.get_quote() and populated
    from WebSocket tick data.
    """

    symbol: str
    exchange: Exchange
    broker_name: str

    # Price levels
    last_price: float = Field(..., description="Last traded price")
    open: float = Field(default=0.0, description="Day open price")
    high: float = Field(default=0.0, description="Day high price")
    low: float = Field(default=0.0, description="Day low price")
    close: float = Field(default=0.0, description="Previous day close")

    # Depth
    bid: float = Field(default=0.0, description="Best bid price")
    ask: float = Field(default=0.0, description="Best ask price")
    bid_qty: int = Field(default=0, description="Bid quantity")
    ask_qty: int = Field(default=0, description="Ask quantity")

    # Volume & OI
    volume: int = Field(default=0, description="Day volume")
    oi: int = Field(default=0, description="Open interest (F&O)")

    # Change
    change: float = Field(default=0.0, description="Price change from prev close")
    change_pct: float = Field(
        default=0.0, description="Percentage change from prev close"
    )

    # Timestamp
    timestamp: Optional[datetime] = Field(
        default=None, description="Quote timestamp in IST"
    )

    @field_validator("symbol")
    @classmethod
    def normalise_symbol(cls, v: str) -> str:
        return v.strip().upper()

    class Config:
        use_enum_values = True


class OHLCV(BaseModel):
    """
    A single OHLCV candlestick bar.
    Returned by broker.get_historical_data().
    """

    datetime: datetime = Field(..., description="Candle open timestamp (IST)")
    open: float
    high: float
    low: float
    close: float
    volume: float = Field(default=0.0)

    # Optional metadata
    symbol: Optional[str] = Field(default=None)
    exchange: Optional[str] = Field(default=None)
    interval: Optional[str] = Field(default=None)
    broker_name: Optional[str] = Field(default=None)

    def to_dict(self) -> dict[str, Any]:
        """Returns OHLCV as a plain dict for InfluxDB writes."""
        return {
            "datetime": self.datetime,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }

    class Config:
        use_enum_values = True


class TickData(BaseModel):
    """
    Raw WebSocket tick data normalised from any broker feed.
    The broker adapter converts SDK-specific tick dicts
    into this model before emitting to subscribers.
    """

    symbol: str
    exchange: str
    broker_name: str

    last_price: float
    bid: float = 0.0
    ask: float = 0.0
    volume: int = 0
    oi: int = 0
    change: float = 0.0
    change_pct: float = 0.0

    timestamp: datetime = Field(
        default_factory=lambda: __import__(
            "utils.time_utils", fromlist=["now_ist"]
        ).now_ist()
    )

    @field_validator("symbol")
    @classmethod
    def normalise_symbol(cls, v: str) -> str:
        return v.strip().upper()


# ═════════════════════════════════════════════
# Section 4 — Position & Fill Models
# ═════════════════════════════════════════════

class Position(BaseModel):
    """
    Open position snapshot returned by broker.get_positions().
    Normalised from broker-specific position formats.
    """

    symbol: str
    exchange: Exchange
    segment: Segment
    broker_name: str
    trading_mode: TradingMode

    # Quantity
    quantity: int = Field(
        ...,
        description="Net quantity (positive=long, negative=short, 0=flat)"
    )
    buy_quantity: int = Field(default=0)
    sell_quantity: int = Field(default=0)

    # Pricing
    average_price: float = Field(default=0.0)
    last_price: float = Field(default=0.0)
    buy_average_price: float = Field(default=0.0)
    sell_average_price: float = Field(default=0.0)

    # P&L
    unrealized_pnl: float = Field(default=0.0)
    realized_pnl: float = Field(default=0.0)
    total_pnl: float = Field(default=0.0)

    # Value
    market_value: float = Field(default=0.0)

    @field_validator("symbol")
    @classmethod
    def normalise_symbol(cls, v: str) -> str:
        return v.strip().upper()

    @property
    def is_long(self) -> bool:
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        return self.quantity < 0

    @property
    def is_flat(self) -> bool:
        return self.quantity == 0

    class Config:
        use_enum_values = True


class Fill(BaseModel):
    """
    A single trade execution / fill record.
    Returned by broker.get_order_status() when
    status == COMPLETE or PARTIAL.
    """

    fill_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="NexaTrade fill UUID",
    )
    order_id: str = Field(..., description="Parent NexaTrade order ID")
    broker_order_id: Optional[str] = Field(default=None)
    broker_name: str

    symbol: str
    exchange: str
    transaction_type: TransactionType
    quantity: int
    price: float
    commission: float = Field(default=0.0)
    trading_mode: TradingMode

    executed_at: datetime = Field(
        default_factory=lambda: __import__(
            "utils.time_utils", fromlist=["now_ist"]
        ).now_ist()
    )

    class Config:
        use_enum_values = True


# ═════════════════════════════════════════════
# Section 5 — Instrument Model
# ═════════════════════════════════════════════

class InstrumentInfo(BaseModel):
    """
    Instrument master record.
    Returned by broker.search_instruments().
    Used for symbol lookup and validation.
    """

    symbol: str
    exchange: Exchange
    segment: Segment
    broker_name: str

    # Identifiers
    isin: Optional[str] = Field(default=None)
    broker_token: Optional[str] = Field(
        default=None,
        description="Broker's internal instrument token/code",
    )

    # Display
    company_name: Optional[str] = Field(default=None)
    instrument_type: Optional[str] = Field(default=None)

    # F&O specific
    expiry: Optional[datetime] = Field(default=None)
    strike_price: Optional[float] = Field(default=None)
    option_type: Optional[str] = Field(default=None)  # CE or PE

    # Lot and tick
    lot_size: int = Field(default=1)
    tick_size: float = Field(default=0.05)

    class Config:
        use_enum_values = True


# ═════════════════════════════════════════════
# Section 6 — Broker Metadata Model
# ═════════════════════════════════════════════

class BrokerInfo(BaseModel):
    """
    Metadata and capability descriptor for a broker adapter.
    Each adapter returns this via broker.get_info().
    Used by the UI to display broker capabilities.
    """

    name: str = Field(..., description="Broker identifier (breeze/zerodha/etc)")
    display_name: str = Field(..., description="Human-readable broker name")
    version: str = Field(default="1.0.0")

    # Capabilities
    supports_websocket: bool = Field(default=True)
    supports_historical_data: bool = Field(default=True)
    supports_paper_trading: bool = Field(default=False)
    supports_options: bool = Field(default=True)
    supports_futures: bool = Field(default=True)
    supports_commodity: bool = Field(default=False)
    supports_order_modify: bool = Field(default=True)
    supports_bracket_orders: bool = Field(default=False)

    # Limits
    max_ws_subscriptions: int = Field(default=50)

    # State
    connection_state: BrokerConnectionState = Field(
        default=BrokerConnectionState.DISCONNECTED
    )
    is_authenticated: bool = Field(default=False)
    last_connected_at: Optional[datetime] = Field(default=None)

    class Config:
        use_enum_values = True


# ═════════════════════════════════════════════
# Section 7 — Strategy Signal Model
# ═════════════════════════════════════════════

class StrategySignal(BaseModel):
    """
    Signal emitted by a strategy.
    Consumed by the order engine to place orders.
    """

    signal_id: str = Field(default_factory=lambda: str(uuid4()))
    strategy_name: str
    symbol: str
    exchange: Exchange
    segment: Segment = Field(default=Segment.EQ)

    direction: SignalDirection
    strength: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Signal confidence (0.0 to 1.0)",
    )

    # Optional order hints
    suggested_price: Optional[float] = Field(default=None)
    suggested_quantity: Optional[int] = Field(default=None)
    stop_loss_price: Optional[float] = Field(default=None)
    target_price: Optional[float] = Field(default=None)

    # Metadata
    reason: str = Field(default="", description="Human-readable signal reason")
    tags: dict[str, Any] = Field(default_factory=dict)
    generated_at: datetime = Field(
        default_factory=lambda: __import__(
            "utils.time_utils", fromlist=["now_ist"]
        ).now_ist()
    )

    @field_validator("symbol")
    @classmethod
    def normalise_symbol(cls, v: str) -> str:
        return v.strip().upper()

    class Config:
        use_enum_values = True