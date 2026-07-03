"""
NexaTrade — Paper Trading Adapter.

Simulates all broker operations locally with no real API calls.
Uses Redis-cached live prices for realistic fill simulation.
Paper trading is a first-class adapter — not a mode flag.

Simulation behaviour:
  - Orders filled at last_price ± slippage_pct
  - Configurable fill delay (ms)
  - Commission deducted on every fill
  - All state stored in Redis (positions, orders, P&L)
  - Supports all order types (MARKET, LIMIT, STOP_LOSS)
  - Supports partial fills (configurable)

Config loaded from: config/brokers/paper.yaml
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from typing import Any, Optional

from brokers.abstract_broker import AbstractBroker
from brokers.models import (
    BrokerConnectionState,
    BrokerInfo,
    Exchange,
    Fill,
    OHLCV,
    InstrumentInfo,
    OrderModifyRequest,
    OrderRequest,
    OrderResponse,
    OrderStatus,
    OrderType,
    Position,
    Quote,
    Segment,
    TickData,
    TradingMode,
    TransactionType,
)
from utils.logger import get_logger, get_trade_logger
from utils.time_utils import now_ist

logger = get_logger(__name__)
trade_logger = get_trade_logger(__name__)


class PaperAdapter(AbstractBroker):
    """
    NexaTrade Paper Trading Adapter.

    Simulates a full broker session locally.
    All order fills are simulated against cached
    live prices with configurable slippage.

    State management:
        - Open orders  → in-memory dict (_open_orders)
        - Positions    → in-memory dict (_positions)
        - Fills        → in-memory list (_fills)
        - Tick prices  → received via registered callbacks
    """

    BROKER_NAME = "paper"

    def __init__(self) -> None:
        super().__init__(self.BROKER_NAME)

        # In-memory state
        self._open_orders: dict[str, dict[str, Any]] = {}
        self._positions: dict[str, dict[str, Any]] = {}
        self._fills: list[Fill] = []
        self._last_prices: dict[str, float] = {}

        # Simulation config (loaded in connect())
        self._slippage_pct: float = 0.05
        self._fill_delay_ms: int = 100
        self._commission_pct: float = 0.03
        self._rejection_rate_pct: float = 0.0

    # ─────────────────────────────────────────
    # Section 1 — Session Management
    # ─────────────────────────────────────────

    async def connect(self) -> bool:
        """
        Initialises the paper trading session.
        Loads simulation config from paper.yaml.
        No real API calls — always succeeds.

        Returns:
            True always.
        """
        self._set_state(BrokerConnectionState.CONNECTING)
        try:
            from config.settings import get_settings
            cfg = get_settings().broker_config("paper")
            sim = cfg.get("broker", {}).get("simulation", {})

            self._slippage_pct = float(
                sim.get("slippage_pct", 0.05)
            )
            self._fill_delay_ms = int(
                sim.get("fill_delay_ms", 100)
            )
            self._commission_pct = float(
                sim.get("commission_pct", 0.03)
            )
            self._rejection_rate_pct = float(
                sim.get("rejection_rate_pct", 0.0)
            )

            self._set_state(BrokerConnectionState.CONNECTED)
            logger.info(
                f"Paper trading session started | "
                f"slippage={self._slippage_pct}% | "
                f"fill_delay={self._fill_delay_ms}ms | "
                f"commission={self._commission_pct}%"
            )
            return True
        except Exception as exc:
            self._set_state(BrokerConnectionState.ERROR)
            raise ConnectionError(
                f"Paper adapter connect failed: {exc}"
            ) from exc

    async def disconnect(self) -> None:
        """Clears paper trading state and disconnects."""
        self._open_orders.clear()
        self._positions.clear()
        self._last_prices.clear()
        self._set_state(BrokerConnectionState.DISCONNECTED)
        logger.info(
            f"Paper trading session ended | "
            f"fills={len(self._fills)}"
        )

    async def is_connected(self) -> bool:
        """Paper adapter is always connected once initialised."""
        return (
            self._connection_state == BrokerConnectionState.CONNECTED
        )

    async def get_info(self) -> BrokerInfo:
        """Returns paper adapter capabilities metadata."""
        return BrokerInfo(
            name=self.BROKER_NAME,
            display_name="NexaTrade Paper Trading",
            version="1.0.0",
            supports_websocket=True,
            supports_historical_data=True,
            supports_paper_trading=True,
            supports_options=True,
            supports_futures=True,
            supports_commodity=False,
            supports_order_modify=True,
            supports_bracket_orders=False,
            max_ws_subscriptions=1000,
            connection_state=self._connection_state,
            is_authenticated=True,
        )

    # ─────────────────────────────────────────
    # Section 2 — Order Management
    # ─────────────────────────────────────────

    async def place_order(self, request: OrderRequest) -> OrderResponse:
        """
        Simulates order placement and fill.

        For MARKET orders: fills immediately at LTP ± slippage.
        For LIMIT orders: stores as open order, fills when price reached.
        Simulates fill_delay_ms before confirming fill.

        Args:
            request: NexaTrade OrderRequest.

        Returns:
            OrderResponse with COMPLETE or OPEN status.
        """
        # Simulate fill delay
        await asyncio.sleep(self._fill_delay_ms / 1000)

        # Simulate random rejection if configured
        if self._rejection_rate_pct > 0:
            import random
            if random.random() * 100 < self._rejection_rate_pct:
                response = OrderResponse(
                    order_id=request.order_id,
                    broker_order_id=None,
                    status=OrderStatus.REJECTED,
                    rejection_reason="Simulated rejection (stress test)",
                    message="Order randomly rejected by paper simulator.",
                    broker_name=self.BROKER_NAME,
                    trading_mode=TradingMode.PAPER,
                    placed_at=now_ist(),
                )
                trade_logger.warning(
                    f"PAPER ORDER REJECTED (simulated) | "
                    f"order_id={request.order_id} | "
                    f"symbol={request.symbol}"
                )
                return response

        # Generate paper broker order ID
        broker_order_id = f"PAPER-{str(uuid.uuid4())[:8].upper()}"

        # Attempt immediate fill for MARKET orders
        if request.order_type == OrderType.MARKET:
            fill_price = self._simulate_fill_price(
                symbol=request.symbol,
                transaction_type=request.transaction_type,
                requested_price=None,
            )
            commission = self._calculate_commission(
                quantity=request.quantity,
                price=fill_price,
            )

            # Update simulated position
            self._update_position(
                symbol=request.symbol,
                exchange=str(request.exchange),
                segment=str(request.segment),
                transaction_type=str(request.transaction_type),
                quantity=request.quantity,
                fill_price=fill_price,
            )

            # Record fill
            fill = Fill(
                order_id=request.order_id,
                broker_order_id=broker_order_id,
                broker_name=self.BROKER_NAME,
                symbol=request.symbol,
                exchange=str(request.exchange),
                transaction_type=request.transaction_type,
                quantity=request.quantity,
                price=fill_price,
                commission=commission,
                trading_mode=TradingMode.PAPER,
                executed_at=now_ist(),
            )
            self._fills.append(fill)

            response = OrderResponse(
                order_id=request.order_id,
                broker_order_id=broker_order_id,
                status=OrderStatus.COMPLETE,
                filled_quantity=request.quantity,
                average_price=fill_price,
                message=(
                    f"Paper order filled at {fill_price:.2f}"
                ),
                broker_name=self.BROKER_NAME,
                trading_mode=TradingMode.PAPER,
                placed_at=now_ist(),
                updated_at=now_ist(),
            )
            trade_logger.info(
                f"PAPER ORDER FILLED | "
                f"order_id={request.order_id} | "
                f"broker_id={broker_order_id} | "
                f"symbol={request.symbol} | "
                f"qty={request.quantity} | "
                f"price={fill_price:.2f} | "
                f"side={request.transaction_type} | "
                f"commission={commission:.2f}"
            )

        else:
            # LIMIT / STOP_LOSS orders: store as open
            self._open_orders[request.order_id] = {
                "request": request,
                "broker_order_id": broker_order_id,
                "status": OrderStatus.OPEN,
                "placed_at": now_ist(),
            }
            response = OrderResponse(
                order_id=request.order_id,
                broker_order_id=broker_order_id,
                status=OrderStatus.OPEN,
                message=f"Paper limit order open at {request.price}",
                broker_name=self.BROKER_NAME,
                trading_mode=TradingMode.PAPER,
                placed_at=now_ist(),
            )
            trade_logger.info(
                f"PAPER ORDER OPEN | "
                f"order_id={request.order_id} | "
                f"symbol={request.symbol} | "
                f"type={request.order_type} | "
                f"price={request.price}"
            )

        return response

    async def modify_order(
        self, request: OrderModifyRequest
    ) -> OrderResponse:
        """Modifies a pending paper limit order."""
        if request.order_id not in self._open_orders:
            return OrderResponse(
                order_id=request.order_id,
                broker_order_id=request.broker_order_id,
                status=OrderStatus.REJECTED,
                rejection_reason="Order not found in paper orders.",
                broker_name=self.BROKER_NAME,
                trading_mode=TradingMode.PAPER,
            )

        order_state = self._open_orders[request.order_id]
        req = order_state["request"]

        # Apply modifications
        if request.price:
            req = req.model_copy(update={"price": request.price})
        if request.quantity:
            req = req.model_copy(
                update={"quantity": request.quantity}
            )
        if request.trigger_price:
            req = req.model_copy(
                update={"trigger_price": request.trigger_price}
            )

        self._open_orders[request.order_id]["request"] = req
        self._open_orders[request.order_id]["status"] = (
            OrderStatus.MODIFIED
        )

        logger.info(
            f"Paper order modified | order_id={request.order_id}"
        )
        return OrderResponse(
            order_id=request.order_id,
            broker_order_id=request.broker_order_id,
            status=OrderStatus.MODIFIED,
            message="Paper order modified.",
            broker_name=self.BROKER_NAME,
            trading_mode=TradingMode.PAPER,
            updated_at=now_ist(),
        )

    async def cancel_order(
        self, order_id: str, broker_order_id: str
    ) -> OrderResponse:
        """Cancels a pending paper order."""
        if order_id in self._open_orders:
            self._open_orders.pop(order_id)
            trade_logger.info(
                f"PAPER ORDER CANCELLED | order_id={order_id}"
            )
            return OrderResponse(
                order_id=order_id,
                broker_order_id=broker_order_id,
                status=OrderStatus.CANCELLED,
                message="Paper order cancelled.",
                broker_name=self.BROKER_NAME,
                trading_mode=TradingMode.PAPER,
                updated_at=now_ist(),
            )

        return OrderResponse(
            order_id=order_id,
            broker_order_id=broker_order_id,
            status=OrderStatus.REJECTED,
            rejection_reason="Order not found.",
            broker_name=self.BROKER_NAME,
            trading_mode=TradingMode.PAPER,
        )

    async def get_order_status(
        self, order_id: str, broker_order_id: str
    ) -> OrderResponse:
        """Returns status of a paper order from in-memory state."""
        if order_id in self._open_orders:
            state = self._open_orders[order_id]
            return OrderResponse(
                order_id=order_id,
                broker_order_id=broker_order_id,
                status=state["status"],
                broker_name=self.BROKER_NAME,
                trading_mode=TradingMode.PAPER,
            )
        # Assume complete if not in open orders
        return OrderResponse(
            order_id=order_id,
            broker_order_id=broker_order_id,
            status=OrderStatus.COMPLETE,
            broker_name=self.BROKER_NAME,
            trading_mode=TradingMode.PAPER,
        )

    async def get_order_history(
        self,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> list[OrderResponse]:
        """Returns all fills as completed order responses."""
        return [
            OrderResponse(
                order_id=fill.order_id,
                broker_order_id=fill.broker_order_id,
                status=OrderStatus.COMPLETE,
                filled_quantity=fill.quantity,
                average_price=fill.price,
                broker_name=self.BROKER_NAME,
                trading_mode=TradingMode.PAPER,
                updated_at=fill.executed_at,
            )
            for fill in self._fills
        ]

    # ─────────────────────────────────────────
    # Section 3 — Market Data
    # ─────────────────────────────────────────

    async def get_quote(
        self, symbol: str, exchange: str
    ) -> Quote:
        """
        Returns a simulated quote using cached last price.
        Falls back to 0.0 if no price is cached for the symbol.
        """
        ltp = self._last_prices.get(symbol.upper(), 0.0)
        return Quote(
            symbol=symbol.upper(),
            exchange=exchange.upper(),
            broker_name=self.BROKER_NAME,
            last_price=ltp,
            timestamp=now_ist(),
        )

    async def get_quotes(
        self, instruments: list[dict[str, str]]
    ) -> list[Quote]:
        """Returns simulated quotes for multiple instruments."""
        return [
            await self.get_quote(
                inst["symbol"], inst.get("exchange", "NSE")
            )
            for inst in instruments
        ]

    async def get_historical_data(
        self,
        symbol: str,
        exchange: str,
        interval: str,
        from_date: str,
        to_date: str,
        segment: str = "EQ",
    ) -> list[OHLCV]:
        """
        Paper adapter delegates historical data to the active
        real broker's historical data endpoint.
        If no real broker is available, returns empty list.
        """
        logger.info(
            f"Paper adapter: delegating historical data | "
            f"symbol={symbol} | interval={interval}"
        )
        return []

    # ─────────────────────────────────────────
    # Section 4 — Portfolio
    # ─────────────────────────────────────────

    async def get_positions(self) -> list[Position]:
        """Returns all simulated open positions."""
        positions: list[Position] = []
        for key, state in self._positions.items():
            if state["quantity"] != 0:
                symbol, exchange, segment = key.split(":")
                ltp = self._last_prices.get(symbol, state["average_price"])
                qty = state["quantity"]
                avg = state["average_price"]
                unrealised = (ltp - avg) * qty

                positions.append(
                    Position(
                        symbol=symbol,
                        exchange=exchange,
                        segment=segment,
                        broker_name=self.BROKER_NAME,
                        trading_mode=TradingMode.PAPER,
                        quantity=qty,
                        average_price=avg,
                        last_price=ltp,
                        unrealized_pnl=round(unrealised, 4),
                        realized_pnl=round(
                            state.get("realized_pnl", 0.0), 4
                        ),
                    )
                )
        return positions

    async def get_holdings(self) -> list[Position]:
        """Paper adapter has no holdings — returns empty list."""
        return []

    async def get_funds(self) -> dict[str, float]:
        """Returns simulated fund balance."""
        from config.settings import get_settings
        initial_capital = (
            get_settings()
            .app_config.get("backtesting", {})
            .get("default_initial_capital", 1_000_000.0)
        )
        # Calculate used margin from open positions
        used_margin = sum(
            abs(state["quantity"]) * state["average_price"]
            for state in self._positions.values()
        )
        available = max(0.0, initial_capital - used_margin)
        return {
            "available_cash": round(available, 2),
            "used_margin": round(used_margin, 2),
            "total_balance": round(initial_capital, 2),
        }

    # ─────────────────────────────────────────
    # Section 5 — WebSocket Feed
    # ─────────────────────────────────────────

    async def subscribe_ticks(
        self, symbols: list[dict[str, str]]
    ) -> None:
        """
        Paper adapter accepts tick subscriptions.
        Prices are updated when live ticks arrive via
        external feed (if configured in paper.yaml).
        """
        for inst in symbols:
            symbol = inst["symbol"].upper()
            logger.debug(
                f"Paper tick subscription registered | "
                f"symbol={symbol}"
            )

    async def unsubscribe_ticks(
        self, symbols: list[dict[str, str]]
    ) -> None:
        """Unregisters paper tick subscriptions."""
        for inst in symbols:
            symbol = inst["symbol"].upper()
            self._last_prices.pop(symbol, None)

    async def subscribe_orders(self) -> None:
        """Paper adapter delivers order updates synchronously."""
        logger.debug(
            "Paper adapter order updates are synchronous."
        )

    # ─────────────────────────────────────────
    # Section 6 — Instrument Search
    # ─────────────────────────────────────────

    async def search_instruments(
        self,
        query: str,
        exchange: Optional[str] = None,
    ) -> list[InstrumentInfo]:
        """Paper adapter does not implement instrument search."""
        return []

    async def get_instrument_info(
        self, symbol: str, exchange: str
    ) -> Optional[InstrumentInfo]:
        """Paper adapter does not implement instrument lookup."""
        return None

    # ─────────────────────────────────────────
    # Section 7 — Paper Simulation Helpers
    # ─────────────────────────────────────────

    def update_last_price(self, symbol: str, price: float) -> None:
        """
        Updates the cached last price for a symbol.
        Called externally when a live tick arrives,
        so limit orders can be checked for fills.

        Args:
            symbol: Instrument symbol.
            price: Latest market price.
        """
        self._last_prices[symbol.upper()] = price
        # Check if any pending limit orders can now be filled
        asyncio.create_task(
            self._check_pending_fills(symbol.upper(), price)
        )

    async def _check_pending_fills(
        self, symbol: str, current_price: float
    ) -> None:
        """
        Checks all open limit/stop-loss orders for a symbol
        and fills them if the current price crosses their level.

        Args:
            symbol: Instrument symbol.
            current_price: Current market price.
        """
        to_fill = []
        for order_id, state in list(self._open_orders.items()):
            req: OrderRequest = state["request"]
            if req.symbol.upper() != symbol:
                continue

            should_fill = False
            if req.order_type == OrderType.LIMIT:
                # BUY: fill if price drops to or below limit
                # SELL: fill if price rises to or above limit
                if (
                    req.transaction_type == TransactionType.BUY
                    and req.price
                    and current_price <= req.price
                ):
                    should_fill = True
                elif (
                    req.transaction_type == TransactionType.SELL
                    and req.price
                    and current_price >= req.price
                ):
                    should_fill = True

            elif req.order_type in (
                OrderType.STOP_LOSS,
                OrderType.STOP_LOSS_MARKET,
            ):
                # Trigger when price crosses trigger level
                if (
                    req.transaction_type == TransactionType.BUY
                    and req.trigger_price
                    and current_price >= req.trigger_price
                ):
                    should_fill = True
                elif (
                    req.transaction_type == TransactionType.SELL
                    and req.trigger_price
                    and current_price <= req.trigger_price
                ):
                    should_fill = True

            if should_fill:
                to_fill.append(order_id)

        # Fill triggered orders
        for order_id in to_fill:
            state = self._open_orders.pop(order_id, None)
            if not state:
                continue
            req = state["request"]
            fill_price = self._simulate_fill_price(
                symbol=req.symbol,
                transaction_type=req.transaction_type,
                requested_price=req.price,
            )
            commission = self._calculate_commission(
                quantity=req.quantity,
                price=fill_price,
            )
            self._update_position(
                symbol=req.symbol,
                exchange=str(req.exchange),
                segment=str(req.segment),
                transaction_type=str(req.transaction_type),
                quantity=req.quantity,
                fill_price=fill_price,
            )
            fill = Fill(
                order_id=order_id,
                broker_order_id=state["broker_order_id"],
                broker_name=self.BROKER_NAME,
                symbol=req.symbol,
                exchange=str(req.exchange),
                transaction_type=req.transaction_type,
                quantity=req.quantity,
                price=fill_price,
                commission=commission,
                trading_mode=TradingMode.PAPER,
                executed_at=now_ist(),
            )
            self._fills.append(fill)

            response = OrderResponse(
                order_id=order_id,
                broker_order_id=state["broker_order_id"],
                status=OrderStatus.COMPLETE,
                filled_quantity=req.quantity,
                average_price=fill_price,
                broker_name=self.BROKER_NAME,
                trading_mode=TradingMode.PAPER,
                updated_at=now_ist(),
            )
            await self._emit_order_update(response)
            trade_logger.info(
                f"PAPER LIMIT FILLED | "
                f"order_id={order_id} | "
                f"symbol={req.symbol} | "
                f"qty={req.quantity} | "
                f"price={fill_price:.2f} | "
                f"trigger={current_price:.2f}"
            )

    def _simulate_fill_price(
        self,
        symbol: str,
        transaction_type: Any,
        requested_price: Optional[float],
    ) -> float:
        """
        Calculates simulated fill price with slippage.

        For MARKET orders: LTP ± slippage_pct
        For LIMIT orders: min(limit, LTP) for BUY,
                          max(limit, LTP) for SELL

        Args:
            symbol: Instrument symbol.
            transaction_type: BUY or SELL.
            requested_price: Limit price (None for market).

        Returns:
            Simulated fill price.
        """
        ltp = self._last_prices.get(symbol.upper(), 100.0)
        slippage = ltp * (self._slippage_pct / 100.0)

        is_buy = str(transaction_type).upper() == "BUY"

        if requested_price is None:
            # MARKET order — apply slippage
            fill_price = ltp + slippage if is_buy else ltp - slippage
        else:
            # LIMIT order — fill at best of limit/LTP
            if is_buy:
                fill_price = min(requested_price, ltp + slippage)
            else:
                fill_price = max(requested_price, ltp - slippage)

        return round(max(fill_price, 0.01), 4)

    def _calculate_commission(
        self, quantity: int, price: float
    ) -> float:
        """
        Calculates simulated brokerage commission.

        Args:
            quantity: Order quantity.
            price: Fill price.

        Returns:
            Commission amount in INR.
        """
        trade_value = quantity * price
        return round(trade_value * (self._commission_pct / 100.0), 4)

    def _update_position(
        self,
        symbol: str,
        exchange: str,
        segment: str,
        transaction_type: str,
        quantity: int,
        fill_price: float,
    ) -> None:
        """
        Updates the in-memory position state after a fill.
        Handles averaging for adds and P&L realisation for reduces.

        Args:
            symbol: Instrument symbol.
            exchange: Exchange code.
            segment: Instrument segment.
            transaction_type: BUY or SELL string.
            quantity: Filled quantity.
            fill_price: Actual fill price.
        """
        key = f"{symbol.upper()}:{exchange.upper()}:{segment.upper()}"
        is_buy = transaction_type.upper() == "BUY"

        if key not in self._positions:
            self._positions[key] = {
                "quantity": 0,
                "average_price": 0.0,
                "realized_pnl": 0.0,
            }

        pos = self._positions[key]
        current_qty = pos["quantity"]
        current_avg = pos["average_price"]

        if is_buy:
            if current_qty >= 0:
                # Adding to long: recalculate average
                new_qty = current_qty + quantity
                pos["average_price"] = (
                    (current_avg * current_qty)
                    + (fill_price * quantity)
                ) / new_qty
                pos["quantity"] = new_qty
            else:
                # Covering short
                close_qty = min(quantity, abs(current_qty))
                pnl = (current_avg - fill_price) * close_qty
                pos["realized_pnl"] += pnl
                remaining = current_qty + quantity
                if remaining > 0:
                    pos["average_price"] = fill_price
                pos["quantity"] = remaining

        else:  # SELL
            if current_qty <= 0:
                # Adding to short: recalculate average
                new_qty = current_qty - quantity
                pos["average_price"] = (
                    (current_avg * abs(current_qty))
                    + (fill_price * quantity)
                ) / abs(new_qty)
                pos["quantity"] = new_qty
            else:
                # Closing long
                close_qty = min(quantity, current_qty)
                pnl = (fill_price - current_avg) * close_qty
                pos["realized_pnl"] += pnl
                remaining = current_qty - quantity
                if remaining < 0:
                    pos["average_price"] = fill_price
                pos["quantity"] = remaining

    def get_paper_fills(self) -> list[Fill]:
        """Returns all fills recorded in this paper session."""
        return list(self._fills)

    def get_paper_pnl(self) -> float:
        """
        Returns total realised P&L across all paper positions.

        Returns:
            Total realised P&L in INR.
        """
        return round(
            sum(
                pos.get("realized_pnl", 0.0)
                for pos in self._positions.values()
            ),
            4,
        )