"""
NexaTrade — Breeze Connect (ICICI Direct) Adapter.

Implements AbstractBroker for Breeze Connect API.
All Breeze SDK calls are isolated to this file.
No Breeze-specific types or SDK objects ever leave this module.

SDK Docs: https://github.com/Idrees-28/BreezeConnect
Credentials read from: settings.broker_credentials("breeze")
Config read from:       settings.broker_config("breeze")
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Optional

from breeze_connect import BreezeConnect

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
    Position,
    ProductType,
    Quote,
    Segment,
    TickData,
    TradingMode,
    TransactionType,
)
from utils.logger import get_logger, get_trade_logger
from utils.time_utils import now_ist, to_ist, date_range_chunks

logger = get_logger(__name__)
trade_logger = get_trade_logger(__name__)


class BreezeAdapter(AbstractBroker):
    """
    Breeze Connect adapter for NexaTrade.

    Translates all NexaTrade broker interface calls
    into Breeze Connect SDK calls and normalises
    responses back to NexaTrade models.

    Session lifecycle:
        connect()    → BreezeConnect.generate_session()
        is_connected → get_customer_details() ping
        disconnect() → ws_disconnect()
    """

    BROKER_NAME = "breeze"

    def __init__(self) -> None:
        super().__init__(self.BROKER_NAME)
        self._client: Optional[BreezeConnect] = None
        self._subscribed_symbols: set[str] = set()

    # ─────────────────────────────────────────
    # Internal — SDK Client Builder
    # ─────────────────────────────────────────

    def _get_client(self) -> BreezeConnect:
        """Returns the Breeze client or raises if not connected."""
        if not self._client:
            raise RuntimeError(
                "BreezeAdapter not connected. "
                "Call await broker.connect() first."
            )
        return self._client

    # ─────────────────────────────────────────
    # Section 1 — Session Management
    # ─────────────────────────────────────────

    async def connect(self) -> bool:
        """
        Authenticates with Breeze Connect using credentials from settings.
        Runs the blocking Breeze SDK in a thread pool executor.

        Returns:
            True if connection successful.

        Raises:
            ConnectionError: If authentication fails.
        """
        self._set_state(BrokerConnectionState.CONNECTING)
        try:
            from config.settings import get_settings
            creds = get_settings().broker_credentials("breeze")

            api_key = creds.api_key.get_secret_value()
            api_secret = creds.api_secret.get_secret_value()
            session_token = creds.session_token.get_secret_value()

            if not all([api_key, api_secret, session_token]):
                raise ConnectionError(
                    "Breeze credentials incomplete. "
                    "Check BREEZE_API_KEY, BREEZE_API_SECRET, "
                    "BREEZE_SESSION_TOKEN in .env"
                )

            # Breeze SDK is synchronous — run in thread pool
            loop = asyncio.get_event_loop()
            self._client = await loop.run_in_executor(
                None,
                self._sync_connect,
                api_key,
                api_secret,
                session_token,
            )

            self._set_state(BrokerConnectionState.CONNECTED)
            logger.info(
                f"Breeze connected | "
                f"api_key={api_key[:8]}..."
            )
            return True

        except ConnectionError:
            self._set_state(BrokerConnectionState.ERROR)
            raise
        except Exception as exc:
            self._set_state(BrokerConnectionState.ERROR)
            raise ConnectionError(
                f"Breeze connection failed: {exc}"
            ) from exc

    def _sync_connect(
        self,
        api_key: str,
        api_secret: str,
        session_token: str,
    ) -> BreezeConnect:
        """
        Synchronous Breeze SDK initialisation.
        Called in thread pool from connect().
        """
        client = BreezeConnect(api_key=api_key)
        client.generate_session(
            api_secret=api_secret,
            session_token=session_token,
        )
        return client

    async def disconnect(self) -> None:
        """Disconnects WebSocket feed and closes Breeze session."""
        if self._client:
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None, self._client.ws_disconnect
                )
            except Exception as exc:
                logger.warning(f"Breeze disconnect warning: {exc}")
            finally:
                self._client = None
                self._subscribed_symbols.clear()
        self._set_state(BrokerConnectionState.DISCONNECTED)
        logger.info("Breeze disconnected.")

    async def is_connected(self) -> bool:
        """Pings Breeze by fetching customer details."""
        if not self._client:
            return False
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self._client.get_customer_details(
                    api_session=self._client.session_key
                ),
            )
            return (
                result is not None
                and result.get("Status") == 200
            )
        except Exception:
            return False

    async def get_info(self) -> BrokerInfo:
        """Returns Breeze broker capabilities metadata."""
        from config.settings import get_settings
        cfg = get_settings().broker_config("breeze")
        ws_cfg = cfg.get("broker", {}).get("websocket", {})

        return BrokerInfo(
            name=self.BROKER_NAME,
            display_name="ICICI Direct — Breeze Connect",
            version="1.0.0",
            supports_websocket=True,
            supports_historical_data=True,
            supports_paper_trading=False,
            supports_options=True,
            supports_futures=True,
            supports_commodity=True,
            supports_order_modify=True,
            supports_bracket_orders=False,
            max_ws_subscriptions=ws_cfg.get("max_subscriptions", 50),
            connection_state=self._connection_state,
            is_authenticated=(
                self._connection_state == BrokerConnectionState.CONNECTED
            ),
        )

    # ─────────────────────────────────────────
    # Section 2 — Order Management
    # ─────────────────────────────────────────

    async def place_order(self, request: OrderRequest) -> OrderResponse:
        """
        Places an order via Breeze Connect.
        Translates OrderRequest → Breeze params → OrderResponse.
        """
        client = self._get_client()
        loop = asyncio.get_event_loop()

        # Translate NexaTrade → Breeze params
        exchange_code = self._get_exchange_code(request.exchange)
        order_type = self._get_order_type_code(request.order_type)
        transaction_type = self._get_transaction_type_code(
            request.transaction_type
        )

        # Map product type
        product_map = {
            ProductType.DELIVERY: "cash",
            ProductType.INTRADAY: "margin",
            ProductType.FUTURES:  "futures",
            ProductType.OPTIONS:  "options",
        }
        product_type = product_map.get(
            request.product_type, "margin"
        )

        try:
            result = await loop.run_in_executor(
                None,
                lambda: client.place_order(
                    stock_code=request.symbol,
                    exchange_code=exchange_code,
                    product=product_type,
                    action=transaction_type,
                    order_type=order_type,
                    quantity=str(request.quantity),
                    price=str(request.price or 0),
                    stoploss=str(request.trigger_price or 0),
                    right="others",
                    validity="day",
                ),
            )
        except Exception as exc:
            raise RuntimeError(
                f"Breeze place_order failed: {exc}"
            ) from exc

        # Normalise response
        if result and result.get("Status") == 200:
            order_data = result.get("Success", {}) or {}
            broker_order_id = str(
                order_data.get("order_id", "")
            )
            response = OrderResponse(
                order_id=request.order_id,
                broker_order_id=broker_order_id,
                status=OrderStatus.OPEN,
                message="Order placed successfully",
                broker_name=self.BROKER_NAME,
                trading_mode=request.trading_mode,
                placed_at=now_ist(),
            )
            trade_logger.info(
                f"ORDER PLACED | "
                f"nexatrade_id={request.order_id} | "
                f"broker_id={broker_order_id} | "
                f"symbol={request.symbol} | "
                f"qty={request.quantity} | "
                f"type={request.transaction_type} | "
                f"price={request.price}"
            )
        else:
            error_msg = str(result.get("Error", "Unknown error"))
            response = OrderResponse(
                order_id=request.order_id,
                broker_order_id=None,
                status=OrderStatus.REJECTED,
                message=error_msg,
                rejection_reason=error_msg,
                broker_name=self.BROKER_NAME,
                trading_mode=request.trading_mode,
                placed_at=now_ist(),
            )
            trade_logger.warning(
                f"ORDER REJECTED | "
                f"nexatrade_id={request.order_id} | "
                f"symbol={request.symbol} | "
                f"reason={error_msg}"
            )

        return response

    async def modify_order(
        self, request: OrderModifyRequest
    ) -> OrderResponse:
        """Modifies an existing Breeze order."""
        client = self._get_client()
        loop = asyncio.get_event_loop()

        try:
            result = await loop.run_in_executor(
                None,
                lambda: client.modify_order(
                    order_id=request.broker_order_id,
                    quantity=str(request.quantity or ""),
                    price=str(request.price or 0),
                    stoploss=str(request.trigger_price or 0),
                    validity="day",
                ),
            )
        except Exception as exc:
            raise RuntimeError(
                f"Breeze modify_order failed: {exc}"
            ) from exc

        status = (
            OrderStatus.OPEN
            if result and result.get("Status") == 200
            else OrderStatus.REJECTED
        )
        return OrderResponse(
            order_id=request.order_id,
            broker_order_id=request.broker_order_id,
            status=status,
            message=str(result.get("Success") or result.get("Error", "")),
            broker_name=self.BROKER_NAME,
            trading_mode=TradingMode.LIVE,
            updated_at=now_ist(),
        )

    async def cancel_order(
        self, order_id: str, broker_order_id: str
    ) -> OrderResponse:
        """Cancels an open Breeze order."""
        client = self._get_client()
        loop = asyncio.get_event_loop()

        try:
            result = await loop.run_in_executor(
                None,
                lambda: client.cancel_order(
                    order_id=broker_order_id
                ),
            )
        except Exception as exc:
            raise RuntimeError(
                f"Breeze cancel_order failed: {exc}"
            ) from exc

        status = (
            OrderStatus.CANCELLED
            if result and result.get("Status") == 200
            else OrderStatus.REJECTED
        )
        trade_logger.info(
            f"ORDER CANCELLED | "
            f"nexatrade_id={order_id} | "
            f"broker_id={broker_order_id} | "
            f"status={status}"
        )
        return OrderResponse(
            order_id=order_id,
            broker_order_id=broker_order_id,
            status=status,
            message=str(result.get("Success") or result.get("Error", "")),
            broker_name=self.BROKER_NAME,
            trading_mode=TradingMode.LIVE,
            updated_at=now_ist(),
        )

    async def get_order_status(
        self, order_id: str, broker_order_id: str
    ) -> OrderResponse:
        """Fetches current status of a Breeze order."""
        client = self._get_client()
        loop = asyncio.get_event_loop()

        try:
            result = await loop.run_in_executor(
                None,
                lambda: client.get_order_detail(
                    exchange_code="NSE",
                    order_id=broker_order_id,
                ),
            )
        except Exception as exc:
            raise RuntimeError(
                f"Breeze get_order_status failed: {exc}"
            ) from exc

        order_data = (
            (result.get("Success") or [{}])[0]
            if result and result.get("Status") == 200
            else {}
        )

        status_map = {
            "Executed":   OrderStatus.COMPLETE,
            "Ordered":    OrderStatus.OPEN,
            "Cancelled":  OrderStatus.CANCELLED,
            "Rejected":   OrderStatus.REJECTED,
            "Modified":   OrderStatus.MODIFIED,
        }
        raw_status = order_data.get("order_status", "")
        status = status_map.get(raw_status, OrderStatus.OPEN)

        return OrderResponse(
            order_id=order_id,
            broker_order_id=broker_order_id,
            status=status,
            filled_quantity=int(
                order_data.get("executed_quantity", 0) or 0
            ),
            average_price=float(
                order_data.get("average_price", 0.0) or 0.0
            ),
            broker_name=self.BROKER_NAME,
            trading_mode=TradingMode.LIVE,
            updated_at=now_ist(),
        )

    async def get_order_history(
        self,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> list[OrderResponse]:
        """Returns Breeze order history for a date range."""
        client = self._get_client()
        loop = asyncio.get_event_loop()

        try:
            result = await loop.run_in_executor(
                None,
                lambda: client.get_order_list(
                    exchange_code="NSE",
                    from_date=from_date or "",
                    to_date=to_date or "",
                ),
            )
        except Exception as exc:
            raise RuntimeError(
                f"Breeze get_order_history failed: {exc}"
            ) from exc

        orders: list[OrderResponse] = []
        order_list = result.get("Success", []) or []
        status_map = {
            "Executed":  OrderStatus.COMPLETE,
            "Ordered":   OrderStatus.OPEN,
            "Cancelled": OrderStatus.CANCELLED,
            "Rejected":  OrderStatus.REJECTED,
        }
        for item in order_list:
            status = status_map.get(
                item.get("order_status", ""), OrderStatus.OPEN
            )
            orders.append(
                OrderResponse(
                    order_id=str(item.get("order_id", "")),
                    broker_order_id=str(item.get("order_id", "")),
                    status=status,
                    filled_quantity=int(
                        item.get("executed_quantity", 0) or 0
                    ),
                    average_price=float(
                        item.get("average_price", 0.0) or 0.0
                    ),
                    broker_name=self.BROKER_NAME,
                    trading_mode=TradingMode.LIVE,
                )
            )
        return orders

    # ─────────────────────────────────────────
    # Section 3 — Market Data
    # ─────────────────────────────────────────

    async def get_quote(self, symbol: str, exchange: str) -> Quote:
        """Fetches a live quote from Breeze."""
        client = self._get_client()
        loop = asyncio.get_event_loop()
        exchange_code = self._get_exchange_code(exchange)

        try:
            result = await loop.run_in_executor(
                None,
                lambda: client.get_quotes(
                    stock_code=symbol.upper(),
                    exchange_code=exchange_code,
                    right="others",
                    product_type="cash",
                    expiry_date="",
                    strike_price="",
                ),
            )
        except Exception as exc:
            raise RuntimeError(
                f"Breeze get_quote failed for {symbol}: {exc}"
            ) from exc

        data = (
            (result.get("Success") or [{}])[0]
            if result and result.get("Status") == 200
            else {}
        )

        ltp = float(data.get("ltp", 0.0) or 0.0)
        prev_close = float(data.get("previous_close", 0.0) or 0.0)
        change = ltp - prev_close
        change_pct = (change / prev_close * 100) if prev_close else 0.0

        return Quote(
            symbol=symbol.upper(),
            exchange=exchange.upper(),
            broker_name=self.BROKER_NAME,
            last_price=ltp,
            open=float(data.get("open", 0.0) or 0.0),
            high=float(data.get("high", 0.0) or 0.0),
            low=float(data.get("low", 0.0) or 0.0),
            close=prev_close,
            bid=float(data.get("best_bid_price", 0.0) or 0.0),
            ask=float(data.get("best_offer_price", 0.0) or 0.0),
            volume=int(data.get("total_quantity_traded", 0) or 0),
            change=round(change, 4),
            change_pct=round(change_pct, 4),
            timestamp=now_ist(),
        )

    async def get_quotes(
        self, instruments: list[dict[str, str]]
    ) -> list[Quote]:
        """Fetches quotes for multiple instruments sequentially."""
        quotes: list[Quote] = []
        for inst in instruments:
            try:
                q = await self.get_quote(
                    inst["symbol"], inst["exchange"]
                )
                quotes.append(q)
            except Exception as exc:
                logger.warning(
                    f"Quote fetch failed | "
                    f"symbol={inst.get('symbol')} | error={exc}"
                )
        return quotes

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
        Fetches OHLCV historical data from Breeze.
        Handles date-range pagination automatically
        using the per-interval limits from breeze.yaml.
        """
        from config.settings import get_settings
        cfg = get_settings().broker_config("breeze")
        hist_limits: dict = (
            cfg.get("broker", {})
            .get("historical", {})
            .get("max_days_per_request", {})
        )
        max_days = hist_limits.get(interval, 30)

        exchange_code = self._get_exchange_code(exchange)
        interval_code = self._get_interval_code(interval)

        # Parse date strings to datetime for chunking
        from datetime import datetime as dt
        start_dt = dt.strptime(from_date, "%Y-%m-%d")
        end_dt = dt.strptime(to_date, "%Y-%m-%d")

        chunks = date_range_chunks(start_dt, end_dt, max_days)
        client = self._get_client()
        loop = asyncio.get_event_loop()
        all_candles: list[OHLCV] = []

        for chunk_start, chunk_end in chunks:
            try:
                result = await loop.run_in_executor(
                    None,
                    lambda cs=chunk_start, ce=chunk_end: (
                        client.get_historical_data_v2(
                            interval=interval_code,
                            from_date=cs.strftime(
                                "%Y-%m-%dT%H:%M:%S.000Z"
                            ),
                            to_date=ce.strftime(
                                "%Y-%m-%dT%H:%M:%S.000Z"
                            ),
                            stock_code=symbol.upper(),
                            exchange_code=exchange_code,
                            product_type=(
                                "futures" if segment == "FUT"
                                else "cash"
                            ),
                        )
                    ),
                )

                candle_list = result.get("Success", []) or []
                for row in candle_list:
                    try:
                        ts = to_ist(
                            datetime.strptime(
                                row["datetime"],
                                "%Y-%m-%d %H:%M:%S",
                            )
                        )
                        all_candles.append(
                            OHLCV(
                                datetime=ts,
                                open=float(row.get("open", 0)),
                                high=float(row.get("high", 0)),
                                low=float(row.get("low", 0)),
                                close=float(row.get("close", 0)),
                                volume=float(row.get("volume", 0)),
                                symbol=symbol.upper(),
                                exchange=exchange.upper(),
                                interval=interval,
                                broker_name=self.BROKER_NAME,
                            )
                        )
                    except Exception as row_exc:
                        logger.warning(
                            f"Skipping malformed candle row: "
                            f"{row_exc}"
                        )
            except Exception as chunk_exc:
                logger.error(
                    f"Historical data chunk failed | "
                    f"symbol={symbol} | "
                    f"chunk={chunk_start}→{chunk_end} | "
                    f"error={chunk_exc}"
                )

        all_candles.sort(key=lambda c: c.datetime)
        logger.debug(
            f"Historical data fetched | broker=breeze | "
            f"symbol={symbol} | interval={interval} | "
            f"candles={len(all_candles)}"
        )
        return all_candles

    # ─────────────────────────────────────────
    # Section 4 — Portfolio
    # ─────────────────────────────────────────

    async def get_positions(self) -> list[Position]:
        """Returns all open positions from Breeze."""
        client = self._get_client()
        loop = asyncio.get_event_loop()

        try:
            result = await loop.run_in_executor(
                None, client.get_portfolio_positions
            )
        except Exception as exc:
            raise RuntimeError(
                f"Breeze get_positions failed: {exc}"
            ) from exc

        positions: list[Position] = []
        pos_list = result.get("Success", []) or []
        for item in pos_list:
            try:
                qty = int(item.get("quantity", 0) or 0)
                positions.append(
                    Position(
                        symbol=str(item.get("stock_code", "")),
                        exchange=item.get("exchange_code", "NSE"),
                        segment=Segment.EQ,
                        broker_name=self.BROKER_NAME,
                        trading_mode=TradingMode.LIVE,
                        quantity=qty,
                        average_price=float(
                            item.get("average_cost", 0.0) or 0.0
                        ),
                        last_price=float(
                            item.get("ltp", 0.0) or 0.0
                        ),
                        unrealized_pnl=float(
                            item.get("unrealized_profit", 0.0) or 0.0
                        ),
                        realized_pnl=float(
                            item.get("realized_profit", 0.0) or 0.0
                        ),
                    )
                )
            except Exception as item_exc:
                logger.warning(
                    f"Skipping malformed position: {item_exc}"
                )
        return positions

    async def get_holdings(self) -> list[Position]:
        """Returns delivery holdings from Breeze."""
        client = self._get_client()
        loop = asyncio.get_event_loop()

        try:
            result = await loop.run_in_executor(
                None, client.get_portfolio_holdings
            )
        except Exception as exc:
            raise RuntimeError(
                f"Breeze get_holdings failed: {exc}"
            ) from exc

        holdings: list[Position] = []
        hold_list = result.get("Success", []) or []
        for item in hold_list:
            try:
                holdings.append(
                    Position(
                        symbol=str(item.get("stock_code", "")),
                        exchange=item.get("exchange_code", "NSE"),
                        segment=Segment.EQ,
                        broker_name=self.BROKER_NAME,
                        trading_mode=TradingMode.LIVE,
                        quantity=int(
                            item.get("quantity", 0) or 0
                        ),
                        average_price=float(
                            item.get("average_cost", 0.0) or 0.0
                        ),
                        last_price=float(
                            item.get("ltp", 0.0) or 0.0
                        ),
                    )
                )
            except Exception as item_exc:
                logger.warning(
                    f"Skipping malformed holding: {item_exc}"
                )
        return holdings

    async def get_funds(self) -> dict[str, float]:
        """Returns available margin and funds from Breeze."""
        client = self._get_client()
        loop = asyncio.get_event_loop()

        try:
            result = await loop.run_in_executor(
                None, client.get_funds
            )
        except Exception as exc:
            raise RuntimeError(
                f"Breeze get_funds failed: {exc}"
            ) from exc

        data = (
            (result.get("Success") or [{}])[0]
            if result and result.get("Status") == 200
            else {}
        )
        available = float(data.get("net_available_balance", 0.0) or 0.0)
        used = float(data.get("margin_utilised", 0.0) or 0.0)
        return {
            "available_cash": available,
            "used_margin": used,
            "total_balance": available + used,
        }

    # ─────────────────────────────────────────
    # Section 5 — WebSocket Feed
    # ─────────────────────────────────────────

    async def subscribe_ticks(
        self, symbols: list[dict[str, str]]
    ) -> None:
        """
        Subscribes to Breeze WebSocket tick feed.
        Sets up on_ticks callback → normalises to TickData
        → emits via self._emit_tick().
        """
        client = self._get_client()
        loop = asyncio.get_event_loop()

        def _on_ticks(tick_data: dict) -> None:
            """Breeze WebSocket callback — normalises tick data."""
            try:
                tick = TickData(
                    symbol=str(
                        tick_data.get("stock_code", "UNKNOWN")
                    ),
                    exchange=str(
                        tick_data.get("exchange_code", "NSE")
                    ),
                    broker_name=self.BROKER_NAME,
                    last_price=float(
                        tick_data.get("ltp", 0.0) or 0.0
                    ),
                    bid=float(
                        tick_data.get("best_bid_price", 0.0) or 0.0
                    ),
                    ask=float(
                        tick_data.get("best_offer_price", 0.0) or 0.0
                    ),
                    volume=int(
                        tick_data.get("total_quantity_traded", 0) or 0
                    ),
                    timestamp=now_ist(),
                )
                # Schedule async emit on the event loop
                asyncio.run_coroutine_threadsafe(
                    self._emit_tick(tick), loop
                )
            except Exception as exc:
                logger.error(f"Tick normalisation error: {exc}")

        # Register callback with Breeze WebSocket
        await loop.run_in_executor(
            None,
            lambda: client.on_ticks.append(_on_ticks),
        )

        # Subscribe each symbol
        for inst in symbols:
            symbol = inst["symbol"].upper()
            exchange_code = self._get_exchange_code(
                inst.get("exchange", "NSE")
            )
            key = f"{symbol}:{exchange_code}"
            if key not in self._subscribed_symbols:
                await loop.run_in_executor(
                    None,
                    lambda s=symbol, e=exchange_code: (
                        client.subscribe_feeds(
                            exchange_code=e,
                            stock_code=s,
                            product_type="cash",
                            get_exchange_quotes=True,
                            get_market_depth=False,
                        )
                    ),
                )
                self._subscribed_symbols.add(key)
                logger.debug(
                    f"Subscribed to tick feed | "
                    f"symbol={symbol} | exchange={exchange_code}"
                )

    async def unsubscribe_ticks(
        self, symbols: list[dict[str, str]]
    ) -> None:
        """Unsubscribes from Breeze tick feed for given symbols."""
        client = self._get_client()
        loop = asyncio.get_event_loop()

        for inst in symbols:
            symbol = inst["symbol"].upper()
            exchange_code = self._get_exchange_code(
                inst.get("exchange", "NSE")
            )
            key = f"{symbol}:{exchange_code}"
            if key in self._subscribed_symbols:
                await loop.run_in_executor(
                    None,
                    lambda s=symbol, e=exchange_code: (
                        client.unsubscribe_feeds(
                            exchange_code=e,
                            stock_code=s,
                            product_type="cash",
                        )
                    ),
                )
                self._subscribed_symbols.discard(key)
                logger.debug(
                    f"Unsubscribed from tick feed | "
                    f"symbol={symbol}"
                )

    async def subscribe_orders(self) -> None:
        """
        Subscribes to Breeze order update notifications.
        Note: Breeze delivers order updates via the same
        WebSocket connection as tick feed.
        """
        logger.info(
            "Breeze order updates delivered via tick WebSocket."
        )

    # ─────────────────────────────────────────
    # Section 6 — Instrument Search
    # ─────────────────────────────────────────

    async def search_instruments(
        self,
        query: str,
        exchange: Optional[str] = None,
    ) -> list[InstrumentInfo]:
        """
        Searches Breeze instrument master for matching symbols.
        Note: Breeze does not expose a search API directly.
        This searches a locally cached instrument master CSV.
        """
        logger.warning(
            "Breeze instrument search requires local master CSV. "
            "Returning empty list — implement master CSV loader."
        )
        return []

    async def get_instrument_info(
        self, symbol: str, exchange: str
    ) -> Optional[InstrumentInfo]:
        """Returns instrument details from Breeze master."""
        logger.warning(
            f"Breeze get_instrument_info not implemented | "
            f"symbol={symbol}"
        )
        return None