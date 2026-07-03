"""
NexaTrade — AbstractBroker Interface.

Every broker adapter MUST inherit from AbstractBroker and
implement ALL abstract methods. The rest of NexaTrade
only ever interacts with this interface — never with
any broker SDK directly.

Contract rules:
    1. Every method is an async coroutine.
    2. All inputs and outputs use models from brokers.models.
    3. No broker-specific types ever escape the adapter.
    4. Adapters handle all SDK exceptions internally and
       re-raise as standard Python exceptions with clear messages.
    5. Adapters must be stateless between calls except for
       the connection/session object stored in self.

Lifecycle contract:
    await broker.connect()          → establish session
    await broker.is_connected()     → health check
    await broker.disconnect()       → clean shutdown

Adding a new broker:
    class MyBrokerAdapter(AbstractBroker):
        async def connect(self): ...
        async def disconnect(self): ...
        async def get_quote(self, symbol, exchange): ...
        # ... implement all abstract methods

That's it. Zero changes anywhere else.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

from brokers.models import (
    BrokerConnectionState,
    BrokerInfo,
    Fill,
    OHLCV,
    InstrumentInfo,
    OrderModifyRequest,
    OrderRequest,
    OrderResponse,
    Position,
    Quote,
    TickData,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class AbstractBroker(ABC):
    """
    NexaTrade Broker Interface — the plug-and-play contract.

    Every broker adapter inherits this class and implements
    all abstract methods. The core engine only calls methods
    defined here — it never imports any broker SDK.

    Attributes:
        broker_name: Broker identifier string (e.g. "breeze").
        _connection_state: Current connection state.
        _tick_callbacks: List of async callbacks for tick events.
        _order_callbacks: List of async callbacks for order events.
    """

    def __init__(self, broker_name: str) -> None:
        self.broker_name: str = broker_name
        self._connection_state: BrokerConnectionState = (
            BrokerConnectionState.DISCONNECTED
        )
        self._tick_callbacks: list[Callable] = []
        self._order_callbacks: list[Callable] = []
        self._logger = get_logger(f"brokers.{broker_name}")

    # ─────────────────────────────────────────
    # Section 1 — Session Management (Abstract)
    # ─────────────────────────────────────────

    @abstractmethod
    async def connect(self) -> bool:
        """
        Establishes a broker session / authenticates.

        Implementations should:
        - Load credentials from settings.broker_credentials()
        - Initialise the broker SDK client
        - Set self._connection_state = CONNECTED on success
        - Return True on success, False on failure

        Returns:
            True if connection was successful.

        Raises:
            ConnectionError: If authentication fails.

        Example implementation:
            creds = get_settings().broker_credentials("breeze")
            self._client = BreezeConnect(api_key=creds.api_key.get_secret_value())
            self._client.generate_session(...)
            self._connection_state = BrokerConnectionState.CONNECTED
            return True
        """
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """
        Gracefully closes the broker session.

        Implementations should:
        - Unsubscribe all WebSocket feeds
        - Close the SDK connection
        - Set self._connection_state = DISCONNECTED

        Example implementation:
            await self._client.ws_disconnect()
            self._connection_state = BrokerConnectionState.DISCONNECTED
        """
        ...

    @abstractmethod
    async def is_connected(self) -> bool:
        """
        Returns True if the broker session is active and healthy.
        Should perform a lightweight liveness check (e.g. fetch profile).

        Returns:
            True if connected and authenticated.

        Example implementation:
            try:
                profile = await self._client.get_customer_details(...)
                return profile is not None
            except Exception:
                return False
        """
        ...

    @abstractmethod
    async def get_info(self) -> BrokerInfo:
        """
        Returns metadata and capabilities for this broker.

        Returns:
            BrokerInfo model describing this adapter.
        """
        ...

    # ─────────────────────────────────────────
    # Section 2 — Order Management (Abstract)
    # ─────────────────────────────────────────

    @abstractmethod
    async def place_order(self, request: OrderRequest) -> OrderResponse:
        """
        Places a new order via the broker.

        Implementations must:
        - Translate OrderRequest → broker SDK params
        - Call the broker SDK's order placement method
        - Translate the response → OrderResponse
        - Never raise SDK-specific exceptions to callers

        Args:
            request: Broker-agnostic OrderRequest model.

        Returns:
            OrderResponse with broker_order_id and status.

        Raises:
            RuntimeError: If order placement fails.
        """
        ...

    @abstractmethod
    async def modify_order(
        self, request: OrderModifyRequest
    ) -> OrderResponse:
        """
        Modifies an existing open or partial order.

        Args:
            request: OrderModifyRequest with new parameters.

        Returns:
            Updated OrderResponse.

        Raises:
            RuntimeError: If modification fails.
        """
        ...

    @abstractmethod
    async def cancel_order(
        self, order_id: str, broker_order_id: str
    ) -> OrderResponse:
        """
        Cancels an open order.

        Args:
            order_id: NexaTrade internal order ID.
            broker_order_id: Broker's own order identifier.

        Returns:
            OrderResponse with CANCELLED status.

        Raises:
            RuntimeError: If cancellation fails.
        """
        ...

    @abstractmethod
    async def get_order_status(
        self, order_id: str, broker_order_id: str
    ) -> OrderResponse:
        """
        Fetches the current status of a placed order.

        Args:
            order_id: NexaTrade internal order ID.
            broker_order_id: Broker's order identifier.

        Returns:
            Latest OrderResponse for this order.
        """
        ...

    @abstractmethod
    async def get_order_history(
        self,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> list[OrderResponse]:
        """
        Returns the broker's order history for a date range.

        Args:
            from_date: Start date string (YYYY-MM-DD).
            to_date: End date string (YYYY-MM-DD).

        Returns:
            List of OrderResponse for all orders in range.
        """
        ...

    # ─────────────────────────────────────────
    # Section 3 — Market Data (Abstract)
    # ─────────────────────────────────────────

    @abstractmethod
    async def get_quote(
        self, symbol: str, exchange: str
    ) -> Quote:
        """
        Fetches a live market quote for an instrument.

        Args:
            symbol: Instrument symbol (e.g. "RELIANCE").
            exchange: Exchange code (e.g. "NSE").

        Returns:
            Normalised Quote model.

        Raises:
            RuntimeError: If quote fetch fails.
        """
        ...

    @abstractmethod
    async def get_quotes(
        self, instruments: list[dict[str, str]]
    ) -> list[Quote]:
        """
        Fetches live quotes for multiple instruments in one call.

        Args:
            instruments: List of {"symbol": ..., "exchange": ...} dicts.

        Returns:
            List of Quote models in the same order as input.
        """
        ...

    @abstractmethod
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
        Fetches OHLCV historical data for an instrument.

        Implementations must handle pagination internally
        if the broker limits per-request date ranges.
        Use utils.time_utils.date_range_chunks() for this.

        Args:
            symbol: Instrument symbol.
            exchange: Exchange code.
            interval: Candle interval (1minute/5minute/1day etc).
            from_date: Start date (YYYY-MM-DD).
            to_date: End date (YYYY-MM-DD).
            segment: Instrument segment (EQ/FUT/OPT).

        Returns:
            List of OHLCV candles sorted by datetime ascending.
        """
        ...

    # ─────────────────────────────────────────
    # Section 4 — Portfolio (Abstract)
    # ─────────────────────────────────────────

    @abstractmethod
    async def get_positions(self) -> list[Position]:
        """
        Fetches all open positions from the broker.

        Returns:
            List of normalised Position models.
            Empty list if no positions.
        """
        ...

    @abstractmethod
    async def get_holdings(self) -> list[Position]:
        """
        Fetches delivery holdings (long-term equity).

        Returns:
            List of Position models for held stocks.
        """
        ...

    @abstractmethod
    async def get_funds(self) -> dict[str, float]:
        """
        Returns available and used margin / funds.

        Returns:
            Dict with at minimum:
                available_cash   → float (INR)
                used_margin      → float (INR)
                total_balance    → float (INR)
        """
        ...

    # ─────────────────────────────────────────
    # Section 5 — WebSocket Feed (Abstract)
    # ─────────────────────────────────────────

    @abstractmethod
    async def subscribe_ticks(
        self, symbols: list[dict[str, str]]
    ) -> None:
        """
        Subscribes to live tick feed for a list of instruments.
        Tick data is delivered via registered tick callbacks.

        The adapter must call self._emit_tick(tick) for each
        received tick to propagate data to all subscribers.

        Args:
            symbols: List of {"symbol": ..., "exchange": ...} dicts.

        Example implementation:
            self._client.subscribe_feeds(
                exchange_code="NSE",
                stock_code="RELIANCE",
                product_type="cash",
                get_exchange_quotes=True,
                get_market_depth=False,
            )
        """
        ...

    @abstractmethod
    async def unsubscribe_ticks(
        self, symbols: list[dict[str, str]]
    ) -> None:
        """
        Unsubscribes from live tick feed for given instruments.

        Args:
            symbols: List of {"symbol": ..., "exchange": ...} dicts.
        """
        ...

    @abstractmethod
    async def subscribe_orders(self) -> None:
        """
        Subscribes to broker order update notifications.
        Order updates are delivered via registered order callbacks.

        The adapter must call self._emit_order_update(response)
        for each received order update.
        """
        ...

    # ─────────────────────────────────────────
    # Section 6 — Instrument Search (Abstract)
    # ─────────────────────────────────────────

    @abstractmethod
    async def search_instruments(
        self,
        query: str,
        exchange: Optional[str] = None,
    ) -> list[InstrumentInfo]:
        """
        Searches for instruments by name or symbol.

        Args:
            query: Search query string.
            exchange: Optional exchange filter.

        Returns:
            List of matching InstrumentInfo models.
        """
        ...

    @abstractmethod
    async def get_instrument_info(
        self, symbol: str, exchange: str
    ) -> Optional[InstrumentInfo]:
        """
        Returns detailed information for a specific instrument.

        Args:
            symbol: Instrument symbol.
            exchange: Exchange code.

        Returns:
            InstrumentInfo or None if not found.
        """
        ...

    # ─────────────────────────────────────────
    # Section 7 — Callback Registration (Concrete)
    # These are final — adapters do NOT override these.
    # ─────────────────────────────────────────

    def register_tick_callback(self, callback: Callable) -> None:
        """
        Registers an async callback for live tick events.
        Multiple callbacks can be registered (fan-out).

        Args:
            callback: Async function accepting a TickData argument.

        Example:
            async def on_tick(tick: TickData):
                await strategy.process_tick(tick)

            broker.register_tick_callback(on_tick)
        """
        if callback not in self._tick_callbacks:
            self._tick_callbacks.append(callback)
            self._logger.debug(
                f"Tick callback registered | fn={callback.__name__}"
            )

    def register_order_callback(self, callback: Callable) -> None:
        """
        Registers an async callback for order update events.

        Args:
            callback: Async function accepting an OrderResponse argument.
        """
        if callback not in self._order_callbacks:
            self._order_callbacks.append(callback)
            self._logger.debug(
                f"Order callback registered | fn={callback.__name__}"
            )

    def unregister_tick_callback(self, callback: Callable) -> None:
        """Removes a previously registered tick callback."""
        if callback in self._tick_callbacks:
            self._tick_callbacks.remove(callback)

    def unregister_order_callback(self, callback: Callable) -> None:
        """Removes a previously registered order callback."""
        if callback in self._order_callbacks:
            self._order_callbacks.remove(callback)

    # ─────────────────────────────────────────
    # Section 8 — Internal Emit Helpers (Concrete)
    # Called by adapters to propagate events.
    # ─────────────────────────────────────────

    async def _emit_tick(self, tick: TickData) -> None:
        """
        Emits a tick event to all registered callbacks.
        Called internally by adapters from their WebSocket handlers.

        Args:
            tick: Normalised TickData model.
        """
        for callback in self._tick_callbacks:
            try:
                await callback(tick)
            except Exception as exc:
                self._logger.error(
                    f"Tick callback error | "
                    f"fn={callback.__name__} | error={exc}"
                )

    async def _emit_order_update(self, response: OrderResponse) -> None:
        """
        Emits an order update to all registered callbacks.
        Called internally by adapters from their order feed handlers.

        Args:
            response: Updated OrderResponse model.
        """
        for callback in self._order_callbacks:
            try:
                await callback(response)
            except Exception as exc:
                self._logger.error(
                    f"Order callback error | "
                    f"fn={callback.__name__} | error={exc}"
                )

    # ─────────────────────────────────────────
    # Section 9 — Connection State Helpers (Concrete)
    # ─────────────────────────────────────────

    @property
    def connection_state(self) -> BrokerConnectionState:
        """Returns the current connection state."""
        return self._connection_state

    @property
    def is_paper(self) -> bool:
        """Returns True if this is a paper trading adapter."""
        return self.broker_name == "paper"

    def _set_state(self, state: BrokerConnectionState) -> None:
        """
        Updates the connection state and logs the transition.
        Used internally by adapters during connect/disconnect.

        Args:
            state: New BrokerConnectionState.
        """
        old_state = self._connection_state
        self._connection_state = state
        if old_state != state:
            self._logger.info(
                f"Broker state transition | "
                f"broker={self.broker_name} | "
                f"{old_state} → {state}"
            )

    # ─────────────────────────────────────────
    # Section 10 — Map Helpers (Concrete)
    # Shared translation helpers for all adapters.
    # ─────────────────────────────────────────

    def _get_exchange_code(self, exchange: str) -> str:
        """
        Translates NexaTrade exchange name to broker-specific code.
        Reads from config/brokers/{broker_name}.yaml exchange_map.

        Args:
            exchange: NexaTrade exchange identifier.

        Returns:
            Broker-specific exchange code string.
        """
        from config.settings import get_settings
        cfg = get_settings().broker_config(self.broker_name)
        exchange_map: dict = (
            cfg.get("broker", {}).get("exchange_map", {})
        )
        return exchange_map.get(exchange.upper(), exchange.upper())

    def _get_order_type_code(self, order_type: str) -> str:
        """
        Translates NexaTrade OrderType to broker-specific string.
        Reads from config/brokers/{broker_name}.yaml order_type_map.
        """
        from config.settings import get_settings
        cfg = get_settings().broker_config(self.broker_name)
        order_type_map: dict = (
            cfg.get("broker", {}).get("order_type_map", {})
        )
        return order_type_map.get(order_type.upper(), order_type.upper())

    def _get_transaction_type_code(self, txn_type: str) -> str:
        """
        Translates NexaTrade TransactionType to broker-specific string.
        Reads from config/brokers/{broker_name}.yaml transaction_type_map.
        """
        from config.settings import get_settings
        cfg = get_settings().broker_config(self.broker_name)
        txn_map: dict = (
            cfg.get("broker", {}).get("transaction_type_map", {})
        )
        return txn_map.get(txn_type.upper(), txn_type.upper())

    def _get_interval_code(self, interval: str) -> str:
        """
        Translates NexaTrade interval string to broker-specific string.
        Reads from config/brokers/{broker_name}.yaml interval_map.
        """
        from config.settings import get_settings
        cfg = get_settings().broker_config(self.broker_name)
        interval_map: dict = (
            cfg.get("broker", {}).get("interval_map", {})
        )
        return interval_map.get(interval, interval)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"broker={self.broker_name!r}, "
            f"state={self._connection_state!r})"
        )