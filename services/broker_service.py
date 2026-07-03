"""
NexaTrade — Broker Service.

The BrokerService is the single application-level
owner of the active broker adapter instance.
It manages the full broker lifecycle and acts as
the gateway for all broker operations.

Architecture:
    Strategy Engine → BrokerService → AbstractBroker → Adapter

This replaces any previous breeze_client.py or similar files.
No code outside brokers/ and services/ should ever hold
a direct reference to a broker adapter.

Lifecycle:
    service = BrokerService()
    await service.start()           → connect + subscribe
    broker = service.broker         → get the active adapter
    await service.stop()            → disconnect + cleanup
    await service.switch_broker(name) → hot-swap adapter

Usage:
    from services.broker_service import BrokerService
    broker_svc = BrokerService()
    await broker_svc.start()
    quote = await broker_svc.broker.get_quote("RELIANCE", "NSE")
"""

from __future__ import annotations

import asyncio
from typing import Optional, Callable

from brokers.abstract_broker import AbstractBroker
from brokers.models import (
    BrokerConnectionState,
    BrokerInfo,
    OrderRequest,
    OrderResponse,
    Quote,
    Position,
    TickData,
)
from brokers.registry import get_broker, list_registered_brokers
from utils.logger import get_logger

logger = get_logger(__name__)


class BrokerService:
    """
    Application-level broker lifecycle manager.

    Responsibilities:
    - Owns the single active broker adapter instance
    - Manages connect / reconnect / disconnect lifecycle
    - Routes tick and order callbacks to subscribers
    - Provides hot-swap broker switching without restart
    - Exposes a clean API for all higher-level services

    Access pattern:
        from services.broker_service import BrokerService
        broker_svc = BrokerService()
        await broker_svc.start()

        # All broker operations via service
        quote = await broker_svc.get_quote("RELIANCE", "NSE")
        positions = await broker_svc.get_positions()

        # Or get raw adapter for advanced use
        broker = broker_svc.broker
    """

    def __init__(self) -> None:
        from config.settings import get_settings
        self._settings = get_settings()
        self._broker: Optional[AbstractBroker] = None
        self._reconnect_task: Optional[asyncio.Task] = None
        self._is_running: bool = False

        # Registered external callbacks
        self._tick_subscribers: list[Callable] = []
        self._order_subscribers: list[Callable] = []

    # ─────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────

    @property
    def broker(self) -> AbstractBroker:
        """
        Returns the active broker adapter.

        Raises:
            RuntimeError: If BrokerService has not been started.
        """
        if not self._broker:
            raise RuntimeError(
                "BrokerService not started. "
                "Call await broker_service.start() first."
            )
        return self._broker

    @property
    def active_broker_name(self) -> str:
        """Returns the name of the currently active broker."""
        if self._broker:
            return self._broker.broker_name
        return self._settings.active_broker

    @property
    def is_connected(self) -> bool:
        """Returns True if the active broker is connected."""
        if not self._broker:
            return False
        return (
            self._broker.connection_state
            == BrokerConnectionState.CONNECTED
        )

    @property
    def is_running(self) -> bool:
        """Returns True if the service is active."""
        return self._is_running

    # ─────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────

    async def start(
        self,
        broker_name: Optional[str] = None,
        auto_reconnect: bool = True,
    ) -> bool:
        """
        Starts the broker service.
        Instantiates the adapter, connects, and registers
        internal callbacks.

        Args:
            broker_name: Broker to use. Defaults to active broker
                         from settings (ACTIVE_BROKER env var).
            auto_reconnect: Whether to auto-reconnect on disconnect.

        Returns:
            True if connected successfully.

        Raises:
            ConnectionError: If broker connection fails.
        """
        name = broker_name or self._settings.active_broker
        logger.info(
            f"BrokerService starting | broker={name}"
        )

        # Instantiate adapter from registry
        self._broker = get_broker(name)

        # Wire internal callbacks
        self._broker.register_tick_callback(
            self._internal_tick_handler
        )
        self._broker.register_order_callback(
            self._internal_order_handler
        )

        # Connect
        connected = await self._broker.connect()
        if connected:
            self._is_running = True
            logger.info(
                f"BrokerService started | "
                f"broker={name} | "
                f"mode={self._settings.trading_mode}"
            )
            # Start auto-reconnect watchdog
            if auto_reconnect and not self._broker.is_paper:
                self._reconnect_task = asyncio.create_task(
                    self._reconnect_watchdog()
                )
        else:
            logger.error(
                f"BrokerService failed to connect | broker={name}"
            )
        return connected

    async def stop(self) -> None:
        """
        Stops the broker service.
        Cancels reconnect watchdog and disconnects cleanly.
        """
        logger.info(
            f"BrokerService stopping | "
            f"broker={self.active_broker_name}"
        )
        self._is_running = False

        # Cancel reconnect watchdog
        if self._reconnect_task:
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass
            self._reconnect_task = None

        # Disconnect broker
        if self._broker:
            await self._broker.disconnect()
            self._broker = None

        logger.info("BrokerService stopped.")

    async def switch_broker(self, new_broker_name: str) -> bool:
        """
        Hot-swaps the active broker at runtime.
        Stops the current broker and starts the new one.
        All registered callbacks are preserved.

        Used from the NexaTrade UI settings panel to
        switch brokers without restarting the application.

        Args:
            new_broker_name: Name of the broker to switch to.

        Returns:
            True if new broker connected successfully.

        Example:
            # From UI settings panel
            success = await broker_svc.switch_broker("paper")
        """
        logger.info(
            f"Broker switch initiated | "
            f"{self.active_broker_name} → {new_broker_name}"
        )

        # Preserve existing subscribers
        tick_subs = list(self._tick_subscribers)
        order_subs = list(self._order_subscribers)

        # Stop current broker
        await self.stop()

        # Start new broker
        connected = await self.start(broker_name=new_broker_name)

        # Re-register external subscribers
        if connected:
            for cb in tick_subs:
                self.register_tick_subscriber(cb)
            for cb in order_subs:
                self.register_order_subscriber(cb)
            logger.info(
                f"Broker switch complete | "
                f"new_broker={new_broker_name} | "
                f"tick_subs={len(tick_subs)} | "
                f"order_subs={len(order_subs)}"
            )
        return connected

    # ─────────────────────────────────────────
    # Reconnect Watchdog
    # ─────────────────────────────────────────

    async def _reconnect_watchdog(self) -> None:
        """
        Background task that monitors broker connectivity.
        Attempts to reconnect if the broker disconnects unexpectedly.
        Reads retry config from app_config.yaml.
        """
        cfg = self._settings.app_config.get("broker", {})
        delay = int(cfg.get("reconnect_delay_seconds", 5))
        max_attempts = int(cfg.get("max_reconnect_attempts", 10))

        logger.debug("Reconnect watchdog started.")

        while self._is_running:
            await asyncio.sleep(delay)

            if not self._broker:
                break

            try:
                alive = await self._broker.is_connected()
            except Exception:
                alive = False

            if not alive and self._is_running:
                logger.warning(
                    f"Broker disconnected unexpectedly | "
                    f"broker={self.active_broker_name} | "
                    f"attempting reconnect..."
                )
                for attempt in range(1, max_attempts + 1):
                    try:
                        reconnected = await self._broker.connect()
                        if reconnected:
                            logger.info(
                                f"Broker reconnected | "
                                f"attempt={attempt}"
                            )
                            break
                    except Exception as exc:
                        logger.error(
                            f"Reconnect attempt {attempt} failed | "
                            f"error={exc}"
                        )
                    await asyncio.sleep(delay * attempt)
                else:
                    logger.critical(
                        f"All reconnect attempts failed | "
                        f"broker={self.active_broker_name} | "
                        f"max_attempts={max_attempts}"
                    )

    # ─────────────────────────────────────────
    # Callback Fan-Out
    # ─────────────────────────────────────────

    async def _internal_tick_handler(self, tick: TickData) -> None:
        """
        Internal tick handler — fans out to all external subscribers.
        Also updates paper adapter's last price cache if active.
        """
        # Update paper adapter's price cache for limit order fills
        if (
            self._broker
            and self._broker.is_paper
            and hasattr(self._broker, "update_last_price")
        ):
            self._broker.update_last_price(tick.symbol, tick.last_price)

        # Fan-out to all external subscribers
        for callback in self._tick_subscribers:
            try:
                await callback(tick)
            except Exception as exc:
                logger.error(
                    f"Tick subscriber error | "
                    f"fn={callback.__name__} | error={exc}"
                )

    async def _internal_order_handler(
        self, response: OrderResponse
    ) -> None:
        """
        Internal order update handler — fans out to all external
        order subscribers.
        """
        for callback in self._order_subscribers:
            try:
                await callback(response)
            except Exception as exc:
                logger.error(
                    f"Order subscriber error | "
                    f"fn={callback.__name__} | error={exc}"
                )

    # ─────────────────────────────────────────
    # Subscriber Registration
    # ─────────────────────────────────────────

    def register_tick_subscriber(self, callback: Callable) -> None:
        """
        Registers an external async callback for tick events.
        Called by the strategy engine and feed service.

        Args:
            callback: Async function accepting TickData.
        """
        if callback not in self._tick_subscribers:
            self._tick_subscribers.append(callback)
            logger.debug(
                f"Tick subscriber registered | "
                f"fn={callback.__name__}"
            )

    def register_order_subscriber(self, callback: Callable) -> None:
        """
        Registers an external async callback for order updates.
        Called by the order engine and position manager.

        Args:
            callback: Async function accepting OrderResponse.
        """
        if callback not in self._order_subscribers:
            self._order_subscribers.append(callback)
            logger.debug(
                f"Order subscriber registered | "
                f"fn={callback.__name__}"
            )

    def unregister_tick_subscriber(self, callback: Callable) -> None:
        """Removes an external tick subscriber."""
        if callback in self._tick_subscribers:
            self._tick_subscribers.remove(callback)

    def unregister_order_subscriber(self, callback: Callable) -> None:
        """Removes an external order subscriber."""
        if callback in self._order_subscribers:
            self._order_subscribers.remove(callback)

    # ─────────────────────────────────────────
    # Convenience Passthrough Methods
    # ─────────────────────────────────────────

    async def get_quote(
        self, symbol: str, exchange: str
    ) -> Quote:
        """Passthrough to active broker.get_quote()."""
        return await self.broker.get_quote(symbol, exchange)

    async def get_positions(self) -> list[Position]:
        """Passthrough to active broker.get_positions()."""
        return await self.broker.get_positions()

    async def get_funds(self) -> dict[str, float]:
        """Passthrough to active broker.get_funds()."""
        return await self.broker.get_funds()

    async def place_order(
        self, request: OrderRequest
    ) -> OrderResponse:
        """Passthrough to active broker.place_order()."""
        return await self.broker.place_order(request)

    async def get_broker_info(self) -> BrokerInfo:
        """Passthrough to active broker.get_info()."""
        return await self.broker.get_info()

    async def health_check(self) -> dict[str, bool]:
        """
        Performs a broker connectivity health check.

        Returns:
            Dict with health status fields.
        """
        connected = await self._broker.is_connected() if self._broker else False
        return {
            "broker_connected": connected,
            "service_running": self._is_running,
            "is_paper": self._broker.is_paper if self._broker else True,
        }

    def __repr__(self) -> str:
        return (
            f"BrokerService("
            f"broker={self.active_broker_name!r}, "
            f"connected={self.is_connected}, "
            f"running={self._is_running})"
        )