"""
NexaTrade — Dependency Injection Container.

The Container is the single source of truth for all
application-level singleton instances. It wires together
all services in the correct dependency order and manages
their lifecycle (start / stop).

Dependency graph:
    Settings
        ├── PostgresClient
        ├── InfluxClient
        ├── RedisClient
        └── BrokerService
                └── FeedService
                        └── DataService
                                └── RiskManager
                                        └── StrategyEngine
                                                └── BacktestRunner

Lifecycle:
    container = Container()
    await container.start()    → initialises all services
    ...application runs...
    await container.stop()     → shuts down all services cleanly

Usage:
    from container import Container

    container = Container()
    await container.start()

    # Access services
    broker  = container.broker_service
    feed    = container.feed_service
    engine  = container.strategy_engine
    runner  = container.backtest_runner

    await container.stop()
"""

from __future__ import annotations

import asyncio
from typing import Optional

from backtesting.backtest_runner import BacktestRunner
from backtesting.backtester import Backtester
from config.settings import get_settings, Settings
from data.storage.influx_client import InfluxClient
from data.storage.postgres_client import PostgresClient
from data.storage.redis_client import RedisClient
from services.broker_service import BrokerService
from services.data_service import DataService
from services.feed_service import FeedService
from strategies.risk_manager import RiskManager
from strategies.strategy_engine import StrategyEngine
from utils.logger import get_logger

logger = get_logger(__name__)


class Container:
    """
    NexaTrade Dependency Injection Container.

    Owns and manages the lifecycle of every application service.
    All services are created once and shared across the application.

    Startup order:
        1. PostgresClient.initialise()
        2. InfluxClient.initialise()
        3. RedisClient.initialise()
        4. BrokerService.start()
        5. FeedService.start()
        6. StrategyEngine.start()

    Shutdown order (reverse):
        1. StrategyEngine.stop()
        2. FeedService.stop()
        3. BrokerService.stop()
        4. RedisClient.shutdown()
        5. InfluxClient.shutdown()
        6. PostgresClient.shutdown()
    """

    def __init__(self) -> None:
        self._settings: Settings = get_settings()
        self._started:  bool     = False

        # ── Storage layer ─────────────────────
        self._pg:     Optional[PostgresClient] = None
        self._influx: Optional[InfluxClient]   = None
        self._redis:  Optional[RedisClient]    = None

        # ── Service layer ─────────────────────
        self._broker_svc:    Optional[BrokerService]   = None
        self._feed_svc:      Optional[FeedService]     = None
        self._data_svc:      Optional[DataService]     = None
        self._risk_mgr:      Optional[RiskManager]     = None
        self._strategy_eng:  Optional[StrategyEngine]  = None
        self._backtest_runner: Optional[BacktestRunner] = None

    # ─────────────────────────────────────────
    # Properties (read-only access)
    # ─────────────────────────────────────────

    @property
    def settings(self) -> Settings:
        """Returns application settings."""
        return self._settings

    @property
    def pg(self) -> PostgresClient:
        """Returns the PostgreSQL client."""
        self._assert_started()
        return self._pg

    @property
    def influx(self) -> InfluxClient:
        """Returns the InfluxDB client."""
        self._assert_started()
        return self._influx

    @property
    def redis(self) -> RedisClient:
        """Returns the Redis client."""
        self._assert_started()
        return self._redis

    @property
    def broker_service(self) -> BrokerService:
        """Returns the active BrokerService."""
        self._assert_started()
        return self._broker_svc

    @property
    def feed_service(self) -> FeedService:
        """Returns the FeedService."""
        self._assert_started()
        return self._feed_svc

    @property
    def data_service(self) -> DataService:
        """Returns the DataService."""
        self._assert_started()
        return self._data_svc

    @property
    def risk_manager(self) -> RiskManager:
        """Returns the RiskManager."""
        self._assert_started()
        return self._risk_mgr

    @property
    def strategy_engine(self) -> StrategyEngine:
        """Returns the StrategyEngine."""
        self._assert_started()
        return self._strategy_eng

    @property
    def backtest_runner(self) -> BacktestRunner:
        """Returns the BacktestRunner."""
        self._assert_started()
        return self._backtest_runner

    # ─────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────

    async def start(self) -> None:
        """
        Initialises and starts all application services.
        Must be called once before accessing any service.

        Raises:
            RuntimeError: If any critical service fails to start.
        """
        if self._started:
            logger.warning("Container already started.")
            return

        logger.info(
            f"NexaTrade Container starting | "
            f"env={self._settings.environment} | "
            f"broker={self._settings.active_broker} | "
            f"mode={self._settings.trading_mode}"
        )

        try:
            await self._init_storage()
            await self._init_broker()
            await self._init_services()
            await self._init_engine()
            self._started = True
            logger.info(
                "✅ NexaTrade Container started successfully."
            )
        except Exception as exc:
            logger.critical(
                f"Container startup failed: {exc}"
            )
            # Attempt clean teardown
            await self._teardown_partial()
            raise RuntimeError(
                f"Container failed to start: {exc}"
            ) from exc

    async def stop(self) -> None:
        """
        Shuts down all services in reverse dependency order.
        Safe to call even if start() was not completed.
        """
        if not self._started:
            return

        logger.info("NexaTrade Container shutting down...")
        await self._teardown_all()
        self._started = False
        logger.info("✅ NexaTrade Container shut down cleanly.")

    async def restart(self) -> None:
        """
        Stops and restarts all services.
        Used for applying config changes without process restart.
        """
        logger.info("Container restarting...")
        await self.stop()
        await asyncio.sleep(1)
        await self.start()

    # ─────────────────────────────────────────
    # Startup Steps
    # ─────────────────────────────────────────

    async def _init_storage(self) -> None:
        """Initialises all three storage clients."""
        logger.info("Initialising storage layer...")

        # PostgreSQL
        self._pg = PostgresClient()
        await self._pg.initialise()
        logger.info(
            f"  ✅ PostgreSQL | "
            f"{self._settings.postgres.dsn_masked}"
        )

        # InfluxDB
        self._influx = InfluxClient()
        await self._influx.initialise()
        logger.info(
            f"  ✅ InfluxDB   | "
            f"{self._settings.influx.url}"
        )

        # Redis
        self._redis = RedisClient()
        await self._redis.initialise()
        logger.info(
            f"  ✅ Redis      | "
            f"{self._settings.redis.host}:{self._settings.redis.port}"
        )

    async def _init_broker(self) -> None:
        """Initialises and starts the BrokerService."""
        logger.info(
            f"Initialising broker service | "
            f"broker={self._settings.active_broker}..."
        )
        self._broker_svc = BrokerService()
        connected = await self._broker_svc.start(
            broker_name=self._settings.active_broker,
            auto_reconnect=True,
        )
        if connected:
            logger.info(
                f"  ✅ BrokerService | "
                f"broker={self._settings.active_broker} | "
                f"mode={self._settings.trading_mode}"
            )
        else:
            logger.warning(
                f"  ⚠️  BrokerService connected with warnings | "
                f"broker={self._settings.active_broker}"
            )

    async def _init_services(self) -> None:
        """Initialises DataService and FeedService."""
        logger.info("Initialising data and feed services...")

        # DataService
        self._data_svc = DataService(
            influx_client=self._influx,
            broker_service=self._broker_svc,
        )
        logger.info("  ✅ DataService")

        # FeedService
        self._feed_svc = FeedService(
            broker_service=self._broker_svc,
            redis_client=self._redis,
            influx_client=self._influx,
        )
        await self._feed_svc.start()
        logger.info("  ✅ FeedService")

        # RiskManager
        self._risk_mgr = RiskManager(
            redis_client=self._redis,
            pg_client=self._pg,
        )
        logger.info("  ✅ RiskManager")

    async def _init_engine(self) -> None:
        """Initialises the StrategyEngine and BacktestRunner."""
        logger.info("Initialising strategy engine...")

        # StrategyEngine
        self._strategy_eng = StrategyEngine(
            broker_service=self._broker_svc,
            feed_service=self._feed_svc,
            data_service=self._data_svc,
            risk_manager=self._risk_mgr,
            redis_client=self._redis,
            pg_client=self._pg,
        )
        await self._strategy_eng.start()
        logger.info(
            f"  ✅ StrategyEngine | "
            f"registered={self._strategy_eng.registered_count}"
        )

        # BacktestRunner
        self._backtest_runner = BacktestRunner(
            data_service=self._data_svc,
            pg_client=self._pg,
        )
        logger.info("  ✅ BacktestRunner")

    # ─────────────────────────────────────────
    # Teardown Steps
    # ─────────────────────────────────────────

    async def _teardown_all(self) -> None:
        """Full teardown in reverse dependency order."""
        steps = [
            ("StrategyEngine",  self._stop_strategy_engine),
            ("FeedService",     self._stop_feed_service),
            ("BrokerService",   self._stop_broker_service),
            ("Redis",           self._stop_redis),
            ("InfluxDB",        self._stop_influx),
            ("PostgreSQL",      self._stop_postgres),
        ]
        for name, step in steps:
            try:
                await step()
                logger.info(f"  ✅ {name} shut down")
            except Exception as exc:
                logger.warning(
                    f"  ⚠️  {name} shutdown warning: {exc}"
                )

    async def _teardown_partial(self) -> None:
        """
        Best-effort teardown for partially started container.
        Called when startup fails mid-way.
        """
        try:
            await self._teardown_all()
        except Exception:
            pass

    async def _stop_strategy_engine(self) -> None:
        if self._strategy_eng:
            await self._strategy_eng.stop()

    async def _stop_feed_service(self) -> None:
        if self._feed_svc:
            await self._feed_svc.stop()

    async def _stop_broker_service(self) -> None:
        if self._broker_svc:
            await self._broker_svc.stop()

    async def _stop_redis(self) -> None:
        if self._redis:
            await self._redis.shutdown()

    async def _stop_influx(self) -> None:
        if self._influx:
            await self._influx.shutdown()

    async def _stop_postgres(self) -> None:
        if self._pg:
            await self._pg.shutdown()

    # ─────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────

    def _assert_started(self) -> None:
        """Raises RuntimeError if container has not been started."""
        if not self._started:
            raise RuntimeError(
                "Container not started. "
                "Call await container.start() first."
            )

    @property
    def is_started(self) -> bool:
        """Returns True if the container is running."""
        return self._started

    async def health_check(self) -> dict[str, bool]:
        """
        Performs a full system health check.

        Returns:
            Dict of {service_name: is_healthy} booleans.
        """
        if not self._started:
            return {"container": False}

        results: dict[str, bool] = {
            "container": True,
        }

        # Storage health
        try:
            results["redis"] = await self._redis.health_check()
        except Exception:
            results["redis"] = False

        try:
            data_health = await self._data_svc.health_check()
            results["influxdb"] = data_health.get("influx_ok", False)
            results["broker"]   = data_health.get("broker_ok", False)
        except Exception:
            results["influxdb"] = False
            results["broker"]   = False

        # Service health
        results["feed_running"]    = self._feed_svc.is_running
        results["engine_running"]  = self._strategy_eng.is_running
        results["broker_connected"] = self._broker_svc.is_connected

        return results

    def get_system_stats(self) -> dict:
        """
        Returns a comprehensive system status snapshot.
        Used by monitoring endpoints and the UI dashboard.

        Returns:
            Dict with stats from all subsystems.
        """
        if not self._started:
            return {"started": False}

        return {
            "started":        self._started,
            "environment":    self._settings.environment,
            "trading_mode":   self._settings.trading_mode,
            "active_broker":  self._settings.active_broker,
            "broker": {
                "connected":  self._broker_svc.is_connected,
                "name":       self._broker_svc.active_broker_name,
            },
            "feed":    self._feed_svc.get_feed_stats(),
            "engine":  self._strategy_eng.get_engine_stats(),
            "risk":    self._risk_mgr.get_stats(),
            "data_cache": self._data_svc.get_cache_stats(),
        }

    def __repr__(self) -> str:
        return (
            f"Container("
            f"started={self._started}, "
            f"env={self._settings.environment!r}, "
            f"broker={self._settings.active_broker!r})"
        )