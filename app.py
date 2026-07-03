"""
NexaTrade — FastAPI Application Entry Point.

This module:
  - Creates the FastAPI app instance
  - Registers all API routers
  - Manages Container lifecycle via startup/shutdown events
  - Configures CORS, exception handlers, and middleware
  - Exposes the Container as app.state.container
    for access in all route handlers

The Container singleton is started on application startup
and stopped on application shutdown — no manual lifecycle
management needed in routes.

Run with:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload

Or via main.py:
    python main.py
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config.settings import get_settings
from container import Container
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


# ─────────────────────────────────────────────
# Lifespan (startup / shutdown)
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.
    Starts the Container on startup and stops it on shutdown.

    The Container (and all its services) live for the
    entire application lifetime — exactly one instance.
    """
    container = Container()
    app.state.container = container

    logger.info(
        f"🚀 Starting NexaTrade "
        f"v{settings.app_version} | "
        f"env={settings.environment} | "
        f"mode={settings.trading_mode}"
    )

    try:
        await container.start()
    except Exception as exc:
        logger.critical(
            f"Fatal: Container startup failed: {exc}"
        )
        raise

    yield  # ← Application runs here

    logger.info("🛑 Shutting down NexaTrade...")
    await container.stop()
    logger.info("✅ NexaTrade shutdown complete.")


# ─────────────────────────────────────────────
# FastAPI App Factory
# ─────────────────────────────────────────────

def create_app() -> FastAPI:
    """
    Creates and configures the FastAPI application.

    Returns:
        Configured FastAPI instance.
    """
    app_config = settings.app_config.get("api", {})

    app = FastAPI(
        title=app_config.get("title", "NexaTrade API"),
        version=app_config.get("version", "v1"),
        description="Algorithmic trading platform for Indian markets",
        docs_url=app_config.get("docs_url", "/docs"),
        redoc_url=app_config.get("redoc_url", "/redoc"),
        openapi_url=app_config.get("openapi_url", "/openapi.json"),
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────
    cors_cfg = settings.app_config.get("cors", {})
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_cfg.get(
            "origins", settings.cors_origins
        ),
        allow_credentials=cors_cfg.get("allow_credentials", True),
        allow_methods=cors_cfg.get("allow_methods", ["*"]),
        allow_headers=cors_cfg.get("allow_headers", ["*"]),
    )

    # ── Request timing middleware ──────────────
    @app.middleware("http")
    async def add_process_time_header(
        request: Request, call_next
    ):
        start_time = time.perf_counter()
        response   = await call_next(request)
        process_ms = (time.perf_counter() - start_time) * 1000
        response.headers["X-Process-Time-Ms"] = (
            f"{process_ms:.2f}"
        )
        return response

    # ── Global exception handler ──────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(
        request: Request, exc: Exception
    ):
        logger.error(
            f"Unhandled exception | "
            f"path={request.url.path} | "
            f"error={exc}"
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error":   "internal_server_error",
                "message": "An unexpected error occurred.",
                "path":    str(request.url.path),
            },
        )

    # ── Register API routers ──────────────────
    _register_routers(app)

    # ── Root health endpoint ──────────────────
    @app.get(
        "/",
        tags=["System"],
        summary="API root — health check",
    )
    async def root():
        return {
            "app":     settings.app_name,
            "version": settings.app_version,
            "status":  "running",
            "mode":    settings.trading_mode,
            "broker":  settings.active_broker,
            "env":     settings.environment,
        }

    @app.get(
        "/health",
        tags=["System"],
        summary="Full system health check",
    )
    async def health(request: Request):
        """
        Performs a health check across all services.
        Returns 200 if all healthy, 503 if any are down.
        """
        container: Container = request.app.state.container
        checks = await container.health_check()
        all_healthy = all(checks.values())
        return JSONResponse(
            status_code=(
                status.HTTP_200_OK
                if all_healthy
                else status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            content={
                "status": "healthy" if all_healthy else "degraded",
                "checks": checks,
            },
        )

    @app.get(
        "/stats",
        tags=["System"],
        summary="Full system statistics",
    )
    async def system_stats(request: Request):
        """Returns a comprehensive system status snapshot."""
        container: Container = request.app.state.container
        return container.get_system_stats()

    return app


def _register_routers(app: FastAPI) -> None:
    """
    Registers all API routers onto the FastAPI app.
    Each router is prefixed and tagged for organisation.

    All routers are imported lazily here to avoid circular imports.
    """
    from api.routes import (
        auth_router,
        broker_router,
        data_router,
        feed_router,
        orders_router,
        positions_router,
        strategies_router,
        backtest_router,
        risk_router,
        websocket_router,
    )

    prefix = "/api/v1"

    app.include_router(
        auth_router,
        prefix=f"{prefix}/auth",
        tags=["Authentication"],
    )
    app.include_router(
        broker_router,
        prefix=f"{prefix}/broker",
        tags=["Broker"],
    )
    app.include_router(
        data_router,
        prefix=f"{prefix}/data",
        tags=["Market Data"],
    )
    app.include_router(
        feed_router,
        prefix=f"{prefix}/feed",
        tags=["Live Feed"],
    )
    app.include_router(
        orders_router,
        prefix=f"{prefix}/orders",
        tags=["Orders"],
    )
    app.include_router(
        positions_router,
        prefix=f"{prefix}/positions",
        tags=["Positions"],
    )
    app.include_router(
        strategies_router,
        prefix=f"{prefix}/strategies",
        tags=["Strategies"],
    )
    app.include_router(
        backtest_router,
        prefix=f"{prefix}/backtest",
        tags=["Backtesting"],
    )
    app.include_router(
        risk_router,
        prefix=f"{prefix}/risk",
        tags=["Risk Management"],
    )
    app.include_router(
        websocket_router,
        prefix=f"{prefix}/ws",
        tags=["WebSocket"],
    )

    logger.info(
        f"API routers registered | prefix={prefix}"
    )


# ─────────────────────────────────────────────
# App instance (module-level for uvicorn)
# ─────────────────────────────────────────────
app = create_app()