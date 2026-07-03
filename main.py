"""
NexaTrade — Application Entry Point.

Run with:
    python main.py                     # Production
    python main.py --reload            # Development hot-reload
    python main.py --port 9000         # Custom port
"""

from __future__ import annotations

import argparse
import sys

import uvicorn

from config.settings import get_settings


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="NexaTrade Algorithmic Trading Platform"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Host to bind (overrides APP_HOST env var)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind (overrides API_PORT env var)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        default=False,
        help="Enable hot-reload (development only)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of uvicorn workers",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["debug", "info", "warning", "error"],
        help="Log level (overrides LOG_LEVEL env var)",
    )
    return parser.parse_args()


def main() -> None:
    """NexaTrade application entry point."""
    args    = parse_args()
    settings = get_settings()

    host      = args.host      or settings.api_host
    port      = args.port      or settings.api_port
    reload    = args.reload    or settings.api_reload
    log_level = args.log_level or settings.log_level.lower()
    workers   = args.workers

    # Prevent reload with multiple workers
    if reload and workers > 1:
        print(
            "⚠️  Warning: --reload is not compatible with "
            "--workers > 1. Using 1 worker."
        )
        workers = 1

    print(
        f"\n"
        f"  ╔══════════════════════════════════════╗\n"
        f"  ║       NexaTrade v{settings.app_version:<6}              ║\n"
        f"  ║  Algorithmic Trading Platform        ║\n"
        f"  ╚══════════════════════════════════════╝\n"
        f"\n"
        f"  Environment : {settings.environment}\n"
        f"  Trading Mode: {settings.trading_mode.upper()}\n"
        f"  Active Broker: {settings.active_broker}\n"
        f"  API         : http://{host}:{port}\n"
        f"  Docs        : http://{host}:{port}/docs\n"
        f"  Log Level   : {log_level.upper()}\n"
    )

    if settings.is_live_trading:
        print(
            "  ⚠️  LIVE TRADING MODE — Real orders will be placed!\n"
        )

    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers,
        log_level=log_level,
        access_log=True,
    )


if __name__ == "__main__":
    main()