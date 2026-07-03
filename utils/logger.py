"""
NexaTrade — Centralised Logging Module.

Architecture:
- Uses Loguru as the logging backend.
- Three log streams:
    1. Console  → human-readable, coloured, INFO+ only
    2. app.log  → full structured application log (DEBUG+)
    3. trade.log → trade-only audit log (INFO+), never deleted

- Logger names follow Python's __name__ convention.
- Trade logger is a specialised sink for order/fill events.
- All logs are written in IST timezone.
- Log rotation, retention, and compression via app_config.yaml.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from loguru import logger as _loguru_logger

# ─────────────────────────────────────────────
# Lazy config import to avoid circular imports
# ─────────────────────────────────────────────
def _get_log_config() -> dict:
    """
    Reads logging config from app_config.yaml.
    Returns defaults if config is not yet loaded.
    """
    try:
        from config.settings import get_settings
        cfg = get_settings().app_config.get("logging", {})
        return cfg
    except Exception:
        return {
            "log_dir": "logs",
            "rotation": "100 MB",
            "retention": "30 days",
            "compression": "gz",
            "format": (
                "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
                "{level:<8} | "
                "{name}:{function}:{line} | "
                "{message}"
            ),
        }


# ─────────────────────────────────────────────
# Sink IDs — track configured sinks
# ─────────────────────────────────────────────
_SINKS_CONFIGURED: bool = False
_CONSOLE_SINK_ID: Optional[int] = None
_APP_SINK_ID: Optional[int] = None
_TRADE_SINK_ID: Optional[int] = None


def _configure_sinks() -> None:
    """
    Configures Loguru sinks on first call.
    Idempotent — safe to call multiple times.
    """
    global _SINKS_CONFIGURED, _CONSOLE_SINK_ID
    global _APP_SINK_ID, _TRADE_SINK_ID

    if _SINKS_CONFIGURED:
        return

    cfg = _get_log_config()
    log_dir = Path(cfg.get("log_dir", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    log_format = cfg.get(
        "format",
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level:<8} | "
        "{name}:{function}:{line} | "
        "{message}",
    )
    rotation = cfg.get("rotation", "100 MB")
    retention = cfg.get("retention", "30 days")
    compression = cfg.get("compression", "gz")

    # ── Remove Loguru default sink ───────────
    _loguru_logger.remove()

    # ── Sink 1: Console (INFO+, coloured) ────
    _CONSOLE_SINK_ID = _loguru_logger.add(
        sys.stdout,
        level="INFO",
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level:<8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
        backtrace=False,
        diagnose=False,
    )

    # ── Sink 2: App log file (DEBUG+) ────────
    _APP_SINK_ID = _loguru_logger.add(
        str(log_dir / "nexatrade_{time:YYYY-MM-DD}.log"),
        level="DEBUG",
        format=log_format,
        rotation=rotation,
        retention=retention,
        compression=compression,
        encoding="utf-8",
        backtrace=True,
        diagnose=True,
        enqueue=True,           # Thread-safe async writes
    )

    # ── Sink 3: Trade log file (INFO+) ───────
    # Dedicated audit trail for all order/fill events.
    # Never compressed or deleted automatically.
    _TRADE_SINK_ID = _loguru_logger.add(
        str(log_dir / "trades_{time:YYYY-MM-DD}.log"),
        level="INFO",
        format=log_format,
        rotation="1 day",
        retention="365 days",   # Keep trade logs for 1 year
        compression=None,       # No compression — legal audit trail
        encoding="utf-8",
        filter=lambda record: "TRADE" in record["extra"],
        backtrace=False,
        diagnose=False,
        enqueue=True,
    )

    _SINKS_CONFIGURED = True


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def get_logger(name: str) -> "logger":
    """
    Returns a Loguru logger bound to the given module name.
    Configures sinks on first call.

    Args:
        name: Typically __name__ of the calling module.

    Returns:
        A Loguru logger with context bound to `name`.

    Example:
        logger = get_logger(__name__)
        logger.info("Strategy started | symbol=RELIANCE")
    """
    _configure_sinks()
    return _loguru_logger.bind(module=name)


def get_trade_logger(name: str) -> "logger":
    """
    Returns a Loguru logger that writes to BOTH the
    standard app log AND the dedicated trade audit log.

    All records emitted through this logger carry the
    'TRADE' extra key, which routes them to the trade sink.

    Args:
        name: Typically __name__ of the calling module.

    Returns:
        A Loguru logger bound with TRADE=True extra.

    Example:
        trade_logger = get_trade_logger(__name__)
        trade_logger.info(
            "ORDER PLACED | symbol=RELIANCE | qty=100 | price=2450.00"
        )
    """
    _configure_sinks()
    return _loguru_logger.bind(module=name, TRADE=True)


def set_log_level(level: str) -> None:
    """
    Dynamically updates the console sink log level at runtime.
    Useful for toggling DEBUG mode from the UI settings panel.

    Args:
        level: Log level string (DEBUG/INFO/WARNING/ERROR/CRITICAL).
    """
    global _CONSOLE_SINK_ID
    _configure_sinks()
    if _CONSOLE_SINK_ID is not None:
        _loguru_logger.remove(_CONSOLE_SINK_ID)
        _CONSOLE_SINK_ID = _loguru_logger.add(
            sys.stdout,
            level=level.upper(),
            format=(
                "<green>{time:HH:mm:ss}</green> | "
                "<level>{level:<8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan> | "
                "<level>{message}</level>"
            ),
            colorize=True,
            backtrace=False,
            diagnose=False,
        )