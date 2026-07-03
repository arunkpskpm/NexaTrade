"""
NexaTrade — Broker Registry & Factory.

The registry is the single lookup table that maps
broker name strings to adapter classes.

To add a new broker:
    1. Create brokers/adapters/my_broker_adapter.py
    2. Import and register it in BROKER_REGISTRY below
    3. Done — zero other changes needed

The factory function get_broker() is the ONLY way
the rest of NexaTrade instantiates a broker adapter.
It reads the active broker from settings and returns
a fully constructed adapter instance.

Usage:
    from brokers.registry import get_broker

    broker = await get_broker()          # active broker
    broker = await get_broker("paper")   # specific broker
    broker = get_broker("breeze", connect=False)  # no auto-connect
"""

from __future__ import annotations

from typing import Optional, Type

from brokers.abstract_broker import AbstractBroker
from utils.logger import get_logger

logger = get_logger(__name__)


# ═════════════════════════════════════════════
# Broker Registry
# Maps broker name → adapter class
#
# To add a new broker:
#   1. Create the adapter file
#   2. Import it below
#   3. Add one line to BROKER_REGISTRY
# ═════════════════════════════════════════════

def _build_registry() -> dict[str, Type[AbstractBroker]]:
    """
    Lazily builds the broker registry dict.
    Imports are deferred to avoid loading all broker SDKs
    at startup — only the active broker's SDK is imported.

    Returns:
        Dict mapping broker name → adapter class.
    """
    from brokers.adapters.breeze_adapter import BreezeAdapter
    from brokers.adapters.paper_adapter import PaperAdapter

    registry: dict[str, Type[AbstractBroker]] = {
        "breeze": BreezeAdapter,
        "paper":  PaperAdapter,

        # ── Add new brokers below ─────────────
        # "zerodha":  ZerodhaAdapter,
        # "angelone": AngelOneAdapter,
        # "upstox":   UpstoxAdapter,
        # ─────────────────────────────────────
    }
    return registry


# Module-level registry instance (built lazily on first access)
_REGISTRY: Optional[dict[str, Type[AbstractBroker]]] = None


@property
def BROKER_REGISTRY(self) -> dict[str, Type[AbstractBroker]]:
    """Lazy accessor — do not call directly. Use get_broker_class()."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _build_registry()
    return _REGISTRY


def get_broker_class(broker_name: str) -> Type[AbstractBroker]:
    """
    Returns the adapter class for the given broker name.

    Args:
        broker_name: Broker identifier string.

    Returns:
        AbstractBroker subclass (not instantiated).

    Raises:
        ValueError: If broker_name is not in the registry.

    Example:
        AdapterClass = get_broker_class("breeze")
        broker = AdapterClass()
    """
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _build_registry()

    name = broker_name.lower().strip()
    if name not in _REGISTRY:
        registered = list(_REGISTRY.keys())
        raise ValueError(
            f"Unknown broker: '{name}'. "
            f"Registered brokers: {registered}. "
            f"To add a new broker, register it in brokers/registry.py."
        )
    return _REGISTRY[name]


def get_broker(
    broker_name: Optional[str] = None,
) -> AbstractBroker:
    """
    Factory — instantiates and returns a broker adapter.

    If broker_name is None, uses the active broker from settings
    (ACTIVE_BROKER env var / app_config.yaml).

    This is the ONLY function the rest of NexaTrade uses
    to obtain a broker instance. Never instantiate adapters directly.

    Args:
        broker_name: Broker identifier. Defaults to active broker.

    Returns:
        Instantiated AbstractBroker adapter (not yet connected).
        Call await broker.connect() to establish session.

    Raises:
        ValueError: If broker_name is not registered.

    Example:
        # Get active broker (from ACTIVE_BROKER env)
        broker = get_broker()
        connected = await broker.connect()

        # Get a specific broker
        paper = get_broker("paper")
        await paper.connect()

        # Swap broker at runtime (e.g. from UI settings)
        broker = get_broker("zerodha")
        await broker.connect()
    """
    from config.settings import get_settings
    settings = get_settings()

    name = (broker_name or settings.active_broker).lower().strip()
    adapter_class = get_broker_class(name)
    instance = adapter_class()

    logger.info(
        f"Broker adapter instantiated | "
        f"broker={name} | "
        f"class={adapter_class.__name__}"
    )
    return instance


def list_registered_brokers() -> list[str]:
    """
    Returns all registered broker names.

    Returns:
        Sorted list of registered broker name strings.

    Example:
        brokers = list_registered_brokers()
        # ["breeze", "paper"]
    """
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _build_registry()
    return sorted(_REGISTRY.keys())


def is_broker_registered(broker_name: str) -> bool:
    """
    Returns True if the given broker name is in the registry.

    Args:
        broker_name: Broker identifier to check.

    Returns:
        True if registered, False otherwise.
    """
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _build_registry()
    return broker_name.lower().strip() in _REGISTRY