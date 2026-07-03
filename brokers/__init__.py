"""
NexaTrade — Broker Abstraction Package.

This package is the heart of NexaTrade's plug-and-play
broker architecture. No code outside this package should
ever import a broker-specific SDK directly.

Public API:
    from brokers import get_broker, AbstractBroker
    from brokers.models import OrderRequest, OrderResponse, Quote, Position

Adding a new broker (3 steps only):
    1. Create  brokers/adapters/my_broker_adapter.py
    2. Add     config/brokers/my_broker.yaml
    3. Register in brokers/registry.py → BROKER_REGISTRY

Zero changes needed in any other file.
"""

from brokers.abstract_broker import AbstractBroker
from brokers.registry import get_broker, get_broker_class, BROKER_REGISTRY

__all__ = [
    "AbstractBroker",
    "get_broker",
    "get_broker_class",
    "BROKER_REGISTRY",
]