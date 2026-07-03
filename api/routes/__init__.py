"""
NexaTrade — API Routes package.
Exports all routers for registration in app.py.
"""

from api.routes.auth      import router as auth_router
from api.routes.broker    import router as broker_router
from api.routes.data      import router as data_router
from api.routes.feed      import router as feed_router
from api.routes.orders    import router as orders_router
from api.routes.positions import router as positions_router
from api.routes.strategies import router as strategies_router
from api.routes.backtest  import router as backtest_router
from api.routes.risk      import router as risk_router
from api.routes.websocket import router as websocket_router

__all__ = [
    "auth_router",
    "broker_router",
    "data_router",
    "feed_router",
    "orders_router",
    "positions_router",
    "strategies_router",
    "backtest_router",
    "risk_router",
    "websocket_router",
]