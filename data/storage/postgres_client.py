"""
NexaTrade — PostgreSQL Async Client.

Wraps asyncpg with a connection pool, schema migration,
and all query methods required by NexaTrade services.

Schema (auto-created on first initialise()):
    users              → auth and user management
    strategies         → registered strategy metadata
    orders             → full order lifecycle audit trail
    positions          → position snapshots
    backtest_runs      → backtest run metadata and metrics
    risk_snapshots     → daily risk metrics snapshots

Connection pooling:
    asyncpg pool — min 2, max 20 connections
    Pool is shared across all coroutines in the process.
    Never create new connections outside this class.

Usage:
    pg = PostgresClient()
    await pg.initialise()

    user  = await pg.get_user_by_username("alice")
    await pg.insert_order(order_id=..., symbol=..., ...)
    await pg.shutdown()
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Optional

import asyncpg

from config.settings import get_settings
from utils.logger import get_logger
from utils.time_utils import now_utc

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# DDL — Schema Creation SQL
# ─────────────────────────────────────────────

_DDL_STATEMENTS = [

    # ── users ──────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS users (
        user_id       UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
        username      VARCHAR(64) UNIQUE NOT NULL,
        email         VARCHAR(128) UNIQUE,
        password_hash TEXT        NOT NULL,
        is_active     BOOLEAN     NOT NULL DEFAULT TRUE,
        is_admin      BOOLEAN     NOT NULL DEFAULT FALSE,
        created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    """,

    # ── strategies ─────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS strategies (
        id            SERIAL      PRIMARY KEY,
        name          VARCHAR(128) UNIQUE NOT NULL,
        display_name  VARCHAR(256),
        module_path   VARCHAR(512),
        class_name    VARCHAR(128),
        description   TEXT,
        parameters    JSONB       NOT NULL DEFAULT '{}',
        is_active     BOOLEAN     NOT NULL DEFAULT FALSE,
        broker_name   VARCHAR(64),
        created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    """,

    # ── orders ─────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS orders (
        id                BIGSERIAL   PRIMARY KEY,
        order_id          VARCHAR(64) UNIQUE NOT NULL,
        broker_order_id   VARCHAR(128),
        broker_name       VARCHAR(64) NOT NULL,
        symbol            VARCHAR(64) NOT NULL,
        exchange          VARCHAR(16) NOT NULL,
        segment           VARCHAR(16) NOT NULL DEFAULT 'EQ',
        order_type        VARCHAR(32) NOT NULL,
        transaction_type  VARCHAR(8)  NOT NULL,
        product_type      VARCHAR(32) NOT NULL DEFAULT 'INTRADAY',
        quantity          INTEGER     NOT NULL,
        price             NUMERIC(14,4),
        trigger_price     NUMERIC(14,4),
        filled_quantity   INTEGER     NOT NULL DEFAULT 0,
        average_price     NUMERIC(14,4),
        status            VARCHAR(32) NOT NULL DEFAULT 'PENDING',
        rejection_reason  TEXT,
        trading_mode      VARCHAR(8)  NOT NULL DEFAULT 'paper',
        strategy_name     VARCHAR(128),
        tags              JSONB       NOT NULL DEFAULT '{}',
        placed_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    """,

    # ── orders indices ─────────────────────────────────────
    """
    CREATE INDEX IF NOT EXISTS idx_orders_broker_name
        ON orders(broker_name);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_orders_symbol
        ON orders(symbol);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_orders_strategy_name
        ON orders(strategy_name);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_orders_placed_at
        ON orders(placed_at DESC);
    """,

    # ── positions ──────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS positions (
        id              BIGSERIAL   PRIMARY KEY,
        broker_name     VARCHAR(64) NOT NULL,
        trading_mode    VARCHAR(8)  NOT NULL DEFAULT 'paper',
        symbol          VARCHAR(64) NOT NULL,
        exchange        VARCHAR(16) NOT NULL,
        segment         VARCHAR(16) NOT NULL DEFAULT 'EQ',
        quantity        INTEGER     NOT NULL DEFAULT 0,
        average_price   NUMERIC(14,4) NOT NULL DEFAULT 0,
        last_price      NUMERIC(14,4) NOT NULL DEFAULT 0,
        realized_pnl    NUMERIC(14,4) NOT NULL DEFAULT 0,
        unrealized_pnl  NUMERIC(14,4) NOT NULL DEFAULT 0,
        strategy_name   VARCHAR(128),
        updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        UNIQUE(broker_name, trading_mode, symbol, exchange)
    );
    """,

    # ── backtest_runs ──────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS backtest_runs (
        id               BIGSERIAL   PRIMARY KEY,
        run_id           VARCHAR(32) UNIQUE NOT NULL,
        strategy_name    VARCHAR(128) NOT NULL,
        symbol           VARCHAR(64) NOT NULL,
        exchange         VARCHAR(16) NOT NULL DEFAULT 'NSE',
        interval         VARCHAR(16) NOT NULL,
        start_date       DATE        NOT NULL,
        end_date         DATE        NOT NULL,
        initial_capital  NUMERIC(16,4) NOT NULL,
        parameters       JSONB       NOT NULL DEFAULT '{}',
        status           VARCHAR(16) NOT NULL DEFAULT 'PENDING',
        broker_name      VARCHAR(64),
        final_capital    NUMERIC(16,4),
        total_pnl        NUMERIC(16,4),
        total_trades     INTEGER,
        win_rate         NUMERIC(6,4),
        max_drawdown     NUMERIC(6,4),
        sharpe_ratio     NUMERIC(10,6),
        created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        completed_at     TIMESTAMPTZ
    );
    """,

    # ── backtest_runs indices ──────────────────────────────
    """
    CREATE INDEX IF NOT EXISTS idx_backtest_strategy
        ON backtest_runs(strategy_name);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_backtest_symbol
        ON backtest_runs(symbol);
    """,

    # ── risk_snapshots ─────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS risk_snapshots (
        id                 BIGSERIAL   PRIMARY KEY,
        broker_name        VARCHAR(64) NOT NULL,
        trading_mode       VARCHAR(8)  NOT NULL DEFAULT 'paper',
        snapshot_date      DATE        NOT NULL DEFAULT CURRENT_DATE,
        daily_pnl          NUMERIC(14,4) NOT NULL DEFAULT 0,
        open_positions     INTEGER     NOT NULL DEFAULT 0,
        total_orders       INTEGER     NOT NULL DEFAULT 0,
        max_drawdown_pct   NUMERIC(6,4) NOT NULL DEFAULT 0,
        portfolio_value    NUMERIC(16,4) NOT NULL DEFAULT 0,
        created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        UNIQUE(broker_name, trading_mode, snapshot_date)
    );
    """,
]


class PostgresClient:
    """
    NexaTrade Async PostgreSQL Client.

    Uses asyncpg connection pool under the hood.
    All public methods are async coroutines.

    Schema is auto-created on initialise().
    """

    def __init__(self) -> None:
        self._pool: Optional[asyncpg.Pool] = None
        self._settings = get_settings()

    # ─────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────

    async def initialise(self) -> None:
        """
        Creates the asyncpg connection pool and runs DDL migrations.

        Raises:
            RuntimeError: If connection fails.
        """
        dsn = self._settings.postgres.raw_dsn
        try:
            self._pool = await asyncpg.create_pool(
                dsn=dsn,
                min_size=2,
                max_size=20,
                command_timeout=30,
                statement_cache_size=100,
            )
            await self._run_migrations()
            logger.info(
                f"PostgreSQL pool ready | "
                f"dsn={self._settings.postgres.dsn_masked}"
            )
        except Exception as exc:
            raise RuntimeError(
                f"PostgreSQL init failed: {exc}"
            ) from exc

    async def shutdown(self) -> None:
        """Closes all pool connections gracefully."""
        if self._pool:
            await self._pool.close()
            logger.info("PostgreSQL pool closed.")

    async def _run_migrations(self) -> None:
        """Executes all DDL statements in order."""
        async with self._pool.acquire() as conn:
            for ddl in _DDL_STATEMENTS:
                await conn.execute(ddl)
        logger.debug("PostgreSQL schema migration complete.")

    async def health_check(self) -> bool:
        """Returns True if a test query succeeds."""
        try:
            async with self._pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception:
            return False

    # ─────────────────────────────────────────
    # Internal Helper
    # ─────────────────────────────────────────

    def _pool_required(self) -> None:
        """Raises if pool has not been initialised."""
        if not self._pool:
            raise RuntimeError(
                "PostgresClient not initialised. "
                "Call await pg.initialise() first."
            )

    # ═════════════════════════════════════════
    # Section 1 — Users
    # ═════════════════════════════════════════

    async def get_user_by_username(
        self, username: str
    ) -> Optional[dict[str, Any]]:
        """
        Fetches a user record by username.

        Args:
            username: Unique username string.

        Returns:
            User dict or None if not found.
        """
        self._pool_required()
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT user_id, username, email,
                       password_hash, is_active, is_admin,
                       created_at, updated_at
                FROM   users
                WHERE  username = $1
                """,
                username,
            )
        return dict(row) if row else None

    async def get_user_by_id(
        self, user_id: str
    ) -> Optional[dict[str, Any]]:
        """Fetches a user record by UUID."""
        self._pool_required()
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT user_id, username, email,
                       is_active, is_admin, created_at
                FROM   users
                WHERE  user_id = $1::uuid
                """,
                user_id,
            )
        return dict(row) if row else None

    async def create_user(
        self,
        username: str,
        password_hash: str,
        email: Optional[str] = None,
        is_admin: bool = False,
    ) -> str:
        """
        Creates a new user and returns the generated UUID.

        Args:
            username: Unique username.
            password_hash: bcrypt hash.
            email: Optional email address.
            is_admin: Admin flag.

        Returns:
            New user UUID string.
        """
        self._pool_required()
        async with self._pool.acquire() as conn:
            user_id = await conn.fetchval(
                """
                INSERT INTO users
                    (username, email, password_hash, is_admin)
                VALUES ($1, $2, $3, $4)
                RETURNING user_id::text
                """,
                username, email, password_hash, is_admin,
            )
        logger.info(f"User created | username={username}")
        return user_id

    async def update_user_password(
        self, user_id: str, new_hash: str
    ) -> None:
        """Updates the password hash for a user."""
        self._pool_required()
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE users
                SET    password_hash = $1, updated_at = NOW()
                WHERE  user_id = $2::uuid
                """,
                new_hash, user_id,
            )

    # ═════════════════════════════════════════
    # Section 2 — Strategies
    # ═════════════════════════════════════════

    async def upsert_strategy(
        self,
        name: str,
        display_name: str,
        module_path: str,
        class_name: str,
        description: str,
        parameters: dict[str, Any],
        is_active: bool,
        broker_name: str,
    ) -> None:
        """
        Upserts a strategy record.
        Called by StrategyEngine after plugin discovery.

        Args:
            name: Unique strategy name (snake_case).
            display_name: Human-readable name.
            module_path: Python module path.
            class_name: Strategy class name.
            description: Strategy description text.
            parameters: Default parameter dict.
            is_active: Current activation status.
            broker_name: Associated broker.
        """
        self._pool_required()
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO strategies
                    (name, display_name, module_path, class_name,
                     description, parameters, is_active,
                     broker_name, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, $8, NOW())
                ON CONFLICT (name) DO UPDATE
                SET display_name = EXCLUDED.display_name,
                    module_path  = EXCLUDED.module_path,
                    class_name   = EXCLUDED.class_name,
                    description  = EXCLUDED.description,
                    parameters   = EXCLUDED.parameters,
                    is_active    = EXCLUDED.is_active,
                    broker_name  = EXCLUDED.broker_name,
                    updated_at   = NOW()
                """,
                name, display_name, module_path, class_name,
                description, json.dumps(parameters),
                is_active, broker_name,
            )

    async def set_strategy_active(
        self, name: str, is_active: bool
    ) -> None:
        """Updates the is_active flag for a strategy."""
        self._pool_required()
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE strategies
                SET    is_active = $1, updated_at = NOW()
                WHERE  name      = $2
                """,
                is_active, name,
            )

    async def get_active_strategies(
        self, broker_name: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """
        Returns all strategies with is_active = TRUE.

        Args:
            broker_name: Optional broker filter.

        Returns:
            List of strategy dicts.
        """
        self._pool_required()
        async with self._pool.acquire() as conn:
            if broker_name:
                rows = await conn.fetch(
                    """
                    SELECT name, display_name, parameters,
                           broker_name
                    FROM   strategies
                    WHERE  is_active   = TRUE
                    AND    broker_name = $1
                    """,
                    broker_name,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT name, display_name, parameters,
                           broker_name
                    FROM   strategies
                    WHERE  is_active = TRUE
                    """,
                )
        results = []
        for row in rows:
            d = dict(row)
            # Deserialise JSONB parameters
            if isinstance(d.get("parameters"), str):
                d["parameters"] = json.loads(d["parameters"])
            results.append(d)
        return results

    # ═════════════════════════════════════════
    # Section 3 — Orders
    # ═════════════════════════════════════════

    async def insert_order(
        self,
        order_id: str,
        broker_name: str,
        symbol: str,
        exchange: str,
        segment: str,
        order_type: str,
        transaction_type: str,
        quantity: int,
        trading_mode: str,
        strategy_name: Optional[str] = None,
        broker_order_id: Optional[str] = None,
        price: Optional[float] = None,
        trigger_price: Optional[float] = None,
        product_type: str = "INTRADAY",
        tags: Optional[dict] = None,
    ) -> None:
        """
        Inserts a new order record into the orders table.

        Args:
            order_id: NexaTrade internal order UUID.
            broker_name: Active broker name.
            symbol: Instrument symbol.
            exchange: Exchange code.
            segment: Market segment.
            order_type: MARKET / LIMIT / STOP_LOSS.
            transaction_type: BUY / SELL.
            quantity: Order quantity.
            trading_mode: paper / live.
            strategy_name: Originating strategy (optional).
            broker_order_id: Broker-assigned order ID.
            price: Limit price (optional).
            trigger_price: Stop trigger price (optional).
            product_type: INTRADAY / DELIVERY / etc.
            tags: Arbitrary JSON tags.
        """
        self._pool_required()
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO orders (
                    order_id, broker_order_id, broker_name,
                    symbol, exchange, segment,
                    order_type, transaction_type, product_type,
                    quantity, price, trigger_price,
                    trading_mode, strategy_name, tags
                ) VALUES (
                    $1,  $2,  $3,  $4,  $5,  $6,
                    $7,  $8,  $9,  $10, $11, $12,
                    $13, $14, $15::jsonb
                )
                ON CONFLICT (order_id) DO NOTHING
                """,
                order_id, broker_order_id, broker_name,
                symbol, exchange, segment,
                order_type, transaction_type, product_type,
                quantity, price, trigger_price,
                trading_mode, strategy_name,
                json.dumps(tags or {}),
            )

    async def update_order_status(
        self,
        order_id: str,
        status: str,
        broker_order_id: Optional[str] = None,
        filled_quantity: int = 0,
        average_price: Optional[float] = None,
        rejection_reason: Optional[str] = None,
    ) -> None:
        """
        Updates the status of an existing order.

        Args:
            order_id: NexaTrade order ID.
            status: New order status string.
            broker_order_id: Broker order ID (if now available).
            filled_quantity: Filled quantity.
            average_price: Average fill price.
            rejection_reason: Rejection message if rejected.
        """
        self._pool_required()
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE orders
                SET    status           = $1,
                       broker_order_id  = COALESCE($2, broker_order_id),
                       filled_quantity  = $3,
                       average_price    = COALESCE($4, average_price),
                       rejection_reason = COALESCE($5, rejection_reason),
                       updated_at       = NOW()
                WHERE  order_id         = $6
                """,
                status, broker_order_id,
                filled_quantity, average_price,
                rejection_reason, order_id,
            )

    async def get_order(
        self, order_id: str
    ) -> Optional[dict[str, Any]]:
        """Returns a single order record by order_id."""
        self._pool_required()
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT * FROM orders WHERE order_id = $1
                """,
                order_id,
            )
        return dict(row) if row else None

    async def get_orders(
        self,
        broker_name: Optional[str] = None,
        trading_mode: Optional[str] = None,
        strategy_name: Optional[str] = None,
        symbol: Optional[str] = None,
        from_dt: Optional[datetime] = None,
        to_dt: Optional[datetime] = None,
        limit: int = 500,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """
        Returns orders matching the given filters.

        Args:
            broker_name: Filter by broker.
            trading_mode: Filter by paper/live.
            strategy_name: Filter by strategy.
            symbol: Filter by symbol.
            from_dt: Filter orders placed after this datetime.
            to_dt: Filter orders placed before this datetime.
            limit: Max rows to return.
            offset: Pagination offset.

        Returns:
            List of order dicts ordered by placed_at DESC.
        """
        self._pool_required()
        conditions = []
        params     = []
        idx        = 1

        if broker_name:
            conditions.append(f"broker_name = ${idx}")
            params.append(broker_name)
            idx += 1
        if trading_mode:
            conditions.append(f"trading_mode = ${idx}")
            params.append(trading_mode)
            idx += 1
        if strategy_name:
            conditions.append(f"strategy_name = ${idx}")
            params.append(strategy_name)
            idx += 1
        if symbol:
            conditions.append(f"symbol = ${idx}")
            params.append(symbol.upper())
            idx += 1
        if from_dt:
            conditions.append(f"placed_at >= ${idx}")
            params.append(from_dt)
            idx += 1
        if to_dt:
            conditions.append(f"placed_at <= ${idx}")
            params.append(to_dt)
            idx += 1

        where  = "WHERE " + " AND ".join(conditions) if conditions else ""
        params += [limit, offset]

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT * FROM orders
                {where}
                ORDER BY placed_at DESC
                LIMIT  ${idx} OFFSET ${idx + 1}
                """,
                *params,
            )
        return [dict(r) for r in rows]

    async def get_today_order_count(
        self,
        broker_name: str,
        trading_mode: str,
    ) -> int:
        """Returns number of orders placed today for a broker."""
        self._pool_required()
        async with self._pool.acquire() as conn:
            count = await conn.fetchval(
                """
                SELECT COUNT(*)
                FROM   orders
                WHERE  broker_name  = $1
                AND    trading_mode = $2
                AND    placed_at   >= CURRENT_DATE
                """,
                broker_name, trading_mode,
            )
        return int(count or 0)

    # ═════════════════════════════════════════
    # Section 4 — Positions
    # ═════════════════════════════════════════

    async def upsert_position(
        self,
        broker_name: str,
        trading_mode: str,
        symbol: str,
        exchange: str,
        segment: str,
        quantity: int,
        average_price: float,
        last_price: float,
        realized_pnl: float,
        unrealized_pnl: float,
        strategy_name: Optional[str] = None,
    ) -> None:
        """
        Upserts a position record.
        Called after every fill to keep positions in sync.
        """
        self._pool_required()
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO positions (
                    broker_name, trading_mode, symbol,
                    exchange, segment, quantity,
                    average_price, last_price,
                    realized_pnl, unrealized_pnl,
                    strategy_name, updated_at
                ) VALUES (
                    $1, $2, $3, $4, $5, $6,
                    $7, $8, $9, $10, $11, NOW()
                )
                ON CONFLICT (broker_name, trading_mode, symbol, exchange)
                DO UPDATE SET
                    quantity       = EXCLUDED.quantity,
                    average_price  = EXCLUDED.average_price,
                    last_price     = EXCLUDED.last_price,
                    realized_pnl   = EXCLUDED.realized_pnl,
                    unrealized_pnl = EXCLUDED.unrealized_pnl,
                    strategy_name  = EXCLUDED.strategy_name,
                    updated_at     = NOW()
                """,
                broker_name, trading_mode, symbol,
                exchange, segment, quantity,
                average_price, last_price,
                realized_pnl, unrealized_pnl, strategy_name,
            )

    async def get_positions(
        self,
        broker_name: str,
        trading_mode: str,
        open_only: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Returns positions for a broker and mode.

        Args:
            broker_name: Broker filter.
            trading_mode: paper / live.
            open_only: If True, only returns non-zero qty.

        Returns:
            List of position dicts.
        """
        self._pool_required()
        qty_filter = "AND quantity != 0" if open_only else ""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT * FROM positions
                WHERE  broker_name  = $1
                AND    trading_mode = $2
                {qty_filter}
                ORDER BY updated_at DESC
                """,
                broker_name, trading_mode,
            )
        return [dict(r) for r in rows]

    # ═════════════════════════════════════════
    # Section 5 — Backtest Runs
    # ═════════════════════════════════════════

    async def insert_backtest_run(
        self,
        run_id: str,
        strategy_name: str,
        symbol: str,
        interval: str,
        start_date: str,
        end_date: str,
        initial_capital: float,
        parameters: dict[str, Any],
        broker_name: Optional[str] = None,
        exchange: str = "NSE",
    ) -> None:
        """Inserts a new backtest run record with PENDING status."""
        self._pool_required()
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO backtest_runs (
                    run_id, strategy_name, symbol, exchange,
                    interval, start_date, end_date,
                    initial_capital, parameters, broker_name
                ) VALUES (
                    $1, $2, $3, $4, $5, $6::date, $7::date,
                    $8, $9::jsonb, $10
                )
                ON CONFLICT (run_id) DO NOTHING
                """,
                run_id, strategy_name, symbol, exchange,
                interval, start_date, end_date,
                initial_capital, json.dumps(parameters),
                broker_name,
            )

    async def update_backtest_run(
        self,
        run_id: str,
        status: str,
        final_capital: Optional[float] = None,
        total_pnl: Optional[float] = None,
        total_trades: Optional[int] = None,
        win_rate: Optional[float] = None,
        max_drawdown: Optional[float] = None,
        sharpe_ratio: Optional[float] = None,
    ) -> None:
        """Updates a backtest run with final metrics on completion."""
        self._pool_required()
        completed_at = (
            now_utc() if status in ("COMPLETE", "FAILED") else None
        )
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE backtest_runs
                SET    status        = $1,
                       final_capital = COALESCE($2, final_capital),
                       total_pnl     = COALESCE($3, total_pnl),
                       total_trades  = COALESCE($4, total_trades),
                       win_rate      = COALESCE($5, win_rate),
                       max_drawdown  = COALESCE($6, max_drawdown),
                       sharpe_ratio  = COALESCE($7, sharpe_ratio),
                       completed_at  = COALESCE($8, completed_at)
                WHERE  run_id        = $9
                """,
                status, final_capital, total_pnl,
                total_trades, win_rate, max_drawdown,
                sharpe_ratio, completed_at, run_id,
            )

    async def get_backtest_runs(
        self,
        strategy_name: Optional[str] = None,
        symbol: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """
        Returns backtest run records ordered by created_at DESC.

        Args:
            strategy_name: Filter by strategy.
            symbol: Filter by symbol.
            limit: Page size.
            offset: Pagination offset.

        Returns:
            List of backtest run dicts.
        """
        self._pool_required()
        conditions: list[str] = []
        params: list[Any]     = []
        idx = 1

        if strategy_name:
            conditions.append(f"strategy_name = ${idx}")
            params.append(strategy_name)
            idx += 1
        if symbol:
            conditions.append(f"symbol = ${idx}")
            params.append(symbol.upper())
            idx += 1

        where = "WHERE " + " AND ".join(conditions) if conditions else ""
        params += [limit, offset]

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT * FROM backtest_runs
                {where}
                ORDER BY created_at DESC
                LIMIT  ${idx} OFFSET ${idx + 1}
                """,
                *params,
            )

        results = []
        for row in rows:
            d = dict(row)
            if isinstance(d.get("parameters"), str):
                d["parameters"] = json.loads(d["parameters"])
            results.append(d)
        return results

    # ═════════════════════════════════════════
    # Section 6 — Risk Snapshots
    # ═════════════════════════════════════════

    async def upsert_risk_snapshot(
        self,
        broker_name: str,
        trading_mode: str,
        daily_pnl: float,
        open_positions: int,
        total_orders: int,
        max_drawdown_pct: float,
        portfolio_value: float,
    ) -> None:
        """
        Upserts today's risk snapshot.
        Called periodically by a background task.
        """
        self._pool_required()
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO risk_snapshots (
                    broker_name, trading_mode,
                    snapshot_date, daily_pnl,
                    open_positions, total_orders,
                    max_drawdown_pct, portfolio_value
                ) VALUES (
                    $1, $2, CURRENT_DATE,
                    $3, $4, $5, $6, $7
                )
                ON CONFLICT (broker_name, trading_mode, snapshot_date)
                DO UPDATE SET
                    daily_pnl       = EXCLUDED.daily_pnl,
                    open_positions  = EXCLUDED.open_positions,
                    total_orders    = EXCLUDED.total_orders,
                    max_drawdown_pct = EXCLUDED.max_drawdown_pct,
                    portfolio_value = EXCLUDED.portfolio_value,
                    created_at      = NOW()
                """,
                broker_name, trading_mode,
                daily_pnl, open_positions, total_orders,
                max_drawdown_pct, portfolio_value,
            )

    async def get_latest_risk_snapshot(
        self, broker_name: str
    ) -> Optional[dict[str, Any]]:
        """Returns the most recent risk snapshot for a broker."""
        self._pool_required()
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT * FROM risk_snapshots
                WHERE  broker_name = $1
                ORDER  BY snapshot_date DESC, created_at DESC
                LIMIT  1
                """,
                broker_name,
            )
        return dict(row) if row else None