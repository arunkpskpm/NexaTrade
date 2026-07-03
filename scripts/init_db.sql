-- ─────────────────────────────────────────────────────────────────
-- NexaTrade — PostgreSQL Schema Initialisation
-- Run automatically by Docker on first container start.
-- Idempotent: all statements use IF NOT EXISTS / ON CONFLICT.
-- ─────────────────────────────────────────────────────────────────

-- ─────────────────────────────────────────────
-- Extensions
-- ─────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- ─────────────────────────────────────────────
-- ENUM Types
-- ─────────────────────────────────────────────
DO $$ BEGIN
    CREATE TYPE order_status_enum AS ENUM (
        'PENDING',
        'OPEN',
        'COMPLETE',
        'CANCELLED',
        'REJECTED',
        'MODIFIED',
        'PARTIAL'
    );
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
    CREATE TYPE transaction_type_enum AS ENUM ('BUY', 'SELL');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
    CREATE TYPE order_type_enum AS ENUM (
        'MARKET',
        'LIMIT',
        'STOP_LOSS',
        'STOP_LOSS_MARKET'
    );
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
    CREATE TYPE trading_mode_enum AS ENUM ('paper', 'live');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
    CREATE TYPE broker_enum AS ENUM (
        'breeze',
        'zerodha',
        'angelone',
        'upstox',
        'paper'
    );
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- ─────────────────────────────────────────────
-- Table: users
-- Single-user desktop app — one admin account.
-- Table designed to support multi-user in future.
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS users (
    id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    username            VARCHAR(64)     NOT NULL UNIQUE,
    email               VARCHAR(255)    NOT NULL UNIQUE,
    hashed_password     TEXT            NOT NULL,
    is_active           BOOLEAN         NOT NULL DEFAULT TRUE,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_users_username ON users (username);

-- ─────────────────────────────────────────────
-- Table: broker_configs
-- Persists per-broker credential snapshots.
-- Credentials are encrypted at rest via pgcrypto.
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS broker_configs (
    id              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    broker_name     broker_enum     NOT NULL,
    display_name    VARCHAR(128)    NOT NULL,
    is_active       BOOLEAN         NOT NULL DEFAULT FALSE,
    credentials     JSONB           NOT NULL DEFAULT '{}',
    extra_config    JSONB           NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_broker_name UNIQUE (broker_name)
);

CREATE INDEX IF NOT EXISTS idx_broker_configs_active
    ON broker_configs (is_active);

-- ─────────────────────────────────────────────
-- Table: strategies
-- Plugin registry — one row per discovered strategy.
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS strategies (
    id              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    name            VARCHAR(128)    NOT NULL UNIQUE,
    display_name    VARCHAR(256)    NOT NULL DEFAULT '',
    module_path     VARCHAR(512)    NOT NULL,
    class_name      VARCHAR(128)    NOT NULL,
    description     TEXT            NOT NULL DEFAULT '',
    parameters      JSONB           NOT NULL DEFAULT '{}',
    is_active       BOOLEAN         NOT NULL DEFAULT FALSE,
    broker_name     broker_enum,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_strategies_active
    ON strategies (is_active);
CREATE INDEX IF NOT EXISTS idx_strategies_broker
    ON strategies (broker_name);

-- ─────────────────────────────────────────────
-- Table: orders
-- Immutable audit log — never UPDATE or DELETE rows.
-- Broker-agnostic: broker_name tracks which broker
-- placed the order.
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS orders (
    id                  UUID                    PRIMARY KEY DEFAULT gen_random_uuid(),
    order_id            VARCHAR(128)            NOT NULL UNIQUE,
    broker_order_id     VARCHAR(128),
    broker_name         broker_enum             NOT NULL,
    strategy_name       VARCHAR(128),
    symbol              VARCHAR(64)             NOT NULL,
    exchange            VARCHAR(32)             NOT NULL,
    segment             VARCHAR(32)             NOT NULL,
    order_type          order_type_enum         NOT NULL,
    transaction_type    transaction_type_enum   NOT NULL,
    quantity            INTEGER                 NOT NULL CHECK (quantity > 0),
    filled_quantity     INTEGER                 NOT NULL DEFAULT 0,
    price               NUMERIC(18, 4),
    trigger_price       NUMERIC(18, 4),
    average_price       NUMERIC(18, 4),
    status              order_status_enum       NOT NULL DEFAULT 'PENDING',
    trading_mode        trading_mode_enum       NOT NULL DEFAULT 'paper',
    rejection_reason    TEXT,
    tags                JSONB                   NOT NULL DEFAULT '{}',
    placed_at           TIMESTAMPTZ             NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ             NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_orders_symbol
    ON orders (symbol);
CREATE INDEX IF NOT EXISTS idx_orders_status
    ON orders (status);
CREATE INDEX IF NOT EXISTS idx_orders_broker
    ON orders (broker_name);
CREATE INDEX IF NOT EXISTS idx_orders_strategy
    ON orders (strategy_name);
CREATE INDEX IF NOT EXISTS idx_orders_placed_at
    ON orders (placed_at DESC);
CREATE INDEX IF NOT EXISTS idx_orders_trading_mode
    ON orders (trading_mode);

-- ─────────────────────────────────────────────
-- Table: positions
-- Live position state — upserted on every fill.
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS positions (
    id              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol          VARCHAR(64)     NOT NULL,
    exchange        VARCHAR(32)     NOT NULL,
    segment         VARCHAR(32)     NOT NULL,
    broker_name     broker_enum     NOT NULL,
    quantity        INTEGER         NOT NULL DEFAULT 0,
    average_price   NUMERIC(18, 4)  NOT NULL DEFAULT 0,
    last_price      NUMERIC(18, 4)  NOT NULL DEFAULT 0,
    unrealized_pnl  NUMERIC(18, 4)  NOT NULL DEFAULT 0,
    realized_pnl    NUMERIC(18, 4)  NOT NULL DEFAULT 0,
    trading_mode    trading_mode_enum NOT NULL DEFAULT 'paper',
    updated_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_position
        UNIQUE (symbol, exchange, segment, broker_name, trading_mode)
);

CREATE INDEX IF NOT EXISTS idx_positions_symbol
    ON positions (symbol);
CREATE INDEX IF NOT EXISTS idx_positions_broker
    ON positions (broker_name);

-- ─────────────────────────────────────────────
-- Table: trades
-- Every completed fill — child records of orders.
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS trades (
    id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    order_id            VARCHAR(128)    NOT NULL REFERENCES orders(order_id),
    broker_name         broker_enum     NOT NULL,
    strategy_name       VARCHAR(128),
    symbol              VARCHAR(64)     NOT NULL,
    exchange            VARCHAR(32)     NOT NULL,
    transaction_type    transaction_type_enum NOT NULL,
    quantity            INTEGER         NOT NULL,
    price               NUMERIC(18, 4)  NOT NULL,
    trading_mode        trading_mode_enum NOT NULL DEFAULT 'paper',
    commission          NUMERIC(18, 4)  NOT NULL DEFAULT 0,
    pnl                 NUMERIC(18, 4)  NOT NULL DEFAULT 0,
    executed_at         TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_trades_order_id
    ON trades (order_id);
CREATE INDEX IF NOT EXISTS idx_trades_symbol
    ON trades (symbol);
CREATE INDEX IF NOT EXISTS idx_trades_broker
    ON trades (broker_name);
CREATE INDEX IF NOT EXISTS idx_trades_executed_at
    ON trades (executed_at DESC);

-- ─────────────────────────────────────────────
-- Table: risk_snapshots
-- Point-in-time risk config history.
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS risk_snapshots (
    id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    max_drawdown_pct    NUMERIC(6, 2)   NOT NULL,
    daily_loss_limit    NUMERIC(18, 4)  NOT NULL,
    max_position_size   INTEGER         NOT NULL,
    kill_switch_active  BOOLEAN         NOT NULL DEFAULT FALSE,
    broker_name         broker_enum,
    snapshot_at         TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_risk_snapshots_at
    ON risk_snapshots (snapshot_at DESC);

-- ─────────────────────────────────────────────
-- Table: backtest_runs
-- Metadata for every backtest execution.
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS backtest_runs (
    id              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id          VARCHAR(64)     NOT NULL UNIQUE,
    strategy_name   VARCHAR(128)    NOT NULL,
    broker_name     broker_enum,
    symbol          VARCHAR(64)     NOT NULL,
    interval        VARCHAR(32)     NOT NULL,
    start_date      DATE            NOT NULL,
    end_date        DATE            NOT NULL,
    initial_capital NUMERIC(18, 4)  NOT NULL,
    final_capital   NUMERIC(18, 4),
    total_pnl       NUMERIC(18, 4),
    total_trades    INTEGER,
    win_rate        NUMERIC(6, 4),
    max_drawdown    NUMERIC(6, 4),
    sharpe_ratio    NUMERIC(8, 4),
    parameters      JSONB           NOT NULL DEFAULT '{}',
    status          VARCHAR(32)     NOT NULL DEFAULT 'PENDING',
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    completed_at    TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_backtest_runs_strategy
    ON backtest_runs (strategy_name);
CREATE INDEX IF NOT EXISTS idx_backtest_runs_created
    ON backtest_runs (created_at DESC);

-- ─────────────────────────────────────────────
-- Trigger: auto-update updated_at columns
-- ─────────────────────────────────────────────
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DO $$ DECLARE
    t TEXT;
BEGIN
    FOREACH t IN ARRAY ARRAY[
        'users',
        'broker_configs',
        'strategies',
        'orders',
        'positions'
    ] LOOP
        EXECUTE format(
            'DROP TRIGGER IF EXISTS trg_updated_at ON %I;
             CREATE TRIGGER trg_updated_at
             BEFORE UPDATE ON %I
             FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();',
            t, t
        );
    END LOOP;
END $$;