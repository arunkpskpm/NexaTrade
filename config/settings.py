"""
NexaTrade — Centralised Settings & Configuration.

All application configuration is loaded here — exactly once —
and served through a singleton get_settings() function.

Configuration sources (in priority order):
    1. Environment variables (.env file)
    2. config/app_config.yaml
    3. config/risk_config.yaml
    4. config/brokers/{broker_name}.yaml (per-broker)

Environment variables always override YAML values.

Design rules:
    - get_settings() is the ONLY way to access config anywhere
    - No code imports from .env or YAML files directly
    - All secrets are SecretStr (never logged or serialised)
    - Settings object is immutable after creation (frozen)
    - Broker credentials are loaded lazily on first access

Usage:
    from config.settings import get_settings

    settings = get_settings()
    db_dsn   = settings.postgres.raw_dsn
    active   = settings.active_broker
    risk     = settings.risk_params
    creds    = settings.broker_credentials("breeze")
    cfg      = settings.broker_config("breeze")
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env before anything else
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_ENV_FILE, override=False)

_CONFIG_DIR  = Path(__file__).resolve().parent
_PROJECT_DIR = _CONFIG_DIR.parent


# ═════════════════════════════════════════════
# Section 1 — Sub-Setting Models
# ═════════════════════════════════════════════

class PostgresSettings(BaseSettings):
    """PostgreSQL connection settings."""

    model_config = SettingsConfigDict(
        env_prefix="POSTGRES_", extra="ignore"
    )

    host:     str = Field(default="localhost")
    port:     int = Field(default=5432)
    user:     str = Field(default="nexatrade")
    password: SecretStr = Field(default=SecretStr("nexatrade"))
    db:       str = Field(default="nexatrade_db")
    schema:   str = Field(default="public")

    @property
    def raw_dsn(self) -> str:
        """Returns asyncpg-compatible DSN string."""
        pwd = self.password.get_secret_value()
        return (
            f"postgresql://{self.user}:{pwd}"
            f"@{self.host}:{self.port}/{self.db}"
        )

    @property
    def dsn_masked(self) -> str:
        """Returns DSN with password masked (safe to log)."""
        return (
            f"postgresql://{self.user}:***"
            f"@{self.host}:{self.port}/{self.db}"
        )


class InfluxSettings(BaseSettings):
    """InfluxDB v2 connection settings."""

    model_config = SettingsConfigDict(
        env_prefix="INFLUX_", extra="ignore"
    )

    url:    str       = Field(default="http://localhost:8086")
    token:  SecretStr = Field(default=SecretStr("nexatrade-token"))
    org:    str       = Field(default="nexatrade")
    bucket: str       = Field(default="market_data")


class RedisSettings(BaseSettings):
    """Redis connection settings."""

    model_config = SettingsConfigDict(
        env_prefix="REDIS_", extra="ignore"
    )

    host:     str = Field(default="localhost")
    port:     int = Field(default=6379)
    db:       int = Field(default=0)
    password: Optional[SecretStr] = Field(default=None)

    @property
    def url(self) -> str:
        """Returns redis:// URL string."""
        pwd = self.password
        if pwd:
            pw_str = pwd.get_secret_value()
            return (
                f"redis://:{pw_str}@{self.host}:{self.port}/{self.db}"
            )
        return f"redis://{self.host}:{self.port}/{self.db}"


class JWTSettings(BaseSettings):
    """JWT authentication settings."""

    model_config = SettingsConfigDict(
        env_prefix="JWT_", extra="ignore"
    )

    secret_key:   SecretStr = Field(
        default=SecretStr("change-me-in-production")
    )
    algorithm:    str       = Field(default="HS256")
    expire_minutes: int     = Field(default=1440)  # 24 hours


class BrokerCredentials(BaseSettings):
    """
    Per-broker authentication credentials.
    Loaded from environment variables with broker-specific prefix.

    Breeze: BREEZE_API_KEY, BREEZE_API_SECRET, BREEZE_SESSION_TOKEN
    """

    model_config = SettingsConfigDict(extra="ignore")

    api_key:       SecretStr = Field(default=SecretStr(""))
    api_secret:    SecretStr = Field(default=SecretStr(""))
    session_token: SecretStr = Field(default=SecretStr(""))
    access_token:  SecretStr = Field(default=SecretStr(""))
    client_id:     Optional[str] = Field(default=None)


# ═════════════════════════════════════════════
# Section 2 — Main Settings Class
# ═════════════════════════════════════════════

class Settings(BaseSettings):
    """
    NexaTrade Main Application Settings.

    Reads from environment variables and YAML config files.
    All sub-settings are lazily instantiated on first access.
    """

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Core application settings ─────────────
    app_name:      str = Field(default="NexaTrade",     env="APP_NAME")
    app_version:   str = Field(default="1.0.0",         env="APP_VERSION")
    environment:   str = Field(default="development",   env="APP_ENV")
    debug:         bool = Field(default=False,           env="DEBUG")
    log_level:     str = Field(default="INFO",           env="LOG_LEVEL")
    secret_key:    SecretStr = Field(
        default=SecretStr("change-me-in-production"),
        env="SECRET_KEY",
    )

    # ── Broker settings ────────────────────────
    active_broker: str = Field(default="paper",         env="ACTIVE_BROKER")
    trading_mode:  str = Field(default="paper",         env="TRADING_MODE")

    # ── API settings ──────────────────────────
    api_host:      str  = Field(default="0.0.0.0",      env="API_HOST")
    api_port:      int  = Field(default=8000,            env="API_PORT")
    api_reload:    bool = Field(default=False,           env="API_RELOAD")

    # ── CORS settings ─────────────────────────
    cors_origins:  list[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173"],
        env="CORS_ORIGINS",
    )

    @field_validator("trading_mode")
    @classmethod
    def validate_trading_mode(cls, v: str) -> str:
        allowed = {"paper", "live"}
        v = v.lower().strip()
        if v not in allowed:
            raise ValueError(
                f"TRADING_MODE must be one of {allowed}, got '{v}'"
            )
        return v

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        allowed = {"development", "staging", "production"}
        v = v.lower().strip()
        if v not in allowed:
            raise ValueError(
                f"APP_ENV must be one of {allowed}"
            )
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v = v.upper().strip()
        if v not in allowed:
            return "INFO"
        return v

    # ─────────────────────────────────────────
    # Sub-Settings (Lazy Properties)
    # ─────────────────────────────────────────

    @property
    def postgres(self) -> PostgresSettings:
        """Returns PostgreSQL connection settings."""
        return PostgresSettings()

    @property
    def influx(self) -> InfluxSettings:
        """Returns InfluxDB connection settings."""
        return InfluxSettings()

    @property
    def redis(self) -> RedisSettings:
        """Returns Redis connection settings."""
        return RedisSettings()

    @property
    def jwt(self) -> JWTSettings:
        """Returns JWT settings."""
        return JWTSettings()

    # ─────────────────────────────────────────
    # YAML Config Loaders
    # ─────────────────────────────────────────

    @property
    def app_config(self) -> dict[str, Any]:
        """
        Returns the parsed app_config.yaml dict.
        Cached after first load.
        """
        return _load_yaml(_CONFIG_DIR / "app_config.yaml")

    @property
    def risk_params(self) -> dict[str, Any]:
        """
        Returns the parsed risk_config.yaml dict.
        Cached after first load.
        """
        return _load_yaml(_CONFIG_DIR / "risk_config.yaml")

    @property
    def feed_config(self) -> dict[str, Any]:
        """Returns the feed section of app_config.yaml."""
        return self.app_config.get("feed", {})

    # ─────────────────────────────────────────
    # Broker Config & Credentials
    # ─────────────────────────────────────────

    def broker_config(self, broker_name: str) -> dict[str, Any]:
        """
        Returns the broker-specific YAML configuration.

        Config file: config/brokers/{broker_name}.yaml

        Args:
            broker_name: Broker identifier string.

        Returns:
            Parsed YAML dict. Empty dict if file not found.

        Example:
            cfg = settings.broker_config("breeze")
            max_subs = cfg["broker"]["websocket"]["max_subscriptions"]
        """
        broker_config_file = (
            _CONFIG_DIR / "brokers" / f"{broker_name}.yaml"
        )
        return _load_yaml(broker_config_file)

    def broker_credentials(
        self, broker_name: str
    ) -> BrokerCredentials:
        """
        Returns credentials for the specified broker.
        Reads from environment variables using broker-specific prefix.

        Env var naming convention:
            {BROKER_NAME_UPPER}_{CREDENTIAL_NAME}
            e.g. BREEZE_API_KEY, BREEZE_API_SECRET

        Args:
            broker_name: Broker identifier string.

        Returns:
            BrokerCredentials model with SecretStr fields.

        Example:
            creds = settings.broker_credentials("breeze")
            key = creds.api_key.get_secret_value()
        """
        prefix = broker_name.upper()
        return BrokerCredentials(
            api_key=SecretStr(
                os.getenv(f"{prefix}_API_KEY", "")
            ),
            api_secret=SecretStr(
                os.getenv(f"{prefix}_API_SECRET", "")
            ),
            session_token=SecretStr(
                os.getenv(f"{prefix}_SESSION_TOKEN", "")
            ),
            access_token=SecretStr(
                os.getenv(f"{prefix}_ACCESS_TOKEN", "")
            ),
            client_id=os.getenv(f"{prefix}_CLIENT_ID"),
        )

    # ─────────────────────────────────────────
    # Derived Properties
    # ─────────────────────────────────────────

    @property
    def is_production(self) -> bool:
        """Returns True if running in production environment."""
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        """Returns True if running in development environment."""
        return self.environment == "development"

    @property
    def is_live_trading(self) -> bool:
        """Returns True if trading mode is live."""
        return self.trading_mode == "live"

    @property
    def is_paper_trading(self) -> bool:
        """Returns True if trading mode is paper."""
        return self.trading_mode == "paper"

    def __repr__(self) -> str:
        return (
            f"Settings("
            f"env={self.environment!r}, "
            f"broker={self.active_broker!r}, "
            f"mode={self.trading_mode!r}, "
            f"debug={self.debug})"
        )


# ═════════════════════════════════════════════
# Section 3 — YAML Loader (Cached)
# ═════════════════════════════════════════════

@lru_cache(maxsize=32)
def _load_yaml(path: Path) -> dict[str, Any]:
    """
    Loads and parses a YAML file.
    Results are cached via lru_cache — each file is read once.

    Args:
        path: Absolute path to YAML file.

    Returns:
        Parsed dict. Empty dict if file does not exist.
    """
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = yaml.safe_load(f)
            return content if isinstance(content, dict) else {}
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning(
            f"Failed to load YAML: {path} | error={exc}"
        )
        return {}


# ═════════════════════════════════════════════
# Section 4 — Singleton Factory
# ═════════════════════════════════════════════

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Returns the singleton Settings instance.

    This is the ONLY function the rest of NexaTrade uses
    to access configuration. Never instantiate Settings directly.

    Returns:
        Cached Settings instance (created on first call).

    Example:
        from config.settings import get_settings

        settings = get_settings()
        print(settings.active_broker)   # "breeze"
        print(settings.trading_mode)    # "live"
        print(settings.postgres.raw_dsn)
        print(settings.risk_params)
    """
    return Settings()