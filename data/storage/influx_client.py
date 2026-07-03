"""
NexaTrade — InfluxDB v2 Async Client.

Manages all time-series data writes and queries for
OHLCV candles and raw ticks.

Architecture:
    Measurement: candles
        Tags:    broker_name, symbol, exchange, interval
        Fields:  open, high, low, close, volume
        Time:    candle datetime (nanosecond precision, UTC)

    Measurement: ticks
        Tags:    broker_name, symbol, exchange
        Fields:  last_price, bid, ask, volume, oi
        Time:    tick timestamp (nanosecond precision, UTC)

Write strategy:
    Synchronous write_api with batching disabled
    (each write_candles call flushes immediately).
    This ensures data is available for immediate re-query.

Query strategy:
    All reads use Flux query language.
    Results are converted to pandas DataFrames with
    IST DatetimeIndex for strategy consumption.

Usage:
    influx = InfluxClient()
    await influx.initialise()

    await influx.write_candles(
        candles=[...],
        broker_name="breeze",
        symbol="RELIANCE",
        exchange="NSE",
        interval="5minute",
    )

    df = await influx.get_candles(
        symbol="RELIANCE",
        exchange="NSE",
        interval="5minute",
        broker_name="breeze",
        from_dt=start,
        to_dt=end,
    )

    await influx.shutdown()
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Optional

import pandas as pd
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

from config.settings import get_settings
from utils.logger import get_logger
from utils.time_utils import IST, to_utc

logger = get_logger(__name__)

# Thread pool for running synchronous InfluxDB SDK calls
_executor = ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="influx"
)


class InfluxClient:
    """
    NexaTrade InfluxDB v2 Client.

    Uses the official influxdb-client-python SDK.
    All I/O is offloaded to a thread pool to avoid
    blocking the asyncio event loop.

    Buckets are auto-created if they don't exist on init.
    """

    def __init__(self) -> None:
        self._client:    Optional[InfluxDBClient]  = None
        self._write_api  = None
        self._query_api  = None
        self._delete_api = None
        self._settings   = get_settings()

    # ─────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────

    async def initialise(self) -> None:
        """
        Creates the InfluxDB client and ensures the bucket exists.

        Raises:
            RuntimeError: If connection or auth fails.
        """
        cfg = self._settings.influx
        try:
            self._client = InfluxDBClient(
                url=cfg.url,
                token=cfg.token.get_secret_value(),
                org=cfg.org,
                timeout=10_000,  # ms
            )
            self._write_api  = self._client.write_api(
                write_options=SYNCHRONOUS
            )
            self._query_api  = self._client.query_api()
            self._delete_api = self._client.delete_api()

            # Verify connection
            await self._run_sync(self._client.ping)

            # Ensure bucket exists
            await self._ensure_bucket(cfg.bucket)

            logger.info(
                f"InfluxDB connected | "
                f"url={cfg.url} | "
                f"org={cfg.org} | "
                f"bucket={cfg.bucket}"
            )
        except Exception as exc:
            raise RuntimeError(
                f"InfluxDB init failed: {exc}"
            ) from exc

    async def shutdown(self) -> None:
        """Closes the InfluxDB client and thread pool."""
        if self._client:
            await self._run_sync(self._client.close)
            logger.info("InfluxDB client closed.")

    async def _ensure_bucket(self, bucket_name: str) -> None:
        """Creates the bucket if it does not already exist."""
        buckets_api = self._client.buckets_api()
        org         = self._settings.influx.org

        def _check_or_create():
            existing = buckets_api.find_buckets(name=bucket_name)
            if not existing or not existing.buckets:
                buckets_api.create_bucket(
                    bucket_name=bucket_name,
                    org=org,
                    retention_rules=[{
                        "type": "expire",
                        "everySeconds": 365 * 24 * 3600 * 5,  # 5 years
                    }],
                )
                logger.info(
                    f"InfluxDB bucket created | name={bucket_name}"
                )

        await self._run_sync(_check_or_create)

    # ─────────────────────────────────────────
    # Write — Candles
    # ─────────────────────────────────────────

    async def write_candles(
        self,
        candles: list[dict[str, Any]],
        broker_name: str,
        symbol: str,
        exchange: str,
        interval: str,
    ) -> None:
        """
        Writes OHLCV candles to InfluxDB.

        Args:
            candles: List of candle dicts with keys:
                     datetime, open, high, low, close, volume.
            broker_name: Source broker tag.
            symbol: Instrument symbol tag.
            exchange: Exchange tag.
            interval: Candle interval tag.
        """
        if not candles:
            return

        points = []
        for c in candles:
            dt = c.get("datetime")
            if dt is None:
                continue
            if hasattr(dt, "tzinfo"):
                ts = to_utc(dt)
            else:
                ts = to_utc(pd.Timestamp(dt).to_pydatetime())

            point = (
                Point("candles")
                .tag("broker_name", broker_name)
                .tag("symbol",      symbol.upper())
                .tag("exchange",    exchange.upper())
                .tag("interval",    interval)
                .field("open",   float(c.get("open",   0)))
                .field("high",   float(c.get("high",   0)))
                .field("low",    float(c.get("low",    0)))
                .field("close",  float(c.get("close",  0)))
                .field("volume", float(c.get("volume", 0)))
                .time(ts, WritePrecision.NANOSECONDS)
            )
            points.append(point)

        if not points:
            return

        bucket = self._settings.influx.bucket
        org    = self._settings.influx.org

        def _write():
            self._write_api.write(
                bucket=bucket, org=org, record=points
            )

        try:
            await self._run_sync(_write)
            logger.debug(
                f"InfluxDB candles written | "
                f"symbol={symbol} | "
                f"interval={interval} | "
                f"points={len(points)}"
            )
        except Exception as exc:
            logger.error(
                f"InfluxDB candle write failed | "
                f"symbol={symbol} | error={exc}"
            )

    # ─────────────────────────────────────────
    # Write — Ticks
    # ─────────────────────────────────────────

    async def write_tick(
        self,
        broker_name: str,
        symbol: str,
        exchange: str,
        last_price: float,
        bid: float = 0.0,
        ask: float = 0.0,
        volume: float = 0.0,
        oi: float = 0.0,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Writes a single tick to InfluxDB.

        Args:
            broker_name: Source broker.
            symbol: Instrument symbol.
            exchange: Exchange code.
            last_price: Last traded price.
            bid: Best bid.
            ask: Best ask.
            volume: Cumulative day volume.
            oi: Open interest.
            timestamp: Tick timestamp. Defaults to now UTC.
        """
        from utils.time_utils import now_utc
        ts    = to_utc(timestamp) if timestamp else now_utc()
        bucket = self._settings.influx.bucket
        org    = self._settings.influx.org

        point = (
            Point("ticks")
            .tag("broker_name", broker_name)
            .tag("symbol",      symbol.upper())
            .tag("exchange",    exchange.upper())
            .field("last_price", float(last_price))
            .field("bid",        float(bid))
            .field("ask",        float(ask))
            .field("volume",     float(volume))
            .field("oi",         float(oi))
            .time(ts, WritePrecision.NANOSECONDS)
        )

        def _write():
            self._write_api.write(
                bucket=bucket, org=org, record=point
            )

        try:
            await self._run_sync(_write)
        except Exception as exc:
            logger.debug(
                f"InfluxDB tick write failed | "
                f"symbol={symbol} | error={exc}"
            )

    # ─────────────────────────────────────────
    # Query — Candles
    # ─────────────────────────────────────────

    async def get_candles(
        self,
        symbol: str,
        exchange: str,
        interval: str,
        broker_name: str,
        from_dt: Optional[datetime] = None,
        to_dt: Optional[datetime] = None,
        limit: int = 10000,
    ) -> pd.DataFrame:
        """
        Queries OHLCV candles from InfluxDB.

        Args:
            symbol: Instrument symbol.
            exchange: Exchange code.
            interval: Candle interval string.
            broker_name: Broker tag filter.
            from_dt: Query start datetime (UTC-aware).
            to_dt: Query end datetime (UTC-aware).
            limit: Maximum rows to return.

        Returns:
            pandas DataFrame with DatetimeIndex (IST timezone)
            and columns [open, high, low, close, volume].
            Empty DataFrame if no data found.
        """
        from utils.time_utils import now_utc
        from datetime import timedelta

        bucket = self._settings.influx.bucket
        org    = self._settings.influx.org

        start  = from_dt or (now_utc() - timedelta(days=30))
        stop   = to_dt   or now_utc()

        start_str = start.strftime("%Y-%m-%dT%H:%M:%SZ")
        stop_str  = stop.strftime("%Y-%m-%dT%H:%M:%SZ")

        flux_query = f"""
        from(bucket: "{bucket}")
          |> range(start: {start_str}, stop: {stop_str})
          |> filter(fn: (r) => r._measurement == "candles")
          |> filter(fn: (r) => r.symbol      == "{symbol.upper()}")
          |> filter(fn: (r) => r.exchange     == "{exchange.upper()}")
          |> filter(fn: (r) => r.interval     == "{interval}")
          |> filter(fn: (r) => r.broker_name  == "{broker_name}")
          |> pivot(
               rowKey:["_time"],
               columnKey: ["_field"],
               valueColumn: "_value"
             )
          |> sort(columns: ["_time"], desc: false)
          |> limit(n: {limit})
        """

        def _query():
            return self._query_api.query_data_frame(
                flux_query, org=org
            )

        try:
            result = await self._run_sync(_query)
        except Exception as exc:
            logger.warning(
                f"InfluxDB candle query failed | "
                f"symbol={symbol} | error={exc}"
            )
            return pd.DataFrame()

        if result is None or (
            isinstance(result, pd.DataFrame) and result.empty
        ):
            return pd.DataFrame()

        # Handle list of DataFrames returned by query_data_frame
        if isinstance(result, list):
            if not result:
                return pd.DataFrame()
            result = pd.concat(result, ignore_index=True)

        # Normalise columns
        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(set(result.columns)):
            return pd.DataFrame()

        # Set DatetimeIndex
        result["_time"] = pd.to_datetime(result["_time"])
        result.set_index("_time", inplace=True)

        # Convert to IST
        if result.index.tz is None:
            result.index = result.index.tz_localize("UTC")
        result.index = result.index.tz_convert(IST)
        result.index.name = "datetime"

        # Return only OHLCV columns
        df = result[["open", "high", "low", "close", "volume"]].copy()
        df = df.astype(float)
        df.sort_index(inplace=True)
        return df

    # ─────────────────────────────────────────
    # Delete — Candles
    # ─────────────────────────────────────────

    async def delete_candles(
        self,
        broker_name: str,
        symbol: str,
        exchange: str,
        interval: str,
        from_dt: datetime,
        to_dt: datetime,
    ) -> None:
        """
        Deletes candle data for a symbol in a time range.
        Used for data correction workflows.

        Args:
            broker_name: Broker tag filter.
            symbol: Instrument symbol.
            exchange: Exchange code.
            interval: Candle interval.
            from_dt: Delete range start.
            to_dt: Delete range end.
        """
        bucket = self._settings.influx.bucket
        org    = self._settings.influx.org

        predicate = (
            f'_measurement="candles" '
            f'AND broker_name="{broker_name}" '
            f'AND symbol="{symbol.upper()}" '
            f'AND exchange="{exchange.upper()}" '
            f'AND interval="{interval}"'
        )

        start_utc = to_utc(from_dt)
        stop_utc  = to_utc(to_dt)

        def _delete():
            self._delete_api.delete(
                start=start_utc,
                stop=stop_utc,
                predicate=predicate,
                bucket=bucket,
                org=org,
            )

        try:
            await self._run_sync(_delete)
            logger.info(
                f"InfluxDB candles deleted | "
                f"symbol={symbol} | "
                f"interval={interval} | "
                f"range={from_dt}→{to_dt}"
            )
        except Exception as exc:
            logger.error(
                f"InfluxDB delete failed | "
                f"symbol={symbol} | error={exc}"
            )

    # ─────────────────────────────────────────
    # Query — Ticks
    # ─────────────────────────────────────────

    async def get_ticks(
        self,
        symbol: str,
        exchange: str,
        broker_name: str,
        from_dt: datetime,
        to_dt: datetime,
        limit: int = 50000,
    ) -> pd.DataFrame:
        """
        Queries raw tick data from InfluxDB.

        Args:
            symbol: Instrument symbol.
            exchange: Exchange code.
            broker_name: Broker tag filter.
            from_dt: Query start (UTC-aware).
            to_dt: Query end (UTC-aware).
            limit: Max rows.

        Returns:
            DataFrame with tick fields or empty DataFrame.
        """
        bucket    = self._settings.influx.bucket
        org       = self._settings.influx.org
        start_str = from_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        stop_str  = to_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        flux_query = f"""
        from(bucket: "{bucket}")
          |> range(start: {start_str}, stop: {stop_str})
          |> filter(fn: (r) => r._measurement == "ticks")
          |> filter(fn: (r) => r.symbol       == "{symbol.upper()}")
          |> filter(fn: (r) => r.exchange      == "{exchange.upper()}")
          |> filter(fn: (r) => r.broker_name   == "{broker_name}")
          |> pivot(
               rowKey:["_time"],
               columnKey: ["_field"],
               valueColumn: "_value"
             )
          |> sort(columns: ["_time"], desc: false)
          |> limit(n: {limit})
        """

        def _query():
            return self._query_api.query_data_frame(
                flux_query, org=org
            )

        try:
            result = await self._run_sync(_query)
        except Exception as exc:
            logger.warning(
                f"InfluxDB tick query failed | "
                f"symbol={symbol} | error={exc}"
            )
            return pd.DataFrame()

        if result is None or (
            isinstance(result, pd.DataFrame) and result.empty
        ):
            return pd.DataFrame()

        if isinstance(result, list):
            if not result:
                return pd.DataFrame()
            result = pd.concat(result, ignore_index=True)

        result["_time"] = pd.to_datetime(result["_time"])
        result.set_index("_time", inplace=True)
        if result.index.tz is None:
            result.index = result.index.tz_localize("UTC")
        result.index = result.index.tz_convert(IST)
        return result

    # ─────────────────────────────────────────
    # Async Thread Pool Helper
    # ─────────────────────────────────────────

    async def _run_sync(self, fn, *args, **kwargs):
        """
        Runs a synchronous callable in the thread pool.
        Prevents blocking the asyncio event loop.

        Args:
            fn: Synchronous callable.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Return value of fn(*args, **kwargs).
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _executor,
            lambda: fn(*args, **kwargs),
        )