"""
NexaTrade — Data Service.

The DataService is the single orchestrator for all
historical market data operations.

Responsibilities:
  - Fetch OHLCV history for any symbol from any broker
  - Serve data as pandas DataFrames with standard columns
  - Manage a two-tier cache:
      Tier 1: InfluxDB  (fast, persistent time-series)
      Tier 2: Broker API (authoritative, slower, rate-limited)
  - Write all broker-fetched data back to InfluxDB (cache fill)
  - Detect and fill gaps in InfluxDB data
  - Validate data quality (OHLCV sanity checks)
  - Provide a pandas DataFrame API for backtesting and strategies

Cache-first strategy:
    1. Check InfluxDB for the requested range
    2. If data is complete → return it
    3. If data is partial → fetch missing range from broker
    4. Merge and write missing data to InfluxDB
    5. Return the merged DataFrame

Usage:
    data_svc = DataService(influx_client, broker_service)

    # Get OHLCV as DataFrame
    df = await data_svc.get_ohlcv(
        "RELIANCE", "NSE", "5minute",
        from_date="2024-01-01", to_date="2024-06-01"
    )

    # Force refresh from broker
    df = await data_svc.get_ohlcv(
        "RELIANCE", "NSE", "5minute",
        from_date="2024-01-01", to_date="2024-06-01",
        force_refresh=True,
    )

    # Get latest N candles
    df = await data_svc.get_latest_candles("NIFTY50", "NSE", "1minute", n=100)
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd

from brokers.models import OHLCV
from data.storage.influx_client import InfluxClient
from services.broker_service import BrokerService
from utils.logger import get_logger
from utils.time_utils import now_ist, to_utc, to_ist, IST

logger = get_logger(__name__)

# ─────────────────────────────────────────────
# Expected DataFrame columns (canonical order)
# ─────────────────────────────────────────────
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


class DataService:
    """
    NexaTrade two-tier historical data orchestrator.

    All returned DataFrames:
        - Index: DatetimeIndex in IST (timezone-aware)
        - Columns: open, high, low, close, volume (float64)
        - Sorted: ascending by datetime
        - Validated: OHLC sanity checks applied

    Adding a new data source:
        Implement a method that returns list[OHLCV]
        and add it as a fallback tier in _fetch_from_broker().
    """

    def __init__(
        self,
        influx_client: InfluxClient,
        broker_service: BrokerService,
    ) -> None:
        self._influx = influx_client
        self._broker_svc = broker_service

        # In-memory DataFrame cache for repeated small queries
        # Key: (symbol, exchange, interval, from_date, to_date)
        self._mem_cache: dict[tuple, pd.DataFrame] = {}
        self._mem_cache_maxsize: int = 50

    # ─────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────

    async def get_ohlcv(
        self,
        symbol: str,
        exchange: str,
        interval: str,
        from_date: str,
        to_date: Optional[str] = None,
        broker_name: Optional[str] = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Returns OHLCV data as a pandas DataFrame.

        Cache-first strategy:
            1. In-memory cache (fastest)
            2. InfluxDB cache (fast, persistent)
            3. Broker API (authoritative, slowest)

        Args:
            symbol: Instrument symbol (e.g. "RELIANCE").
            exchange: Exchange code (e.g. "NSE").
            interval: Candle interval string (e.g. "5minute").
            from_date: Start date string (YYYY-MM-DD).
            to_date: End date string. Defaults to today.
            broker_name: Override broker source.
                         Defaults to active broker.
            force_refresh: If True, bypasses all caches and
                           fetches fresh from broker API.

        Returns:
            DataFrame with DatetimeIndex (IST) and
            columns [open, high, low, close, volume].
            Empty DataFrame if no data is available.

        Example:
            df = await data_svc.get_ohlcv(
                "RELIANCE", "NSE", "15minute",
                from_date="2024-01-01",
                to_date="2024-03-31",
            )
            # Compute indicators
            df["sma20"] = df["close"].rolling(20).mean()
        """
        symbol   = symbol.upper()
        exchange = exchange.upper()
        to_date  = to_date or now_ist().strftime("%Y-%m-%d")
        broker   = broker_name or self._broker_svc.active_broker_name

        cache_key = (symbol, exchange, interval, from_date, to_date)

        # 1 ── In-memory cache ─────────────────
        if not force_refresh and cache_key in self._mem_cache:
            logger.debug(
                f"DataService mem-cache hit | "
                f"symbol={symbol} | interval={interval}"
            )
            return self._mem_cache[cache_key].copy()

        # 2 ── InfluxDB cache ──────────────────
        if not force_refresh:
            df = await self._fetch_from_influx(
                symbol=symbol,
                exchange=exchange,
                interval=interval,
                from_date=from_date,
                to_date=to_date,
                broker_name=broker,
            )
            if not df.empty:
                gap_df = await self._fill_gaps(
                    df=df,
                    symbol=symbol,
                    exchange=exchange,
                    interval=interval,
                    from_date=from_date,
                    to_date=to_date,
                    broker_name=broker,
                )
                result = self._validate_and_clean(gap_df)
                self._cache_set(cache_key, result)
                return result.copy()

        # 3 ── Broker API ──────────────────────
        df = await self._fetch_from_broker(
            symbol=symbol,
            exchange=exchange,
            interval=interval,
            from_date=from_date,
            to_date=to_date,
            broker_name=broker,
        )

        if not df.empty:
            # Write to InfluxDB for future cache hits
            await self._write_to_influx(
                df=df,
                symbol=symbol,
                exchange=exchange,
                interval=interval,
                broker_name=broker,
            )

        result = self._validate_and_clean(df)
        self._cache_set(cache_key, result)
        return result.copy()

    async def get_latest_candles(
        self,
        symbol: str,
        exchange: str,
        interval: str,
        n: int = 200,
        broker_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Returns the latest N candles for an instrument.

        Calculates the required lookback date automatically
        based on the interval and requested bar count.

        Args:
            symbol: Instrument symbol.
            exchange: Exchange code.
            interval: Candle interval.
            n: Number of recent candles to return.
            broker_name: Override broker source.

        Returns:
            DataFrame with up to n rows (may be fewer
            if insufficient history exists).

        Example:
            df = await data_svc.get_latest_candles(
                "NIFTY50", "NSE", "1minute", n=100
            )
        """
        from services.feed_service import INTERVAL_MINUTES
        interval_mins = INTERVAL_MINUTES.get(interval, 1)
        # Request 20% extra bars to account for market holidays
        lookback_bars = int(n * 1.20)
        lookback_minutes = max(interval_mins * lookback_bars, 1440)

        to_dt   = now_ist()
        from_dt = to_dt - timedelta(minutes=lookback_minutes)

        df = await self.get_ohlcv(
            symbol=symbol,
            exchange=exchange,
            interval=interval,
            from_date=from_dt.strftime("%Y-%m-%d"),
            to_date=to_dt.strftime("%Y-%m-%d"),
            broker_name=broker_name,
        )

        # Return last n rows
        return df.tail(n) if len(df) >= n else df

    async def get_intraday(
        self,
        symbol: str,
        exchange: str,
        interval: str = "1minute",
        date: Optional[str] = None,
        broker_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Returns all intraday OHLCV candles for a single day.

        Args:
            symbol: Instrument symbol.
            exchange: Exchange code.
            interval: Intraday interval (1minute, 5minute, etc).
            date: Target date (YYYY-MM-DD). Defaults to today.
            broker_name: Override broker source.

        Returns:
            DataFrame filtered to the requested trading day.
        """
        date = date or now_ist().strftime("%Y-%m-%d")
        df = await self.get_ohlcv(
            symbol=symbol,
            exchange=exchange,
            interval=interval,
            from_date=date,
            to_date=date,
            broker_name=broker_name,
            force_refresh=True,  # Intraday always fresh
        )
        return df

    async def get_multiple_symbols(
        self,
        symbols: list[dict[str, str]],
        interval: str,
        from_date: str,
        to_date: Optional[str] = None,
        broker_name: Optional[str] = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Fetches OHLCV data for multiple instruments concurrently.

        Args:
            symbols: List of {"symbol": ..., "exchange": ...} dicts.
            interval: Candle interval.
            from_date: Start date (YYYY-MM-DD).
            to_date: End date (YYYY-MM-DD). Defaults to today.
            broker_name: Override broker source.

        Returns:
            Dict mapping "{symbol}:{exchange}" → DataFrame.

        Example:
            dfs = await data_svc.get_multiple_symbols(
                [
                    {"symbol": "RELIANCE", "exchange": "NSE"},
                    {"symbol": "TCS", "exchange": "NSE"},
                ],
                interval="5minute",
                from_date="2024-01-01",
            )
            reliance_df = dfs["RELIANCE:NSE"]
        """
        tasks = {
            f"{inst['symbol'].upper()}:{inst.get('exchange','NSE').upper()}": (
                self.get_ohlcv(
                    symbol=inst["symbol"],
                    exchange=inst.get("exchange", "NSE"),
                    interval=interval,
                    from_date=from_date,
                    to_date=to_date,
                    broker_name=broker_name,
                )
            )
            for inst in symbols
        }

        results: dict[str, pd.DataFrame] = {}
        for key, coro in tasks.items():
            try:
                results[key] = await coro
            except Exception as exc:
                logger.error(
                    f"Multi-fetch failed | key={key} | error={exc}"
                )
                results[key] = pd.DataFrame(
                    columns=OHLCV_COLUMNS
                )

        logger.info(
            f"Multi-symbol fetch complete | "
            f"symbols={len(results)} | interval={interval}"
        )
        return results

    # ─────────────────────────────────────────
    # Data Fetching — Tier Implementations
    # ─────────────────────────────────────────

    async def _fetch_from_influx(
        self,
        symbol: str,
        exchange: str,
        interval: str,
        from_date: str,
        to_date: str,
        broker_name: str,
    ) -> pd.DataFrame:
        """
        Fetches OHLCV data from InfluxDB.

        Args:
            symbol: Instrument symbol.
            exchange: Exchange code.
            interval: Candle interval.
            from_date: Start date string.
            to_date: End date string.
            broker_name: Broker tag filter.

        Returns:
            DataFrame or empty DataFrame if not found.
        """
        try:
            from_dt = datetime.strptime(from_date, "%Y-%m-%d")
            to_dt   = (
                datetime.strptime(to_date, "%Y-%m-%d")
                + timedelta(days=1)
            )
            df = await self._influx.get_candles(
                symbol=symbol,
                exchange=exchange,
                interval=interval,
                broker_name=broker_name,
                from_dt=to_utc(from_dt),
                to_dt=to_utc(to_dt),
                limit=10000,
            )
            if not df.empty:
                logger.debug(
                    f"InfluxDB cache hit | "
                    f"symbol={symbol} | interval={interval} | "
                    f"rows={len(df)}"
                )
            return df
        except Exception as exc:
            logger.warning(
                f"InfluxDB fetch failed | "
                f"symbol={symbol} | error={exc}"
            )
            return pd.DataFrame(columns=OHLCV_COLUMNS)

    async def _fetch_from_broker(
        self,
        symbol: str,
        exchange: str,
        interval: str,
        from_date: str,
        to_date: str,
        broker_name: str,
    ) -> pd.DataFrame:
        """
        Fetches OHLCV data from the broker historical API.
        Converts the list[OHLCV] response to a DataFrame.

        Args:
            symbol: Instrument symbol.
            exchange: Exchange code.
            interval: Candle interval.
            from_date: Start date string (YYYY-MM-DD).
            to_date: End date string (YYYY-MM-DD).
            broker_name: Broker identifier.

        Returns:
            DataFrame or empty DataFrame on failure.
        """
        try:
            logger.info(
                f"Fetching from broker API | "
                f"broker={broker_name} | symbol={symbol} | "
                f"interval={interval} | "
                f"range={from_date}→{to_date}"
            )
            candles: list[OHLCV] = (
                await self._broker_svc.broker.get_historical_data(
                    symbol=symbol,
                    exchange=exchange,
                    interval=interval,
                    from_date=from_date,
                    to_date=to_date,
                )
            )
            if not candles:
                return pd.DataFrame(columns=OHLCV_COLUMNS)

            return self._candles_to_dataframe(candles)
        except Exception as exc:
            logger.error(
                f"Broker API fetch failed | "
                f"symbol={symbol} | error={exc}"
            )
            return pd.DataFrame(columns=OHLCV_COLUMNS)

    async def _fill_gaps(
        self,
        df: pd.DataFrame,
        symbol: str,
        exchange: str,
        interval: str,
        from_date: str,
        to_date: str,
        broker_name: str,
    ) -> pd.DataFrame:
        """
        Detects and fills data gaps in the InfluxDB DataFrame.

        A gap is defined as: the InfluxDB result covers less than
        80% of the requested trading time range.

        If a gap is detected, fetches the missing portion from
        the broker API and merges the two DataFrames.

        Args:
            df: InfluxDB DataFrame (may be partial).
            symbol: Instrument symbol.
            exchange: Exchange code.
            interval: Candle interval.
            from_date: Requested start date.
            to_date: Requested end date.
            broker_name: Broker identifier.

        Returns:
            Merged DataFrame with gaps filled.
        """
        from services.feed_service import INTERVAL_MINUTES
        interval_mins = INTERVAL_MINUTES.get(interval, 1)
        if interval_mins == 0:
            return df  # Sub-minute: no gap detection

        requested_start = datetime.strptime(from_date, "%Y-%m-%d")
        requested_end   = datetime.strptime(to_date,   "%Y-%m-%d")
        requested_days  = (requested_end - requested_start).days + 1

        # Estimate expected bars (rough: 375 min/day for NSE intraday)
        trading_mins_per_day = 375 if interval_mins < 1440 else 1
        expected_bars = (
            requested_days * trading_mins_per_day
        ) // max(interval_mins, 1)

        coverage = len(df) / max(expected_bars, 1)
        if coverage >= 0.80:
            return df  # Good enough coverage

        logger.info(
            f"Data gap detected | "
            f"symbol={symbol} | interval={interval} | "
            f"coverage={coverage:.0%} | "
            f"fetching from broker..."
        )

        broker_df = await self._fetch_from_broker(
            symbol=symbol,
            exchange=exchange,
            interval=interval,
            from_date=from_date,
            to_date=to_date,
            broker_name=broker_name,
        )

        if broker_df.empty:
            return df

        # Write new data to InfluxDB
        await self._write_to_influx(
            df=broker_df,
            symbol=symbol,
            exchange=exchange,
            interval=interval,
            broker_name=broker_name,
        )

        # Merge: broker data takes precedence for overlapping bars
        merged = pd.concat([df, broker_df])
        merged = merged[~merged.index.duplicated(keep="last")]
        merged.sort_index(inplace=True)
        return merged

    async def _write_to_influx(
        self,
        df: pd.DataFrame,
        symbol: str,
        exchange: str,
        interval: str,
        broker_name: str,
    ) -> None:
        """
        Writes a DataFrame to InfluxDB as OHLCV candles.

        Args:
            df: DataFrame to persist.
            symbol: Instrument symbol.
            exchange: Exchange code.
            interval: Candle interval.
            broker_name: Source broker tag.
        """
        if df.empty:
            return
        try:
            records = [
                {
                    "datetime": idx,
                    "open":     float(row["open"]),
                    "high":     float(row["high"]),
                    "low":      float(row["low"]),
                    "close":    float(row["close"]),
                    "volume":   float(row["volume"]),
                }
                for idx, row in df.iterrows()
            ]
            await self._influx.write_candles(
                candles=records,
                broker_name=broker_name,
                symbol=symbol,
                exchange=exchange,
                interval=interval,
            )
            logger.debug(
                f"Written to InfluxDB | "
                f"symbol={symbol} | interval={interval} | "
                f"rows={len(records)}"
            )
        except Exception as exc:
            logger.warning(
                f"InfluxDB write failed | "
                f"symbol={symbol} | error={exc}"
            )

    # ─────────────────────────────────────────
    # Data Quality
    # ─────────────────────────────────────────

    def _validate_and_clean(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Validates and cleans an OHLCV DataFrame.

        Cleaning steps:
            1. Ensure all required columns exist
            2. Cast all columns to float64
            3. Drop rows where close == 0 or close is NaN
            4. Fix OHLC violations (high < low, etc.)
            5. Sort by datetime index ascending
            6. Remove duplicate timestamps
            7. Ensure timezone-aware IST index

        Args:
            df: Input OHLCV DataFrame.

        Returns:
            Cleaned DataFrame.
        """
        if df.empty:
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        # 1 ── Ensure all columns present ─────
        for col in OHLCV_COLUMNS:
            if col not in df.columns:
                df[col] = 0.0

        # 2 ── Cast to float64 ─────────────────
        for col in OHLCV_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # 3 ── Drop invalid close prices ───────
        df = df[df["close"].notna() & (df["close"] > 0)]

        # 4 ── Fix OHLC violations ─────────────
        df["high"]  = df[["open", "high", "close"]].max(axis=1)
        df["low"]   = df[["open", "low",  "close"]].min(axis=1)

        # 5 ── Sort by datetime ────────────────
        df.sort_index(inplace=True)

        # 6 ── Remove duplicate timestamps ─────
        df = df[~df.index.duplicated(keep="last")]

        # 7 ── Ensure IST timezone ─────────────
        if df.index.tz is None:
            df.index = df.index.tz_localize(IST)
        else:
            df.index = df.index.tz_convert(IST)

        return df[OHLCV_COLUMNS]

    def _candles_to_dataframe(
        self, candles: list[OHLCV]
    ) -> pd.DataFrame:
        """
        Converts a list[OHLCV] to a pandas DataFrame.

        Args:
            candles: List of OHLCV model instances.

        Returns:
            DataFrame with DatetimeIndex (IST) and OHLCV columns.
        """
        if not candles:
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        records = [
            {
                "datetime": to_ist(c.datetime),
                "open":     c.open,
                "high":     c.high,
                "low":      c.low,
                "close":    c.close,
                "volume":   c.volume,
            }
            for c in candles
        ]
        df = pd.DataFrame(records)
        df.set_index("datetime", inplace=True)
        df.index = pd.DatetimeIndex(df.index)
        df = df[OHLCV_COLUMNS]
        return df

    # ─────────────────────────────────────────
    # Memory Cache Helpers
    # ─────────────────────────────────────────

    def _cache_set(
        self, key: tuple, df: pd.DataFrame
    ) -> None:
        """
        Stores a DataFrame in the in-memory cache.
        Evicts oldest entry if cache is full (FIFO).

        Args:
            key: Cache key tuple.
            df: DataFrame to cache.
        """
        if len(self._mem_cache) >= self._mem_cache_maxsize:
            oldest_key = next(iter(self._mem_cache))
            del self._mem_cache[oldest_key]
        self._mem_cache[key] = df

    def invalidate_cache(
        self,
        symbol: Optional[str] = None,
        exchange: Optional[str] = None,
        interval: Optional[str] = None,
    ) -> int:
        """
        Invalidates in-memory cache entries matching the filter.
        Call after a data correction or forced refresh.

        Args:
            symbol: Symbol filter. Invalidates all if None.
            exchange: Exchange filter.
            interval: Interval filter.

        Returns:
            Number of cache entries removed.
        """
        keys_to_remove = []
        for key in self._mem_cache:
            sym, exch, ivl, _, _ = key
            if symbol and sym != symbol.upper():
                continue
            if exchange and exch != exchange.upper():
                continue
            if interval and ivl != interval:
                continue
            keys_to_remove.append(key)

        for key in keys_to_remove:
            del self._mem_cache[key]

        if keys_to_remove:
            logger.info(
                f"Cache invalidated | "
                f"entries_removed={len(keys_to_remove)} | "
                f"filter=({symbol},{exchange},{interval})"
            )
        return len(keys_to_remove)

    def get_cache_stats(self) -> dict[str, Any]:
        """
        Returns in-memory cache statistics.

        Returns:
            Dict with cache size and entry details.
        """
        return {
            "cache_size":    len(self._mem_cache),
            "max_size":      self._mem_cache_maxsize,
            "cached_keys": [
                {
                    "symbol":   k[0],
                    "exchange": k[1],
                    "interval": k[2],
                    "from":     k[3],
                    "to":       k[4],
                }
                for k in self._mem_cache.keys()
            ],
        }

    # ─────────────────────────────────────────
    # Utility Methods
    # ─────────────────────────────────────────

    async def delete_symbol_data(
        self,
        symbol: str,
        exchange: str,
        interval: str,
        broker_name: Optional[str] = None,
        from_dt: Optional[datetime] = None,
        to_dt: Optional[datetime] = None,
    ) -> None:
        """
        Deletes OHLCV data from InfluxDB for a symbol.
        Used for data correction — re-fetch will repopulate.

        Args:
            symbol: Instrument symbol.
            exchange: Exchange code.
            interval: Candle interval.
            broker_name: Broker tag filter.
            from_dt: Delete start boundary.
            to_dt: Delete end boundary.
        """
        broker_name = (
            broker_name or self._broker_svc.active_broker_name
        )
        from_dt = from_dt or datetime(2020, 1, 1)
        to_dt   = to_dt   or now_ist()

        await self._influx.delete_candles(
            broker_name=broker_name,
            symbol=symbol,
            exchange=exchange,
            interval=interval,
            from_dt=from_dt,
            to_dt=to_dt,
        )
        self.invalidate_cache(symbol=symbol, exchange=exchange)
        logger.info(
            f"Symbol data deleted | "
            f"symbol={symbol} | interval={interval}"
        )

    async def health_check(self) -> dict[str, bool]:
        """
        Checks connectivity of all data service dependencies.

        Returns:
            Dict with InfluxDB and broker connectivity status.
        """
        influx_ok = False
        broker_ok = False

        try:
            test_df = await self._influx.get_candles(
                symbol="TEST",
                exchange="NSE",
                interval="1minute",
                broker_name="paper",
                limit=1,
            )
            influx_ok = True
        except Exception:
            influx_ok = False

        try:
            broker_ok = await self._broker_svc.broker.is_connected()
        except Exception:
            broker_ok = False

        return {
            "influx_ok": influx_ok,
            "broker_ok": broker_ok,
        }