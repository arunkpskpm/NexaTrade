"""
NexaTrade — Time Utilities.

All time operations in NexaTrade use these helpers.
No raw datetime.now() calls anywhere else in the codebase.

IST = Asia/Kolkata (UTC+5:30)

Rules:
    - All timestamps stored in DB are UTC
    - All timestamps shown to users are IST
    - All strategy/market checks use IST
    - InfluxDB writes use UTC nanoseconds
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Generator

import pytz

# Canonical timezone objects
IST = pytz.timezone("Asia/Kolkata")
UTC = timezone.utc


def now_ist() -> datetime:
    """
    Returns the current datetime in IST (timezone-aware).

    Returns:
        Current datetime localised to Asia/Kolkata.
    """
    return datetime.now(IST)


def now_utc() -> datetime:
    """
    Returns the current datetime in UTC (timezone-aware).

    Returns:
        Current datetime in UTC.
    """
    return datetime.now(UTC)


def to_ist(dt: datetime) -> datetime:
    """
    Converts any datetime to IST.
    Assumes UTC if the datetime is timezone-naive.

    Args:
        dt: Input datetime (naive or aware).

    Returns:
        IST-aware datetime.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(IST)


def to_utc(dt: datetime) -> datetime:
    """
    Converts any datetime to UTC.
    Assumes IST if the datetime is timezone-naive.

    Args:
        dt: Input datetime (naive or aware).

    Returns:
        UTC-aware datetime.
    """
    if dt.tzinfo is None:
        dt = IST.localize(dt)
    return dt.astimezone(UTC)


def floor_to_minute(
    dt: datetime,
    interval_minutes: int,
) -> datetime:
    """
    Floors a datetime to the nearest candle boundary.

    Examples:
        floor_to_minute(09:17:45, 5)  → 09:15:00
        floor_to_minute(10:32:00, 15) → 10:30:00
        floor_to_minute(14:59:30, 1)  → 14:59:00

    Args:
        dt: Input datetime.
        interval_minutes: Candle interval in minutes.

    Returns:
        Datetime floored to the candle boundary.
    """
    if interval_minutes <= 0:
        return dt.replace(second=0, microsecond=0)

    total_minutes = dt.hour * 60 + dt.minute
    floored       = (total_minutes // interval_minutes) * interval_minutes
    return dt.replace(
        hour=floored // 60,
        minute=floored % 60,
        second=0,
        microsecond=0,
    )


def is_market_open(dt: Optional[datetime] = None) -> bool:
    """
    Returns True if the NSE market is currently open.

    NSE trading hours: 09:15 – 15:30 IST, Mon–Fri
    Excludes public holidays (basic weekday check only).

    Args:
        dt: Datetime to check. Defaults to now IST.

    Returns:
        True if within NSE trading hours on a weekday.
    """
    from typing import Optional
    now = to_ist(dt) if dt else now_ist()

    # Weekend check (Mon=0 ... Sun=6)
    if now.weekday() >= 5:
        return False

    # Market hours: 09:15 – 15:30
    market_open  = now.replace(hour=9,  minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)

    return market_open <= now <= market_close


def date_range_chunks(
    start: datetime,
    end: datetime,
    chunk_days: int,
) -> Generator[tuple[datetime, datetime], None, None]:
    """
    Yields (chunk_start, chunk_end) tuples for a date range.
    Used for paginating broker historical data API calls.

    Args:
        start: Range start datetime.
        end: Range end datetime.
        chunk_days: Max days per chunk.

    Yields:
        Tuple of (chunk_start, chunk_end) datetimes.

    Example:
        for cs, ce in date_range_chunks(start, end, 30):
            candles = await client.get_history(cs, ce)
    """
    current = start
    while current < end:
        chunk_end = min(
            current + timedelta(days=chunk_days), end
        )
        yield current, chunk_end
        current = chunk_end


from typing import Optional