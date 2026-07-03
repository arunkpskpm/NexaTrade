"""
NexaTrade — Technical Indicator Library.

Pure-function indicators built on pandas and numpy.
No external TA library dependencies.

All functions accept pandas Series or DataFrame inputs
and return Series outputs — composable and chainable.

Indicators provided:
    Trend:     ema, sma, wma, dema, tema
    Momentum:  rsi, macd, stochastic, williams_r, cci
    Volatility: atr, bollinger_bands, keltner_channels, stddev
    Volume:    volume_ratio, obv, vwap
    Signal:    crossover, crossunder, above, below
    Pattern:   is_doji, is_hammer, is_engulfing
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd


# ═════════════════════════════════════════════
# Section 1 — Trend Indicators
# ═════════════════════════════════════════════

def ema(
    series: pd.Series,
    period: int,
    adjust: bool = False,
) -> pd.Series:
    """
    Exponential Moving Average.

    Args:
        series: Price series (typically close).
        period: EMA period.
        adjust: Use adjusted EWM (see pandas docs).

    Returns:
        EMA series (same index as input).
    """
    return series.ewm(span=period, adjust=adjust).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    """
    Simple Moving Average.

    Args:
        series: Price series.
        period: SMA period.

    Returns:
        SMA series.
    """
    return series.rolling(window=period).mean()


def wma(series: pd.Series, period: int) -> pd.Series:
    """
    Weighted Moving Average.
    More recent prices receive linearly higher weights.

    Args:
        series: Price series.
        period: WMA period.

    Returns:
        WMA series.
    """
    weights = np.arange(1, period + 1, dtype=float)

    def _wma(x: np.ndarray) -> float:
        return np.dot(x, weights) / weights.sum()

    return series.rolling(window=period).apply(_wma, raw=True)


def dema(series: pd.Series, period: int) -> pd.Series:
    """
    Double Exponential Moving Average.
    Reduces lag compared to EMA.
    DEMA = 2 × EMA(n) − EMA(EMA(n))
    """
    e1 = ema(series, period)
    e2 = ema(e1, period)
    return 2 * e1 - e2


def tema(series: pd.Series, period: int) -> pd.Series:
    """
    Triple Exponential Moving Average.
    TEMA = 3 × EMA − 3 × EMA(EMA) + EMA(EMA(EMA))
    """
    e1 = ema(series, period)
    e2 = ema(e1, period)
    e3 = ema(e2, period)
    return 3 * e1 - 3 * e2 + e3


# ═════════════════════════════════════════════
# Section 2 — Momentum Indicators
# ═════════════════════════════════════════════

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index (Wilder's smoothing).

    Args:
        series: Close price series.
        period: RSI lookback period.

    Returns:
        RSI series (0–100).
    """
    delta   = series.diff()
    gain    = delta.clip(lower=0)
    loss    = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD — Moving Average Convergence Divergence.

    Args:
        series: Close price series.
        fast: Fast EMA period.
        slow: Slow EMA period.
        signal: Signal line EMA period.

    Returns:
        Tuple of (macd_line, signal_line, histogram).
    """
    fast_ema   = ema(series, fast)
    slow_ema   = ema(series, slow)
    macd_line  = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


def stochastic(
    df: pd.DataFrame,
    k_period: int = 14,
    d_period: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """
    Stochastic Oscillator (%K and %D).

    Args:
        df: DataFrame with high, low, close columns.
        k_period: %K lookback period.
        d_period: %D smoothing period.

    Returns:
        Tuple of (stoch_k, stoch_d) series (0–100).
    """
    low_min  = df["low"].rolling(window=k_period).min()
    high_max = df["high"].rolling(window=k_period).max()
    stoch_k  = 100 * (df["close"] - low_min) / (
        (high_max - low_min).replace(0, np.nan)
    )
    stoch_d = stoch_k.rolling(window=d_period).mean()
    return stoch_k.fillna(50), stoch_d.fillna(50)


def williams_r(
    df: pd.DataFrame, period: int = 14
) -> pd.Series:
    """
    Williams %R Oscillator.

    Args:
        df: DataFrame with high, low, close columns.
        period: Lookback period.

    Returns:
        Williams %R series (-100 to 0).
    """
    high_max = df["high"].rolling(window=period).max()
    low_min  = df["low"].rolling(window=period).min()
    wr = -100 * (high_max - df["close"]) / (
        (high_max - low_min).replace(0, np.nan)
    )
    return wr.fillna(-50)


def cci(
    df: pd.DataFrame, period: int = 20
) -> pd.Series:
    """
    Commodity Channel Index.

    Args:
        df: DataFrame with high, low, close columns.
        period: CCI period.

    Returns:
        CCI series.
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    mean_tp       = typical_price.rolling(window=period).mean()
    mean_dev      = typical_price.rolling(window=period).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True
    )
    return (typical_price - mean_tp) / (0.015 * mean_dev.replace(0, np.nan))


# ═════════════════════════════════════════════
# Section 3 — Volatility Indicators
# ═════════════════════════════════════════════

def atr(
    df: pd.DataFrame, period: int = 14
) -> pd.Series:
    """
    Average True Range (Wilder's smoothing).

    True Range = max(
        high − low,
        |high − prev_close|,
        |low  − prev_close|
    )

    Args:
        df: DataFrame with high, low, close columns.
        period: ATR period.

    Returns:
        ATR series.
    """
    high      = df["high"]
    low       = df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def bollinger_bands(
    series: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands.

    Args:
        series: Close price series.
        period: SMA period.
        std_dev: Number of standard deviations.

    Returns:
        Tuple of (upper_band, middle_band, lower_band).
    """
    middle = sma(series, period)
    std    = series.rolling(window=period).std()
    upper  = middle + std_dev * std
    lower  = middle - std_dev * std
    return upper, middle, lower


def keltner_channels(
    df: pd.DataFrame,
    ema_period: int = 20,
    atr_period: int = 10,
    multiplier: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Keltner Channels.

    Args:
        df: DataFrame with high, low, close.
        ema_period: EMA period for midline.
        atr_period: ATR period.
        multiplier: ATR multiplier for channel width.

    Returns:
        Tuple of (upper, middle, lower).
    """
    middle = ema(df["close"], ema_period)
    atr_sr = atr(df, atr_period)
    upper  = middle + multiplier * atr_sr
    lower  = middle - multiplier * atr_sr
    return upper, middle, lower


def stddev(series: pd.Series, period: int = 20) -> pd.Series:
    """Rolling standard deviation."""
    return series.rolling(window=period).std()


# ═════════════════════════════════════════════
# Section 4 — Volume Indicators
# ═════════════════════════════════════════════

def volume_ratio(
    df: pd.DataFrame, period: int = 20
) -> pd.Series:
    """
    Volume Ratio — current volume vs rolling average.
    Values > 1.5 indicate above-average volume.

    Args:
        df: DataFrame with volume column.
        period: Rolling average period.

    Returns:
        Volume ratio series.
    """
    avg_vol = df["volume"].rolling(window=period).mean()
    return df["volume"] / avg_vol.replace(0, np.nan)


def obv(df: pd.DataFrame) -> pd.Series:
    """
    On-Balance Volume.

    Args:
        df: DataFrame with close and volume columns.

    Returns:
        OBV series.
    """
    direction = np.sign(df["close"].diff())
    direction.iloc[0] = 0
    return (direction * df["volume"]).cumsum()


def vwap(df: pd.DataFrame) -> pd.Series:
    """
    Volume Weighted Average Price.
    Resets each trading day (use intraday data only).

    Args:
        df: DataFrame with high, low, close, volume columns.

    Returns:
        VWAP series.
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    cum_tp_vol    = (typical_price * df["volume"]).cumsum()
    cum_vol       = df["volume"].cumsum()
    return cum_tp_vol / cum_vol.replace(0, np.nan)


# ═════════════════════════════════════════════
# Section 5 — Signal Helpers
# ═════════════════════════════════════════════

def crossover(
    fast: pd.Series, slow: pd.Series
) -> pd.Series:
    """
    Returns True on bars where fast crosses above slow.
    (fast[-1] > slow[-1]) AND (fast[-2] <= slow[-2])

    Args:
        fast: Fast series.
        slow: Slow series.

    Returns:
        Boolean Series — True on crossover bars.
    """
    return (fast > slow) & (fast.shift(1) <= slow.shift(1))


def crossunder(
    fast: pd.Series, slow: pd.Series
) -> pd.Series:
    """
    Returns True on bars where fast crosses below slow.

    Args:
        fast: Fast series.
        slow: Slow series.

    Returns:
        Boolean Series — True on crossunder bars.
    """
    return (fast < slow) & (fast.shift(1) >= slow.shift(1))


def above(series: pd.Series, level: float) -> pd.Series:
    """Returns True where series is above level."""
    return series > level


def below(series: pd.Series, level: float) -> pd.Series:
    """Returns True where series is below level."""
    return series < level


# ═════════════════════════════════════════════
# Section 6 — Candlestick Pattern Detectors
# ═════════════════════════════════════════════

def is_doji(
    df: pd.DataFrame,
    threshold_pct: float = 0.1,
) -> pd.Series:
    """
    Detects Doji candles.
    Body (|open-close|) < threshold_pct % of range.

    Args:
        df: DataFrame with open, high, low, close.
        threshold_pct: Max body-to-range ratio for doji.

    Returns:
        Boolean Series.
    """
    body  = (df["close"] - df["open"]).abs()
    rng   = df["high"] - df["low"]
    return (body / rng.replace(0, np.nan)) < (threshold_pct / 100)


def is_hammer(
    df: pd.DataFrame,
    body_ratio: float = 0.3,
    shadow_ratio: float = 2.0,
) -> pd.Series:
    """
    Detects Hammer candles (bullish reversal).
    Small body at top, long lower shadow.

    Args:
        df: DataFrame with open, high, low, close.
        body_ratio: Max body / total range ratio.
        shadow_ratio: Min lower shadow / body ratio.

    Returns:
        Boolean Series.
    """
    body       = (df["close"] - df["open"]).abs()
    total_rng  = (df["high"] - df["low"]).replace(0, np.nan)
    lower_sh   = df[["open", "close"]].min(axis=1) - df["low"]
    upper_sh   = df["high"] - df[["open", "close"]].max(axis=1)

    return (
        (body / total_rng < body_ratio)
        & (lower_sh > shadow_ratio * body.replace(0, np.nan))
        & (upper_sh < body)
    )


def is_engulfing(
    df: pd.DataFrame,
    bullish: bool = True,
) -> pd.Series:
    """
    Detects Bullish or Bearish Engulfing candles.

    Args:
        df: DataFrame with open, close.
        bullish: True for bullish engulfing, False for bearish.

    Returns:
        Boolean Series.
    """
    prev_open  = df["open"].shift(1)
    prev_close = df["close"].shift(1)

    if bullish:
        # Current green candle body engulfs prior red candle body
        return (
            (df["close"] > df["open"])            # Current bullish
            & (prev_close < prev_open)             # Prior bearish
            & (df["open"]  <= prev_close)          # Open below prior close
            & (df["close"] >= prev_open)           # Close above prior open
        )
    else:
        # Current red candle body engulfs prior green candle body
        return (
            (df["close"] < df["open"])             # Current bearish
            & (prev_close > prev_open)             # Prior bullish
            & (df["open"]  >= prev_close)          # Open above prior close
            & (df["close"] <= prev_open)           # Close below prior open
        )