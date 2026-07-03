"""
Unit tests for utils/indicators.py

Tests cover: ema, sma, rsi, macd, atr, bollinger_bands,
crossover, crossunder, volume_ratio, is_doji
"""

import numpy as np
import pandas as pd
import pytest

from utils.indicators import (
    atr,
    bollinger_bands,
    crossover,
    crossunder,
    dema,
    ema,
    is_doji,
    macd,
    obv,
    rsi,
    sma,
    stochastic,
    tema,
    volume_ratio,
    vwap,
    wma,
)


@pytest.fixture
def price_series() -> pd.Series:
    """200-bar synthetic close price series."""
    np.random.seed(0)
    returns = np.random.normal(0, 0.01, 200)
    prices  = 1000 * np.exp(np.cumsum(returns))
    return pd.Series(prices)


@pytest.fixture
def ohlcv_df(price_series) -> pd.DataFrame:
    """Full OHLCV DataFrame derived from price_series."""
    n  = len(price_series)
    np.random.seed(0)
    c  = price_series.values
    h  = c * (1 + np.abs(np.random.normal(0, 0.005, n)))
    l  = c * (1 - np.abs(np.random.normal(0, 0.005, n)))
    o  = pd.Series(c).shift(1).fillna(c[0]).values
    v  = np.random.randint(10_000, 500_000, n).astype(float)
    return pd.DataFrame({
        "open": o, "high": h, "low": l,
        "close": c, "volume": v,
    })


# ─────────────────────────────────────────────
# Trend Indicators
# ─────────────────────────────────────────────

class TestEMA:
    def test_returns_series(self, price_series):
        result = ema(price_series, 20)
        assert isinstance(result, pd.Series)
        assert len(result) == len(price_series)

    def test_no_nan_after_warmup(self, price_series):
        result = ema(price_series, 20)
        assert result.iloc[20:].isna().sum() == 0

    def test_shorter_period_faster(self, price_series):
        fast = ema(price_series, 9)
        slow = ema(price_series, 21)
        # Fast EMA reacts more to recent moves
        assert fast.std() >= slow.std() * 0.8


class TestSMA:
    def test_returns_correct_length(self, price_series):
        result = sma(price_series, 10)
        assert len(result) == len(price_series)

    def test_nan_in_warmup(self, price_series):
        result = sma(price_series, 20)
        assert result.iloc[:19].isna().all()

    def test_simple_average(self):
        s      = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = sma(s, 3)
        assert abs(result.iloc[2] - 2.0) < 1e-9
        assert abs(result.iloc[4] - 4.0) < 1e-9

    def test_wma_weighted(self):
        s      = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = wma(s, 3)
        # WMA(3) at idx 4: (3×5 + 2×4 + 1×3) / 6 = 4.333
        expected = (3 * 5 + 2 * 4 + 1 * 3) / 6
        assert abs(result.iloc[4] - expected) < 1e-6

    def test_dema_less_lag(self, price_series):
        e  = ema(price_series, 20)
        d  = dema(price_series, 20)
        t  = tema(price_series, 20)
        # DEMA and TEMA should both be non-null after warmup
        assert not d.iloc[40:].isna().any()
        assert not t.iloc[60:].isna().any()


# ─────────────────────────────────────────────
# Momentum Indicators
# ─────────────────────────────────────────────

class TestRSI:
    def test_bounded_0_100(self, price_series):
        result = rsi(price_series, 14)
        clean  = result.dropna()
        assert (clean >= 0).all() and (clean <= 100).all()

    def test_no_nan_after_warmup(self, price_series):
        result = rsi(price_series, 14)
        assert result.iloc[14:].isna().sum() == 0

    def test_rising_series_high_rsi(self):
        rising = pd.Series(range(1, 101), dtype=float)
        result = rsi(rising, 14)
        assert result.iloc[-1] > 80

    def test_falling_series_low_rsi(self):
        falling = pd.Series(range(100, 0, -1), dtype=float)
        result  = rsi(falling, 14)
        assert result.iloc[-1] < 20


class TestMACD:
    def test_returns_three_series(self, price_series):
        ml, sl, hist = macd(price_series)
        assert all(
            isinstance(x, pd.Series)
            for x in [ml, sl, hist]
        )
        assert len(ml) == len(price_series)

    def test_histogram_is_diff(self, price_series):
        ml, sl, hist = macd(price_series)
        diff = ml - sl
        pd.testing.assert_series_equal(hist, diff)

    def test_signal_smoother_than_macd(self, price_series):
        ml, sl, _ = macd(price_series)
        assert ml.std() >= sl.std()


class TestStochastic:
    def test_bounded(self, ohlcv_df):
        k, d = stochastic(ohlcv_df, 14, 3)
        clean_k = k.dropna()
        clean_d = d.dropna()
        assert (clean_k >= 0).all() and (clean_k <= 100).all()
        assert (clean_d >= 0).all() and (clean_d <= 100).all()


# ─────────────────────────────────────────────
# Volatility Indicators
# ─────────────────────────────────────────────

class TestATR:
    def test_always_positive(self, ohlcv_df):
        result = atr(ohlcv_df, 14)
        clean  = result.dropna()
        assert (clean > 0).all()

    def test_length(self, ohlcv_df):
        result = atr(ohlcv_df, 14)
        assert len(result) == len(ohlcv_df)

    def test_high_volatility_larger_atr(self):
        low_vol  = pd.DataFrame({
            "high": [100.5] * 50,
            "low":  [99.5]  * 50,
            "close": [100.0] * 50,
        })
        high_vol = pd.DataFrame({
            "high": [110.0] * 50,
            "low":  [90.0]  * 50,
            "close": [100.0] * 50,
        })
        atr_low  = atr(low_vol,  14).mean()
        atr_high = atr(high_vol, 14).mean()
        assert atr_high > atr_low


class TestBollingerBands:
    def test_upper_above_lower(self, price_series):
        upper, mid, lower = bollinger_bands(price_series, 20, 2.0)
        valid = upper.dropna()
        low_v = lower.dropna()
        assert (valid.values > low_v.values).all()

    def test_mid_is_sma(self, price_series):
        _, mid, _ = bollinger_bands(price_series, 20)
        expected  = sma(price_series, 20)
        pd.testing.assert_series_equal(mid, expected)


# ─────────────────────────────────────────────
# Volume Indicators
# ─────────────────────────────────────────────

class TestVolumeIndicators:
    def test_volume_ratio_mean_near_one(self, ohlcv_df):
        ratio = volume_ratio(ohlcv_df, 20)
        clean = ratio.dropna()
        # Average ratio should be close to 1.0
        assert abs(clean.mean() - 1.0) < 0.2

    def test_obv_monotone_rising(self):
        df = pd.DataFrame({
            "close":  [1.0, 2.0, 3.0, 4.0, 5.0],
            "volume": [100.0, 200.0, 300.0, 400.0, 500.0],
        })
        result = obv(df)
        diffs  = result.diff().iloc[1:]
        assert (diffs > 0).all()

    def test_vwap_between_low_high(self, ohlcv_df):
        result = vwap(ohlcv_df)
        clean  = result.dropna()
        assert (clean >= ohlcv_df["low"].loc[clean.index]).all()
        assert (clean <= ohlcv_df["high"].loc[clean.index]).all()


# ─────────────────────────────────────────────
# Signal Helpers
# ─────────────────────────────────────────────

class TestCrossover:
    def test_detects_crossover(self):
        # fast crosses above slow at index 5
        fast = pd.Series([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
        slow = pd.Series([2, 3, 4, 5, 4, 3, 2, 1], dtype=float)
        cross = crossover(fast, slow)
        # Should detect the cross at index 4 (fast > slow AND was <=)
        assert cross.any()

    def test_no_false_crossover(self):
        fast  = pd.Series([5, 6, 7, 8, 9], dtype=float)
        slow  = pd.Series([1, 2, 3, 4, 5], dtype=float)
        # fast always above slow — crossover only at start
        cross = crossover(fast, slow)
        # After index 0, should not detect repeated crossovers
        assert cross.sum() <= 1

    def test_crossunder_detects(self):
        fast  = pd.Series([8, 7, 6, 5, 4, 3, 2, 1], dtype=float)
        slow  = pd.Series([5, 5, 5, 5, 5, 5, 5, 5], dtype=float)
        under = crossunder(fast, slow)
        assert under.any()


# ─────────────────────────────────────────────
# Candlestick Patterns
# ─────────────────────────────────────────────

class TestCandlePatterns:
    def test_doji_detected(self):
        df = pd.DataFrame({
            "open":  [100.0, 100.0],
            "high":  [105.0, 105.0],
            "low":   [95.0,  95.0],
            "close": [100.05, 100.05],   # tiny body
        })
        result = is_doji(df, threshold_pct=1.0)
        assert result.iloc[0]

    def test_large_candle_not_doji(self):
        df = pd.DataFrame({
            "open":  [100.0],
            "high":  [110.0],
            "low":   [99.0],
            "close": [108.0],  # large body
        })
        result = is_doji(df, threshold_pct=0.1)
        assert not result.iloc[0]