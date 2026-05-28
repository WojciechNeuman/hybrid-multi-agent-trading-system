"""Data pipeline: indicator computation and MarketSnapshot construction."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from hmats.agents.base import MarketSnapshot


# ---------------------------------------------------------------------------
# Technical indicators
# ---------------------------------------------------------------------------


def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = _ema(series, fast)
    ema_slow = _ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _bollinger_bands(
    series: pd.Series, window: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = _sma(series, window)
    std = series.rolling(window=window, min_periods=window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicator columns to an OHLCV DataFrame *in place* and return it."""
    close = df["Close"]
    volume = df["Volume"]

    df["SMA_20"] = _sma(close, 20)
    df["SMA_50"] = _sma(close, 50)
    df["SMA_200"] = _sma(close, 200)

    df["EMA_12"] = _ema(close, 12)
    df["EMA_26"] = _ema(close, 26)

    df["RSI_14"] = _rsi(close, 14)

    macd_line, signal_line, histogram = _macd(close)
    df["MACD"] = macd_line
    df["MACD_Signal"] = signal_line
    df["MACD_Hist"] = histogram

    bb_upper, bb_mid, bb_lower = _bollinger_bands(close)
    df["BB_Upper"] = bb_upper
    df["BB_Mid"] = bb_mid
    df["BB_Lower"] = bb_lower

    df["Volume_Delta"] = volume.diff()

    return df


# ---------------------------------------------------------------------------
# Snapshot factory
# ---------------------------------------------------------------------------

_INDICATOR_COLS = [
    "SMA_20",
    "SMA_50",
    "SMA_200",
    "EMA_12",
    "EMA_26",
    "RSI_14",
    "MACD",
    "MACD_Signal",
    "MACD_Hist",
    "BB_Upper",
    "BB_Mid",
    "BB_Lower",
    "Volume_Delta",
]


def build_snapshot(
    ticker: str,
    df: pd.DataFrame,
    *,
    lookback: int = 200,
    timestamp: datetime | None = None,
) -> MarketSnapshot:
    """Build a :class:`MarketSnapshot` from the tail of an indicator-enriched DataFrame.

    Parameters
    ----------
    ticker:
        Ticker symbol, e.g. ``"BTC-USD"``.
    df:
        DataFrame produced by :func:`compute_indicators`.
    lookback:
        Number of trailing rows to include in ``ohlcv``.
    timestamp:
        Override timestamp; defaults to the last index value.
    """
    tail = df.tail(lookback).copy()
    last_row = df.iloc[-1]

    indicators = {col: float(last_row[col]) for col in _INDICATOR_COLS if pd.notna(last_row[col])}

    if timestamp is None:
        ts = df.index[-1]
        timestamp = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else datetime.now()

    return MarketSnapshot(
        ticker=ticker,
        timestamp=timestamp,
        ohlcv=tail,
        indicators=indicators,
    )
