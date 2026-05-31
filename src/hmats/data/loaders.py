"""Unified data loader for 5-minute (and multi-timeframe) Binance klines.

The primary entry point for the 5-minute microstructure benchmark is
``load_5m_extended()``, which returns the full nine-column DataFrame
required by ``engineering_5m.make_features_5m()``.

Column schema
-------------
open, high, low, close, volume          -- standard OHLCV (float64)
quote_volume                             -- quote-asset volume (USDT)
num_trades                               -- number of individual trades
taker_buy_base_volume                    -- taker-buy volume in base asset
taker_buy_quote_volume                   -- taker-buy volume in quote asset
"""

from __future__ import annotations

import pandas as pd

from hmats.data.binance_store import (
    _EXTENDED_COLS,
    _OHLCV_COLS,
    fetch_and_store,
    load,
)

_ALL_COLS = _OHLCV_COLS + _EXTENDED_COLS


def load_5m_extended(
    symbol: str = "BTCUSDT",
    start: str | None = None,
    end: str | None = None,
    store_dir: str = "data/raw",
    fetch_if_missing: bool = False,
    fetch_start: str = "2020-01-01",
) -> pd.DataFrame:
    """Return 5-minute klines with all nine microstructure columns.

    If the Parquet file is missing and *fetch_if_missing* is True, the data
    is downloaded from Binance and stored before returning.  Otherwise a
    ``FileNotFoundError`` is raised.

    If the stored file predates the extended-column schema (i.e. it only
    contains the five OHLCV columns), a ``ValueError`` is raised — re-fetch
    with ``fetch_and_store(..., interval="5m", extended=True)`` first.
    """
    try:
        df = load(symbol, "5m", start=start, end=end, store_dir=store_dir)
    except FileNotFoundError:
        if not fetch_if_missing:
            raise
        df = fetch_and_store(
            symbol=symbol,
            interval="5m",
            start=fetch_start,
            end=end,
            store_dir=store_dir,
            extended=True,
        )
        if start is not None:
            df = df[df.index >= pd.Timestamp(start, tz="UTC")]
        if end is not None:
            df = df[df.index < pd.Timestamp(end, tz="UTC")]

    missing = [c for c in _EXTENDED_COLS if c not in df.columns]
    if missing:
        if fetch_if_missing:
            df = fetch_and_store(
                symbol=symbol,
                interval="5m",
                start=fetch_start,
                end=end,
                store_dir=store_dir,
                extended=True,
            )
            if start is not None:
                df = df[df.index >= pd.Timestamp(start, tz="UTC")]
            if end is not None:
                df = df[df.index < pd.Timestamp(end, tz="UTC")]
        else:
            raise ValueError(
                f"Stored file for {symbol}_5m.parquet is missing extended columns "
                f"{missing}. Re-fetch with fetch_and_store(..., extended=True)."
            )

    return df[_ALL_COLS].copy()


def load_hourly(
    symbol: str = "BTCUSDT",
    start: str | None = None,
    end: str | None = None,
    store_dir: str = "data/raw",
) -> pd.DataFrame:
    """Return hourly OHLCV klines (standard five columns)."""
    return load(symbol, "1h", start=start, end=end, store_dir=store_dir)


def load_eth_btc_ratio_hourly(
    start: str | None = None,
    end: str | None = None,
    store_dir: str = "data/raw",
) -> pd.Series:
    """ETH/BTC close-price ratio on 1-hour bars (used for ADF regime feature).

    Loads ETHUSDT and BTCUSDT hourly data and returns the ratio series.
    If ETHUSDT data is unavailable the function returns None so callers
    can degrade gracefully.
    """
    try:
        eth = load("ETHUSDT", "1h", start=start, end=end, store_dir=store_dir)
        btc = load("BTCUSDT", "1h", start=start, end=end, store_dir=store_dir)
    except FileNotFoundError:
        return None

    ratio = eth["close"] / btc["close"]
    ratio.name = "eth_btc_ratio"
    return ratio
