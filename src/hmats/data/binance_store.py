"""Binance kline data — append-capable Parquet store.

One Parquet file per ``(symbol, interval)`` pair under ``store_dir``.
``fetch_and_store()`` does an incremental update: only rows newer than
the last stored timestamp are fetched from the Binance REST API.
``load()`` reads the Parquet and filters by date range.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import requests

BINANCE_BASE_URL = "https://api.binance.com"
KLINES_ENDPOINT = "/api/v3/klines"

_OHLCV_COLS = ["open", "high", "low", "close", "volume"]
_EXTENDED_COLS = [
    "quote_volume",
    "num_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
]


def _to_millis(ts: str) -> int:
    dt = pd.to_datetime(ts, utc=True)
    return int(dt.value // 10**6)


def _parquet_path(store_dir: str, symbol: str, interval: str) -> str:
    return os.path.join(store_dir, f"{symbol}_{interval}.parquet")


def _fetch_pages(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    limit: int,
    sleep_s: float,
) -> list[list]:
    """Paginated kline download from Binance REST API."""
    rows: list[list] = []
    cur = start_ms

    while True:
        params: dict = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cur,
            "endTime": end_ms,
            "limit": limit,
        }
        r = requests.get(BINANCE_BASE_URL + KLINES_ENDPOINT, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()

        if not data:
            break
        rows.extend(data)

        last_open_time = int(data[-1][0])
        next_cur = last_open_time + 1
        if next_cur >= end_ms:
            break
        cur = next_cur
        time.sleep(sleep_s)

    return rows


def _rows_to_df(rows: list[list], extended: bool = False) -> pd.DataFrame:
    """Convert raw Binance kline rows to a clean OHLCV DataFrame.

    When *extended* is True the four microstructure columns are included:
    quote_volume, num_trades, taker_buy_base_volume, taker_buy_quote_volume.
    """
    df = pd.DataFrame(
        rows,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "num_trades",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
            "ignore",
        ],
    )
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    keep_cols = _OHLCV_COLS + (_EXTENDED_COLS if extended else [])
    for c in keep_cols:
        df[c] = df[c].astype(np.float64)

    return (
        df[["open_time", *keep_cols]]
        .set_index("open_time")
        .sort_index()
    )


def fetch_and_store(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    start: str = "2020-01-01",
    end: str | None = None,
    store_dir: str = "data/raw",
    limit: int = 1000,
    sleep_s: float = 0.15,
    extended: bool = False,
) -> pd.DataFrame:
    """Fetch Binance klines and write/append to a Parquet file.

    If the Parquet file already exists, only rows newer than the last
    stored timestamp are downloaded (incremental update).  Deduplication
    is applied on the ``open_time`` index.

    Set *extended=True* to also persist quote_volume, num_trades,
    taker_buy_base_volume, and taker_buy_quote_volume.

    Returns the full stored DataFrame.
    """
    os.makedirs(store_dir, exist_ok=True)
    path = _parquet_path(store_dir, symbol, interval)

    end_str = end or pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    end_ms = _to_millis(end_str)

    existing: pd.DataFrame | None = None
    if os.path.exists(path):
        existing = pd.read_parquet(path)
        if not isinstance(existing.index, pd.DatetimeIndex):
            existing.index = pd.to_datetime(existing.index, utc=True)
        elif existing.index.tz is None:
            existing.index = existing.index.tz_localize("UTC")

        max_ts_ms = int(existing.index.max().value // 10**6)
        fetch_start_ms = max_ts_ms + 1
    else:
        fetch_start_ms = _to_millis(start)

    if fetch_start_ms >= end_ms:
        return existing if existing is not None else pd.DataFrame()

    rows = _fetch_pages(symbol, interval, fetch_start_ms, end_ms, limit, sleep_s)

    if rows:
        new_df = _rows_to_df(rows, extended=extended)
        if existing is not None:
            combined = pd.concat([existing, new_df])
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        else:
            combined = new_df
    elif existing is not None:
        combined = existing
    else:
        all_cols = _OHLCV_COLS + (_EXTENDED_COLS if extended else [])
        combined = pd.DataFrame(columns=all_cols)
        combined.index.name = "open_time"

    combined.to_parquet(path)
    return combined


def load(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    start: str | None = None,
    end: str | None = None,
    store_dir: str = "data/raw",
) -> pd.DataFrame:
    """Read the stored Parquet file, optionally filtering by date range.

    Raises ``FileNotFoundError`` if the Parquet file does not exist.
    Call ``fetch_and_store()`` first to populate the store.
    """
    path = _parquet_path(store_dir, symbol, interval)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No data file at {path}. Run fetch_and_store() first."
        )

    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    elif df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    if start is not None:
        df = df[df.index >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        df = df[df.index < pd.Timestamp(end, tz="UTC")]

    return df
