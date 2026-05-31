"""Fetch and store Binance 1h klines with full microstructure columns.

Spot klines (``fetch_taker_volume``)
-------------------------------------
Re-fetches the standard ``/api/v3/klines`` endpoint but keeps ALL 12 columns:
  - ``volume``                 — base asset volume
  - ``quote_asset_volume``     — USDT volume (used for true VWAP)
  - ``num_trades``             — trade count per bar (institutional vs. retail signature)
  - ``taker_buy_base_volume``  — BTC bought aggressively (market orders)
  - ``taker_buy_quote_volume`` — USDT spent by aggressive buyers (slippage/urgency)

Futures klines (``fetch_mark_price_klines``, ``fetch_index_price_klines``)
---------------------------------------------------------------------------
Fetches OHLC from Binance Futures sister endpoints:
  - ``/fapi/v1/markPriceKlines``   — Binance mark price (manipulation-resistant)
  - ``/fapi/v1/indexPriceKlines``  — Underlying spot index (basket of exchanges)

Basis features (derived in v4_features.py):
  - ``basis``            = mark_close − spot_close      (funding pressure)
  - ``futures_premium``  = basis / spot_close           (market sentiment dial)
  - ``index_deviation``  = index_close − spot_close     (lead/lag vs. composite)
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import requests

# ── Endpoints ─────────────────────────────────────────────────────────────────
SPOT_BASE    = "https://api.binance.com"
FUTURES_BASE = "https://fapi.binance.com"
SPOT_KLINES_EP   = "/api/v3/klines"
MARK_KLINES_EP   = "/fapi/v1/markPriceKlines"
INDEX_KLINES_EP  = "/fapi/v1/indexPriceKlines"

# ── Spot kline column layout (Binance /api/v3/klines) ─────────────────────────
_ALL_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "num_trades",
    "taker_buy_base_volume", "taker_buy_quote_volume", "ignore",
]
_KEEP_COLS = [
    "open", "high", "low", "close", "volume",
    "quote_asset_volume", "num_trades",
    "taker_buy_base_volume", "taker_buy_quote_volume",
]
_FLOAT_COLS = [
    "open", "high", "low", "close", "volume",
    "quote_asset_volume", "taker_buy_base_volume", "taker_buy_quote_volume",
]

# ── Futures kline column layout (mark price / index price) ────────────────────
# Same 12-column array but indices 5-11 are zeros / irrelevant for mark/index.
_FUTURES_OHLC_KEEP = ["open", "high", "low", "close"]


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _to_millis(ts: str) -> int:
    return int(pd.to_datetime(ts, utc=True).value // 10**6)


def _fetch_pages(
    base_url: str,
    endpoint: str,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    limit: int = 1000,
    sleep_s: float = 0.15,
    symbol_param: str = "symbol",
) -> list[list]:
    """Generic paginated kline fetcher for any Binance kline endpoint.

    ``symbol_param`` controls the query-string key for the ticker:
    - ``"symbol"`` — used by spot klines and mark-price klines
    - ``"pair"``   — required by ``/fapi/v1/indexPriceKlines``
    """
    rows: list[list] = []
    cur = start_ms
    while True:
        r = requests.get(
            base_url + endpoint,
            params={symbol_param: symbol, "interval": interval,
                    "startTime": cur, "endTime": end_ms, "limit": limit},
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        rows.extend(data)
        last_open = int(data[-1][0])
        next_cur = last_open + 1
        if next_cur >= end_ms:
            break
        cur = next_cur
        time.sleep(sleep_s)
    return rows


def _spot_rows_to_df(rows: list[list]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=_ALL_COLS)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["num_trades"] = df["num_trades"].astype(int)
    for c in _FLOAT_COLS:
        df[c] = df[c].astype(np.float64)
    return df[["open_time", *_KEEP_COLS]].set_index("open_time").sort_index()


def _futures_rows_to_df(rows: list[list]) -> pd.DataFrame:
    """Parse mark-price / index-price klines — only OHLC is meaningful."""
    df = pd.DataFrame(rows, columns=_ALL_COLS)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in _FUTURES_OHLC_KEEP:
        df[c] = df[c].astype(np.float64)
    return df[["open_time", *_FUTURES_OHLC_KEEP]].set_index("open_time").sort_index()


def _incremental_fetch_and_store(
    path: str,
    base_url: str,
    endpoint: str,
    symbol: str,
    interval: str,
    start: str,
    end: str | None,
    rows_to_df,
    limit: int,
    sleep_s: float,
    verbose: bool,
    label: str,
    symbol_param: str = "symbol",
) -> pd.DataFrame:
    """Shared incremental download + parquet cache logic."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    end_str = end or pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    end_ms  = _to_millis(end_str)

    existing: pd.DataFrame | None = None
    if os.path.exists(path):
        existing = pd.read_parquet(path)
        if existing.index.tz is None:
            existing.index = existing.index.tz_localize("UTC")
        fetch_start_ms = int(existing.index.max().value // 10**6) + 1
        if verbose:
            print(f"  {label}: {len(existing):,} rows cached up to "
                  f"{existing.index.max().date()}")
    else:
        fetch_start_ms = _to_millis(start)

    if verbose:
        print(f"  {label}: fetching from "
              f"{pd.to_datetime(fetch_start_ms, unit='ms', utc=True).date()} ...")

    rows = _fetch_pages(base_url, endpoint, symbol, interval,
                        fetch_start_ms, end_ms, limit, sleep_s,
                        symbol_param=symbol_param)

    if rows:
        new_df = rows_to_df(rows)
        if verbose:
            print(f"  {label}: +{len(new_df):,} new rows")
        combined = pd.concat([existing, new_df]) if existing is not None else new_df
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    else:
        combined = existing if existing is not None else pd.DataFrame()
        if verbose:
            print(f"  {label}: already up to date")

    combined.to_parquet(path)
    return combined


def _load_parquet(path: str, label: str, tz_strip: bool) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No cached file at '{path}'. "
            f"Run the corresponding fetch function first."
        )
    df = pd.read_parquet(path)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    if tz_strip:
        df.index = df.index.tz_localize(None)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Public API — Spot taker volume
# ══════════════════════════════════════════════════════════════════════════════

def fetch_taker_volume(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    start: str = "2017-01-01",
    end: str | None = None,
    store_dir: str = "data/raw",
    limit: int = 1000,
    sleep_s: float = 0.15,
    verbose: bool = True,
) -> pd.DataFrame:
    """Download (incrementally) Binance spot klines with all microstructure columns.

    Saves to ``{store_dir}/{symbol}_{interval}_taker.parquet``.
    Incremental — only bars newer than last stored timestamp are fetched.

    Columns saved
    -------------
    open, high, low, close, volume          — standard OHLCV
    quote_asset_volume                      — USDT volume (for true VWAP)
    num_trades                              — trade count (institutional signature)
    taker_buy_base_volume                   — aggressive buy volume in BTC (TFI)
    taker_buy_quote_volume                  — aggressive buy volume in USDT (slippage)
    """
    path = os.path.join(store_dir, f"{symbol}_{interval}_taker.parquet")
    return _incremental_fetch_and_store(
        path=path,
        base_url=SPOT_BASE, endpoint=SPOT_KLINES_EP,
        symbol=symbol, interval=interval, start=start, end=end,
        rows_to_df=_spot_rows_to_df,
        limit=limit, sleep_s=sleep_s, verbose=verbose,
        label=f"{symbol} {interval} taker",
    )


def load_taker_volume(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    store_dir: str = "data/raw",
    tz_strip: bool = True,
) -> pd.DataFrame:
    """Load cached taker-volume parquet. Pass store_dir as an absolute path."""
    path = os.path.join(store_dir, f"{symbol}_{interval}_taker.parquet")
    return _load_parquet(path, "taker volume", tz_strip)


# ══════════════════════════════════════════════════════════════════════════════
# Public API — Futures mark price klines
# ══════════════════════════════════════════════════════════════════════════════

def fetch_mark_price_klines(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    start: str = "2019-09-01",   # mark price available from Binance Futures launch
    end: str | None = None,
    store_dir: str = "data/raw",
    limit: int = 1000,
    sleep_s: float = 0.15,
    verbose: bool = True,
) -> pd.DataFrame:
    """Download Binance Futures mark price klines (OHLC only).

    Mark price is Binance's manipulation-resistant price, computed from
    multiple spot exchanges. Used to calculate the futures basis:
        basis = mark_close - spot_close

    Saves to ``{store_dir}/{symbol}_{interval}_mark_price.parquet``.
    """
    path = os.path.join(store_dir, f"{symbol}_{interval}_mark_price.parquet")
    return _incremental_fetch_and_store(
        path=path,
        base_url=FUTURES_BASE, endpoint=MARK_KLINES_EP,
        symbol=symbol, interval=interval, start=start, end=end,
        rows_to_df=_futures_rows_to_df,
        limit=limit, sleep_s=sleep_s, verbose=verbose,
        label=f"{symbol} {interval} mark price",
    )


def load_mark_price_klines(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    store_dir: str = "data/raw",
    tz_strip: bool = True,
) -> pd.DataFrame:
    path = os.path.join(store_dir, f"{symbol}_{interval}_mark_price.parquet")
    return _load_parquet(path, "mark price", tz_strip)


# ══════════════════════════════════════════════════════════════════════════════
# Public API — Futures index price klines
# ══════════════════════════════════════════════════════════════════════════════

def fetch_index_price_klines(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    start: str = "2019-09-01",
    end: str | None = None,
    store_dir: str = "data/raw",
    limit: int = 1000,
    sleep_s: float = 0.15,
    verbose: bool = True,
) -> pd.DataFrame:
    """Download Binance Futures index price klines (OHLC only).

    Index price is the underlying spot basket from multiple exchanges.
    Used to detect divergences between our spot feed and the composite index.
        index_deviation = index_close - spot_close

    Saves to ``{store_dir}/{symbol}_{interval}_index_price.parquet``.
    """
    path = os.path.join(store_dir, f"{symbol}_{interval}_index_price.parquet")
    return _incremental_fetch_and_store(
        path=path,
        base_url=FUTURES_BASE, endpoint=INDEX_KLINES_EP,
        symbol=symbol, interval=interval, start=start, end=end,
        rows_to_df=_futures_rows_to_df,
        limit=limit, sleep_s=sleep_s, verbose=verbose,
        label=f"{symbol} {interval} index price",
        symbol_param="pair",   # indexPriceKlines uses 'pair=' not 'symbol='
    )


def load_index_price_klines(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    store_dir: str = "data/raw",
    tz_strip: bool = True,
) -> pd.DataFrame:
    path = os.path.join(store_dir, f"{symbol}_{interval}_index_price.parquet")
    return _load_parquet(path, "index price", tz_strip)
