"""V4 feature engineering — three new feature groups.

Task 1 — Trade Flow Imbalance (TFI) + Fractional Differentiation
Task 3 — Regime Detection (Hurst, rolling ADF on ETH/BTC, BB-width percentile, sideways flag)

Task 2 (meta-labeling) lives in ``hmats.data.meta_labeling``.

Quick-start
-----------
    from hmats.data.v4_features import build_v4_features, find_min_fracdiff_d
    import pandas as pd

    v1  = pd.read_parquet("data/features/BTCUSDT_1h_features.parquet")
    v3  = pd.read_parquet("data/features/BTCUSDT_1h_v3_features.parquet")

    # Optional — requires BTCUSDT_1h_taker.parquet (run fetch_taker_volume first)
    from hmats.data.fetch_taker_volume import load_taker_volume
    taker = load_taker_volume()

    v4 = build_v4_features(v1, v3, taker_df=taker)
    v4.to_parquet("data/features/BTCUSDT_1h_v4_features.parquet")
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ──────────────────────────────────────────────────────────────────────────────
# Task 1a — Trade Flow Imbalance
# ──────────────────────────────────────────────────────────────────────────────

def compute_tfi(taker_df: pd.DataFrame) -> pd.DataFrame:
    """Compute microstructure features from full Binance kline columns.

    Requires columns: volume, quote_asset_volume, num_trades,
                      taker_buy_base_volume, taker_buy_quote_volume.

    Returns
    -------
    Group A — Trade Flow Imbalance (TFI)
        tfi_pct        TFI = (buy_vol - sell_vol) / total_vol  ∈ (-1, 1)
        tfi_z_24h      24h z-score of tfi_pct
        tfi_z_72h      72h z-score
        tfi_z_168h     168h z-score
        tfi_ema_12     12-bar EMA of tfi_pct (momentum smoother)
        tfi_ema_24     24-bar EMA of tfi_pct

    Group B — Average Trade Size (institutional vs. retail signature)
        avg_trade_size      volume / num_trades  (BTC per execution)
        avg_trade_size_z24  24h z-score — spikes = whale block trades

    Group C — True VWAP (precise money-weighted price)
        true_vwap           quote_asset_volume / volume  (USDT/BTC)
        close_vs_true_vwap  (close - true_vwap) / true_vwap — deviation %

    Group D — Aggressor Execution Price (slippage / urgency dial)
        taker_buy_price     avg price paid by market buyers (USDT/BTC)
        taker_sell_price    avg price received by market sellers
        taker_price_premium (taker_buy_price / taker_sell_price) - 1
                            > 0 → buyers eating into book, strong momentum
                            < 0 → sellers desperate, distribution signal
    """
    vol       = taker_df["volume"]
    quote_vol = taker_df["quote_asset_volume"]
    n_trades  = taker_df["num_trades"].replace(0, np.nan)
    tb_base   = taker_df["taker_buy_base_volume"]
    tb_quote  = taker_df["taker_buy_quote_volume"]

    ts_base  = vol   - tb_base    # taker sell base
    ts_quote = quote_vol - tb_quote  # taker sell quote

    out = pd.DataFrame(index=taker_df.index)

    # ── Group A: TFI ──────────────────────────────────────────────────────────
    tfi_pct = ((tb_base - ts_base) / (vol + 1e-12)).clip(-1, 1)
    out["tfi_pct"] = tfi_pct
    for w, name in [(24, "24h"), (72, "72h"), (168, "168h")]:
        mu  = tfi_pct.rolling(w, min_periods=w // 2).mean()
        sig = tfi_pct.rolling(w, min_periods=w // 2).std()
        out[f"tfi_z_{name}"] = ((tfi_pct - mu) / (sig + 1e-12)).clip(-5, 5)
    out["tfi_ema_12"] = tfi_pct.ewm(span=12, adjust=False).mean()
    out["tfi_ema_24"] = tfi_pct.ewm(span=24, adjust=False).mean()

    # ── Group B: Average trade size ───────────────────────────────────────────
    avg_sz = (vol / n_trades).fillna(0)
    out["avg_trade_size"] = avg_sz
    mu24  = avg_sz.rolling(24, min_periods=12).mean()
    sig24 = avg_sz.rolling(24, min_periods=12).std()
    out["avg_trade_size_z24"] = ((avg_sz - mu24) / (sig24 + 1e-12)).clip(-5, 5)

    # ── Group C: True VWAP ────────────────────────────────────────────────────
    true_vwap = quote_vol / (vol + 1e-12)
    out["true_vwap"]          = true_vwap
    out["close_vs_true_vwap"] = (taker_df["close"] / (true_vwap + 1e-12) - 1).clip(-0.05, 0.05)

    # ── Group D: Taker execution prices ───────────────────────────────────────
    taker_buy_px  = tb_quote  / (tb_base  + 1e-12)
    taker_sell_px = ts_quote  / (ts_base  + 1e-12)
    # zero out bars where taker volume was essentially zero
    taker_buy_px  = taker_buy_px.where(tb_base  > 1e-6, np.nan)
    taker_sell_px = taker_sell_px.where(ts_base > 1e-6, np.nan)
    premium = (taker_buy_px / (taker_sell_px + 1e-12) - 1).clip(-0.01, 0.01)

    out["taker_buy_price"]     = taker_buy_px
    out["taker_sell_price"]    = taker_sell_px
    out["taker_price_premium"] = premium

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Task 1b — Fractional Differentiation (López de Prado, Ch. 5)
# ──────────────────────────────────────────────────────────────────────────────

def _fracdiff_weights(
    d: float, size: int, threshold: float = 1e-5, max_window: int = 200
) -> np.ndarray:
    """Fixed-width fractional differencing weights (López de Prado § 5.4).

    w_0 = 1,  w_k = -w_{k-1} * (d - k + 1) / k

    Stops when ``|w_k| < threshold`` OR ``k >= max_window``.
    The hard cap on ``max_window`` makes the effective lookback predictable
    and avoids the weight vector growing to ``n`` for small d.
    Returns weights oldest-first for direct dot-product with a sliding window.
    """
    w = [1.0]
    for k in range(1, min(size, max_window)):
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < threshold:
            break
        w.append(w_k)
    return np.array(w[::-1])          # oldest weight first → dot with oldest data


def fracdiff_series(
    series: pd.Series,
    d: float,
    threshold: float = 1e-5,
    max_window: int = 200,
) -> pd.Series:
    """Apply fixed-width fractional differencing of order ``d`` to ``series``.

    Uses a fixed maximum lookback (``max_window``) so the warm-up period is
    bounded and predictable regardless of ``d``.  This follows the practical
    recommendation in AFML Ch. 5 (López de Prado).

    Parameters
    ----------
    series:     Price series (log-price recommended for numerical stability).
    d:          Differencing order ∈ (0, 1].  d=1 → standard first diff.
    threshold:  Weight truncation threshold.
    max_window: Hard cap on the weight vector length (default 200 bars).
    """
    vals   = series.values.astype(float)
    n      = len(vals)
    full_w = _fracdiff_weights(d, n, threshold, max_window)
    p      = len(full_w)           # effective window = min(threshold_cutoff, max_window)
    out    = np.full(n, np.nan)

    for t in range(p - 1, n):
        window = vals[t - p + 1 : t + 1]
        if not np.isnan(window).any():
            out[t] = float(np.dot(full_w, window))

    return pd.Series(out, index=series.index, name=f"fracdiff_{d:.2f}")


def find_min_fracdiff_d(
    series: pd.Series,
    d_values: Optional[list[float]] = None,
    adf_pvalue_threshold: float = 0.05,
    threshold: float = 1e-5,
    verbose: bool = True,
) -> tuple[float, pd.Series]:
    """Sweep ``d`` to find minimum order that makes ``series`` stationary.

    Stationarity criterion: ADF p-value < ``adf_pvalue_threshold``.

    Parameters
    ----------
    series:
        Raw price series (log-price recommended).
    d_values:
        Candidate ``d`` values to try in ascending order.
        Defaults to [0.1, 0.2, ..., 0.9].
    adf_pvalue_threshold:
        ADF rejection threshold (default 0.05 for 95% confidence).
    verbose:
        Print sweep table.

    Returns
    -------
    (min_d, frac_diff_series)
        ``min_d`` — smallest d achieving stationarity.
        ``frac_diff_series`` — the corresponding differenced series.
    """
    from statsmodels.tsa.stattools import adfuller  # lazy import

    if d_values is None:
        d_values = [round(x, 1) for x in np.arange(0.1, 1.0, 0.1)]

    log_series = np.log(series.replace(0, np.nan)).dropna()

    if verbose:
        print(f"{'d':>6}  {'ADF stat':>10}  {'p-value':>10}  {'Stationary':>12}")
        print("─" * 45)

    min_d: float = float(d_values[-1])
    best_series: pd.Series = fracdiff_series(log_series, min_d, threshold)

    for d in d_values:
        fd = fracdiff_series(log_series, d, threshold).dropna()
        if len(fd) < 50:
            continue
        adf_stat, pval, *_ = adfuller(fd, maxlag=1, regression="c", autolag=None)
        stationary = pval < adf_pvalue_threshold
        if verbose:
            mark = "✓" if stationary else ""
            print(f"{d:>6.1f}  {adf_stat:>10.4f}  {pval:>10.6f}  {mark:>12}")
        if stationary:
            min_d = d
            best_series = fracdiff_series(log_series, d, threshold).reindex(series.index)
            break

    if verbose:
        print(f"\nMinimum stationary d = {min_d:.1f}")

    return min_d, best_series


# ──────────────────────────────────────────────────────────────────────────────
# Futures basis features (mark price + index price)
# ──────────────────────────────────────────────────────────────────────────────

def compute_basis_features(
    spot_close: pd.Series,
    mark_df: Optional[pd.DataFrame] = None,
    index_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Compute futures basis and index deviation features.

    Parameters
    ----------
    spot_close:
        Spot close price series (from V1 / raw OHLCV).
    mark_df:
        Mark price OHLC DataFrame from ``fetch_mark_price_klines()``.
        Only the ``close`` column is used.
    index_df:
        Index price OHLC DataFrame from ``fetch_index_price_klines()``.
        Only the ``close`` column is used.

    Returns
    -------
    DataFrame with columns (each is NaN if the source DataFrame is None):

    From mark price:
        mark_basis         mark_close − spot_close  (positive → futures premium)
        mark_premium_pct   mark_basis / spot_close × 100  (% funding pressure)
        mark_premium_z24   24h z-score of mark_premium_pct
        mark_premium_z168  168h z-score

    From index price:
        index_deviation     index_close − spot_close
        index_deviation_pct index_deviation / spot_close × 100
        index_lead_1h       index_close shifted -1 vs spot (lead/lag detection)
    """
    out = pd.DataFrame(index=spot_close.index)

    if mark_df is not None:
        mark_close = mark_df["close"].reindex(spot_close.index)
        basis      = mark_close - spot_close
        prem_pct   = (basis / (spot_close + 1e-12) * 100)

        out["mark_basis"]        = basis
        out["mark_premium_pct"]  = prem_pct.clip(-5, 5)

        for w, name in [(24, "z24"), (168, "z168")]:
            mu  = prem_pct.rolling(w, min_periods=w // 2).mean()
            sig = prem_pct.rolling(w, min_periods=w // 2).std()
            out[f"mark_premium_{name}"] = ((prem_pct - mu) / (sig + 1e-12)).clip(-5, 5)

    if index_df is not None:
        idx_close  = index_df["close"].reindex(spot_close.index)
        idx_dev    = idx_close - spot_close
        idx_pct    = (idx_dev / (spot_close + 1e-12) * 100)

        out["index_deviation"]     = idx_dev
        out["index_deviation_pct"] = idx_pct.clip(-5, 5)
        # Positive lead_1h means index moved up before spot (predictive)
        out["index_lead_1h"] = (idx_close.shift(-1) - spot_close).clip(-500, 500)

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Task 3a — Rolling Hurst Exponent
# ──────────────────────────────────────────────────────────────────────────────

def _hurst_rs(ts: np.ndarray) -> float:
    """R/S Hurst exponent for a 1-D array.

    Uses log-log regression of (window_size, R/S) across multiple lags.
    Returns NaN if computation fails or series is too short / constant.
    """
    n = len(ts)
    if n < 20:
        return np.nan
    ts = ts.astype(float)
    if np.std(ts) < 1e-12:
        return np.nan

    lags = np.unique(np.geomspace(10, n // 2, num=12).astype(int))
    lags = lags[lags >= 4]

    rs_vals = []
    lag_vals = []
    for lag in lags:
        n_segs = n // lag
        if n_segs < 2:
            continue
        rs_seg = []
        for seg in range(n_segs):
            chunk = ts[seg * lag : (seg + 1) * lag]
            mean  = chunk.mean()
            dev   = np.cumsum(chunk - mean)
            R     = dev.max() - dev.min()
            S     = chunk.std(ddof=1)
            if S > 1e-12:
                rs_seg.append(R / S)
        if rs_seg:
            rs_vals.append(np.mean(rs_seg))
            lag_vals.append(lag)

    if len(lag_vals) < 3:
        return np.nan

    slope, *_ = scipy_stats.linregress(np.log(lag_vals), np.log(rs_vals))
    return float(np.clip(slope, 0.0, 1.0))


def rolling_hurst(
    close: pd.Series,
    windows: list[int] = (24, 72, 168),
    min_periods_frac: float = 0.5,
) -> pd.DataFrame:
    """Compute rolling Hurst exponent for each window size.

    Returns columns ``hurst_{w}h`` for each w in windows.

    Interpretation:
      H > 0.5  →  trending (persistent)
      H ≈ 0.5  →  random walk
      H < 0.5  →  mean-reverting (anti-persistent)
    """
    log_ret = np.log(close).diff()
    out = pd.DataFrame(index=close.index)

    for w in windows:
        min_p = max(20, int(w * min_periods_frac))
        col   = f"hurst_{w}h"
        values = np.full(len(close), np.nan)

        arr = log_ret.values
        for i in range(len(arr)):
            if i < min_p:
                continue
            start = max(0, i - w + 1)
            chunk = arr[start : i + 1]
            chunk = chunk[~np.isnan(chunk)]
            if len(chunk) >= min_p:
                values[i] = _hurst_rs(chunk)

        out[col] = values

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Task 3b — Rolling ADF on ETH/BTC cointegration ratio
# ──────────────────────────────────────────────────────────────────────────────

def rolling_adf(
    series: pd.Series,
    windows: list[int] = (168, 336, 720),
    maxlag: int = 1,
) -> pd.DataFrame:
    """Rolling ADF test on a spread / ratio series.

    Outputs for each window:
      ``adf_tstat_{w}h``  — ADF t-statistic (more negative = more stationary)
      ``adf_pval_{w}h``   — ADF p-value (low = stationary mean-reverting)

    Lower p-value signals the ratio is tightly cointegrated and mean-reverting.
    """
    from statsmodels.tsa.stattools import adfuller

    out = pd.DataFrame(index=series.index)
    arr = series.values.astype(float)
    n   = len(arr)

    for w in windows:
        tstat_vals = np.full(n, np.nan)
        pval_vals  = np.full(n, np.nan)

        for i in range(w - 1, n):
            chunk = arr[i - w + 1 : i + 1]
            if np.isnan(chunk).any() or chunk.std() < 1e-12:
                continue
            try:
                res = adfuller(chunk, maxlag=maxlag, regression="c", autolag=None)
                tstat_vals[i] = res[0]
                pval_vals[i]  = res[1]
            except Exception:
                pass

        out[f"adf_tstat_{w}h"] = tstat_vals
        out[f"adf_pval_{w}h"]  = pval_vals

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Task 3c — Bollinger Band Width Percentile + Sideways Flag
# ──────────────────────────────────────────────────────────────────────────────

def bb_width_percentile(
    close: pd.Series,
    bb_window: int = 20,
    bb_std: float = 2.0,
    baseline_window: int = 180 * 24,   # 180 days at 1h bars
    pct_squeeze_threshold: float = 0.10,
) -> pd.DataFrame:
    """Compute Bollinger Band width percentile relative to rolling baseline.

    Returns:
      ``bb_width_raw``    — raw BB width = (upper - lower) / middle
      ``bb_width_pct``    — percentile of bb_width_raw within baseline_window ∈ [0, 1]
      ``bb_squeeze_flag`` — 1 if bb_width_pct < pct_squeeze_threshold
    """
    mid    = close.rolling(bb_window, min_periods=bb_window).mean()
    sigma  = close.rolling(bb_window, min_periods=bb_window).std(ddof=1)
    upper  = mid + bb_std * sigma
    lower  = mid - bb_std * sigma
    width  = (upper - lower) / (mid + 1e-12)

    # Rolling percentile rank
    width_arr  = width.values.astype(float)
    n          = len(width_arr)
    pct_arr    = np.full(n, np.nan)

    for i in range(n):
        start = max(0, i - baseline_window + 1)
        chunk = width_arr[start : i + 1]
        chunk = chunk[~np.isnan(chunk)]
        if len(chunk) >= 2:
            pct_arr[i] = float(scipy_stats.percentileofscore(chunk, chunk[-1])) / 100.0

    out = pd.DataFrame(index=close.index)
    out["bb_width_raw"]    = width
    out["bb_width_pct"]    = pct_arr
    out["bb_squeeze_flag"] = (pct_arr < pct_squeeze_threshold).astype(np.float32)

    return out


def sideways_flag(
    close: pd.Series,
    vol_of_vol_col: Optional[pd.Series] = None,
    vol_of_vol_window: int = 72,
    bb_window: int = 20,
    baseline_window: int = 180 * 24,
    squeeze_pct: float = 0.10,
    vov_pct_threshold: float = 0.20,
) -> pd.DataFrame:
    """Combined sideways / compression flag.

    sideways = (bb_width_pct < squeeze_pct) AND (vol_of_vol_pct < vov_pct_threshold)

    When both BB width AND volatility-of-volatility are in their lowest
    historical percentiles, the market is in deep compression — ripe for
    a breakout move.

    Returns:
      ``bb_width_pct``   — from bb_width_percentile()
      ``vov_72h_pct``    — vol-of-vol percentile over baseline
      ``sideways_flag``  — 1 if both < threshold, else 0
    """
    bb_df = bb_width_percentile(
        close,
        bb_window=bb_window,
        baseline_window=baseline_window,
        pct_squeeze_threshold=squeeze_pct,
    )

    # Vol-of-vol: std of 1h-return std over vol_of_vol_window bars
    log_ret = np.log(close).diff()
    vol_24h = log_ret.rolling(24).std()

    if vol_of_vol_col is not None:
        vov = vol_of_vol_col
    else:
        vov = vol_24h.rolling(vol_of_vol_window, min_periods=vol_of_vol_window // 2).std()

    vov_arr = vov.values.astype(float)
    n       = len(vov_arr)
    vov_pct = np.full(n, np.nan)

    for i in range(n):
        start = max(0, i - baseline_window + 1)
        chunk = vov_arr[start : i + 1]
        chunk = chunk[~np.isnan(chunk)]
        if len(chunk) >= 2:
            vov_pct[i] = scipy_stats.percentileofscore(chunk, chunk[-1]) / 100.0

    out = bb_df[["bb_width_pct"]].copy()
    out["vov_72h_pct"]  = vov_pct
    out["sideways_flag"] = (
        (bb_df["bb_width_pct"].values < squeeze_pct) &
        (vov_pct < vov_pct_threshold)
    ).astype(np.float32)

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Main builder
# ──────────────────────────────────────────────────────────────────────────────

def build_v4_features(
    v1_df: pd.DataFrame,
    v3_df: pd.DataFrame,
    taker_df: Optional[pd.DataFrame] = None,
    mark_df: Optional[pd.DataFrame] = None,
    index_df: Optional[pd.DataFrame] = None,
    fracdiff_d: Optional[float] = None,
    fracdiff_threshold: float = 1e-5,
    hurst_windows: list[int] = (24, 72),
    adf_windows: list[int] = (168, 336, 720),
    verbose: bool = True,
) -> pd.DataFrame:
    """Build the full V4 feature DataFrame.

    Parameters
    ----------
    v1_df:
        V1 features parquet (must contain ``close``).
    v3_df:
        V3 features parquet (must contain ``cross_eth_btc_ratio``).
    taker_df:
        Full spot kline parquet with ``taker_buy_base_volume``, ``volume``,
        ``quote_asset_volume``, ``num_trades``, ``taker_buy_quote_volume``.
        If None, all microstructure features (TFI, true VWAP, avg trade size,
        taker price premium) are skipped.
    mark_df:
        Mark price OHLC DataFrame from ``fetch_mark_price_klines()``.
        If None, mark price basis features are skipped.
    index_df:
        Index price OHLC DataFrame from ``fetch_index_price_klines()``.
        If None, index deviation features are skipped.
    fracdiff_d:
        Fixed d for fractional differentiation. If None, the minimum
        stationary d is found automatically via ADF sweep.
    hurst_windows:
        Rolling windows for Hurst exponent (24h and 72h; 168h already in V1).
    adf_windows:
        Rolling windows for the ETH/BTC ratio ADF test.
    verbose:
        Print progress.

    Returns
    -------
    DataFrame with all V4 features, indexed to the same timestamps as v1_df.
    """
    close = v1_df["close"]
    idx   = v1_df.index
    parts: list[pd.DataFrame] = []

    # ── Microstructure (TFI + true VWAP + avg trade size + taker premium) ────
    if taker_df is not None:
        if verbose:
            print("Computing microstructure features (TFI, true VWAP, avg trade size, "
                  "taker price premium)...")
        micro_df = compute_tfi(taker_df).reindex(idx)
        parts.append(micro_df)
        if verbose:
            print(f"  → {micro_df.shape[1]} microstructure columns")
    else:
        warnings.warn(
            "taker_df is None — microstructure features skipped "
            "(TFI, true VWAP, avg trade size, taker price premium). "
            "Run Phase F in 00_data_ingestion_v2.ipynb to enable.",
            UserWarning,
            stacklevel=2,
        )

    # ── Futures basis (mark price + index price) ──────────────────────────────
    if mark_df is not None or index_df is not None:
        if verbose:
            sources = []
            if mark_df  is not None: sources.append("mark price")
            if index_df is not None: sources.append("index price")
            print(f"Computing basis features ({', '.join(sources)})...")
        basis_df = compute_basis_features(close, mark_df=mark_df, index_df=index_df)
        parts.append(basis_df.reindex(idx))
        if verbose:
            print(f"  → {basis_df.shape[1]} basis columns")
    else:
        warnings.warn(
            "mark_df and index_df are None — basis/premium features skipped. "
            "Run Phases H and I in 00_data_ingestion_v2.ipynb to enable.",
            UserWarning,
            stacklevel=2,
        )

    # ── Task 1b: Fractional Differentiation ───────────────────────────────────
    if verbose:
        print("Computing fractional differentiation...")
    if fracdiff_d is None:
        fracdiff_d, fd_series = find_min_fracdiff_d(
            close, verbose=verbose
        )
    else:
        log_close = np.log(close.replace(0, np.nan))
        fd_series = fracdiff_series(log_close, fracdiff_d, fracdiff_threshold)

    fd_series.name = f"fracdiff_close_d{fracdiff_d:.1f}"
    parts.append(fd_series.reindex(idx).to_frame())
    if verbose:
        print(f"  → fracdiff d={fracdiff_d:.1f}, col: {fd_series.name}")

    # ── Task 3a: Rolling Hurst (24h, 72h — 168h already in V1) ───────────────
    if verbose:
        print(f"Computing rolling Hurst {hurst_windows}h ...")
    hurst_df = rolling_hurst(close, windows=list(hurst_windows))
    parts.append(hurst_df.reindex(idx))
    if verbose:
        print(f"  → {hurst_df.shape[1]} Hurst columns")

    # ── Task 3b: Rolling ADF on ETH/BTC ratio ─────────────────────────────────
    if "cross_eth_btc_ratio" in v3_df.columns:
        if verbose:
            print(f"Computing rolling ADF on cross_eth_btc_ratio {adf_windows}h ...")
        ratio  = v3_df["cross_eth_btc_ratio"].reindex(idx)
        adf_df = rolling_adf(ratio, windows=list(adf_windows))
        parts.append(adf_df)
        if verbose:
            print(f"  → {adf_df.shape[1]} ADF columns")
    else:
        warnings.warn(
            "cross_eth_btc_ratio not found in v3_df — ADF features skipped.",
            UserWarning,
            stacklevel=2,
        )

    # ── Task 3c: BB width percentile + sideways flag ──────────────────────────
    if verbose:
        print("Computing BB width percentile and sideways flag...")
    vov_col = v1_df.get("vol_of_vol_72h")  # already in V1; reuse
    sw_df   = sideways_flag(close, vol_of_vol_col=vov_col)
    parts.append(sw_df.reindex(idx))
    if verbose:
        print(f"  → {sw_df.shape[1]} sideways columns")

    # ── Concatenate ───────────────────────────────────────────────────────────
    v4 = pd.concat(parts, axis=1)
    v4.index.name = "open_time"

    if verbose:
        print(f"\nV4 feature set: {v4.shape[1]} columns, {len(v4):,} rows")
        print(f"Columns: {list(v4.columns)}")
        nan_pct = v4.isna().mean() * 100
        high_nan = nan_pct[nan_pct > 20]
        if len(high_nan) > 0:
            print(f"\nHigh NaN columns (>20%):")
            for c, pct in high_nan.items():
                print(f"  {c}: {pct:.1f}%")

    return v4


# ──────────────────────────────────────────────────────────────────────────────
# Feature list (for feature registry / selector)
# ──────────────────────────────────────────────────────────────────────────────

V4_TFI_COLS = [
    "tfi_raw", "tfi_pct",
    "tfi_z_24h", "tfi_z_72h", "tfi_z_168h",
    "tfi_ema_12", "tfi_ema_24",
]

V4_FRACDIFF_COLS_TEMPLATE = "fracdiff_close_d{d:.1f}"   # d filled in at build time

V4_HURST_COLS = ["hurst_24h", "hurst_72h"]     # 168h already in V1

V4_ADF_COLS = [
    "adf_tstat_168h", "adf_pval_168h",
    "adf_tstat_336h", "adf_pval_336h",
    "adf_tstat_720h", "adf_pval_720h",
]

V4_REGIME_COLS = [
    "bb_width_raw", "bb_width_pct",
    "vov_72h_pct", "sideways_flag",
]
