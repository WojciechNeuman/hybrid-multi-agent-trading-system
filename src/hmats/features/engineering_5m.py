"""5-minute microstructure feature engineering for HMATS.

Three feature groups
--------------------
M1 – Trade-Flow Imbalance & Microstructure
    Derived directly from the Binance kline microstructure columns
    (quote_volume, num_trades, taker_buy_*).  Zero lookahead.

M2 – Long Memory & Advanced Technicals
    Fractional differentiation of the close series (FFD, minimum-d
    stationarity criterion) plus classic OHLCV-based technical features
    calibrated on 5-minute bars.

M3 – Statistical Regime Dials
    Rolling Hurst exponent (R/S method), Bollinger-Band width percentile,
    and ETH/BTC ADF p-value (forward-filled from hourly to 5-minute).
    All rolling windows operate at three lookbacks: 24 h, 72 h, 168 h
    (288, 864, 2016 five-minute bars respectively).

Public API
----------
make_features_5m(df, eth_btc_hourly=None) -> pd.DataFrame
    Master function.  Accepts the nine-column DataFrame from
    ``loaders.load_5m_extended()`` and an optional hourly ETH/BTC ratio
    series.  Returns a DataFrame of engineered features aligned to the
    same index as *df*, with NaN rows at the head where rolling windows
    have insufficient history.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BARS_PER_HOUR = 12          # 5-min bars in one hour
BARS_24H  = 24  * BARS_PER_HOUR   # 288
BARS_72H  = 72  * BARS_PER_HOUR   # 864
BARS_168H = 168 * BARS_PER_HOUR   # 2016
BARS_180D = 180 * 24 * BARS_PER_HOUR  # 51840 – baseline for BB percentile

# Fractional-differentiation
FFD_MIN_D     = 0.1
FFD_MAX_D     = 0.9
FFD_STEP_D    = 0.05
FFD_WEIGHT_THRESH = 1e-5
FFD_CALIB_WINDOW  = 90 * 24 * BARS_PER_HOUR   # 90-day calibration window

# ATR period (5-min bars equivalent to 1 h = 12 bars, use 14×12 ≈ 168)
ATR_PERIOD = 14 * BARS_PER_HOUR


# ===========================================================================
# Internal helpers
# ===========================================================================


def _safe_div(a: np.ndarray, b: np.ndarray, fill: float = np.nan) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(np.abs(b) > 1e-12, a / b, fill)
    return out.astype(np.float64)


def _ema(series: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1)
    out = np.empty_like(series, dtype=np.float64)
    out[0] = series[0]
    for i in range(1, len(series)):
        out[i] = alpha * series[i] + (1 - alpha) * out[i - 1]
    return out


def _rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    delta = np.diff(close, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = _ema(gain, period)
    avg_loss = _ema(loss, period)
    rs = _safe_div(avg_gain, avg_loss, fill=0.0)
    return 100.0 - (100.0 / (1.0 + rs))


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
         period: int = ATR_PERIOD) -> np.ndarray:
    prev_close = np.roll(close, 1); prev_close[0] = close[0]
    tr = np.maximum(high - low,
         np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    return _ema(tr, period)


# ===========================================================================
# M1 – Microstructure features
# ===========================================================================


def compute_m1(df: pd.DataFrame) -> pd.DataFrame:
    """Trade-flow imbalance and microstructure features.

    Requires columns: volume, quote_volume, num_trades,
    taker_buy_base_volume, taker_buy_quote_volume, close.
    """
    v   = df["volume"].values
    qv  = df["quote_volume"].values
    nt  = df["num_trades"].values
    tbb = df["taker_buy_base_volume"].values
    tbq = df["taker_buy_quote_volume"].values
    c   = df["close"].values

    # Trade-flow imbalance: net buy pressure in base-asset units
    tfi = tbb * 2.0 - v   # = tbb - (v - tbb)

    # Average trade size (base asset per trade)
    avg_ts = _safe_div(v, nt)

    # True intra-bar VWAP from quote and base volumes
    vwap = _safe_div(qv, v)
    vwap_dev = _safe_div(c - vwap, vwap)

    # Taker slippage / urgency: ratio of avg buy price to avg sell price
    sell_base  = v - tbb
    sell_quote = qv - tbq
    buy_price  = _safe_div(tbq, tbb)
    sell_price = _safe_div(sell_quote, sell_base)
    tsu = _safe_div(buy_price, sell_price)

    out = pd.DataFrame(index=df.index)
    out["tfi_5m"]                  = tfi
    out["avg_trade_size_5m"]       = avg_ts
    out["true_vwap_dev_5m"]        = vwap_dev
    out["taker_slippage_urgency_5m"] = tsu
    return out


# ===========================================================================
# M2 – Long Memory & Advanced Technicals
# ===========================================================================


def _ffd_weights(d: float, max_size: int) -> np.ndarray:
    """Fixed-width-window fractional differentiation weights (oldest first)."""
    w = [1.0]
    for k in range(1, max_size):
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < FFD_WEIGHT_THRESH:
            break
        w.append(w_k)
    return np.array(w[::-1], dtype=np.float64)


def _apply_ffd(log_close: np.ndarray, d: float) -> np.ndarray:
    """Apply FFD with given d; returns NaN where window is incomplete."""
    w = _ffd_weights(d, len(log_close))
    width = len(w)
    out = np.full(len(log_close), np.nan)
    for i in range(width - 1, len(log_close)):
        out[i] = float(w @ log_close[i - width + 1: i + 1])
    return out


def _find_min_d(log_close: np.ndarray) -> float:
    """Sweep d ∈ [FFD_MIN_D, FFD_MAX_D] and return smallest d where
    the ADF test rejects the unit-root null (p < 0.05)."""
    from statsmodels.tsa.stattools import adfuller

    for d in np.arange(FFD_MIN_D, FFD_MAX_D + FFD_STEP_D / 2, FFD_STEP_D):
        fd = _apply_ffd(log_close, round(d, 4))
        fd_clean = fd[~np.isnan(fd)]
        if len(fd_clean) < 20:
            continue
        try:
            _, pval, *_ = adfuller(fd_clean, autolag="AIC")
        except Exception:
            continue
        if pval < 0.05:
            return round(d, 4)
    return FFD_MAX_D  # fallback: fully differenced is always stationary


def compute_m2(df: pd.DataFrame) -> pd.DataFrame:
    """Long-memory and technical features on 5-minute close series."""
    c   = df["close"].values
    h   = df["high"].values
    lo  = df["low"].values
    log_c = np.log(np.maximum(c, 1e-12))

    # --- fractional differentiation ---
    calib_len = min(FFD_CALIB_WINDOW, len(log_c))
    d_star = _find_min_d(log_c[:calib_len])
    frac_diff = _apply_ffd(log_c, d_star)

    # --- ATR (percentage) ---
    atr_abs = _atr(h, lo, c)
    atr_pct = _safe_div(atr_abs, c)

    # --- RSI (14 × 12 = 168 bars ≈ 14 h) ---
    rsi = _rsi(c, period=ATR_PERIOD)

    # --- MACD on 5-min bars (12h / 26h / 9h in 5-min units) ---
    ema12 = _ema(c, 12 * BARS_PER_HOUR)
    ema26 = _ema(c, 26 * BARS_PER_HOUR)
    macd_line  = ema12 - ema26
    macd_signal = _ema(macd_line, 9 * BARS_PER_HOUR)
    macd_hist  = macd_line - macd_signal

    # --- Bollinger Band width (20h period) ---
    bb_period = 20 * BARS_PER_HOUR
    bb_mid  = pd.Series(c).rolling(bb_period).mean().values
    bb_std  = pd.Series(c).rolling(bb_period).std(ddof=1).values
    bb_width = _safe_div(2.0 * bb_std, bb_mid)

    # --- Log returns ---
    log_ret = np.concatenate([[np.nan], np.diff(log_c)])

    # --- Candle structure ---
    body      = np.abs(c - df["open"].values)
    full_range = np.maximum(h - lo, 1e-12)
    body_ratio = body / full_range
    upper_wick = (h - np.maximum(c, df["open"].values)) / full_range
    lower_wick = (np.minimum(c, df["open"].values) - lo) / full_range
    bullish = (c > df["open"].values).astype(np.float32)

    out = pd.DataFrame(index=df.index)
    out["frac_diff_close_5m"]   = frac_diff
    out["frac_diff_d_star"]     = d_star          # constant – useful for audit
    out["atr_pct_5m"]           = atr_pct
    out["rsi_5m"]               = rsi
    out["macd_5m"]              = macd_line
    out["macd_signal_5m"]       = macd_signal
    out["macd_hist_5m"]         = macd_hist
    out["bb_width_5m"]          = bb_width
    out["log_ret_5m"]           = log_ret
    out["body_ratio_5m"]        = body_ratio
    out["upper_wick_5m"]        = upper_wick
    out["lower_wick_5m"]        = lower_wick
    out["bullish_5m"]           = bullish
    return out


# ===========================================================================
# M3 – Regime Dials
# ===========================================================================


def _hurst_rs(arr: np.ndarray) -> float:
    """Hurst exponent via rescaled-range (R/S) analysis.

    Returns 0.5 (random-walk default) when the window is too short or
    the R/S regression is degenerate.
    """
    n = len(arr)
    if n < 32:
        return 0.5

    max_lag = min(n // 4, 256)
    lags  = []
    rs_vals = []
    for lag in range(8, max_lag + 1, max(1, max_lag // 32)):
        n_chunks = n // lag
        if n_chunks < 2:
            continue
        rs_chunk = []
        for i in range(n_chunks):
            chunk = arr[i * lag: (i + 1) * lag]
            m = chunk.mean()
            dev = np.cumsum(chunk - m)
            R = dev.max() - dev.min()
            S = chunk.std(ddof=1)
            if S > 1e-12:
                rs_chunk.append(R / S)
        if rs_chunk:
            rs_vals.append(np.mean(rs_chunk))
            lags.append(lag)

    if len(lags) < 4:
        return 0.5
    slope, *_ = np.polyfit(np.log(lags), np.log(rs_vals), 1)
    return float(np.clip(slope, 0.0, 1.0))


def _rolling_hurst(series: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(series), np.nan)
    for i in range(window - 1, len(series)):
        out[i] = _hurst_rs(series[i - window + 1: i + 1])
    return out


def _adf_pvalue(series: np.ndarray) -> float:
    from statsmodels.tsa.stattools import adfuller
    if len(series) < 10 or np.all(series == series[0]):
        return 1.0
    try:
        _, pval, *_ = adfuller(series, autolag="AIC", maxlag=10)
        return float(pval)
    except Exception:
        return 1.0


def _rolling_bb_percentile(close: np.ndarray,
                            bb_period: int,
                            baseline: int) -> np.ndarray:
    """Bollinger Band width mapped to its percentile over a rolling baseline."""
    bb_mid = pd.Series(close).rolling(bb_period).mean().values
    bb_std = pd.Series(close).rolling(bb_period).std(ddof=1).values
    bb_w   = _safe_div(2.0 * bb_std, bb_mid, fill=np.nan)

    out = np.full(len(close), np.nan)
    for i in range(baseline - 1, len(close)):
        window_data = bb_w[i - baseline + 1: i + 1]
        valid = window_data[~np.isnan(window_data)]
        if len(valid) < 2:
            continue
        out[i] = float(scipy_stats.percentileofscore(valid, bb_w[i], kind="rank"))
    return out


def compute_m3(
    df: pd.DataFrame,
    eth_btc_hourly: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Regime-dial features at three lookback horizons (24 h, 72 h, 168 h).

    *eth_btc_hourly* is an optional hourly ETH/BTC ratio Series; if
    provided it is resampled / forward-filled to the 5-minute index.
    """
    c = df["close"].values
    out = pd.DataFrame(index=df.index)

    # --- Rolling Hurst exponent ---
    for window, label in [(BARS_24H, "24h"), (BARS_72H, "72h"), (BARS_168H, "168h")]:
        out[f"hurst_{label}_5m"] = _rolling_hurst(c, window)

    # --- Bollinger Band width percentile (180-day baseline) ---
    bb_period_5m = 20 * BARS_PER_HOUR  # same 20-hour period as M2
    out["bb_width_pct_5m"] = _rolling_bb_percentile(c, bb_period_5m, BARS_180D)

    # --- ETH/BTC ADF p-value (forward-fill hourly → 5-min) ---
    if eth_btc_hourly is not None:
        # Reindex to 5-minute frequency with forward fill
        ratio_5m = (
            eth_btc_hourly
            .reindex(df.index, method="ffill")
            .values
        )
        adf_vals = np.full(len(df), np.nan)
        for window, label in [(BARS_24H, "24h"), (BARS_72H, "72h"), (BARS_168H, "168h")]:
            arr = np.full(len(df), np.nan)
            for i in range(window - 1, len(df)):
                arr[i] = _adf_pvalue(ratio_5m[i - window + 1: i + 1])
            out[f"eth_btc_adf_pval_{label}_5m"] = arr
    else:
        for label in ["24h", "72h", "168h"]:
            out[f"eth_btc_adf_pval_{label}_5m"] = np.nan

    return out


# ===========================================================================
# Labels
# ===========================================================================


def make_tbm_labels_5m(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    atr_pct: np.ndarray,
    atr_mult: float = 2.0,
    horizon_bars: int = 144,   # 12 hours in 5-min bars
) -> np.ndarray:
    """Triple Barrier Method on 5-minute bars.

    Class 1 = Long  (upper barrier close*(1+atr_mult*atr_pct) hit first).
    Class 0 = Short (lower barrier close*(1-atr_mult*atr_pct) hit first).
    Class 2 = Neutral (time barrier hit first).
    Last *horizon_bars* rows are set to -1 (insufficient lookahead, drop them).
    """
    N = len(close)
    labels = np.full(N, 2, dtype=np.int8)
    for i in range(N - horizon_bars):
        c     = close[i]
        atr   = atr_pct[i]
        upper = c * (1.0 + atr_mult * atr)
        lower = c * (1.0 - atr_mult * atr)
        for j in range(i + 1, i + horizon_bars + 1):
            if high[j] >= upper:
                labels[i] = 1
                break
            elif low[j] <= lower:
                labels[i] = 0
                break
    labels[N - horizon_bars:] = -1
    return labels


def make_fixed_horizon_labels(
    close: np.ndarray,
    horizon_bars: int = 12,      # 1 hour forward
    up_thresh: float = 0.003,    # +0.3%
    down_thresh: float = -0.003, # -0.3%
) -> np.ndarray:
    """Fixed-horizon label: forward return relative to thresholds.

    Class 1 = Long  (forward return  >  up_thresh).
    Class 0 = Short (forward return  < down_thresh).
    Class 2 = Neutral (otherwise).
    Last *horizon_bars* rows set to -1.
    """
    N = len(close)
    labels = np.full(N, 2, dtype=np.int8)
    for i in range(N - horizon_bars):
        fwd_ret = (close[i + horizon_bars] - close[i]) / close[i]
        if fwd_ret > up_thresh:
            labels[i] = 1
        elif fwd_ret < down_thresh:
            labels[i] = 0
    labels[N - horizon_bars:] = -1
    return labels


# ===========================================================================
# Master feature builder
# ===========================================================================


def make_features_5m(
    df: pd.DataFrame,
    eth_btc_hourly: Optional[pd.Series] = None,
    include_m3_hurst: bool = True,
    tbm_atr_mult: float = 2.0,
    tbm_horizon_bars: int = 144,
    fh_horizon_bars: int = 12,
    fh_up_thresh: float = 0.003,
    fh_down_thresh: float = -0.003,
) -> pd.DataFrame:
    """Compute the full 5-minute microstructure feature matrix.

    Parameters
    ----------
    df : DataFrame
        Nine-column 5-minute kline data from ``loaders.load_5m_extended()``.
    eth_btc_hourly : Series, optional
        Hourly ETH/BTC ratio from ``loaders.load_eth_btc_ratio_hourly()``.
        Required for the ADF regime feature; silently omitted if None.
    include_m3_hurst : bool
        Set False to skip the computationally expensive rolling Hurst
        computation (useful for fast iteration).
    tbm_atr_mult, tbm_horizon_bars : TBM label parameters.
    fh_horizon_bars, fh_up_thresh, fh_down_thresh : Fixed-horizon label
        parameters.

    Returns
    -------
    DataFrame
        Feature matrix plus ``tbm_label``, ``fh_label``, and raw OHLCV
        columns (``close``, ``high``, ``low``, ``atr_pct_5m``) needed
        by the backtester.  Rows with ``tbm_label == -1`` must be dropped
        by the caller before training.
    """
    m1 = compute_m1(df)
    m2 = compute_m2(df)

    if include_m3_hurst:
        m3 = compute_m3(df, eth_btc_hourly)
    else:
        # Lighter M3: skip Hurst, only BB percentile + ADF
        c_vals = df["close"].values
        bb_period_5m = 20 * BARS_PER_HOUR
        m3 = pd.DataFrame(index=df.index)
        m3["bb_width_pct_5m"] = _rolling_bb_percentile(c_vals, bb_period_5m, BARS_180D)
        # Hurst columns filled with NaN so downstream code sees consistent schema
        for label in ["24h", "72h", "168h"]:
            m3[f"hurst_{label}_5m"] = np.nan
        if eth_btc_hourly is not None:
            ratio_5m = (
                eth_btc_hourly.reindex(df.index, method="ffill").values
            )
            for window, label in [(BARS_24H, "24h"), (BARS_72H, "72h"), (BARS_168H, "168h")]:
                arr = np.full(len(df), np.nan)
                for i in range(window - 1, len(df)):
                    arr[i] = _adf_pvalue(ratio_5m[i - window + 1: i + 1])
                m3[f"eth_btc_adf_pval_{label}_5m"] = arr
        else:
            for label in ["24h", "72h", "168h"]:
                m3[f"eth_btc_adf_pval_{label}_5m"] = np.nan

    feat = pd.concat([m1, m2, m3], axis=1)

    # Labels
    c_arr    = df["close"].values
    h_arr    = df["high"].values
    lo_arr   = df["low"].values
    atr_arr  = m2["atr_pct_5m"].values

    feat["tbm_label"] = make_tbm_labels_5m(
        c_arr, h_arr, lo_arr, atr_arr,
        atr_mult=tbm_atr_mult,
        horizon_bars=tbm_horizon_bars,
    )
    feat["fh_label"] = make_fixed_horizon_labels(
        c_arr,
        horizon_bars=fh_horizon_bars,
        up_thresh=fh_up_thresh,
        down_thresh=fh_down_thresh,
    )

    # Pass-through columns required by the backtester
    feat["close"]    = df["close"]
    feat["high"]     = df["high"]
    feat["low"]      = df["low"]
    feat["open"]     = df["open"]

    return feat
