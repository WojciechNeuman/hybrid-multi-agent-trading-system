"""01 — Advanced Structural Features (1h BTCUSDT)

Adapts the 5m structural feature pipeline (01_structural_features.ipynb)
to 1-hour bars. Window parameters are remapped so each group captures the
same economic time horizons as the 5m version.

Parameter mapping (5m → 1h):
  SWING_ORDER_S : 12 bars = 1h  → 12 bars = 12h  (minor structure)
  SWING_ORDER_L : 48 bars = 4h  → 48 bars = 48h  (major structure)
  VOC_WIN_S     : 72 bars = 6h  → 24 bars = 24h  (short liquidity)
  VOC_WIN_L     : 288 bars= 24h → 168 bars= 168h (weekly liquidity)
  ATR_WIN_S     : 20 bars       → 20 bars  (20h)
  ATR_WIN_L     : 72 bars       → 72 bars  (72h = 3 days)
  BB_WIN        : 20 bars       → 20 bars  (20h)
  BURN_IN       : 2500 5m bars  → 200 1h bars

Group D MTF context changes: since we are already on 1h, the higher
timeframes are 4h and daily (instead of 1h and 4h in the 5m version).

Input : data/raw/BTCUSDT_1h.parquet
Output: data/features/BTCUSDT_1h_structural.parquet
"""
from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.signal as ss

warnings.filterwarnings("ignore")
print(f"pandas {pd.__version__} | numpy {np.__version__}")


# ── Repo root ─────────────────────────────────────────────────────────────────
def _repo_root() -> Path:
    p = Path.cwd()
    while p != p.parent:
        if (p / "pyproject.toml").exists():
            return p
        p = p.parent
    raise RuntimeError("pyproject.toml not found")

REPO_ROOT    = _repo_root()
RAW_DIR      = REPO_ROOT / "data" / "raw"
FEATURES_DIR = REPO_ROOT / "data" / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

OUT_PARQUET  = FEATURES_DIR / "BTCUSDT_1h_structural.parquet"

# ── Window parameters (all in 1h bars) ───────────────────────────────────────
SWING_ORDER_S = 12    # ±12h each side — minor structure
SWING_ORDER_L = 48    # ±48h each side — major / swing structure
NEAR_THRESH   = 0.003 # 0.3% proximity flag

VOC_WIN_S     = 24    # 24h  — short liquidity window
VOC_WIN_L     = 168   # 168h — weekly liquidity window
POC_BINS      = 30

BB_WIN        = 20    # 20h
ATR_WIN_S     = 20    # 20h
ATR_WIN_L     = 72    # 72h = 3 days

H4_EMA_FAST   = 20
H4_EMA_SLOW   = 50
RSI_PERIOD    = 14

BARS_PER_YEAR = 365.25 * 24   # 8 766 bars/year at 1h

BURN_IN       = 200   # ≈ 8 days of 1h data warm-up

print(f"REPO_ROOT : {REPO_ROOT}")
print(f"OUTPUT    : {OUT_PARQUET}")


# ── Load raw 1h OHLCV ─────────────────────────────────────────────────────────
def _load(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run 00_data_ingestion first.")
    df = pd.read_parquet(path)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index.name = "open_time"
    df = df[["open", "high", "low", "close", "volume"]].astype("float64")
    print(f"  1h BTC: {len(df):>9,} rows | "
          f"{df.index[0].date()} → {df.index[-1].date()}")
    return df

print("\nLoading raw data...")
df1h = _load(RAW_DIR / "BTCUSDT_1h.parquet")


# ══════════════════════════════════════════════════════════════════════════════
# Group A — Micro-Structural Price Action (struct_)
# ══════════════════════════════════════════════════════════════════════════════
def build_group_A(df: pd.DataFrame, order_s: int, order_l: int,
                  near_thresh: float) -> pd.DataFrame:
    close_arr = df["close"].values
    high_arr  = df["high"].values
    low_arr   = df["low"].values
    n         = len(df)

    feats = pd.DataFrame(index=df.index, dtype="float32")

    def _confirmed_extrema(arr, comparator, order):
        idx = ss.argrelextrema(arr, comparator, order=order)[0]
        price_series = np.full(n, np.nan)
        for raw_i in idx:
            confirmed_at = raw_i + order
            if confirmed_at < n:
                price_series[confirmed_at] = arr[raw_i]
        s = pd.Series(price_series, index=df.index)
        ffilled = s.ffill()
        is_new  = s.notna()
        counter = is_new.cumsum()
        age     = is_new.groupby(counter).cumcount()
        age     = age.where(counter > 0).astype("float32")
        return ffilled.astype("float32"), age

    sh_s, sh_s_age = _confirmed_extrema(high_arr, np.greater_equal, order_s)
    sl_s, sl_s_age = _confirmed_extrema(low_arr,  np.less_equal,    order_s)
    sh_l, _        = _confirmed_extrema(high_arr, np.greater_equal, order_l)
    sl_l, _        = _confirmed_extrema(low_arr,  np.less_equal,    order_l)

    close = df["close"]
    feats["struct_dist_swing_high_s"] = (sh_s - close) / close
    feats["struct_dist_swing_low_s"]  = (close - sl_s) / close
    feats["struct_dist_swing_high_l"] = (sh_l - close) / close
    feats["struct_dist_swing_low_l"]  = (close - sl_l) / close

    feats["struct_near_swing_high_s"] = (
        feats["struct_dist_swing_high_s"].abs() < near_thresh
    ).astype("float32")
    feats["struct_near_swing_low_s"] = (
        feats["struct_dist_swing_low_s"].abs() < near_thresh
    ).astype("float32")

    feats["struct_swing_high_age_s"] = sh_s_age
    feats["struct_swing_low_age_s"]  = sl_s_age

    total_range = (df["high"] - df["low"]).replace(0, np.nan)
    body_top    = df[["open", "close"]].max(axis=1)
    body_bot    = df[["open", "close"]].min(axis=1)
    feats["struct_lower_wick_ratio"] = ((body_bot - df["low"])  / total_range).astype("float32")
    feats["struct_upper_wick_ratio"] = ((df["high"] - body_top) / total_range).astype("float32")
    feats["struct_body_ratio"]       = ((df["close"] - df["open"]).abs() / total_range).astype("float32")

    return feats

print("\nBuilding Group A (Structure)...")
grp_A = build_group_A(df1h, SWING_ORDER_S, SWING_ORDER_L, NEAR_THRESH)
print(f"  Features : {list(grp_A.columns)}")
print(f"  NaN rows : {grp_A.isna().any(axis=1).sum():,}")


# ══════════════════════════════════════════════════════════════════════════════
# Group B — Liquidity & Order Flow Proxies (liq_)
# ══════════════════════════════════════════════════════════════════════════════
def _rolling_vwap(df: pd.DataFrame, window: int) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    pv = tp * df["volume"]
    return (pv.rolling(window, min_periods=1).sum()
            / df["volume"].rolling(window, min_periods=1).sum())

def _anchored_vwap(df: pd.DataFrame, freq: str) -> pd.Series:
    tp  = (df["high"] + df["low"] + df["close"]) / 3
    pv  = tp * df["volume"]
    if freq == "D":
        key = df.index.floor("D")
    else:
        key = df.index.to_period("W").start_time.tz_localize("UTC")
    cum_pv  = pv.groupby(key).cumsum()
    cum_vol = df["volume"].groupby(key).cumsum()
    return (cum_pv / cum_vol.replace(0, np.nan)).astype("float64")

def _rolling_poc(close_arr: np.ndarray, vol_arr: np.ndarray,
                 window: int, n_bins: int) -> np.ndarray:
    n = len(close_arr)
    p_min, p_max = close_arr.min(), close_arr.max()
    edges   = np.linspace(p_min, p_max, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    bar_bin = np.clip(np.digitize(close_arr, edges) - 1, 0, n_bins - 1)
    vol_mat = np.zeros((n, n_bins), dtype=np.float32)
    vol_mat[np.arange(n), bar_bin] = vol_arr.astype(np.float32)
    cum_vol = np.cumsum(vol_mat, axis=0)
    pad     = np.zeros((window, n_bins), dtype=np.float32)
    lagged  = np.vstack([pad, cum_vol[:-window]])
    win_vol = cum_vol - lagged
    poc_bin = np.argmax(win_vol, axis=1)
    poc_arr = centers[poc_bin].astype(np.float32)
    poc_arr[:window] = np.nan
    return poc_arr

def build_group_B(df: pd.DataFrame, win_s: int, win_l: int,
                  poc_bins: int) -> pd.DataFrame:
    close  = df["close"]
    volume = df["volume"]
    feats  = pd.DataFrame(index=df.index, dtype="float32")

    for tag, series in [
        ("daily",        _anchored_vwap(df, "D")),
        ("weekly",       _anchored_vwap(df, "W")),
        (f"{win_s}h",    _rolling_vwap(df, win_s)),
        (f"{win_l}h",    _rolling_vwap(df, win_l)),
    ]:
        feats[f"liq_vwap_dev_{tag}"] = ((close - series) / series).astype("float32")

    close_np = close.to_numpy(dtype=np.float64)
    vol_np   = volume.to_numpy(dtype=np.float64)
    for win, tag in [(win_s, f"{win_s}h"), (win_l, f"{win_l}h")]:
        poc = _rolling_poc(close_np, vol_np, win, poc_bins)
        feats[f"liq_poc_dist_{tag}"] = ((close_np - poc) / poc).astype("float32")

    for win, tag in [(win_s, f"{win_s}h"), (win_l, f"{win_l}h")]:
        v_mean = volume.rolling(win, min_periods=win // 2).mean().shift(1)
        v_std  = volume.rolling(win, min_periods=win // 2).std().shift(1)
        z      = (volume - v_mean) / v_std.replace(0, np.nan)
        feats[f"liq_vol_z_{tag}"] = z.clip(-5, 5).astype("float32")

    z_l  = feats[f"liq_vol_z_{win_l}h"]
    body = (df["close"] - df["open"]).astype("float32")
    feats["liq_exhaustion_bull"] = ((z_l > 3) & (body > 0)).astype("float32")
    feats["liq_exhaustion_bear"] = ((z_l > 3) & (body < 0)).astype("float32")

    return feats

print("Building Group B (Liquidity)...")
grp_B = build_group_B(df1h, VOC_WIN_S, VOC_WIN_L, POC_BINS)
print(f"  Features : {list(grp_B.columns)}")
print(f"  NaN rows : {grp_B.isna().any(axis=1).sum():,}")


# ══════════════════════════════════════════════════════════════════════════════
# Group C — Volatility Squeeze & Regime Vectors (volat_)
# ══════════════════════════════════════════════════════════════════════════════
def build_group_C(df: pd.DataFrame, bb_win: int,
                  atr_win_s: int, atr_win_l: int,
                  bars_per_year: float) -> pd.DataFrame:
    feats      = pd.DataFrame(index=df.index, dtype="float32")
    close      = df["close"]
    high, low  = df["high"], df["low"]
    open_      = df["open"]
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    for win, tag in [(atr_win_s, f"{atr_win_s}"), (atr_win_l, f"{atr_win_l}")]:
        atr = tr.rolling(win, min_periods=win // 2).mean()
        feats[f"volat_atr_{tag}_pct"] = (atr / close).astype("float32")

    atr_s = tr.rolling(atr_win_s, min_periods=atr_win_s // 2).mean()

    sma20    = close.rolling(bb_win, min_periods=bb_win // 2).mean()
    std20    = close.rolling(bb_win, min_periods=bb_win // 2).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    bb_width = 4 * std20

    feats["volat_bb_width_20"]    = (bb_width / close).astype("float32")
    feats["volat_bb_position_20"] = (
        (close - bb_lower) / bb_width.replace(0, np.nan)
    ).clip(0, 1).astype("float32")

    ema20    = close.ewm(span=bb_win, adjust=False).mean()
    kc_width = 2 * 1.5 * atr_s
    squeeze_ratio = bb_width / kc_width.replace(0, np.nan)
    feats["volat_bk_squeeze"] = squeeze_ratio.clip(0, 3).astype("float32")
    feats["volat_squeeze_on"] = (squeeze_ratio < 1.0).astype("float32")

    log_hl = np.log(high / low.replace(0, np.nan))
    log_co = np.log(close / open_.replace(0, np.nan))
    gk_bar = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2

    for win, tag in [(atr_win_s, f"{atr_win_s}"), (atr_win_l, f"{atr_win_l}")]:
        gk_roll = gk_bar.rolling(win, min_periods=win // 2).mean()
        feats[f"volat_gk_{tag}"] = np.sqrt(gk_roll * bars_per_year).astype("float32")

    return feats

print("Building Group C (Volatility)...")
grp_C = build_group_C(df1h, BB_WIN, ATR_WIN_S, ATR_WIN_L, BARS_PER_YEAR)
print(f"  Features : {list(grp_C.columns)}")
print(f"  NaN rows : {grp_C.isna().any(axis=1).sum():,}")


# ══════════════════════════════════════════════════════════════════════════════
# Group D — Macro Context & MTF Alignment (mtf_)
# On 1h bars the higher timeframes are 4h and daily (vs 1h+4h on 5m).
# ══════════════════════════════════════════════════════════════════════════════
def _wilder_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    up    = delta.clip(lower=0)
    dn    = (-delta).clip(lower=0)
    ema_u = up.ewm(com=period - 1, adjust=False, min_periods=period).mean()
    ema_d = dn.ewm(com=period - 1, adjust=False, min_periods=period).mean()
    rs    = ema_u / ema_d.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).astype("float32")

def _ema_spread(close: pd.Series, fast: int, slow: int) -> pd.Series:
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    return ((ema_f - ema_s) / close).astype("float32")

def build_group_D(df1h: pd.DataFrame, ema_fast: int, ema_slow: int,
                  rsi_period: int) -> pd.DataFrame:
    """MTF signals: 4h and daily context mapped to 1h index + session timing."""
    feats = pd.DataFrame(index=df1h.index, dtype="float32")
    idx1h = df1h.index

    # ── 4-Hour features ───────────────────────────────────────────────────────
    h4_close  = df1h["close"].resample("4h", closed="left", label="left").last().dropna()
    h4_spread = _ema_spread(h4_close, ema_fast, ema_slow)
    h4_rsi    = _wilder_rsi(h4_close, rsi_period)
    h4_above  = (h4_close > h4_close.ewm(span=ema_slow, adjust=False).mean()
                 ).astype("float32")

    # shift(1): the 4h bar labeled "08:00" closes at ~11:59.
    # Without shift, 1h bars 08:00–11:00 see future data from that 4h bar.
    feats["mtf_h4_ema_signal"]  = h4_spread.shift(1).reindex(idx1h, method="ffill")
    feats["mtf_h4_rsi"]         = (h4_rsi / 100).shift(1).reindex(idx1h, method="ffill")
    feats["mtf_h4_above_ema50"] = h4_above.shift(1).reindex(idx1h, method="ffill")

    # ── Daily features ────────────────────────────────────────────────────────
    h1d_close  = df1h["close"].resample("1D", closed="left", label="left").last().dropna()
    h1d_spread = _ema_spread(h1d_close, ema_fast, ema_slow)
    h1d_rsi    = _wilder_rsi(h1d_close, rsi_period)

    feats["mtf_h1d_ema_signal"] = h1d_spread.shift(1).reindex(idx1h, method="ffill")
    feats["mtf_h1d_rsi"]        = (h1d_rsi / 100).shift(1).reindex(idx1h, method="ffill")

    # ── Composite: weighted 4h (40%) + daily (60%) ────────────────────────────
    h4_clip  = feats["mtf_h4_ema_signal"].clip(-0.05, 0.05) / 0.05
    h1d_clip = feats["mtf_h1d_ema_signal"].clip(-0.05, 0.05) / 0.05
    feats["mtf_alignment"] = (0.4 * h4_clip + 0.6 * h1d_clip).astype("float32")

    # ── Session timing (UTC) ──────────────────────────────────────────────────
    hour = idx1h.hour
    dow  = idx1h.dayofweek
    feats["mtf_session_hour_sin"] = np.sin(2 * np.pi * hour / 24).astype("float32")
    feats["mtf_session_hour_cos"] = np.cos(2 * np.pi * hour / 24).astype("float32")
    feats["mtf_session_dow_sin"]  = np.sin(2 * np.pi * dow  / 7).astype("float32")
    feats["mtf_session_dow_cos"]  = np.cos(2 * np.pi * dow  / 7).astype("float32")

    return feats

print("Building Group D (MTF Context: 4h + daily)...")
grp_D = build_group_D(df1h, H4_EMA_FAST, H4_EMA_SLOW, RSI_PERIOD)
print(f"  Features : {list(grp_D.columns)}")
print(f"  NaN rows : {grp_D.isna().any(axis=1).sum():,}")


# ══════════════════════════════════════════════════════════════════════════════
# Assemble & save
# ══════════════════════════════════════════════════════════════════════════════
print("\nAssembling feature matrix...")
feat_df = pd.concat([grp_A, grp_B, grp_C, grp_D], axis=1)

feat_df = feat_df.iloc[BURN_IN:].copy()
n_before = len(feat_df)
feat_df  = feat_df.ffill()
feat_df  = feat_df.dropna()
n_after  = len(feat_df)

print(f"  After burn-in ({BURN_IN} bars) : {n_before:>10,} rows")
print(f"  After dropna               : {n_after:>10,} rows")
print(f"  Rows removed               : {n_before - n_after:>10,}")
print(f"  Total features             : {feat_df.shape[1]}")
print(f"  Date range : {feat_df.index[0].date()} → {feat_df.index[-1].date()}")

assert not feat_df.index.duplicated().any(), "Duplicate timestamps!"
assert feat_df.isna().sum().sum() == 0, "NaN values remain!"

feat_df.to_parquet(OUT_PARQUET)
print(f"\nSaved → {OUT_PARQUET}")

# ── Feature registry entry ────────────────────────────────────────────────────
all_cols = list(feat_df.columns)
groups = {
    "A_structure": [c for c in all_cols if c.startswith("struct_")],
    "B_liquidity": [c for c in all_cols if c.startswith("liq_")],
    "C_volatility": [c for c in all_cols if c.startswith("volat_")],
    "D_mtf": [c for c in all_cols if c.startswith("mtf_")],
}

registry = {
    "schema_version": 2,
    "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "description": "Structural features for BTCUSDT 1h — adapted from 5m pipeline.",
    "source_timeframe": "1h",
    "source_assets": ["BTCUSDT"],
    "output_parquet": str(OUT_PARQUET),
    "burn_in_bars": BURN_IN,
    "total_features": len(all_cols),
    "row_count": len(feat_df),
    "date_range": {
        "start": str(feat_df.index[0]),
        "end":   str(feat_df.index[-1]),
    },
    "window_params": {
        "swing_order_minor": SWING_ORDER_S,
        "swing_order_major": SWING_ORDER_L,
        "near_threshold":    NEAR_THRESH,
        "voc_window_short":  VOC_WIN_S,
        "voc_window_long":   VOC_WIN_L,
        "poc_bins":          POC_BINS,
        "bb_window":         BB_WIN,
        "atr_win_s":         ATR_WIN_S,
        "atr_win_l":         ATR_WIN_L,
    },
    "feature_groups": {
        grp: {
            "features": cols,
            "count": len(cols),
        }
        for grp, cols in groups.items()
    },
}

reg_path = FEATURES_DIR / "feature_registry_v2_1h.json"
with open(reg_path, "w") as f:
    json.dump(registry, f, indent=2)
print(f"Registry → {reg_path}")

print("\nDone.")
print("Feature counts by group:")
for grp, cols in groups.items():
    print(f"  {grp:15s}: {len(cols)} features")
print(f"  {'TOTAL':15s}: {len(all_cols)} features")
