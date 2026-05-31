#!/usr/bin/env python
# coding: utf-8

# # 02 -- LGBM Dynamic Walk-Forward Optimization (5-Minute Microstructure Benchmark)
#
# ## Architecture
# ```
#                   ROLLING WALK-FORWARD LOOP STEPPING BY 7 DAYS
# |<─────────────────── 90-Day In-Sample Window ──────────────────>| [ 7-Day OOS ]
#
#   Stage 1: Low-variance drop  +  Spearman collinearity filter (ρ > 0.85)
#   Stage 2: Mutual Information Rank  →  Top 60
#   Stage 3: Multi-window Stability Filter (3 sub-windows, majority vote)
#   Stage 4: LightGBM Permutation Pruning  →  keep ~15 features
#                              │
#                              ▼
#               Train local LightGBM Classifier (multiclass, 3 classes)
#                              │
#                              ▼
#          Inference on the 7-Day Out-of-Sample step
#                              │
#                              ▼
#       Backtest (0-fee benchmark + Futures-fee realistic case)
# ```
#
# ## Labels
#   T1 – Triple Barrier Method  (±2xATR, 12-hour / 144-bar horizon)
#   T2 – Fixed Horizon          (+0.3% / -0.3%, 1-hour / 12-bar horizon)
#
# ## Fee scenarios
#   A – Zero-fee (baseline)
#   B – Binance Futures (0.02% maker, 0.05% taker)
#
# ## Output
#   artifacts/02_lgbm_dynamic_wfo_5m/results.json
#

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SYMBOL        = "BTCUSDT"
INTERVAL      = "5m"
TRAIN_START   = "2020-09-01"   # first date with sufficient liquidity data
STORE_DIR     = "data/raw"
FEATURES_DIR  = "data/features"
ARTIFACT_DIR  = "artifacts/02_lgbm_dynamic_wfo_5m"

# Walk-Forward
BARS_PER_HOUR     = 12
IS_DAYS           = 90
OOS_DAYS          = 7
IS_BARS           = IS_DAYS  * 24 * BARS_PER_HOUR   # 25,920
OOS_BARS          = OOS_DAYS * 24 * BARS_PER_HOUR   # 2,016
EMBARGO_BARS      = 6 * BARS_PER_HOUR               # 6-hour embargo

# Feature selection
STAGE2_TOP_K         = 60   # MI top-K after variance + Spearman filter
STAGE3_N_SUBWINDOWS  = 3    # stability sub-windows
STAGE3_MIN_WINS      = 2    # feature must rank in top-K in ≥ min_wins sub-windows
STAGE4_TOP_K         = 15   # permutation-pruned final set

# LightGBM base
BASE_LGB_PARAMS = {
    "objective":        "multiclass",
    "num_class":        3,
    "metric":           "multi_logloss",
    "n_estimators":     500,
    "num_leaves":       63,
    "min_child_samples": 40,
    "learning_rate":    0.02,
    "max_depth":        -1,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "random_state":     42,
    "n_jobs":           -1,
    "verbose":          -1,
}
EARLY_STOP_ROUNDS = 30
INTERNAL_VAL_FRAC = 0.10     # last 10% of IS used for early stopping

# Execution fees
MAKER_FEE_FUTURES = 0.0002   # 0.02%
TAKER_FEE_FUTURES = 0.0005   # 0.05%
SPOT_TAKER_FEE    = 0.0005   # 0.05%
MAKER_FEE_SPOT    = 0.0000   # 0.00%
SHORT_FUNDING_H   = 0.0000077  # +0.00077%/h received on short futures
BUFFER            = 0.0005   # 5bp penetration buffer

# Trading signal parameters (fixed for this benchmark)
LONG_THRESHOLD  = 0.45
SHORT_THRESHOLD = 0.45
ENTRY_ATR_MULT  = 0.5
SL_ATR_MULT     = 2.0
TP_ATR_MULT     = 2.5
MIN_SL          = 0.010
MIN_HOLD_BARS   = 12     # 1 h
MAX_HOLD_BARS   = 144    # 12 h
COOLDOWN_BARS   = 6      # 30 min

# Label choice for training: "tbm" | "fh"
LABEL_CHOICE = "tbm"

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import itertools
import json
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Repo paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
if not (REPO_ROOT / "pyproject.toml").exists():
    REPO_ROOT = Path.cwd()

_STORE_DIR    = REPO_ROOT / STORE_DIR
_FEATURES_DIR = REPO_ROOT / FEATURES_DIR
_ART_DIR      = REPO_ROOT / ARTIFACT_DIR
_ART_DIR.mkdir(parents=True, exist_ok=True)

FIGURES_DIR = _ART_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelsize": 10, "axes.titlesize": 11,
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
})
ACCENT = "#F7931A"; BLUE = "#2962FF"; GREY = "#9E9E9E"
RED = "#EF5350"; GREEN = "#26A69A"

print(f"REPO_ROOT  : {REPO_ROOT}")
print(f"ARTIFACT   : {_ART_DIR}")

# ---------------------------------------------------------------------------
# Data loading + feature engineering
# ---------------------------------------------------------------------------

from hmats.data.loaders import load_5m_extended, load_eth_btc_ratio_hourly
from hmats.features.engineering_5m import make_features_5m

print("\nLoading 5-minute data …")
raw_5m = load_5m_extended(
    symbol=SYMBOL, store_dir=str(_STORE_DIR), fetch_if_missing=True,
    fetch_start=TRAIN_START,
)

eth_btc = load_eth_btc_ratio_hourly(store_dir=str(_STORE_DIR))
if eth_btc is None:
    print("  ETHUSDT hourly not available — ADF feature will be NaN")

print(f"  Raw 5m shape : {raw_5m.shape}  "
      f"({raw_5m.index[0].date()} → {raw_5m.index[-1].date()})")

print("\nEngineering features (this takes a few minutes for M3 Hurst) …")
t0 = time.perf_counter()
feat_df = make_features_5m(
    raw_5m,
    eth_btc_hourly=eth_btc,
    include_m3_hurst=True,
    tbm_atr_mult=2.0,
    tbm_horizon_bars=144,
    fh_horizon_bars=12,
    fh_up_thresh=0.003,
    fh_down_thresh=-0.003,
)
feat_df.index = feat_df.index.tz_localize(None) if feat_df.index.tz else feat_df.index
print(f"  Feature shape : {feat_df.shape}  ({time.perf_counter()-t0:.0f}s)")

# Save features parquet for re-use
feat_path = _FEATURES_DIR / f"{SYMBOL}_{INTERVAL}_features_5m.parquet"
feat_df.to_parquet(feat_path)
print(f"  Features saved → {feat_path}")

# ---------------------------------------------------------------------------
# Prepare master arrays
# ---------------------------------------------------------------------------

LABEL_COL = "tbm_label" if LABEL_CHOICE == "tbm" else "fh_label"

# Drop bars with invalid labels (-1 = insufficient lookahead)
valid_df = feat_df[feat_df[LABEL_COL] >= 0].copy()

_EXCLUDE_COLS = {
    "open", "high", "low", "close", "volume",
    "quote_volume", "num_trades", "taker_buy_base_volume", "taker_buy_quote_volume",
    "tbm_label", "fh_label", "label", "return", "log_return", "target",
    "frac_diff_d_star",   # constant — not a predictive feature
}
feature_cols = [
    c for c in valid_df.columns
    if c not in _EXCLUDE_COLS and pd.api.types.is_numeric_dtype(valid_df[c])
]

print(f"\nTotal features available : {len(feature_cols)}")
print(f"Total valid bars         : {len(valid_df):,}")
label_vc = valid_df[LABEL_COL].value_counts().sort_index()
for cls, cnt in label_vc.items():
    print(f"  Class {cls} : {cnt:,} ({cnt/len(valid_df):.1%})")

# Numpy arrays for fast slicing
X_all    = valid_df[feature_cols].values.astype(np.float32)
y_all    = valid_df[LABEL_COL].values.astype(np.int32)
ts_all   = valid_df.index
close_all = valid_df["close"].values.astype(np.float64)
high_all  = valid_df["high"].values.astype(np.float64)
low_all   = valid_df["low"].values.astype(np.float64)
atr_all   = valid_df["atr_pct_5m"].values.astype(np.float64)


# ===========================================================================
# 4-Stage Feature Selection  (called INSIDE the WFO loop)
# ===========================================================================


def _variance_filter(X: np.ndarray, cols: list[str],
                     var_thresh: float = 1e-6) -> list[str]:
    keep = [c for c, v in zip(cols, X.var(axis=0)) if v > var_thresh]
    return keep


def _spearman_collinearity_filter(X: np.ndarray, cols: list[str],
                                  rho_thresh: float = 0.85) -> list[str]:
    """Greedy collinearity filter: keep highest-MI-scoring feature among
    each correlated cluster.  Uses absolute Spearman rank correlation."""
    from scipy.stats import spearmanr
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr_mat, _ = spearmanr(X)
    if X.shape[1] == 1:
        return cols
    corr_mat = np.abs(corr_mat)
    np.fill_diagonal(corr_mat, 0.0)
    kept = list(range(len(cols)))
    removed = set()
    for i in range(len(cols)):
        if i in removed:
            continue
        for j in range(i + 1, len(cols)):
            if j in removed:
                continue
            if corr_mat[i, j] > rho_thresh:
                removed.add(j)
    return [cols[i] for i in kept if i not in removed]


def stage1_filter(X_is: np.ndarray, y_is: np.ndarray,
                  cols: list[str]) -> list[str]:
    """Stage 1: variance drop + Spearman collinearity filter (ρ > 0.85)."""
    cols1 = _variance_filter(X_is, cols)
    idx1  = [cols.index(c) for c in cols1]
    cols2 = _spearman_collinearity_filter(X_is[:, idx1], cols1)
    return cols2


def stage2_mi_rank(X_is: np.ndarray, y_is: np.ndarray,
                   cols: list[str], top_k: int = STAGE2_TOP_K) -> list[str]:
    """Stage 2: mutual information rank → Top K."""
    mi = mutual_info_classif(X_is, y_is, discrete_features=False, random_state=42)
    ranked = sorted(zip(cols, mi), key=lambda t: t[1], reverse=True)
    return [c for c, _ in ranked[:top_k]]


def stage3_stability_filter(X_is: np.ndarray, y_is: np.ndarray,
                             cols: list[str],
                             n_windows: int = STAGE3_N_SUBWINDOWS,
                             min_wins: int = STAGE3_MIN_WINS,
                             top_k: int = STAGE2_TOP_K) -> list[str]:
    """Stage 3: multi-window MI stability.

    Split the IS window into *n_windows* equal sub-windows and count how
    many times each feature appears in the top-K ranking.  Features that
    appear in at least *min_wins* out of *n_windows* sub-windows are kept.
    """
    n       = len(X_is)
    sub_sz  = n // n_windows
    win_counts: dict[str, int] = {c: 0 for c in cols}

    for w in range(n_windows):
        sl = slice(w * sub_sz, (w + 1) * sub_sz if w < n_windows - 1 else n)
        Xw = X_is[sl]; yw = y_is[sl]
        if len(np.unique(yw)) < 2:
            continue
        mi  = mutual_info_classif(Xw, yw, discrete_features=False, random_state=42)
        top = sorted(zip(cols, mi), key=lambda t: t[1], reverse=True)[:top_k]
        for c, _ in top:
            win_counts[c] += 1

    return [c for c in cols if win_counts[c] >= min_wins]


def stage4_lgb_permutation_prune(X_is: np.ndarray, y_is: np.ndarray,
                                  cols: list[str],
                                  lgb_params: dict,
                                  n_repeats: int = 5,
                                  top_k: int = STAGE4_TOP_K) -> list[str]:
    """Stage 4: train a quick LGB, compute permutation importance, keep top K."""
    n_val = max(int(0.10 * len(X_is)), 30)
    X_tr, y_tr = X_is[:-n_val], y_is[:-n_val]
    X_vl, y_vl = X_is[-n_val:], y_is[-n_val:]

    ds_tr = lgb.Dataset(X_tr, label=y_tr, feature_name=cols)
    ds_vl = lgb.Dataset(X_vl, label=y_vl, reference=ds_tr)
    pruning_params = {**lgb_params,
                      "n_estimators": 200, "learning_rate": 0.05}
    model = lgb.train(
        pruning_params, ds_tr,
        valid_sets=[ds_vl], valid_names=["val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=20, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    baseline = _logloss(model.predict(X_vl), y_vl, lgb_params["num_class"])
    imp = np.zeros(len(cols))
    rng = np.random.default_rng(42)
    for _ in range(n_repeats):
        for j in range(len(cols)):
            X_perm = X_vl.copy(); rng.shuffle(X_perm[:, j])
            imp[j] += _logloss(model.predict(X_perm), y_vl,
                                lgb_params["num_class"]) - baseline
    imp /= n_repeats

    ranked = sorted(zip(cols, imp), key=lambda t: t[1], reverse=True)
    del model
    return [c for c, _ in ranked[:top_k]]


def _logloss(probs: np.ndarray, y: np.ndarray, n_class: int) -> float:
    N = len(y)
    if probs.ndim == 1:
        probs = probs.reshape(N, -1)
    probs = np.clip(probs, 1e-7, 1.0)
    ll = -np.mean(np.log(probs[np.arange(N), y]))
    return float(ll)


def select_features_dynamic(X_is: np.ndarray, y_is: np.ndarray,
                             cols: list[str],
                             lgb_params: dict) -> list[str]:
    """Full 4-stage pipeline.  Returns final feature list (≤ STAGE4_TOP_K)."""
    # Stage 1
    cols1 = stage1_filter(X_is, y_is, cols)
    if not cols1:
        return cols[:STAGE4_TOP_K]
    idx1 = [cols.index(c) for c in cols1]
    X1   = X_is[:, idx1]

    # Stage 2
    cols2 = stage2_mi_rank(X1, y_is, cols1, top_k=STAGE2_TOP_K)
    if not cols2:
        return cols1[:STAGE4_TOP_K]
    idx2 = [cols1.index(c) for c in cols2]
    X2   = X1[:, idx2]

    # Stage 3
    cols3 = stage3_stability_filter(X2, y_is, cols2,
                                     n_windows=STAGE3_N_SUBWINDOWS,
                                     min_wins=STAGE3_MIN_WINS,
                                     top_k=STAGE2_TOP_K)
    if len(cols3) < STAGE4_TOP_K:
        cols3 = cols2  # fall back to Stage-2 output if too few survive
    idx3 = [cols2.index(c) for c in cols3]
    X3   = X2[:, idx3]

    # Stage 4
    cols4 = stage4_lgb_permutation_prune(X3, y_is, cols3, lgb_params,
                                          top_k=STAGE4_TOP_K)
    return cols4 if cols4 else cols3[:STAGE4_TOP_K]


# ===========================================================================
# Backtester (5-minute aware)
# ===========================================================================


def run_backtest_5m(
        p_up: np.ndarray, p_down: np.ndarray,
        close_arr: np.ndarray, high_arr: np.ndarray,
        low_arr: np.ndarray, atr_arr: np.ndarray,
        long_thr: float = LONG_THRESHOLD,
        short_thr: float = SHORT_THRESHOLD,
        entry_atr: float = ENTRY_ATR_MULT,
        sl_atr: float = SL_ATR_MULT,
        tp_atr: float = TP_ATR_MULT,
        min_sl: float = MIN_SL,
        min_hold: int = MIN_HOLD_BARS,
        max_hold: int = MAX_HOLD_BARS,
        cooldown: int = COOLDOWN_BARS,
        maker_fee: float = 0.0,
        taker_fee_long: float = 0.0,
        taker_fee_short: float = 0.0,
        short_fund_per_bar: float = 0.0,
        buf: float = BUFFER,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Event-driven backtester for 5-minute bars.

    Longs  → Spot execution model.
    Shorts → Futures execution model.
    Returns (equity_curve, trade_log_df).
    """
    exit_long  = 1.0 - long_thr
    exit_short = 1.0 - short_thr

    cash = 1.0; units = 0.0; entry_cash = 0.0
    in_pos = False; direction = None
    entry_px = 0.0; sl_v = 0.0; tp_v = 0.0
    hold_count = 0; cd = 0; entry_bar = -1
    pending = None

    equity = [1.0]
    trades = []
    n_sig = 0; n_fill = 0; n_exp = 0

    N = len(close_arr)
    for i in range(N):
        px  = close_arr[i]; hi = high_arr[i]; lo = low_arr[i]
        pu  = p_up[i];      pd_ = p_down[i]; atr = atr_arr[i]
        if cd > 0:
            cd -= 1

        # -- 1. Fill / expire pending limit order (TIF = 1 bar) --
        if pending is not None:
            lp = pending["lp"]
            if pending["dir"] == "long":
                if lo < lp * (1.0 - buf):
                    units = cash * (1.0 - maker_fee) / lp
                    cash = 0.0; in_pos = True; direction = "long"
                    entry_px = lp; entry_bar = i; hold_count = 0
                    sl_v = pending["sl"]; tp_v = pending["tp"]
                    n_fill += 1
                else:
                    n_exp += 1
            else:
                if hi > lp * (1.0 + buf):
                    entry_cash = cash * (1.0 - maker_fee)
                    cash = 0.0; units = entry_cash / lp
                    in_pos = True; direction = "short"
                    entry_px = lp; entry_bar = i; hold_count = 0
                    sl_v = pending["sl"]; tp_v = pending["tp"]
                    n_fill += 1
                else:
                    n_exp += 1
            pending = None

        # -- 2. Manage open position --
        if in_pos and i > entry_bar:
            hold_count += 1
            if direction == "short":
                entry_cash *= (1.0 + short_fund_per_bar)

            reason = None; pnl = 0.0

            if direction == "long":
                sl_px = entry_px * (1.0 - sl_v)
                tp_px = entry_px * (1.0 + tp_v)
                sl_hit = lo <= sl_px
                tp_hit = hi > tp_px * (1.0 + buf)
                if sl_hit and tp_hit:
                    reason = "sl"; pnl = (sl_px - entry_px) / entry_px
                    cash = units * sl_px * (1.0 - taker_fee_long); units = 0.0
                elif sl_hit:
                    reason = "sl"; pnl = (sl_px - entry_px) / entry_px
                    cash = units * sl_px * (1.0 - taker_fee_long); units = 0.0
                elif tp_hit:
                    reason = "tp"; pnl = (tp_px - entry_px) / entry_px
                    cash = units * tp_px * (1.0 - maker_fee); units = 0.0
                elif hold_count >= max_hold:
                    reason = "max_hold"; pnl = (px - entry_px) / entry_px
                    cash = units * px * (1.0 - taker_fee_long); units = 0.0
                elif hold_count >= min_hold and pu < exit_long:
                    reason = "conf"; pnl = (px - entry_px) / entry_px
                    cash = units * px * (1.0 - taker_fee_long); units = 0.0
            else:  # short
                sl_px = entry_px * (1.0 + sl_v)
                tp_px = entry_px * (1.0 - tp_v)
                sl_hit = hi >= sl_px
                tp_hit = lo < tp_px * (1.0 - buf)
                gross = 0.0
                if sl_hit and tp_hit:
                    reason = "sl"; gross = (entry_px - sl_px) / entry_px
                    cash = entry_cash * (1.0 + gross) * (1.0 - taker_fee_short)
                elif sl_hit:
                    reason = "sl"; gross = (entry_px - sl_px) / entry_px
                    cash = entry_cash * (1.0 + gross) * (1.0 - taker_fee_short)
                elif tp_hit:
                    reason = "tp"; gross = (entry_px - tp_px) / entry_px
                    cash = entry_cash * (1.0 + gross) * (1.0 - maker_fee)
                elif hold_count >= max_hold:
                    reason = "max_hold"; gross = (entry_px - px) / entry_px
                    cash = entry_cash * (1.0 + gross) * (1.0 - taker_fee_short)
                elif hold_count >= min_hold and pd_ < exit_short:
                    reason = "conf"; gross = (entry_px - px) / entry_px
                    cash = entry_cash * (1.0 + gross) * (1.0 - taker_fee_short)
                pnl = gross

            if reason:
                trades.append({
                    "direction": direction, "pnl_pct": pnl,
                    "hold_bars": hold_count, "reason": reason,
                })
                in_pos = False; direction = None; hold_count = 0; cd = cooldown

        # -- 3. Place new pending limit order if flat --
        if not in_pos and pending is None and cd == 0:
            atr_sl = max(sl_atr * atr, min_sl)
            atr_tp = tp_atr * atr
            if pu >= long_thr:
                lp = px * (1.0 - entry_atr * atr)
                pending = {"dir": "long",  "lp": lp, "sl": atr_sl, "tp": atr_tp}
                n_sig += 1
            elif pd_ >= short_thr:
                lp = px * (1.0 + entry_atr * atr)
                pending = {"dir": "short", "lp": lp, "sl": atr_sl, "tp": atr_tp}
                n_sig += 1

        # -- Equity mark-to-market --
        if in_pos and direction == "long":
            equity.append(units * px)
        elif in_pos and direction == "short":
            equity.append(entry_cash * (1.0 + (entry_px - px) / entry_px))
        else:
            equity.append(cash)

    # Force-close at series end
    if in_pos:
        px = close_arr[-1]
        if direction == "long":
            gross = (px - entry_px) / entry_px
            cash  = units * px * (1.0 - taker_fee_long)
        else:
            gross = (entry_px - px) / entry_px
            cash  = entry_cash * (1.0 + gross) * (1.0 - taker_fee_short)
        trades.append({
            "direction": direction, "pnl_pct": gross,
            "hold_bars": hold_count, "reason": "eod",
        })
        equity[-1] = cash

    tdf = pd.DataFrame(trades)
    tdf.attrs["n_signals"] = n_sig
    tdf.attrs["n_fills"]   = n_fill
    tdf.attrs["n_expires"] = n_exp
    return np.array(equity[1:]), tdf


def compute_metrics(equity: np.ndarray, tdf: pd.DataFrame) -> dict:
    """Compute performance metrics from equity curve and trade log."""
    if tdf.empty or len(equity) < 2:
        return {"sharpe": np.nan, "total_return": np.nan, "max_dd": np.nan,
                "win_rate": np.nan, "n_trades": 0, "fill_rate": np.nan}
    ann = 24 * 365 * BARS_PER_HOUR   # 5-min annualisation factor
    eq  = np.maximum(equity, 1e-12)
    ret = np.log(eq[1:] / eq[:-1])
    pk  = np.maximum.accumulate(eq)
    sharpe = float(ret.mean() / (ret.std(ddof=1) + 1e-12) * np.sqrt(ann))
    mdd    = float(((eq - pk) / pk).min())
    fill_r = (tdf.attrs.get("n_fills", 0) /
              max(tdf.attrs.get("n_signals", 1), 1))
    return {
        "sharpe":       sharpe,
        "total_return": float(eq[-1] - 1.0),
        "max_dd":       mdd,
        "win_rate":     float((tdf["pnl_pct"] > 0).mean()),
        "n_trades":     len(tdf),
        "fill_rate":    fill_r,
        "n_long":       int((tdf["direction"] == "long").sum()),
        "n_short":      int((tdf["direction"] == "short").sum()),
    }


# ===========================================================================
# Main Walk-Forward Loop
# ===========================================================================

n_total = len(X_all)

# Find the earliest valid IS start
if n_total < IS_BARS + OOS_BARS:
    raise ValueError(
        f"Insufficient data: {n_total} bars available, need "
        f"at least {IS_BARS + OOS_BARS} ({IS_BARS} IS + {OOS_BARS} OOS)."
    )

# Collect per-fold results
fold_results: list[dict] = []

# Accumulators for the full OOS equity curve
oos_probs_0fee   = []
oos_probs_futures = []

# Store OOS index for plotting
oos_index_list: list[pd.DatetimeIndex] = []

print(f"\n{'='*70}")
print(f"Rolling WFO  |  IS={IS_DAYS}d ({IS_BARS:,} bars)  "
      f"OOS={OOS_DAYS}d ({OOS_BARS:,} bars)")
print(f"4-Stage selection: S1=var+Spearman  S2=MI→{STAGE2_TOP_K}  "
      f"S3=stability  S4=LGB-perm→{STAGE4_TOP_K}")
print(f"Label: {LABEL_COL}  |  Fee scenarios: 0-fee + Futures")
print(f"{'='*70}\n")

fold = 0
t0_wfo = time.perf_counter()

for oos_start in range(IS_BARS, n_total - OOS_BARS + 1, OOS_BARS):
    is_start  = max(0, oos_start - IS_BARS)
    is_end    = oos_start - EMBARGO_BARS
    oos_end   = min(oos_start + OOS_BARS, n_total)

    if is_end - is_start < IS_BARS // 2:
        continue   # not enough IS data

    # --- IS slice ---
    X_is  = X_all[is_start:is_end]
    y_is  = y_all[is_start:is_end]

    # Only proceed if there are at least 2 classes in IS
    if len(np.unique(y_is)) < 2:
        continue

    fold += 1
    fold_ts_start = ts_all[oos_start]
    fold_ts_end   = ts_all[oos_end - 1]
    tqdm.write(f"Fold {fold:>3}  IS=[{ts_all[is_start].date()},{ts_all[is_end-1].date()}]"
               f"  OOS=[{fold_ts_start.date()},{fold_ts_end.date()}]")

    # ------------------------------------------------------------------
    # Feature selection  (entirely within IS window — NO OOS data)
    # ------------------------------------------------------------------
    t_sel = time.perf_counter()
    sel_cols = select_features_dynamic(X_is, y_is, feature_cols, BASE_LGB_PARAMS)
    fi        = [feature_cols.index(c) for c in sel_cols]
    t_sel_ms  = (time.perf_counter() - t_sel) * 1000
    tqdm.write(f"         Selection: {len(sel_cols)} features in {t_sel_ms:.0f}ms"
               f"  [{', '.join(sel_cols[:5])}{'…' if len(sel_cols)>5 else ''}]")

    X_is_sel = X_is[:, fi]

    # ------------------------------------------------------------------
    # Train LightGBM on IS data
    # ------------------------------------------------------------------
    n_val_is = max(int(INTERNAL_VAL_FRAC * len(X_is_sel)), 30)
    X_tr, y_tr = X_is_sel[:-n_val_is], y_is[:-n_val_is]
    X_vl, y_vl = X_is_sel[-n_val_is:], y_is[-n_val_is:]

    ds_tr = lgb.Dataset(X_tr, label=y_tr, feature_name=sel_cols)
    ds_vl = lgb.Dataset(X_vl, label=y_vl, reference=ds_tr, feature_name=sel_cols)

    t_train = time.perf_counter()
    model = lgb.train(
        BASE_LGB_PARAMS, ds_tr,
        valid_sets=[ds_tr, ds_vl], valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=EARLY_STOP_ROUNDS, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    t_train_ms = (time.perf_counter() - t_train) * 1000

    # ------------------------------------------------------------------
    # OOS inference
    # ------------------------------------------------------------------
    X_oos = X_all[oos_start:oos_end, :][:, fi]
    oos_probs = model.predict(X_oos)   # (OOS_BARS, 3)

    oos_probs_0fee.append(oos_probs)
    oos_probs_futures.append(oos_probs)   # same probs, different backtest fees
    oos_index_list.append(ts_all[oos_start:oos_end])

    # ------------------------------------------------------------------
    # Per-fold backtest
    # ------------------------------------------------------------------
    oos_close = close_all[oos_start:oos_end]
    oos_high  = high_all[oos_start:oos_end]
    oos_low   = low_all[oos_start:oos_end]
    oos_atr   = atr_all[oos_start:oos_end]
    pu_oos    = oos_probs[:, 1]
    pd_oos    = oos_probs[:, 0]

    # Scenario A: zero fees
    eq_0, tdf_0 = run_backtest_5m(
        pu_oos, pd_oos, oos_close, oos_high, oos_low, oos_atr,
        maker_fee=0.0, taker_fee_long=0.0, taker_fee_short=0.0,
        short_fund_per_bar=0.0,
    )

    # Scenario B: Binance Futures fees
    eq_f, tdf_f = run_backtest_5m(
        pu_oos, pd_oos, oos_close, oos_high, oos_low, oos_atr,
        maker_fee=MAKER_FEE_FUTURES,
        taker_fee_long=SPOT_TAKER_FEE,
        taker_fee_short=TAKER_FEE_FUTURES,
        short_fund_per_bar=SHORT_FUNDING_H / BARS_PER_HOUR,
    )

    m0 = compute_metrics(eq_0,  tdf_0)
    mf = compute_metrics(eq_f,  tdf_f)
    bh = float(oos_close[-1] / oos_close[0] - 1.0)

    fold_results.append({
        "fold":          fold,
        "oos_start":     str(fold_ts_start.date()),
        "oos_end":       str(fold_ts_end.date()),
        "n_features":    len(sel_cols),
        "best_iter":     model.best_iteration,
        "train_ms":      t_train_ms,
        "bh_return":     bh,
        "0fee":          m0,
        "futures_fee":   mf,
        "features":      sel_cols,
        # raw equity stored separately for stitching (not serialised)
        "_eq_0fee":      eq_0,
        "_eq_futures":   eq_f,
    })

    tqdm.write(
        f"         0-fee  Sharpe={m0['sharpe']:+.3f}  "
        f"Ret={m0['total_return']:+.2%}  DD={m0['max_dd']:.2%}  "
        f"Trades={m0['n_trades']}"
    )
    tqdm.write(
        f"         Fut-fee Sharpe={mf['sharpe']:+.3f}  "
        f"Ret={mf['total_return']:+.2%}  DD={mf['max_dd']:.2%}  "
        f"Trades={mf['n_trades']}"
    )
    del model

wfo_elapsed = time.perf_counter() - t0_wfo
print(f"\nWFO complete: {fold} folds  {wfo_elapsed:.1f}s ({wfo_elapsed/60:.1f} min)")


# ===========================================================================
# Aggregate metrics — stitch equity from stored per-fold curves
# ===========================================================================

# Re-run full OOS pass to build stitched equity curves (needed for aggregate metrics)
print("\nStitching equity curves from fold results …")
all_oos_ts: list = []
eq_0_all_parts: list[np.ndarray] = []
eq_f_all_parts: list[np.ndarray] = []
eq_bh_all_parts: list[np.ndarray] = []

fold_ptr = 0
for oos_start in range(IS_BARS, n_total - OOS_BARS + 1, OOS_BARS):
    is_start = max(0, oos_start - IS_BARS)
    is_end   = oos_start - EMBARGO_BARS
    oos_end  = min(oos_start + OOS_BARS, n_total)
    if is_end - is_start < IS_BARS // 2:
        continue
    if fold_ptr >= len(fold_results):
        break

    res = fold_results[fold_ptr]; fold_ptr += 1
    eq_0_all_parts.append(res["_eq_0fee"])
    eq_f_all_parts.append(res["_eq_futures"])

    oos_close = close_all[oos_start:oos_end]
    eq_bh_all_parts.append(oos_close / oos_close[0])
    all_oos_ts.extend(ts_all[oos_start:oos_end].tolist())


def _chain(parts: list[np.ndarray]) -> np.ndarray:
    """Multiplicatively chain equity sub-curves."""
    out = [1.0]
    level = 1.0
    for eq in parts:
        if len(eq) == 0:
            continue
        ratio = eq / max(eq[0], 1e-12)
        scaled = ratio * level
        out.extend(scaled.tolist())
        level = out[-1]
    return np.array(out[1:]) if len(out) > 1 else np.array([1.0])


eq_0_full  = _chain(eq_0_all_parts)
eq_f_full  = _chain(eq_f_all_parts)
eq_bh_full = _chain(eq_bh_all_parts)
ts_oos_full = np.array(all_oos_ts)

print(f"  OOS bars stitched : {len(eq_0_full):,}")


def _agg_metrics(eq: np.ndarray) -> dict:
    ann = 24 * 365 * BARS_PER_HOUR
    eq = np.maximum(eq, 1e-12)
    ret = np.log(eq[1:] / eq[:-1])
    pk  = np.maximum.accumulate(eq)
    return {
        "sharpe":       float(ret.mean() / (ret.std(ddof=1) + 1e-12) * np.sqrt(ann)),
        "total_return": float(eq[-1] - 1.0),
        "max_dd":       float(((eq - pk) / pk).min()),
    }


agg_0fee    = _agg_metrics(eq_0_full)
agg_futures = _agg_metrics(eq_f_full)

print(f"\n{'─'*60}")
print(f"{'Aggregate OOS metrics':^60}")
print(f"{'─'*60}")
print(f"{'':30s}  {'0-fee':>12}  {'Futures-fee':>12}")
print(f"{'Sharpe ratio':30s}  {agg_0fee['sharpe']:>12.4f}  {agg_futures['sharpe']:>12.4f}")
print(f"{'Total return':30s}  {agg_0fee['total_return']:>11.2%}  {agg_futures['total_return']:>11.2%}")
print(f"{'Max drawdown':30s}  {agg_0fee['max_dd']:>11.2%}  {agg_futures['max_dd']:>11.2%}")
print(f"{'─'*60}")

# Per-fold win rates
fold_sharpes_0fee = [r["0fee"]["sharpe"] for r in fold_results if not np.isnan(r["0fee"]["sharpe"])]
fold_sharpes_fut  = [r["futures_fee"]["sharpe"] for r in fold_results if not np.isnan(r["futures_fee"]["sharpe"])]
pct_pos_0fee = None; pct_pos_fut = None
if fold_sharpes_0fee:
    pct_pos_0fee = sum(s > 0 for s in fold_sharpes_0fee) / len(fold_sharpes_0fee)
    pct_pos_fut  = sum(s > 0 for s in fold_sharpes_fut)  / len(fold_sharpes_fut)
    print(f"{'% positive-Sharpe folds':30s}  {pct_pos_0fee:>11.1%}  {pct_pos_fut:>11.1%}")
    print(f"{'Median fold Sharpe':30s}  {np.median(fold_sharpes_0fee):>12.3f}  {np.median(fold_sharpes_fut):>12.3f}")


# ===========================================================================
# Save results
# ===========================================================================

results = {
    "config": {
        "symbol":         SYMBOL,
        "interval":       INTERVAL,
        "label":          LABEL_COL,
        "is_days":        IS_DAYS,
        "oos_days":       OOS_DAYS,
        "stage4_top_k":   STAGE4_TOP_K,
        "label_choice":   LABEL_CHOICE,
    },
    "aggregate": {
        "0fee":        agg_0fee,
        "futures_fee": agg_futures,
        "n_folds":     fold,
        "pct_pos_sharpe_0fee":    float(pct_pos_0fee) if fold_sharpes_0fee else None,
        "pct_pos_sharpe_futures": float(pct_pos_fut)  if fold_sharpes_fut  else None,
    },
    "per_fold": [
        {
            "fold":        r["fold"],
            "oos_start":   r["oos_start"],
            "oos_end":     r["oos_end"],
            "n_features":  r["n_features"],
            "best_iter":   r["best_iter"],
            "bh_return":   r["bh_return"],
            "0fee":        r["0fee"],
            "futures_fee": r["futures_fee"],
            "top_features": r["features"][:10],
        }
        for r in fold_results
    ],
}

results_path = _ART_DIR / "results.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved → {results_path}")


# ===========================================================================
# Plots
# ===========================================================================

# -- 1. Full stitched OOS equity curves --
fig, axes = plt.subplots(2, 1, figsize=(14, 9),
                          gridspec_kw={"height_ratios": [3, 1.3], "hspace": 0.07})
ax = axes[0]
min_len = min(len(ts_oos_full), len(eq_0_full), len(eq_f_full), len(eq_bh_full))
ts_plot = ts_oos_full[:min_len]
ax.plot(ts_plot, eq_0_full[:min_len],  color=ACCENT, lw=1.3, label="0-fee WFO")
ax.plot(ts_plot, eq_f_full[:min_len],  color=GREEN,  lw=1.3, label="Futures-fee WFO")
ax.plot(ts_plot, eq_bh_full[:min_len], color=BLUE,   lw=1.0, ls="--", label="Buy & Hold")
ax.axhline(1.0, color=GREY, lw=0.7, ls=":")
ax.set_ylabel("Portfolio value"); ax.legend(); ax.grid(axis="y", alpha=0.3)
ax.set_title(
    f"5-Min Microstructure WFO | {SYMBOL} | Label={LABEL_COL} | "
    f"IS={IS_DAYS}d OOS={OOS_DAYS}d | Dynamic 4-stage selection",
    fontweight="bold",
)

ax = axes[1]
pk0 = np.maximum.accumulate(eq_0_full[:min_len])
pkf = np.maximum.accumulate(eq_f_full[:min_len])
ax.fill_between(ts_plot, (eq_0_full[:min_len] - pk0) / (pk0 + 1e-12) * 100,
                0, color=ACCENT, alpha=0.45, label="0-fee")
ax.fill_between(ts_plot, (eq_f_full[:min_len] - pkf) / (pkf + 1e-12) * 100,
                0, color=GREEN,  alpha=0.35, label="Futures-fee")
ax.set_ylabel("Drawdown (%)"); ax.legend(); ax.grid(axis="y", alpha=0.3)
for ax_ in axes:
    ax_.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax_.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax_.xaxis.get_majorticklabels(), rotation=30, ha="right")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "wfo_equity_5m.png")
plt.show()

# -- 2. Per-fold Sharpe bar chart --
if fold_results:
    folds_     = [r["fold"] for r in fold_results]
    sh_0fee_   = [r["0fee"]["sharpe"]    for r in fold_results]
    sh_fut_    = [r["futures_fee"]["sharpe"] for r in fold_results]
    x = np.arange(len(folds_))
    w = 0.35
    fig, ax = plt.subplots(figsize=(max(12, len(folds_) * 0.6), 4))
    ax.bar(x - w/2, sh_0fee_, w, color=ACCENT, alpha=0.8, label="0-fee")
    ax.bar(x + w/2, sh_fut_,  w, color=GREEN,  alpha=0.8, label="Futures-fee")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(
        [r["oos_start"] for r in fold_results], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Sharpe (annualised)"); ax.legend()
    ax.set_title("Per-fold OOS Sharpe — 5m Dynamic WFO", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "per_fold_sharpe_5m.png")
    plt.show()

# -- 3. Feature frequency heat-map --
if fold_results:
    from collections import Counter
    freq = Counter()
    for r in fold_results:
        freq.update(r["features"])
    top_feats = [f for f, _ in freq.most_common(30)]
    heat_data = np.zeros((len(top_feats), len(fold_results)))
    for fi_, feat in enumerate(top_feats):
        for fj_, r in enumerate(fold_results):
            if feat in r["features"]:
                heat_data[fi_, fj_] = 1.0
    fig, ax = plt.subplots(figsize=(max(10, len(fold_results) * 0.5), 8))
    im = ax.imshow(heat_data, aspect="auto", cmap="YlOrRd", interpolation="none")
    ax.set_yticks(range(len(top_feats))); ax.set_yticklabels(top_feats, fontsize=8)
    ax.set_xticks(range(len(fold_results)))
    ax.set_xticklabels([r["oos_start"] for r in fold_results],
                        rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("Fold (OOS start)"); ax.set_title(
        "Top-30 features selected per fold", fontweight="bold")
    plt.colorbar(im, ax=ax, label="Selected (1=yes)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_selection_heatmap_5m.png")
    plt.show()

print("\nAll figures saved to", FIGURES_DIR)
print("Done.")
