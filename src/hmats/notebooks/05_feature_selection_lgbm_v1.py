"""06 — V3 Feature Engineering + Selection Pipeline

Builds ~22 new features from external data sources:
  Group 1: Cross-Asset (from multi-coin OHLCV)
  Group 2: Market Structure (from market caps)
  Group 3: Sentiment (Fear & Greed)
  Group 4: Microstructure (from BTC OHLCV)
  Group 5: Enhanced OHLCV

Merges with existing 195 V1 features, runs 4-stage selection, trains LGBM.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, classification_report

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[3]
RAW_DIR = REPO / "data" / "raw"
EXT_DIR = REPO / "data" / "external"
FEAT_DIR = REPO / "data" / "features"
OUT_DIR = REPO / "lab" / "figures" / "05_feature_selection_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
]
ALTS = [s for s in SYMBOLS if s != "BTCUSDT"]

print("=" * 70)
print("06 — V3 Feature Engineering + Selection Pipeline")
print("=" * 70)


# ══════════════════════════════════════════════════════════════════════
# LOAD BASE DATA
# ══════════════════════════════════════════════════════════════════════
print("\nLoading base data...")

btc = pd.read_parquet(RAW_DIR / "BTCUSDT_1h.parquet")
btc.index = btc.index.tz_localize(None) if btc.index.tz else btc.index

alt_closes = {}
for sym in ALTS:
    p = RAW_DIR / f"{sym}_1h.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        df.index = df.index.tz_localize(None) if df.index.tz else df.index
        alt_closes[sym] = df["close"].reindex(btc.index, method="ffill")

fng_df = pd.read_parquet(EXT_DIR / "fear_greed_index.parquet")
fng_df.index = fng_df.index.tz_localize(None) if fng_df.index.tz else fng_df.index

approx_mcap = pd.read_parquet(EXT_DIR / "approx_market_caps.parquet")
approx_mcap["date"] = pd.to_datetime(approx_mcap["date"]).dt.tz_localize(None)

cg_mcap = pd.read_parquet(EXT_DIR / "coingecko_market_caps.parquet")
cg_mcap["date"] = pd.to_datetime(cg_mcap["date"]).dt.tz_localize(None)

print(f"  BTC: {len(btc)} hourly bars ({btc.index.min().date()} → {btc.index.max().date()})")
print(f"  Altcoins loaded: {list(alt_closes.keys())}")
print(f"  Fear & Greed: {len(fng_df)} daily records")
print(f"  Approx market caps: {len(approx_mcap)} records")
print(f"  CoinGecko market caps: {len(cg_mcap)} records")

v3_features = pd.DataFrame(index=btc.index)


# ══════════════════════════════════════════════════════════════════════
# GROUP 1: Cross-Asset Features (hourly resolution)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("GROUP 1: Cross-Asset Features")
print("=" * 70)

eth_close = alt_closes.get("ETHUSDT")
if eth_close is not None:
    eth_btc = eth_close / btc["close"]
    v3_features["cross_eth_btc_ratio"] = eth_btc
    v3_features["cross_eth_btc_mom_24h"] = eth_btc.pct_change(24)
    v3_features["cross_eth_btc_mom_72h"] = eth_btc.pct_change(72)

alt_rets_24h = pd.DataFrame()
for sym, close in alt_closes.items():
    alt_rets_24h[sym] = close.pct_change(24)

if not alt_rets_24h.empty:
    v3_features["cross_altcoin_breadth_24h"] = (alt_rets_24h > 0).mean(axis=1)
    avg_alt_ret = alt_rets_24h.mean(axis=1)
    btc_ret_24h = btc["close"].pct_change(24)
    v3_features["cross_btc_relative_strength"] = btc_ret_24h - avg_alt_ret
    v3_features["cross_alt_correlation_24h"] = btc_ret_24h.rolling(24).corr(avg_alt_ret)

g1_cols = [c for c in v3_features.columns if c.startswith("cross_")]
print(f"  Built {len(g1_cols)} cross-asset features: {g1_cols}")


# ══════════════════════════════════════════════════════════════════════
# GROUP 2: Market Structure Features (daily, forward-filled)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("GROUP 2: Market Structure Features")
print("=" * 70)

mcap_pivot = approx_mcap.pivot_table(
    index="date", columns="symbol", values="approx_market_cap", aggfunc="last"
)

cg_stable = cg_mcap[cg_mcap["symbol"].isin(["USDTUSDT", "USDCUSDT"])]
if not cg_stable.empty:
    stable_daily = cg_stable.pivot_table(
        index=cg_stable["date"].dt.normalize(),
        columns="symbol", values="market_cap", aggfunc="last"
    )
    stable_total = stable_daily.sum(axis=1)
else:
    stable_total = pd.Series(dtype=float)

our_coins = [s for s in SYMBOLS if s in mcap_pivot.columns]
total_mcap = mcap_pivot[our_coins].sum(axis=1)

btc_dom = mcap_pivot.get("BTCUSDT", pd.Series(dtype=float)) / total_mcap
eth_dom = mcap_pivot.get("ETHUSDT", pd.Series(dtype=float)) / total_mcap

mkt_daily = pd.DataFrame(index=mcap_pivot.index)
mkt_daily["mkt_btc_dominance"] = btc_dom
mkt_daily["mkt_btc_dominance_chg_7d"] = btc_dom.diff(7)
mkt_daily["mkt_eth_dominance"] = eth_dom
mkt_daily["mkt_total_mcap_chg_24h"] = total_mcap.pct_change(1)

if not stable_total.empty:
    stable_reindexed = stable_total.reindex(mcap_pivot.index, method="ffill")
    total_plus_stable = total_mcap.add(stable_reindexed, fill_value=0)
    mkt_daily["mkt_stablecoin_pct"] = stable_reindexed / total_plus_stable
else:
    mkt_daily["mkt_stablecoin_pct"] = np.nan

mkt_hourly = mkt_daily.reindex(btc.index, method="ffill")
for col in mkt_hourly.columns:
    v3_features[col] = mkt_hourly[col]

g2_cols = [c for c in v3_features.columns if c.startswith("mkt_")]
print(f"  Built {len(g2_cols)} market structure features: {g2_cols}")


# ══════════════════════════════════════════════════════════════════════
# GROUP 3: Sentiment Features (daily, forward-filled)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("GROUP 3: Sentiment Features")
print("=" * 70)

fng_daily = fng_df[["value"]].copy()
fng_daily.columns = ["sent_fear_greed"]
fng_daily["sent_fear_greed"] = fng_daily["sent_fear_greed"] / 100.0
fng_daily["sent_fear_greed_ma7"] = fng_daily["sent_fear_greed"].rolling(7).mean()
fng_daily["sent_fear_greed_chg_7d"] = fng_daily["sent_fear_greed"].diff(7)

fng_hourly = fng_daily.reindex(btc.index, method="ffill")
for col in fng_hourly.columns:
    v3_features[col] = fng_hourly[col]

g3_cols = [c for c in v3_features.columns if c.startswith("sent_")]
print(f"  Built {len(g3_cols)} sentiment features: {g3_cols}")


# ══════════════════════════════════════════════════════════════════════
# GROUP 4: Microstructure Features (hourly from BTC OHLCV)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("GROUP 4: Microstructure Features")
print("=" * 70)

close = btc["close"]
volume = btc["volume"]
dollar_vol = close * volume
ret = close.pct_change()
abs_ret = ret.abs()
delta_price = close.diff().abs()

MICRO_WIN = 24

v3_features["micro_amihud_illiq"] = (abs_ret / dollar_vol.replace(0, np.nan)).rolling(MICRO_WIN).mean()

def rolling_kyle_lambda(delta_p, vol, win):
    result = pd.Series(np.nan, index=delta_p.index)
    for i in range(win, len(delta_p)):
        dp = delta_p.iloc[i - win:i].values
        v = vol.iloc[i - win:i].values
        mask = np.isfinite(dp) & np.isfinite(v) & (v > 0)
        if mask.sum() > 5:
            v_m = v[mask]
            dp_m = dp[mask]
            if v_m.std() > 0:
                result.iloc[i] = np.polyfit(v_m, dp_m, 1)[0]
    return result

print("  Computing Kyle's lambda (this takes ~30s)...", flush=True)
v3_features["micro_kyle_lambda"] = rolling_kyle_lambda(delta_price, volume, MICRO_WIN)

cov_serial = ret.rolling(MICRO_WIN).cov(ret.shift(1))
roll_spread = np.where(cov_serial < 0, np.sqrt(-2 * cov_serial), 0.0)
v3_features["micro_roll_spread"] = pd.Series(roll_spread, index=btc.index)

vol_mean = volume.rolling(MICRO_WIN).mean()
vol_std = volume.rolling(MICRO_WIN).std()
v3_features["micro_volume_clock"] = vol_std / vol_mean.replace(0, np.nan)

g4_cols = [c for c in v3_features.columns if c.startswith("micro_")]
print(f"  Built {len(g4_cols)} microstructure features: {g4_cols}")


# ══════════════════════════════════════════════════════════════════════
# GROUP 5: Enhanced OHLCV Features
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("GROUP 5: Enhanced OHLCV Features")
print("=" * 70)

vol_24 = ret.rolling(24).std()
vol_168 = ret.rolling(168).std()
v3_features["vol_term_structure"] = vol_24 / vol_168.replace(0, np.nan)

ret_24 = close.pct_change(24)
ret_72 = close.pct_change(72)
v3_features["mom_normalized_24h"] = ret_24 / vol_24.replace(0, np.nan)
v3_features["mom_normalized_72h"] = ret_72 / vol_168.replace(0, np.nan)

g5_cols = [c for c in v3_features.columns if c.startswith("vol_term") or c.startswith("mom_norm")]
print(f"  Built {len(g5_cols)} enhanced OHLCV features: {g5_cols}")


# ══════════════════════════════════════════════════════════════════════
# SUMMARY OF V3 FEATURES
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("V3 FEATURE SUMMARY")
print("=" * 70)

all_v3_cols = list(v3_features.columns)
print(f"  Total V3 features: {len(all_v3_cols)}")
for col in all_v3_cols:
    valid = v3_features[col].dropna()
    print(f"    {col:<35s}  valid={len(valid):>6} / {len(v3_features)}  "
          f"mean={valid.mean():>10.4f}  std={valid.std():>10.4f}")

v3_out = FEAT_DIR / "BTCUSDT_1h_v3_features.parquet"
v3_features.to_parquet(v3_out)
print(f"\n  Saved V3 features: {v3_out}")


# ══════════════════════════════════════════════════════════════════════
# MERGE WITH V1 FEATURES + RUN SELECTION PIPELINE
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("MERGING V3 + V1 FEATURES")
print("=" * 70)

v1_df = pd.read_parquet(FEAT_DIR / "BTCUSDT_1h_features.parquet")
v1_df.index = v1_df.index.tz_localize(None) if v1_df.index.tz else v1_df.index

with open(FEAT_DIR / "feature_registry.json") as f:
    registry = json.load(f)

BACKTEST_COLS = registry["backtest_only_cols"]
v1_feature_cols = [c for c in v1_df.columns if c not in BACKTEST_COLS + ["label"]]

merged = v1_df.copy()
for col in v3_features.columns:
    merged[col] = v3_features[col].reindex(merged.index)

all_feature_cols = v1_feature_cols + all_v3_cols
print(f"  V1 features: {len(v1_feature_cols)}")
print(f"  V3 features: {len(all_v3_cols)}")
print(f"  Combined pool: {len(all_feature_cols)}")

from hmats.data.splits import calendar_split

TRAIN_END = "2024-01-01"
VAL_END = "2025-01-01"

train_df, val_df, test_df = calendar_split(merged, train_end=TRAIN_END, val_end=VAL_END)

print(f"\n  Train: {len(train_df):>7,}  {train_df.index.min().date()} → {train_df.index.max().date()}")
print(f"  Val:   {len(val_df):>7,}  {val_df.index.min().date()} → {val_df.index.max().date()}")
print(f"  Test:  {len(test_df):>7,}  {test_df.index.min().date()} → {test_df.index.max().date()}")


# ══════════════════════════════════════════════════════════════════════
# 4-STAGE FEATURE SELECTION
# ══════════════════════════════════════════════════════════════════════

# --- STAGE 1: Statistical Filter ---
print("\n" + "=" * 70)
print("STAGE 1: Statistical Filter")
print("=" * 70)

VAR_THRESHOLD = 1e-6
CORR_THRESHOLD = 0.85

valid_cols = [f for f in all_feature_cols if train_df[f].notna().sum() > len(train_df) * 0.5]
print(f"  Features with >50% non-null: {len(valid_cols)} / {len(all_feature_cols)}")

variances = train_df[valid_cols].var()
low_var = variances[variances < VAR_THRESHOLD].index.tolist()
candidates = [f for f in valid_cols if f not in low_var]
print(f"  Variance filter: {len(low_var)} removed")

train_filled = train_df[candidates].fillna(0)
corr_matrix = train_filled.corr(method="spearman").abs()
target_corr = train_filled.corrwith(train_df["label"], method="spearman").abs()

to_drop = set()
for i in range(len(candidates)):
    if candidates[i] in to_drop:
        continue
    for j in range(i + 1, len(candidates)):
        if candidates[j] in to_drop:
            continue
        if corr_matrix.iloc[i, j] > CORR_THRESHOLD:
            drop_feat = candidates[j] if target_corr.get(candidates[i], 0) >= target_corr.get(candidates[j], 0) else candidates[i]
            to_drop.add(drop_feat)

stage1 = [f for f in candidates if f not in to_drop]
print(f"  Correlation filter: {len(to_drop)} removed (ρ > {CORR_THRESHOLD})")
print(f"  Stage 1 survivors: {len(stage1)}")

v3_in_stage1 = [f for f in stage1 if f in all_v3_cols]
print(f"  V3 features surviving Stage 1: {len(v3_in_stage1)} / {len(all_v3_cols)}: {v3_in_stage1}")

# --- STAGE 2: MI Ranking ---
print("\n" + "=" * 70)
print("STAGE 2: Univariate Ranking (Mutual Information)")
print("=" * 70)

TOP_K_MI = 60

X_s1 = train_df[stage1].fillna(0).values
y_s1 = train_df["label"].values

mi_scores = mutual_info_classif(X_s1, y_s1, n_neighbors=5, random_state=42)
ranking = pd.DataFrame({
    "feature": stage1,
    "MI": mi_scores,
    "is_v3": [f in all_v3_cols for f in stage1],
}).sort_values("MI", ascending=False).reset_index(drop=True)

print(f"\n  Top 30 features by MI:")
print(ranking.head(30).to_string(index=False))

v3_in_top = ranking.head(TOP_K_MI)
v3_count = v3_in_top["is_v3"].sum()
print(f"\n  V3 features in top {TOP_K_MI}: {v3_count}")
print(f"  V3 features in ranking:")
v3_ranking = ranking[ranking["is_v3"]].head(20)
print(v3_ranking.to_string(index=False))

stage2 = ranking.head(TOP_K_MI)["feature"].tolist()

# --- STAGE 3: Walk-Forward Stability ---
print("\n" + "=" * 70)
print("STAGE 3: Walk-Forward Feature Stability")
print("=" * 70)

WINDOW_SIZE = 2160
STEP_SIZE = 720
MIN_STABLE_FRAC = 0.5
TOP_K_PER_WINDOW = 35

n = len(train_df)
windows = []
start_idx = 0
while start_idx + WINDOW_SIZE <= n:
    windows.append((start_idx, start_idx + WINDOW_SIZE))
    start_idx += STEP_SIZE

print(f"  {len(windows)} windows, size {WINDOW_SIZE}, step {STEP_SIZE}")

appearance = {f: 0 for f in stage2}
for s, e in windows:
    chunk = train_df.iloc[s:e]
    if chunk["label"].nunique() < 2:
        continue
    X_c = chunk[stage2].fillna(0).values
    y_c = chunk["label"].values
    mi = mutual_info_classif(X_c, y_c, n_neighbors=5, random_state=42)
    top_idx = np.argsort(mi)[-TOP_K_PER_WINDOW:]
    for idx in top_idx:
        appearance[stage2[idx]] += 1

stability = pd.DataFrame({
    "feature": list(appearance.keys()),
    "appearances": list(appearance.values()),
    "frac": [c / len(windows) for c in appearance.values()],
    "is_v3": [f in all_v3_cols for f in appearance.keys()],
}).sort_values("frac", ascending=False).reset_index(drop=True)

stage3 = stability[stability["frac"] >= MIN_STABLE_FRAC]["feature"].tolist()
print(f"  Stable features (>= {MIN_STABLE_FRAC*100:.0f}%): {len(stage3)}")
print(f"\n  Stability ranking (top 40):")
print(stability.head(40).to_string(index=False))

v3_in_stage3 = [f for f in stage3 if f in all_v3_cols]
print(f"\n  V3 features surviving Stage 3: {len(v3_in_stage3)}: {v3_in_stage3}")

# --- STAGE 4: Permutation Importance Pruning ---
print("\n" + "=" * 70)
print("STAGE 4: Model-Based Pruning (Permutation Importance)")
print("=" * 70)

PERM_THRESHOLD = 0.0005

X_train_s4 = train_df[stage3].fillna(0).values
y_train_s4 = train_df["label"].values
X_val_s4 = val_df[stage3].fillna(0).values
y_val_s4 = val_df["label"].values

model_s4 = lgb.LGBMClassifier(
    n_estimators=500, learning_rate=0.03, num_leaves=31,
    max_depth=6, subsample=0.7, colsample_bytree=0.7,
    min_child_samples=50, reg_alpha=0.1, reg_lambda=1.0,
    verbose=-1, random_state=42, n_jobs=-1,
)
model_s4.fit(
    X_train_s4, y_train_s4,
    eval_set=[(X_val_s4, y_val_s4)],
    callbacks=[lgb.early_stopping(30, verbose=False)],
)

baseline_auc = roc_auc_score(y_val_s4, model_s4.predict_proba(X_val_s4)[:, 1])
print(f"  Baseline AUC (Stage 3 features): {baseline_auc:.4f}")

perm_result = permutation_importance(
    model_s4, X_val_s4, y_val_s4,
    scoring="roc_auc", n_repeats=10, random_state=42, n_jobs=-1,
)

perm_df = pd.DataFrame({
    "feature": stage3,
    "importance_mean": perm_result.importances_mean,
    "importance_std": perm_result.importances_std,
    "is_v3": [f in all_v3_cols for f in stage3],
}).sort_values("importance_mean", ascending=False).reset_index(drop=True)

print(f"\n  Permutation importance:")
print(perm_df.to_string(index=False))

stage4 = perm_df[perm_df["importance_mean"] > PERM_THRESHOLD]["feature"].tolist()
v3_in_stage4 = [f for f in stage4 if f in all_v3_cols]
print(f"\n  Stage 4: {len(stage4)} features kept")
print(f"  V3 features in final set: {len(v3_in_stage4)}: {v3_in_stage4}")


# ══════════════════════════════════════════════════════════════════════
# PIPELINE SUMMARY
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PIPELINE SUMMARY")
print("=" * 70)
print(f"  Stage 0 (Pool):      {len(all_feature_cols)} features ({len(v1_feature_cols)} V1 + {len(all_v3_cols)} V3)")
print(f"  Stage 1 (Filter):    {len(stage1)} features ({len(v3_in_stage1)} V3)")
n_v3_s2 = sum(1 for f in stage2 if f in all_v3_cols)
print(f"  Stage 2 (MI Rank):   {len(stage2)} features ({n_v3_s2} V3)")
print(f"  Stage 3 (Stability): {len(stage3)} features ({len(v3_in_stage3)} V3)")
print(f"  Stage 4 (Prune):     {len(stage4)} features ({len(v3_in_stage4)} V3)")
print(f"\n  Final feature set:")
for i, f in enumerate(stage4):
    tag = " [V3-NEW]" if f in all_v3_cols else ""
    mi_val = ranking[ranking["feature"] == f]["MI"].values
    mi_str = f"{mi_val[0]:.4f}" if len(mi_val) > 0 else "N/A"
    stab = stability[stability["feature"] == f]["frac"].values
    stab_str = f"{stab[0]:.2f}" if len(stab) > 0 else "N/A"
    perm = perm_df[perm_df["feature"] == f]["importance_mean"].values
    perm_str = f"{perm[0]:.4f}" if len(perm) > 0 else "N/A"
    print(f"    {i+1:>2}. {f:<35s}  MI={mi_str}  Stab={stab_str}  Perm={perm_str}{tag}")


# ══════════════════════════════════════════════════════════════════════
# FINAL MODEL
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("FINAL MODEL: LightGBM on Selected Features")
print("=" * 70)

final_features = stage4 if len(stage4) >= 5 else stage3[:25]
print(f"  Training with {len(final_features)} features")

X_train = train_df[final_features].fillna(0).values
y_train = train_df["label"].values
X_val = val_df[final_features].fillna(0).values
y_val = val_df["label"].values
X_test = test_df[final_features].fillna(0).values
y_test = test_df["label"].values

final_model = lgb.LGBMClassifier(
    n_estimators=1000, learning_rate=0.02, num_leaves=31,
    max_depth=6, min_child_samples=50, subsample=0.8,
    colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
    verbose=-1, random_state=42, n_jobs=-1,
)
final_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50, verbose=False)],
)

probs_val = final_model.predict_proba(X_val)[:, 1]
probs_test = final_model.predict_proba(X_test)[:, 1]
auc_val = roc_auc_score(y_val, probs_val)
auc_test = roc_auc_score(y_test, probs_test)

print(f"\n  Best iteration: {final_model.best_iteration_}")
print(f"  Validation AUC: {auc_val:.4f}")
print(f"  Test AUC:       {auc_test:.4f}")
print(f"  AUC gap:        {abs(auc_val - auc_test):.4f}")

BASELINE_AUC_VAL = 0.5555
BASELINE_AUC_TEST = 0.5395
print(f"\n  COMPARISON vs V1-only baseline:")
print(f"    Val AUC:  {BASELINE_AUC_VAL:.4f} → {auc_val:.4f}  (Δ = {auc_val - BASELINE_AUC_VAL:+.4f})")
print(f"    Test AUC: {BASELINE_AUC_TEST:.4f} → {auc_test:.4f}  (Δ = {auc_test - BASELINE_AUC_TEST:+.4f})")

lgb_imp = pd.DataFrame({
    "feature": final_features,
    "gain": final_model.feature_importances_,
    "is_v3": [f in all_v3_cols for f in final_features],
}).sort_values("gain", ascending=False)
print(f"\n  LGBM Gain Importance:")
print(lgb_imp.to_string(index=False))

print(f"\n  Classification Report (Test):")
preds_test = (probs_test > 0.5).astype(int)
print(classification_report(y_test, preds_test, target_names=["Down", "Up"]))


# ══════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════

# Pipeline funnel
fig, ax = plt.subplots(figsize=(12, 5))
stage_labels = [
    f"Pool\n({len(all_feature_cols)})",
    f"Filter\n({len(stage1)})",
    f"MI Rank\n({len(stage2)})",
    f"Stability\n({len(stage3)})",
    f"Prune\n({len(stage4)})",
]
counts = [len(all_feature_cols), len(stage1), len(stage2), len(stage3), len(stage4)]
v3_counts = [
    len(all_v3_cols),
    len(v3_in_stage1),
    n_v3_s2,
    len(v3_in_stage3),
    len(v3_in_stage4),
]
x = np.arange(len(stage_labels))
bars1 = ax.bar(x, counts, color="#90CAF9", label="V1 features")
bars2 = ax.bar(x, v3_counts, color="#F57C00", label="V3 features (new)")
for bar, c, v in zip(bars1, counts, v3_counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f"{c} ({v} new)", ha="center", fontweight="bold", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(stage_labels, fontsize=10)
ax.set_ylabel("Feature Count")
ax.set_title("Feature Selection Pipeline — V1 + V3 Combined", fontweight="bold", fontsize=12)
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "01_pipeline_summary.png", dpi=150, bbox_inches="tight")
plt.close()

# Feature importance
fig, ax = plt.subplots(figsize=(10, max(6, len(final_features) * 0.35)))
colors = ["#F57C00" if v3 else "#2962FF" for v3 in lgb_imp["is_v3"]]
ax.barh(range(len(lgb_imp)), lgb_imp["gain"].values, color=colors)
ax.set_yticks(range(len(lgb_imp)))
ax.set_yticklabels(lgb_imp["feature"].values, fontsize=9)
ax.set_xlabel("LightGBM Gain")
ax.set_title(f"Final Feature Importance (val AUC={auc_val:.4f}, test AUC={auc_test:.4f})", fontweight="bold")
ax.invert_yaxis()
import matplotlib.patches as mpatches
ax.legend(handles=[
    mpatches.Patch(color="#2962FF", label="V1 (existing OHLCV)"),
    mpatches.Patch(color="#F57C00", label="V3 (new external)"),
])
plt.tight_layout()
plt.savefig(OUT_DIR / "02_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()

# Probability distributions
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, probs, y, title in [
    (axes[0], probs_val, y_val, "Validation"),
    (axes[1], probs_test, y_test, "Test"),
]:
    ax.hist(probs[y == 0], bins=50, alpha=0.6, label="Down", color="#EF5350", density=True)
    ax.hist(probs[y == 1], bins=50, alpha=0.6, label="Up", color="#26A69A", density=True)
    auc = roc_auc_score(y, probs)
    ax.set_title(f"{title} — AUC={auc:.4f}", fontweight="bold")
    ax.set_xlabel("P(Up)")
    ax.set_ylabel("Density")
    ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "03_probability_distributions.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"\n  Figures saved to: {OUT_DIR}")

# Save results
results = {
    "pipeline": {
        "stage0_pool": len(all_feature_cols),
        "stage0_v1": len(v1_feature_cols),
        "stage0_v3": len(all_v3_cols),
        "stage1_filter": len(stage1),
        "stage2_mi_rank": len(stage2),
        "stage3_stability": len(stage3),
        "stage4_prune": len(stage4),
        "v3_in_final": len(v3_in_stage4),
    },
    "final_features": final_features,
    "v3_features_in_final": v3_in_stage4,
    "val_auc": float(auc_val),
    "test_auc": float(auc_test),
    "baseline_val_auc": BASELINE_AUC_VAL,
    "baseline_test_auc": BASELINE_AUC_TEST,
    "best_iteration": int(final_model.best_iteration_),
}
with open(OUT_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"  Results saved to: {OUT_DIR / 'results.json'}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
