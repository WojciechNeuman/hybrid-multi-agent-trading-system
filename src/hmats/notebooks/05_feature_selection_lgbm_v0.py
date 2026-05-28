"""05 — Rigorous Feature Selection + LightGBM (1h BTCUSDT)

Professional feature selection pipeline:
  Stage 1: Statistical filter (variance + correlation)
  Stage 2: Univariate ranking (Mutual Information)
  Stage 3: Walk-forward stability test
  Stage 4: Permutation importance pruning

Then train final LGBM on selected features with proper evaluation.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, classification_report

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[3]
FEATURES_DIR = REPO / "data" / "features"
OUT_DIR = REPO / "lab" / "figures" / "05_feature_selection"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_PARQUET = FEATURES_DIR / "BTCUSDT_1h_features.parquet"
REGISTRY_PATH = FEATURES_DIR / "feature_registry.json"

TRAIN_END = "2024-01-01"
VAL_END = "2025-01-01"

print("=" * 70)
print("05 — Feature Selection + LightGBM Pipeline")
print("=" * 70)

feat_df = pd.read_parquet(FEATURES_PARQUET)
feat_df.index = feat_df.index.tz_localize(None) if feat_df.index.tz else feat_df.index

with open(REGISTRY_PATH) as f:
    registry = json.load(f)

BACKTEST_COLS = registry["backtest_only_cols"]
feature_cols = [c for c in feat_df.columns if c not in BACKTEST_COLS + ["label"]]

print(f"\nLoaded {len(feat_df):,} rows  {feat_df.index.min().date()} → {feat_df.index.max().date()}")
print(f"ML features: {len(feature_cols)}")
print(f"Label distribution: {feat_df['label'].value_counts().to_dict()}")

from hmats.data.splits import calendar_split
train_df, val_df, test_df = calendar_split(feat_df, train_end=TRAIN_END, val_end=VAL_END)

print(f"\nTrain: {len(train_df):>7,}  {train_df.index.min().date()} → {train_df.index.max().date()}  label={train_df['label'].mean():.3f}")
print(f"Val:   {len(val_df):>7,}  {val_df.index.min().date()} → {val_df.index.max().date()}  label={val_df['label'].mean():.3f}")
print(f"Test:  {len(test_df):>7,}  {test_df.index.min().date()} → {test_df.index.max().date()}  label={test_df['label'].mean():.3f}")


# ══════════════════════════════════════════════════════════════════════
# STAGE 1: Statistical Filter
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STAGE 1: Statistical Filter")
print("=" * 70)

VAR_THRESHOLD = 1e-6
CORR_THRESHOLD = 0.85

variances = train_df[feature_cols].var()
low_var = variances[variances < VAR_THRESHOLD].index.tolist()
candidates = [f for f in feature_cols if f not in low_var]
print(f"  Variance filter: {len(low_var)} removed (threshold={VAR_THRESHOLD})")
if low_var:
    print(f"    Removed: {low_var}")

corr_matrix = train_df[candidates].corr(method="spearman").abs()
target_corr = train_df[candidates].corrwith(train_df["label"], method="spearman").abs()

to_drop = set()
for i in range(len(candidates)):
    if candidates[i] in to_drop:
        continue
    for j in range(i + 1, len(candidates)):
        if candidates[j] in to_drop:
            continue
        if corr_matrix.iloc[i, j] > CORR_THRESHOLD:
            drop_feat = candidates[j] if target_corr[candidates[i]] >= target_corr[candidates[j]] else candidates[i]
            to_drop.add(drop_feat)

stage1_survivors = [f for f in candidates if f not in to_drop]
print(f"  Correlation filter: {len(to_drop)} removed (threshold={CORR_THRESHOLD})")
print(f"  Stage 1 survivors: {len(stage1_survivors)} / {len(feature_cols)}")

fig, ax = plt.subplots(figsize=(12, 10))
sample_feats = stage1_survivors[:40]
corr_sub = train_df[sample_feats].corr(method="spearman")
mask = np.triu(np.ones_like(corr_sub, dtype=bool))
sns.heatmap(corr_sub, mask=mask, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.5, ax=ax, annot=False,
            cbar_kws={"shrink": 0.6, "label": "Spearman ρ"})
ax.set_title("Post-filter Correlation Matrix (top 40 survivors)", fontweight="bold", fontsize=12)
plt.tight_layout()
plt.savefig(OUT_DIR / "01_correlation_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUT_DIR / '01_correlation_matrix.png'}")


# ══════════════════════════════════════════════════════════════════════
# STAGE 2: Univariate Ranking (Mutual Information)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STAGE 2: Univariate Ranking (Mutual Information)")
print("=" * 70)

TOP_K_MI = 50

X_s1 = train_df[stage1_survivors].fillna(0).values
y_s1 = train_df["label"].values

mi_scores = mutual_info_classif(X_s1, y_s1, n_neighbors=5, random_state=42)
spearman_rho = train_df[stage1_survivors].corrwith(train_df["label"], method="spearman")

ranking = pd.DataFrame({
    "feature": stage1_survivors,
    "MI": mi_scores,
    "spearman_rho": spearman_rho.values,
    "abs_rho": spearman_rho.abs().values,
}).sort_values("MI", ascending=False).reset_index(drop=True)
ranking["rank"] = range(1, len(ranking) + 1)

print(f"\n  Top 30 features by Mutual Information:")
print(ranking.head(30).to_string(index=False))

stage2_features = ranking.head(TOP_K_MI)["feature"].tolist()
print(f"\n  Stage 2: Top {TOP_K_MI} features selected by MI")

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
top30 = ranking.head(30)

ax = axes[0]
colors = ["#2962FF" if mi > ranking["MI"].median() else "#90CAF9" for mi in top30["MI"]]
ax.barh(range(len(top30)), top30["MI"], color=colors)
ax.set_yticks(range(len(top30)))
ax.set_yticklabels(top30["feature"], fontsize=8)
ax.set_xlabel("Mutual Information")
ax.set_title("Top 30 Features by MI", fontweight="bold")
ax.invert_yaxis()

ax = axes[1]
colors2 = ["#EF5350" if rho < 0 else "#26A69A" for rho in top30["spearman_rho"]]
ax.barh(range(len(top30)), top30["spearman_rho"], color=colors2)
ax.set_yticks(range(len(top30)))
ax.set_yticklabels(top30["feature"], fontsize=8)
ax.set_xlabel("Spearman ρ with target")
ax.set_title("Spearman Correlation with Label", fontweight="bold")
ax.axvline(0, color="gray", ls="-", lw=0.5)
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(OUT_DIR / "02_mi_ranking.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUT_DIR / '02_mi_ranking.png'}")


# ══════════════════════════════════════════════════════════════════════
# STAGE 3: Walk-Forward Feature Stability
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STAGE 3: Walk-Forward Feature Stability")
print("=" * 70)

WINDOW_SIZE = 2160  # ~3 months at 1h
STEP_SIZE = 720     # ~1 month
MIN_STABLE_FRAC = 0.5
TOP_K_PER_WINDOW = 30

n = len(train_df)
windows = []
start = 0
while start + WINDOW_SIZE <= n:
    windows.append((start, start + WINDOW_SIZE))
    start += STEP_SIZE

print(f"  Walk-forward: {len(windows)} windows of {WINDOW_SIZE} bars, step {STEP_SIZE}")

appearance_count = {f: 0 for f in stage2_features}
window_rankings = {}

for i, (s, e) in enumerate(windows):
    chunk = train_df.iloc[s:e]
    if chunk["label"].nunique() < 2:
        continue
    X_chunk = chunk[stage2_features].fillna(0).values
    y_chunk = chunk["label"].values
    mi = mutual_info_classif(X_chunk, y_chunk, n_neighbors=5, random_state=42)
    top_idx = np.argsort(mi)[-TOP_K_PER_WINDOW:]
    for idx in top_idx:
        appearance_count[stage2_features[idx]] += 1
    window_rankings[i] = {stage2_features[idx]: mi[idx] for idx in range(len(stage2_features))}

total_windows = len(windows)
stability_df = pd.DataFrame({
    "feature": list(appearance_count.keys()),
    "appearances": list(appearance_count.values()),
    "frac": [c / total_windows for c in appearance_count.values()],
}).sort_values("frac", ascending=False).reset_index(drop=True)

stage3_features = stability_df[stability_df["frac"] >= MIN_STABLE_FRAC]["feature"].tolist()
print(f"  Stable features (>= {MIN_STABLE_FRAC*100:.0f}% of windows): {len(stage3_features)} / {len(stage2_features)}")
print(f"\n  Stability ranking:")
print(stability_df.head(40).to_string(index=False))

fig, ax = plt.subplots(figsize=(14, 8))
top_stab = stability_df.head(40)
colors3 = ["#2962FF" if f >= MIN_STABLE_FRAC else "#BBDEFB" for f in top_stab["frac"]]
ax.barh(range(len(top_stab)), top_stab["frac"], color=colors3)
ax.set_yticks(range(len(top_stab)))
ax.set_yticklabels(top_stab["feature"], fontsize=8)
ax.axvline(MIN_STABLE_FRAC, color="red", ls="--", lw=1.5, label=f"Stability threshold ({MIN_STABLE_FRAC*100:.0f}%)")
ax.set_xlabel("Fraction of windows in top-K")
ax.set_title("Walk-Forward Feature Stability", fontweight="bold")
ax.legend(loc="lower right")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(OUT_DIR / "03_wf_stability.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUT_DIR / '03_wf_stability.png'}")


# ══════════════════════════════════════════════════════════════════════
# STAGE 4: Model-Based Pruning (Permutation Importance)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STAGE 4: Model-Based Pruning (Permutation Importance)")
print("=" * 70)

PERM_THRESHOLD = 0.0005

X_train_s4 = train_df[stage3_features].fillna(0).values
y_train_s4 = train_df["label"].values
X_val_s4 = val_df[stage3_features].fillna(0).values
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
print(f"  Best iteration: {model_s4.best_iteration_}")

perm_result = permutation_importance(
    model_s4, X_val_s4, y_val_s4,
    scoring="roc_auc", n_repeats=10, random_state=42, n_jobs=-1,
)

perm_df = pd.DataFrame({
    "feature": stage3_features,
    "importance_mean": perm_result.importances_mean,
    "importance_std": perm_result.importances_std,
}).sort_values("importance_mean", ascending=False).reset_index(drop=True)

print(f"\n  Permutation importance (top 30):")
print(perm_df.head(30).to_string(index=False))

stage4_features = perm_df[perm_df["importance_mean"] > PERM_THRESHOLD]["feature"].tolist()
print(f"\n  Stage 4: {len(stage4_features)} features kept (perm importance > {PERM_THRESHOLD})")

fig, ax = plt.subplots(figsize=(12, 8))
top_perm = perm_df.head(30)
colors4 = ["#2962FF" if imp > PERM_THRESHOLD else "#FFCDD2" for imp in top_perm["importance_mean"]]
ax.barh(range(len(top_perm)), top_perm["importance_mean"], xerr=top_perm["importance_std"],
        color=colors4, capsize=3)
ax.set_yticks(range(len(top_perm)))
ax.set_yticklabels(top_perm["feature"], fontsize=8)
ax.axvline(PERM_THRESHOLD, color="red", ls="--", lw=1.5, label=f"Threshold ({PERM_THRESHOLD})")
ax.axvline(0, color="gray", ls="-", lw=0.5)
ax.set_xlabel("Permutation Importance (AUC drop)")
ax.set_title("Permutation Importance on Validation Set", fontweight="bold")
ax.legend(loc="lower right")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(OUT_DIR / "04_permutation_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUT_DIR / '04_permutation_importance.png'}")


# ══════════════════════════════════════════════════════════════════════
# PIPELINE SUMMARY
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PIPELINE SUMMARY")
print("=" * 70)
print(f"  Stage 0 (Pool):      {len(feature_cols)} features")
print(f"  Stage 1 (Filter):    {len(stage1_survivors)} features")
print(f"  Stage 2 (MI Rank):   {len(stage2_features)} features")
print(f"  Stage 3 (Stability): {len(stage3_features)} features")
print(f"  Stage 4 (Prune):     {len(stage4_features)} features")
print(f"\n  Final feature set:")
for i, f in enumerate(stage4_features):
    mi_val = ranking[ranking["feature"] == f]["MI"].values
    mi_str = f"{mi_val[0]:.4f}" if len(mi_val) > 0 else "N/A"
    stab = stability_df[stability_df["feature"] == f]["frac"].values
    stab_str = f"{stab[0]:.2f}" if len(stab) > 0 else "N/A"
    perm = perm_df[perm_df["feature"] == f]["importance_mean"].values
    perm_str = f"{perm[0]:.4f}" if len(perm) > 0 else "N/A"
    print(f"    {i+1:>2}. {f:<30s}  MI={mi_str}  Stab={stab_str}  Perm={perm_str}")


# ══════════════════════════════════════════════════════════════════════
# FINAL MODEL: Train LGBM on Selected Features
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("FINAL MODEL: LightGBM on Selected Features")
print("=" * 70)

final_features = stage4_features if len(stage4_features) >= 5 else stage3_features[:25]
print(f"  Training with {len(final_features)} features")

X_train_final = train_df[final_features].fillna(0).values
y_train_final = train_df["label"].values
X_val_final = val_df[final_features].fillna(0).values
y_val_final = val_df["label"].values
X_test_final = test_df[final_features].fillna(0).values
y_test_final = test_df["label"].values

LGB_PARAMS = {
    "n_estimators": 1000,
    "learning_rate": 0.02,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "verbose": -1,
    "random_state": 42,
    "n_jobs": -1,
}

final_model = lgb.LGBMClassifier(**LGB_PARAMS)
final_model.fit(
    X_train_final, y_train_final,
    eval_set=[(X_val_final, y_val_final)],
    callbacks=[lgb.early_stopping(50, verbose=False)],
)

probs_val = final_model.predict_proba(X_val_final)[:, 1]
probs_test = final_model.predict_proba(X_test_final)[:, 1]

auc_val = roc_auc_score(y_val_final, probs_val)
auc_test = roc_auc_score(y_test_final, probs_test)

print(f"\n  Best iteration: {final_model.best_iteration_}")
print(f"  Validation AUC: {auc_val:.4f}")
print(f"  Test AUC:       {auc_test:.4f}")
print(f"  AUC gap:        {abs(auc_val - auc_test):.4f}")

# LGBM native importance
lgb_imp = pd.DataFrame({
    "feature": final_features,
    "gain": final_model.feature_importances_,
}).sort_values("gain", ascending=False).reset_index(drop=True)

print(f"\n  LGBM Gain Importance (top 20):")
print(lgb_imp.head(20).to_string(index=False))

# Probability distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, probs, y, title in [
    (axes[0], probs_val, y_val_final, "Validation"),
    (axes[1], probs_test, y_test_final, "Test"),
]:
    ax.hist(probs[y == 0], bins=50, alpha=0.6, label="Label=0", color="#EF5350", density=True)
    ax.hist(probs[y == 1], bins=50, alpha=0.6, label="Label=1", color="#26A69A", density=True)
    sep = probs[y == 1].mean() - probs[y == 0].mean()
    auc = roc_auc_score(y, probs)
    ax.set_title(f"{title} — AUC={auc:.4f}, Sep={sep:.4f}", fontweight="bold")
    ax.set_xlabel("Predicted P(label=1)")
    ax.set_ylabel("Density")
    ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "05_probability_distributions.png", dpi=150, bbox_inches="tight")
plt.close()

# Feature importance plot (final model)
fig, ax = plt.subplots(figsize=(10, 8))
top_lgb = lgb_imp.head(25)
ax.barh(range(len(top_lgb)), top_lgb["gain"], color="#2962FF", alpha=0.8)
ax.set_yticks(range(len(top_lgb)))
ax.set_yticklabels(top_lgb["feature"], fontsize=9)
ax.set_xlabel("LightGBM Gain")
ax.set_title(f"Final Model Feature Importance (AUC val={auc_val:.4f}, test={auc_test:.4f})", fontweight="bold")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(OUT_DIR / "06_final_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()

# Classification report
print(f"\n  Classification Report (Validation):")
preds_val = (probs_val > 0.5).astype(int)
print(classification_report(y_val_final, preds_val, target_names=["Down", "Up"]))

print(f"  Classification Report (Test):")
preds_test = (probs_test > 0.5).astype(int)
print(classification_report(y_test_final, preds_test, target_names=["Down", "Up"]))

# Win rate by probability bucket
print(f"\n  Win Rate by Probability Bucket (Validation):")
bucket_edges = [0.0, 0.35, 0.40, 0.45, 0.48, 0.50, 0.52, 0.55, 0.60, 0.65, 1.0]
val_analysis = pd.DataFrame({"prob": probs_val, "label": y_val_final})
val_analysis["bucket"] = pd.cut(val_analysis["prob"], bins=bucket_edges)
bucket_stats = val_analysis.groupby("bucket", observed=False).agg(
    count=("label", "count"),
    win_rate=("label", "mean"),
).reset_index()
print(bucket_stats.to_string(index=False))

# Pipeline summary figure
fig, ax = plt.subplots(figsize=(10, 5))
stages = ["Pool\n(196)", "Filter\n(" + str(len(stage1_survivors)) + ")",
          "MI Rank\n(" + str(len(stage2_features)) + ")",
          "Stability\n(" + str(len(stage3_features)) + ")",
          "Prune\n(" + str(len(stage4_features)) + ")"]
counts = [len(feature_cols), len(stage1_survivors), len(stage2_features), len(stage3_features), len(stage4_features)]
colors_pipe = ["#E0E0E0", "#BBDEFB", "#90CAF9", "#42A5F5", "#1565C0"]
bars = ax.bar(range(len(stages)), counts, color=colors_pipe, edgecolor="white", linewidth=2)
for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            str(count), ha="center", fontweight="bold", fontsize=12)
ax.set_xticks(range(len(stages)))
ax.set_xticklabels(stages, fontsize=10)
ax.set_ylabel("Feature Count")
ax.set_title("Feature Selection Pipeline — Filter → Rank → Validate → Prune", fontweight="bold", fontsize=12)
plt.tight_layout()
plt.savefig(OUT_DIR / "07_pipeline_summary.png", dpi=150, bbox_inches="tight")
plt.close()

# Save results
results = {
    "pipeline": {
        "stage0_pool": len(feature_cols),
        "stage1_filter": len(stage1_survivors),
        "stage2_mi_rank": len(stage2_features),
        "stage3_stability": len(stage3_features),
        "stage4_prune": len(stage4_features),
    },
    "final_features": final_features,
    "val_auc": float(auc_val),
    "test_auc": float(auc_test),
    "best_iteration": int(final_model.best_iteration_),
    "lgb_params": LGB_PARAMS,
}
results_path = OUT_DIR / "results.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n  Results saved to: {results_path}")

# Save feature list
features_path = OUT_DIR / "selected_features.csv"
pd.DataFrame({"feature": final_features}).to_csv(features_path, index=False)
print(f"  Features saved to: {features_path}")

# Save ranking
ranking_path = OUT_DIR / "full_ranking.csv"
ranking.to_csv(ranking_path, index=False)
print(f"  Full ranking saved to: {ranking_path}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
