"""data/load_features.py — Load all feature parquets and (optionally) merge them.

Outputs:
  df_1h  — 282 features, 1h bars, aligned index (2017-11-15 → 2026-05-16)
  df_5m  — 67 features, 5m bars, aligned index (2017-08-25 → 2026-05-27)

Run as a script or execute cells in Jupyter / VSCode.
"""

# %% [markdown]
# # Load Feature Parquets
# Merges all feature files for 1h and 5m timeframes into two clean DataFrames.

# %% ── Imports ─────────────────────────────────────────────────────────────────
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
FEAT = REPO / "data" / "features"

# %% ── Load 1h files ──────────────────────────────────────────────────────────
print("Loading 1h feature files ...")

df_1h_v1     = pd.read_parquet(FEAT / "BTCUSDT_1h_features.parquet")
df_1h_v3     = pd.read_parquet(FEAT / "BTCUSDT_1h_v3_features.parquet")
df_1h_v4     = pd.read_parquet(FEAT / "BTCUSDT_1h_v4_features.parquet")
df_1h_struct = pd.read_parquet(FEAT / "BTCUSDT_1h_structural.parquet")

# Normalise index timezone (all to tz-naive UTC)
for _df in (df_1h_v1, df_1h_v3, df_1h_v4, df_1h_struct):
    if _df.index.tz is not None:
        _df.index = _df.index.tz_convert("UTC").tz_localize(None)

# Merge on inner join → keeps only rows present in all four files
df_1h = (
    df_1h_v1
    .join(df_1h_v3,     how="inner")
    .join(df_1h_v4,     how="inner")
    .join(df_1h_struct, how="inner")
)

print(
    f"  df_1h : {df_1h.shape[0]:>7,} rows × {df_1h.shape[1]} cols"
    f"  |  {df_1h.index[0].date()} → {df_1h.index[-1].date()}"
)

# %% ── Load 5m files ──────────────────────────────────────────────────────────
print("Loading 5m feature files ...")

df_5m_feat   = pd.read_parquet(FEAT / "BTCUSDT_5m_features_5m.parquet")
df_5m_struct = pd.read_parquet(FEAT / "BTCUSDT_5m_structural_clean.parquet")

for _df in (df_5m_feat, df_5m_struct):
    if _df.index.tz is not None:
        _df.index = _df.index.tz_convert("UTC").tz_localize(None)

df_5m = df_5m_feat.join(df_5m_struct, how="inner")

print(
    f"  df_5m : {df_5m.shape[0]:>7,} rows × {df_5m.shape[1]} cols"
    f"  |  {df_5m.index[0].date()} → {df_5m.index[-1].date()}"
)

# %% ── Quick sanity checks ────────────────────────────────────────────────────
import json

registry_path = FEAT / "feature_registry_v3_2026-06-02.json"
with open(registry_path) as f:
    registry = json.load(f)

expected_1h = registry["summary"]["total_feature_columns_1h"]
expected_5m = registry["summary"]["total_feature_columns_5m"]

# label col is included in df_1h, subtract it for feature count
feature_cols_1h = [c for c in df_1h.columns if c != "label"]
feature_cols_5m = [c for c in df_5m.columns if c not in ("tbm_label", "fh_label", "open", "high", "low", "close")]

print(f"\nSanity check:")
print(f"  1h features in registry : {expected_1h}   in DataFrame : {len(feature_cols_1h)}")
print(f"  5m features in registry : {expected_5m}   in DataFrame : {len(feature_cols_5m)}")

nan_1h = df_1h[feature_cols_1h].isna().mean().sort_values(ascending=False).head(5)
nan_5m = df_5m[feature_cols_5m].isna().mean().sort_values(ascending=False).head(5)

print(f"\nTop NaN rates — 1h:\n{nan_1h.to_string()}")
print(f"\nTop NaN rates — 5m:\n{nan_5m.to_string()}")

# %% ── Feature group lookup ───────────────────────────────────────────────────
def features_in_group(group_name: str) -> list[str]:
    """Return feature list for a named group from the registry."""
    groups = registry["feature_groups"]
    if group_name not in groups:
        available = ", ".join(sorted(groups.keys()))
        raise KeyError(f"Unknown group '{group_name}'. Available: {available}")
    return groups[group_name]["features"]


def features_by_prefix(prefix: str, df: pd.DataFrame) -> list[str]:
    """Return all columns in df whose name starts with prefix."""
    return [c for c in df.columns if c.startswith(prefix)]


# %% ── Example usage ──────────────────────────────────────────────────────────
# Uncomment as needed:

# X_1h = df_1h[feature_cols_1h].dropna()
# y_1h = df_1h.loc[X_1h.index, "label"]

# struct_feats = features_in_group("1h_struct_structure")
# X_struct = df_1h[struct_feats].dropna()

# X_5m = df_5m[feature_cols_5m].dropna()
# y_5m = df_5m.loc[X_5m.index, "tbm_label"]

print("\nDone. Use df_1h and df_5m in subsequent cells.")
