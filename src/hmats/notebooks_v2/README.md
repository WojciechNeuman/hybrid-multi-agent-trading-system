# notebooks_v2 — Unified Thesis Pipeline

This folder is the **canonical, final** set of notebooks for the master's thesis.
Each notebook has a single, well-defined responsibility and they form a strict
execution chain.

---

## Runtime modules live in `hmats.mas` — notebooks are *generated* from them

The five runtime modules that implement the multi-agent pipeline were moved out of
this folder into a proper importable package, **`src/hmats/mas/`**:

```
src/hmats/mas/
├── __init__.py            # re-exports the engine API + the four submodules
├── mas07.py               # engine + Coordinator (the single source of truth)
├── rule_agents.py         # trend / mean-reversion / vol-breakout rule agents
├── crossasset_agent.py    # learned cross-asset / sentiment / flow agent
├── agent_eval.py          # evaluation helpers
└── coordinators.py        # alternative capital-allocation rules
```

Import it like any other project package (same convention as `hmats.data`):

```python
from hmats.mas import mas07, rule_agents, crossasset_agent, agent_eval, coordinators
```

`rule_agents`, `crossasset_agent`, `agent_eval` do `from .mas07 import ...`;
`coordinators` does `from . import mas07 as m`. `mas07.AGENT_DIR` and every artifact
path are **unchanged** — only the modules' on-disk location and their intra-package
imports moved.

### The non-obvious coupling: notebooks inline these modules' *source*

Notebooks `05_rule_agents_v1`, `06_crossasset_v1` and `07_multi_agent_v1` are
**self-contained** — they import nothing at runtime. They are *generated* by the
builder scripts that stay in this folder:

| Builder | Reads (from `hmats/mas/`) | Writes (here) |
|---------|---------------------------|----------------|
| `_build_rule_nb.py`       | `mas07.py` + `rule_agents.py`      | `05_rule_agents_v1.ipynb` |
| `_build_crossasset_nb.py` | `mas07.py` + `crossasset_agent.py` | `06_crossasset_v1.ipynb`  |
| `_build_mas07_nb.py`      | `mas07.py` + `agent_eval.py` + `coordinators.py` | `07_multi_agent_v1.ipynb` |

`_nbinline.py` is the shared inliner. It reads each module's **source text** from
`MODDIR = src/hmats/mas/`, then:

- strips the module docstring, `from __future__ import …`, and the `if __name__ ==
  "__main__"` trailer;
- strips the **engine imports** so the engine (inlined once, in scope) isn't
  re-imported — it removes `from .mas07 import (...)` / `from .mas07 import …`,
  `from . import mas07 as m`, the legacy bare-`mas07` forms, and any leftover
  `import sys` / `sys.path.insert(...)` hack;
- rewrites the `m.` alias prefix (`coordinators` / `agent_eval`) back to bare names.

**Therefore: if you move these modules again or change how they import each other,
update both `MODDIR` and the strip-regexes in `_nbinline.py`, or notebook generation
will silently inline broken (or duplicate-import) cells.**

### Regenerating the notebooks

```bash
cd src/hmats/notebooks_v2
python _build_rule_nb.py
python _build_crossasset_nb.py
python _build_mas07_nb.py
```

Each builder reads the live module source, so editing logic in `hmats/mas/` and
re-running the builder is the canonical way to update the notebooks. After
regenerating, the inlined cells contain **no** `from .mas07` / `import mas07` /
`sys.path` lines and parse cleanly; `05_` was executed end-to-end to confirm runtime
self-containment is intact.

---

## Execution Order

```
00_data_ingestion_v1
  → 01_lgbm_v1 · 02_mamba_v1 · 03_tcn_v1 · 04_patchtst_v1   (learned agents)
  → 05_rule_agents_v1 · 06_crossasset_v1                     (rule + cross-asset agents)
  → 07_multi_agent_v1   (coordinator: merges all 8 agents, loads their artifacts)
```

The final pipeline is exactly eight notebooks, contiguous `00`–`07`. Older
variants (LGBM copies, `02_mamba_v2`, the Colab A100 mamba, TCN copies, the
averaging meta-learner `04_meta_learning_v1`, `05_patchtst_v1_40pp_profit`, and
the signal bake-off `06_ensemble_v1`) were archived to
`lab/notebooks_v2_archive/`.

---

## Notebook number ↔ artifact directory decoupling

The renumbered notebooks intentionally **do not** share the numeric prefix of the
artifact directory they write to. The artifact dirs under
`artifacts/notebooks_v2/` were left at their original names so the Colab notebooks
and `mas07.AGENT_DIR` keep working without edits:

| Notebook | Artifact directory |
|----------|--------------------|
| `04_patchtst_v1`   | `05_patchtst` |
| `05_rule_agents_v1`| `08_trend`, `08_meanrev`, `08_volbreak` |
| `06_crossasset_v1` | `09_crossasset` |

`00`–`03` (`01_lgbm`, `02_mamba`, `03_tcn`) still match. This decoupling is
deliberate: renaming the artifact dirs would break `mas07.AGENT_DIR` and the
save paths baked into the Colab notebooks, so only the notebook filenames were
renumbered.

---

## Notebooks

### `00_data_ingestion_v1.ipynb` — Unified Parquet
**Run once. Prerequisite for all model notebooks.**

Merges all feature sources into a single parquet:

| Input | Description |
|-------|-------------|
| `data/features/BTCUSDT_1h_features.parquet` | 199 V1 features + directional `label` |
| `data/features/BTCUSDT_1h_v4_features.parquet` | 25 V4 microstructure/regime features |
| `data/features/BTCUSDT_1h_structural.parquet` | 39 structural features |
| `data/features/BTCUSDT_1h_microstructure.parquet` | 4 complexity features (Roll, Amihud, etc.) |
| `data/raw/BTCUSDT_1h.parquet` | Raw OHLCV — supplies `open`, `high`, `low`, `volume` |

**Output:** `data/features/BTCUSDT_1h_unified.parquet`  
~145 MB, 74 000 rows (2017-11-15 → 2026-05-16), ~270 columns.

---

### `01_lgbm_v1.ipynb` — LightGBM Directional Model
**Local machine. Runtime: ~15 min.**

- **Model:** LightGBM binary classifier predicting `label` (next-bar direction).
- **Method:** M1Y expanding walk-forward, monthly step (mirrors the proven v12 approach).
- **Features:** 11 Boruta-selected features (locked from notebook v12 experiments).
- **Grid search:** 2022–2023 validation window, exhaustive ATR-bracket parameter sweep.
- **OOS:** 2024-01-01 onward.

**Artifacts → `artifacts/notebooks_v2/01_lgbm/`:**
```
oos_probs.npy   — float32 array of P(Up), aligned to OOS bar index
oos_index.npy   — int64 nanosecond timestamps (pd.to_datetime compatible)
model.txt       — LightGBM model (last WFO fold, text format)
results.json    — Sharpe, return, MaxDD, win-rate, trades, monthly stats
01_equity_drawdown.png
02_monthly_returns.png
03_monthly_heatmap.png
```

---

### `02_mamba_v1.ipynb` — Mamba SSM (Google Colab)
**Colab: A100 GPU (~30–45 min) or T4 (~60–90 min with reduced settings).**

- **Model:** Mamba state-space sequence classifier (pure-PyTorch, chunked SSM scan).
- **Target:** Directional `label` — same as LGBM, enabling direct comparison.
- **Method:** Semi-annual expanding walk-forward, QuantileTransformer per fold.
- **Features:** 39 Boruta-against-TBM validated features (baked in, skip Boruta by default).
- **Grid search:** Same ATR-bracket grid as LGBM → comparable Sharpe/return/MaxDD.
- **Input:** Upload only `BTCUSDT_1h_unified.parquet` to Colab (single file).

**T4 settings** (in config cell): `BATCH=512`, `D_STATE=8`, `STRIDE=2`, `RETRAIN_MONTHS=12`.

**After Colab run:** download `mamba_artifacts.zip`, unzip, copy contents into:
```
artifacts/notebooks_v2/02_mamba/
```

**Artifacts (same schema as LGBM):**
```
oos_probs.npy, oos_index.npy, model_lastfold.pt, results.json, trades_wfees.csv, PNGs
```

---

### `03_tcn_v1.ipynb` — Temporal Convolutional Network
**Local machine (GPU/MPS/CPU). Runtime: ~20–40 min.**

- **Model:** Multi-task TCN — classifies direction (Up/Down) and regresses forward volatility.
- **Label:** Triple Barrier Method ±2σ volatility bands, 24-bar vertical barrier.
- **Architecture:** 4 causal blocks (channels [64,64,64,64], dilation 1/2/4/8, kernel 3).
- **Splits:** Train ≤ 2022-12-31 · Grid-val 2023 · OOS 2024-01-01+ (aligned with LGBM/Mamba).
- **Features:** 11 LGBM core + 11 extended V1 + 7 V4 + 4 structural + 1 fractionally-differenced log-price.

**Artifacts → `artifacts/notebooks_v2/03_tcn/`:**
```
oos_probs.npy, oos_index.npy, model.pt, results.json, PNGs
```

---

### `04_meta_learning_v1.ipynb` — Three-Agent Ensemble Supervisor
**Local machine. Run after 01, 02, 03.**

- **Input:** Loads `oos_probs.npy` from each model directory. Works with 2 of 3 models if Mamba is not yet available.
- **Ensemble signal:** mean(lgbm_p_up, mamba_p_up, tcn_p_up) > 0.56 → Long; < 0.44 → Short.
- **Meta-target:** TBM label at each signal bar (TP=1, SL=0) with SL=1.5×ATR, TP=2.5×ATR, max_hold=48h.
- **Meta-model:** LightGBM binary classifier, features = model probs + regime context (ATR, Hurst, RSI, vol ratio, etc.).
- **Walk-forward:** Expanding window, 3-month step, 48h embargo.
- **Sizing:** position = side × (meta_prob − 0.5) × 2 when meta_prob > 0.55.
- **Comparison figure:** overlays all model equity curves + meta-agent on the same axes.

**Artifacts → `artifacts/notebooks_v2/04_meta/`:**
```
results.json, 01_equity_comparison.png, 02_monthly_returns.png, 03_feature_importance.png
```

---

## Artifact Contract

Every model notebook writes these files in a consistent schema so the meta-learning
notebook can load any of them without knowing which model produced them:

```
artifacts/notebooks_v2/{model_dir}/
├── oos_probs.npy      # np.float32, shape (n_oos_bars,)
├── oos_index.npy      # np.int64, nanosecond timestamps → pd.to_datetime(...)
├── model.*            # native model file
└── results.json       # standard fields (see below)
```

**Standard `results.json` fields:**

```json
{
  "notebook": "0X_model_v1",
  "created":  "<ISO timestamp>",
  "oos_period": "2024-01-01→...",
  "oos_auc": 0.5xx,
  "best_params": { ... },
  "backtest_wfees": {
    "n_trades": N, "n_long": L, "n_short": S,
    "win_rate": 0.xx, "total_ret": 0.xx, "sharpe": 0.xx, "maxdd": -0.xx
  },
  "backtest_0fee":  { ... },
  "monthly": { "mean_pct": x.x, "positive_months": N, "total_months": N },
  "artifacts": { "oos_probs": "oos_probs.npy", "oos_index": "oos_index.npy", ... }
}
```

---

## Split Alignment

All three models use the **same OOS period** (2024-01-01 onward):

| Period | Dates | Purpose |
|--------|-------|---------|
| Pre-history | 2017-11 → 2021-12 | Training data |
| Grid-val | 2022-01 → 2023-12 | Backtest parameter selection |
| OOS | 2024-01-01 → latest | True out-of-sample evaluation |

---

## What Changed from notebooks/ (v1)

| Old | New |
|-----|-----|
| Load 2–3 separate parquets per notebook | Load **one** `BTCUSDT_1h_unified.parquet` |
| Model notebooks versioned to v12, v1, v4… | All reset to `_v1` in this folder |
| Meta-learning re-ran all agents live | Meta-learning **loads pre-computed artifacts** |
| DRL (PPO) and GP included in meta-ensemble | Removed — only LGBM, Mamba, TCN |
| TCN OOS starts 2024-11-10 (short window) | TCN OOS aligned to 2024-01-01 |
| Mamba needed 3 separate parquets on Colab | Mamba needs 1 file on Colab |
| No standard artifact schema | All models write `oos_probs.npy` + `oos_index.npy` |
