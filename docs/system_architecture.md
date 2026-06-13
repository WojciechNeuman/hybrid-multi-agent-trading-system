# Hybrid Multi-Agent Trading System — Architecture (notebooks_v2)

> **Status 2026-06-11.** This document describes the **as-built `notebooks_v2/` pipeline**:
> four base agents (LGBM, Mamba, TCN, PatchTST) feeding a LightGBM meta-learner (`04_meta`),
> evaluated on the unified 2-year OOS window **2024-05-31 → 2026-05-16**.
> The earlier DRL+GP / `08_meta` / OOS-2024-01 design has been retired from the live ensemble
> (DRL and GP remain documented in the thesis as standalone experiments only).
> Companion charts: `docs/diagrams/{agent_communication,feature_pipeline,meta_dataflow}.md`.
> Full diagnosis of current meta performance: `docs/meta_analysis_2026-06-11.md`.

## Full Pipeline

```mermaid
flowchart TD
    %% ── Raw data sources ─────────────────────────────────────────────────────
    subgraph DATA["Data Ingestion (00_data_ingestion_v1.ipynb)"]
        direction TB
        BINANCE["Binance REST API\n/api/v3/klines\n/fapi/v1/markPriceKlines\n/fapi/v1/indexPriceKlines"]
        CG["CoinGecko API\nMarket caps / dominance"]
        AFG["Alternative.me\nFear & Greed Index"]

        BINANCE --> OHLCV["BTC/USDT 1h OHLCV\n~74,366 bars\n2017-11 → 2026-05"]
        BINANCE --> TAKER["Taker-volume feed\nmark price · index price"]
        CG --> MCAP["Market caps · dominance\nalt-coin breadth (lagged 1d)"]
        AFG --> SENT["Fear & Greed (daily, lagged 1d)\nsentiment z-score"]
    end

    %% ── Unified feature parquet ──────────────────────────────────────────────
    subgraph FEAT["Feature Engineering → BTCUSDT_1h_unified.parquet"]
        direction TB
        V1["V1 Technical Library\n196 features / 22 groups"]
        V3["V3 External Library\n20 features (daily, lagged 1d)"]
        V4["V4 Microstructure Library\n25 features"]
        STRUCT["Structural Library\n39 features"]
        MICRO["Microstructure Library\n4 features (Roll/Amihud/SampEn)"]

        OHLCV & TAKER & MCAP & SENT --> V1 & V3 & V4 & STRUCT & MICRO
        V1 & V3 & V4 & STRUCT & MICRO --> POOL["285 ML-eligible features\n(leak audit: mkt_total_mcap_chg_24h excluded)"]
    end

    %% ── Per-agent feature selection ──────────────────────────────────────────
    subgraph FSEL["Agent-Specific Feature Selection"]
        direction LR
        FS_LGBM["LGBM\n4-stage Boruta\n→ ~15 features"]
        FS_MAMBA["Mamba\nBoruta-vs-TBM\n→ 39 features (V1+V4)"]
        FS_TCN["TCN\ncurated sequential diet\n→ 32 features + fracdiff"]
        FS_PATCH["PatchTST\nsame 32-feature diet as TCN\n(controlled comparison)"]
    end

    POOL --> FS_LGBM & FS_MAMBA & FS_TCN & FS_PATCH

    %% ── Base agents ──────────────────────────────────────────────────────────
    subgraph AGENTS["Base Agents — Walk-Forward OOS p(up), unified test 2024-05-31 → 2026-05-16"]
        direction TB

        LGBM["01_lgbm_v1\nLightGBM (tree)\nlabel = next-bar direction\nrolling 1y WFO, monthly step\nOOS AUC 0.536 · +12.0% · Sharpe 0.24"]

        MAMBA["02_mamba_v1 (Colab A100)\nMamba SSM (pure-torch scan)\nlabel = TBM ±2σ/24h\nsliding 12-mo WFO, 3-mo step\nOOS AUC 0.532 · −27.0% · Sharpe −0.52"]

        TCN["03_tcn_v1\nDilated causal TCN (multitask)\nlabel = TBM ±2σ/24h\ntrain ≤2022 → predict 2023+\nOOS AUC 0.513 · +73.8% · Sharpe 1.14"]

        PATCH["05_patchtst_v1\nPatchTST (transformer, temp-scaled)\nlabel = TBM ±2σ/24h\ntrain ≤2022 → predict 2023+\nOOS AUC 0.509 · +36.1% · Sharpe 0.87"]
    end

    FS_LGBM --> LGBM
    FS_MAMBA --> MAMBA
    FS_TCN  --> TCN
    FS_PATCH --> PATCH

    %% ── OOS signal artefacts ─────────────────────────────────────────────────
    subgraph SIG["Per-agent artifacts  artifacts/notebooks_v2/NN_*/"]
        direction LR
        S_LGBM["01_lgbm/\nwfo_probs.npy (2018→2026)\noos_probs.npy · results.json"]
        S_MAMBA["02_mamba/\nwfo_probs.npy (2022→2026)\noos_probs.npy · results.json"]
        S_TCN["03_tcn/\nwfo_probs.npy (2023→2026)\noos_probs.npy · results.json"]
        S_PATCH["05_patchtst/\nwfo_probs.npy (2023→2026)\noos_probs.npy · results.json"]
    end

    LGBM --> S_LGBM
    MAMBA --> S_MAMBA
    TCN  --> S_TCN
    PATCH --> S_PATCH

    %% ── Meta-learning supervisor ─────────────────────────────────────────────
    subgraph META["Meta-Learning Supervisor (04_meta_learning_v1.ipynb)"]
        direction TB

        LOAD["Signal load + align\n_load_model_probs(): prefer wfo_probs ▸ oos_probs\nMETA_TRAIN_START = 2nd-earliest start +1mo"]

        ENS["Ensemble p(up)\n= mean of available models / bar\nprimary_long if >0.56 & n≥2\nprimary_short if <0.44 & n≥2"]

        TBM["Triple Barrier meta-label (meta_y)\nTP 2.5·ATR / SL 1.5·ATR / 48h\non the primary SIDE"]

        FIT["Meta-Classifier (LightGBM)\nfeatures: 4 model probs + ensemble\n+ n_models + regime context\nexpanding WFO, 3-mo step, 48h embargo\nOOS AUC 0.464 (< 0.5) ⚠"]

        SIZE["Position Sizing (capital bug fixed)\nposition = side·(meta_prob−0.5)·2\ngate: meta_prob > 0.55"]
    end

    S_LGBM & S_MAMBA & S_TCN & S_PATCH --> LOAD
    POOL -. regime + ATR context .-> FIT
    LOAD --> ENS --> TBM --> FIT --> SIZE

    %% ── Execution & evaluation ───────────────────────────────────────────────
    subgraph EXEC["Execution & Evaluation"]
        direction LR
        BACKTEST["Sized backtest\ntaker 0.05% · maker 0%\nshort funding +0.00077%/h"]
        METRICS["Metrics\nSharpe · Return · MaxDD\nWin Rate · Trades · per-regime"]
        BENCH["Benchmark\nBTC Buy-and-Hold"]
    end

    SIZE --> BACKTEST --> METRICS
    BENCH --> METRICS
```

## Agent Signal Types

| Agent | Artifact signal | Range | Paradigm | Label |
|-------|-----------------|-------|----------|-------|
| 01 LGBM | `lgbm_p_up` | [0, 1] continuous | Gradient boosting (tabular) | next-bar direction |
| 02 Mamba | `mamba_p_up` | [0, 1] continuous | Selective state-space (SSM) | TBM ±2σ / 24h |
| 03 TCN | `tcn_p_up` | [0, 1] continuous | Dilated causal CNN (multitask) | TBM ±2σ / 24h |
| 05 PatchTST | `patch_p_up` | [0, 1] continuous (temp-scaled) | Patch transformer | TBM ±2σ / 24h |
| **04 Meta** | **position** | **[−1, +1] continuous** | **Meta-labeling (LightGBM)** | TBM on primary side |

> ⚠ **Label heterogeneity:** LGBM predicts a *next-bar direction* event; the three deep
> models predict a *TBM ±2σ/24h* event. `ensemble_p_up = mean(...)` averages probabilities
> of different events — a known limitation (analysis §3c, task T4).

## Walk-Forward OOS Configuration

| Agent | Train scheme | Step | wfo_probs coverage | Reproducible? |
|-------|--------------|------|--------------------|---------------|
| 01 LGBM | expanding 1y rolling WFO | monthly | 2018 → 2026 | yes (`random_state`) |
| 02 Mamba | sliding 12-mo WFO | 3 months | 2022 → 2026 | **no — torch unseeded (T2)** |
| 03 TCN | chronological (train ≤2022) | — | 2023 → 2026 | mostly (DataLoader gen — T3) |
| 05 PatchTST | chronological (train ≤2022) | — | 2023 → 2026 | mostly (DataLoader gen — T3) |
| 04 Meta | expanding WFO on signals | 3 months | trains <2024-05-31, tests after | yes (`random_state`) |

## Current Status & Known Issues (2026-06-11)

The meta-learner currently **underperforms every base model** (OOS AUC 0.464 < 0.5, return −4.1%).
This is an *integration*-layer problem, not a base-model problem (TCN +73.8% / PatchTST +36.1%
standalone). Three compounding causes, with fixes tracked in `docs/TASKS_meta_overhaul.md`:

1. **Regime memorisation** — top meta feature is `halving_cycle_pos` (monotonic time index) → **T7**.
2. **Short-biased primary signal** — base grids tuned on a 2022→2024-05 bear/chop window → **T6**.
3. **Discarded tuned edge + mixed label semantics** — meta ignores each agent's `best_params`
   and averages heterogeneous-label probabilities → **T4 / T5**.

The −100% sizing wipeout (capital-accounting bug) was fixed on 2026-06-11 (**T1**).
