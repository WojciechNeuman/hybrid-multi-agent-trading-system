# Agent Communication — `notebooks_v2` Ensemble (2026-06-11)

How the four base agents and the meta-learner exchange data. This reflects the
**current** `notebooks_v2/` pipeline (not the older `notebooks/` DRL+GP design in
`system_architecture.md`, which is stale).

```mermaid
flowchart TD
    PARQUET["BTCUSDT_1h_unified.parquet\n74,366 bars · 292 cols\n2017-11 → 2026-05"]:::data

    subgraph BASE["Base agents — each emits a calibrated p(up) stream"]
        direction TB
        LGBM["01_lgbm_v1\nLightGBM (tree)\nlabel = next-bar direction\nrolling 1y WFO, monthly step\n11 Boruta-locked features"]:::model
        MAMBA["02_mamba_v1 (Colab)\nMamba SSM\nlabel = TBM ±2σ/24h\nsliding WFO\n39 Boruta-vs-TBM features"]:::model
        TCN["03_tcn_v1\nTCN (causal conv, multitask)\nlabel = TBM ±2σ/24h\ntrain≤2022 → predict 2023+\n32 features + fracdiff"]:::model
        PATCH["05_patchtst_v1\nPatchTST (transformer)\nlabel = TBM ±2σ/24h + temp-scaling\ntrain≤2022 → predict 2023+\n32 features + fracdiff"]:::model
    end

    PARQUET --> LGBM & MAMBA & TCN & PATCH

    subgraph ART["Per-agent artifacts  artifacts/notebooks_v2/NN_*/"]
        direction LR
        A1["01_lgbm/\nwfo_probs.npy (2018→2026)\noos_probs.npy\nresults.json (BEST grid)"]:::art
        A2["02_mamba/\nwfo_probs.npy (2022→2026)\noos_probs.npy\nresults.json"]:::art
        A3["03_tcn/\nwfo_probs.npy (2023→2026)\noos_probs.npy\nresults.json"]:::art
        A5["05_patchtst/\nwfo_probs.npy (2023→2026)\noos_probs.npy\nresults.json"]:::art
    end

    LGBM --> A1
    MAMBA --> A2
    TCN --> A3
    PATCH --> A5

    subgraph MET
        direction TB
        LOAD["_load_model_probs()\nprefer wfo_probs ▸ oos_probs\nauto-set META_TRAIN_START\n= 2nd-earliest model start +1mo"]:::meta
        ENS["Ensemble p(up)\n= mean of available models / bar\nprimary_long  if >0.56 & n≥2\nprimary_short if <0.44 & n≥2"]:::meta
        TBM["Meta-label (meta_y)\nTBM on the primary SIDE\nTP=2.5·ATR / SL=1.5·ATR / 48h"]:::meta
        FIT["Meta-classifier (LightGBM)\nfeatures: 4 model probs +\nensemble + n_models + regime ctx\nWFO expanding, 3-mo step, 48h embargo"]:::meta
        SIZE["Sizing\nposition = side·(meta_prob−0.5)·2\ngate: meta_prob > 0.55"]:::meta
    end

    A1 & A2 & A3 & A5 --> LOAD --> ENS --> TBM --> FIT --> SIZE
    PARQUET -. regime + ATR context .-> FIT
    SIZE --> BT["Sized backtest\nOOS 2024-05-31 → 2026-05-16\nSharpe / Return / MaxDD"]:::exec

    classDef data fill:#1f2937,stroke:#9ca3af,color:#fff;
    classDef model fill:#0b3d5c,stroke:#2962FF,color:#fff;
    classDef art fill:#374151,stroke:#9E9E9E,color:#fff;
    classDef meta fill:#14532d,stroke:#26A69A,color:#fff;
    classDef exec fill:#7f1d1d,stroke:#EF5350,color:#fff;
```

## Communication contract

| Edge | Payload | Format | Coverage required |
|---|---|---|---|
| base → artifact | `p(up)` per bar | `wfo_probs.npy` + `wfo_index.npy` (int64 ns) | **Must span meta-train + OOS** |
| base → artifact | OOS `p(up)` | `oos_probs.npy` + `oos_index.npy` | OOS only |
| base → meta | tuned trading rule | `results.json["best_params"]` | **Currently unused by meta** ⚠ |
| artifact → meta | aligned probs | reindexed to parquet index, NaN where uncovered | ≥2 models per bar to fire |
| meta → exec | continuous position | `[-1,+1]` | OOS window |

⚠ **Key gap:** each base agent also publishes a *tuned* `best_params` trading rule
(thresholds, SL/TP, holds) that produced its standalone backtest edge — but the meta
ignores it and re-derives signals from raw probabilities with one fixed untuned rule.
See `docs/meta_analysis_2026-06-11.md` §3.
