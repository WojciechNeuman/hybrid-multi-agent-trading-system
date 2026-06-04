# Hybrid Multi-Agent Trading System — Architecture Diagram

## Full Pipeline

```mermaid
flowchart TD
    %% ── Raw data sources ─────────────────────────────────────────────────────
    subgraph DATA["Data Ingestion (00_data_ingestion_v3.ipynb)"]
        direction TB
        BINANCE["Binance REST API\n/api/v3/klines\n/fapi/v1/markPriceKlines\n/fapi/v1/indexPriceKlines"]
        CG["CoinGecko API\nMarket caps / dominance"]
        AFG["Alternative.me\nFear & Greed Index"]
        YF["yfinance\nS&P 500 benchmark"]

        BINANCE --> OHLCV["BTC/USDT 1h OHLCV\n74,366 bars\n2017-11 → 2026-05"]
        BINANCE --> TAKER["Taker-volume feed\nmark price · index price"]
        CG --> MCAP["Market caps · dominance\nalt-coin breadth"]
        AFG --> SENT["Fear & Greed (daily)\nsentiment z-score"]
        YF --> SPY["SPY daily\nbenchmark"]
    end

    %% ── Feature libraries ────────────────────────────────────────────────────
    subgraph FEAT["Feature Engineering"]
        direction TB
        V1["V1 Technical Library\n196 features / 22 groups\nreturns · volatility · MA · oscillators\nIchimoku · SuperTrend · Fibonacci\ncalendar · regime · statistical"]
        V3["V3 External Library\n21 features\ncross-asset · market-structure\nsentiment · microstructure proxies"]
        V4["V4 Microstructure Library\n25 features\nTFI · VWAP · taker price premium\nHurst · ADF · fracdiff · sideways-flag"]
        STRUCT["Structural Library\n39 features\nswing anchors · POC · anchored-VWAP\nBollinger-Keltner squeeze · MTF alignment"]

        OHLCV & TAKER & MCAP & SENT --> V1 & V3 & V4 & STRUCT
    end

    %% ── Per-agent feature selection ──────────────────────────────────────────
    subgraph FSEL["Agent-Specific Feature Selection"]
        direction LR
        FS_LGBM["LGBM selection\nRF importance rank\n+ Pearson ρ filter\n→ 50 features from V1"]
        FS_TCN["TCN selection\ncurated 32-feature subset\n9 LGBM-core + 11 V1 + 7 V4\n+ 4 structural + fracdiff"]
        FS_DRL["DRL selection\n19-feature DRL diet\nprice · volatility · regime\nmomentum · time encodings"]
        FS_GP["GP selection\n12 bounded oscillators\nrsi · stoch · mfi · cmf\nmacd · hl_position · obv_z · vol_z"]
        FS_MAMBA["MAMBA selection\n[placeholder]\n~32 features\nsequential architecture"]
    end

    V1 & V3 & V4 & STRUCT --> FS_LGBM & FS_TCN & FS_DRL & FS_GP & FS_MAMBA

    %% ── Base agents ──────────────────────────────────────────────────────────
    subgraph AGENTS["Base Agents (Walk-Forward OOS: 2024-01-01 → 2026-05-31)"]
        direction TB

        LGBM["LGBM v12\nGradient Boosting\nbinary P(up) ∈ [0,1]\nchronological 3-split\n93,312 configs searched\nSharpe 1.661 · +149.6% return"]

        TCN["TCN v0\nDilated Causal CNN\n3-class P(up/flat/down)\nTriple Barrier labels\nMulti-task + vol aux\nSharpe 0.457 · +29.5% return"]

        DRL["DRL PPO v3\nProximal Policy Optimisation\naction ∈ {-1, 0, 1}\nmin_hold=6h hard constraint\nM1Y WFO · 92 folds · 500k ts\nSharpe -3.05 · -92.5% (w/fees)"]

        GP["GP v2\nGenetic Programming (DEAP)\naction ∈ {-1, 0, 1}\nBoolean-gate-only pset\n2-week step · 196 folds\nresults: placeholder"]

        MAMBA["MAMBA v1\nSelective State Space Model\nP(up) ∈ [0,1]\nsequential architecture\nresults: placeholder"]
    end

    FS_LGBM --> LGBM
    FS_TCN  --> TCN
    FS_DRL  --> DRL
    FS_GP   --> GP
    FS_MAMBA --> MAMBA

    %% ── OOS signal artefacts ─────────────────────────────────────────────────
    subgraph SIG["OOS Signal Artefacts (2024-01-01 →)"]
        direction LR
        S_LGBM["lgbm_p_up\nlgbm_oos_signals.parquet"]
        S_TCN["tcn_p_up / tcn_p_down\ntcn_oos_signals.parquet"]
        S_DRL["drl_action {-1,0,1}\ndrl_oos_signals.parquet"]
        S_GP["gp_action {-1,0,1}\ngp_oos_signals.parquet"]
        S_MAMBA["mamba_p_up\nmamba_oos_signals.parquet\n[placeholder]"]
    end

    LGBM --> S_LGBM
    TCN  --> S_TCN
    DRL  --> S_DRL
    GP   --> S_GP
    MAMBA --> S_MAMBA

    %% ── Meta-learning supervisor ─────────────────────────────────────────────
    subgraph META["Meta-Learning Supervisor (08_meta_learning_v1.ipynb)"]
        direction TB

        ALIGN["Signal Alignment\nbuild_signal_df()\ncommon DatetimeIndex\nprimary_signal + primary_side\n2022-01-01 → 2026-05-31"]

        CORR["Diversity Analysis\nSpearman ρ heatmap\nlow ρ ≈ 0.2–0.4\nconfirms agent independence"]

        TBM["Triple Barrier Labelling\nATR-based TP/SL barriers\n48h max hold\nmeta_label ∈ {0, 1}"]

        META_FIT["Meta-Classifier\nLightGBM\nfeatures: all 5 base signals\n+ volatility · regime context\ntrained on 2022–2024 signals\nWFO step = 3 months"]

        SIZE["Position Sizing\npos = sign × meta_prob\ncontinuous ∈ [-1, 1]\nthreshold = 0.55"]
    end

    S_LGBM & S_TCN & S_DRL & S_GP & S_MAMBA --> ALIGN
    V1 & V4 & STRUCT --> ALIGN
    ALIGN --> CORR
    ALIGN --> TBM --> META_FIT --> SIZE

    %% ── Execution & evaluation ───────────────────────────────────────────────
    subgraph EXEC["Execution & Evaluation"]
        direction LR
        BACKTEST["Continuous-Position Backtest\nrun_sized_backtest()\ntaker 0.05% · maker 0%\nshort funding +0.00077%/h"]
        METRICS["Performance Metrics\nSharpe · Return · MaxDD\nWin Rate · Trades · Alpha"]
        BENCH["Benchmarks\nBTC Buy-and-Hold\nS&P 500 (SPY)"]
    end

    SIZE --> BACKTEST --> METRICS
    BENCH --> METRICS
```

## Agent Signal Types

| Agent | Signal | Range | Paradigm |
|-------|--------|-------|----------|
| LGBM v12 | `lgbm_p_up` | [0, 1] continuous | Gradient boosting (tabular) |
| TCN v0 | `tcn_p_up`, `tcn_p_down` | [0, 1] continuous | Dilated causal CNN (sequential) |
| DRL PPO v3 | `drl_action` | {−1, 0, +1} discrete | Reinforcement learning |
| GP v2 | `gp_action` | {−1, 0, +1} discrete | Evolutionary symbolic regression |
| MAMBA v1 | `mamba_p_up` | [0, 1] continuous | Selective state-space model |
| **Meta-Supervisor** | **position** | **[−1, +1] continuous** | **Meta-labeling (LightGBM)** |

## Walk-Forward OOS Configuration

| Agent | Train window | Step size | Folds (total) | OOS folds (from 2024-01) |
|-------|-------------|-----------|--------------|--------------------------|
| LGBM v12 | chronological split | — | 1 | 1 test set |
| TCN v0 | pre-2024 | — | 1 | 1 test set |
| DRL PPO v3 | 8,760h (1 year) | 720h (1 month) | 92 | 29 |
| GP v2 | 8,760h (1 year) | 336h (2 weeks) | 196 | 62 |
| MAMBA v1 | pre-2024 | — | 1 | 1 test set |
| Meta-Supervisor | 2022–2024 signals | 3 months | WFO | OOS 2024-01 → |
