# Feature Creation Pipeline (2026-06-11)

The unified parquet (`BTCUSDT_1h_unified.parquet`, 292 columns) is assembled from four
feature libraries built on raw data, then each agent draws its own subset. Group counts
are from `docs/features.md`.

```mermaid
flowchart LR
    subgraph SRC["Raw sources (00_data_ingestion)"]
        direction TB
        BIN["Binance 1h OHLCV\n+ taker / mark / index"]:::src
        CG["CoinGecko\nmcap · dominance · breadth"]:::src
        FG["Alternative.me\nFear & Greed"]:::src
    end

    subgraph LIB["Feature libraries → unified parquet"]
        direction TB
        V1["V1 Technical — 196 feats / 22 groups"]:::lib
        V3["V3 External — 21 feats"]:::lib
        V4["V4 Microstructure — 25 feats"]:::lib
        STR["Structural — 39 feats"]:::lib
    end

    BIN --> V1 & V4 & STR
    BIN & CG & FG --> V3

    subgraph G1["V1 groups (price/vol/trend/osc)"]
        direction TB
        GA["returns 18 · volatility 10 · vol_regime 11"]:::grp
        GB["ma_ratios 12 · long_cycle_ma 11 · ma_crosses 9"]:::grp
        GC["bollinger 4 · macd 4 · oscillators 6"]:::grp
        GD["volume 9 · volume_profile 10 · candles 12"]:::grp
        GE["S/R 13 · fibonacci 10 · ichimoku 10 · supertrend 8"]:::grp
        GF["divergences 5 · statistical 13 · composite 7 · calendar 11"]:::grp
    end
    V1 --> GA & GB & GC & GD & GE & GF

    subgraph G3["V3 groups (causal/external)"]
        direction TB
        H1["cross-asset (ETH/BTC, breadth)"]:::grp
        H2["market structure (dominance, mcap)\n⚠ mkt_total_mcap_chg_24h LEAKS — excluded"]:::leak
        H3["sentiment (fear&greed)"]:::grp
        H4["microstructure (Amihud, Kyle, Roll)"]:::grp
    end
    V3 --> H1 & H2 & H3 & H4

    subgraph G4["V4 + Structural"]
        direction TB
        K1["V4: TFI · anchored VWAP · Hurst · ADF\nfracdiff · sideways-flag"]:::grp
        K2["Struct: swing anchors · POC · BB-Keltner squeeze\nMTF (4h/1D) alignment"]:::grp
    end
    V4 --> K1
    STR --> K2

    GA & GB & GC & GD & GE & GF & H1 & H3 & H4 & K1 & K2 --> SEL

    subgraph SEL["Per-agent feature selection"]
        direction TB
        SL["LGBM: 11 Boruta-locked\n(stoch,ret,rsi_div,macd_hist,hurst,ad_z,...)"]:::sel
        SM["Mamba: 39 Boruta-vs-TBM\n(ret_48/72/168h, supertrend, fib, ...)"]:::sel
        ST["TCN / PatchTST: curated 32\n9 LGBM-core + 11 V1 + 7 V4 + 4 struct + fracdiff"]:::sel
    end

    classDef src fill:#1f2937,stroke:#9ca3af,color:#fff;
    classDef lib fill:#0b3d5c,stroke:#2962FF,color:#fff;
    classDef grp fill:#374151,stroke:#9E9E9E,color:#fff;
    classDef leak fill:#7f1d1d,stroke:#EF5350,color:#fff;
    classDef sel fill:#14532d,stroke:#26A69A,color:#fff;
```

## Excluded-from-ML columns

`{open, high, low, close, volume, label}` (targets/raw OHLCV used only for
labelling + backtest) and **`mkt_total_mcap_chg_24h`** (confirmed forward-leak:
+0.44 corr with the future 24h move — see `MEMORY.md` / `project_mcap_feature_leak`).
