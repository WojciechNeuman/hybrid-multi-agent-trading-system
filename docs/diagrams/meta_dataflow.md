# Meta-Learning Data Flow & Walk-Forward Timeline (2026-06-11)

## 1 · Meta-labeling data flow

```mermaid
flowchart TD
    P1["4 base p(up) streams\naligned to parquet index"]:::in
    P2["ensemble_p_up = mean / bar"]:::step
    P3{"primary signal?\n>0.56 long · <0.44 short\n& n_models ≥ 2"}:::dec
    P4["primary_side ∈ {−1,+1}\nonly on signal bars (4.4% of bars)"]:::step
    P5["meta_y = TBM outcome of that side\nTP 2.5·ATR before SL 1.5·ATR within 48h\n1 = win, 0 = loss"]:::step
    P6["Meta features:\n4 probs + ensemble + n_models + 14 regime"]:::step
    P7["LightGBM meta-classifier\nexpanding WFO, 3-mo step, 48h embargo"]:::step
    P8["meta_prob → gate(>0.55) → size = side·(p−0.5)·2"]:::out

    P1-->P2-->P3-->|yes|P4-->P5-->P7
    P4-->P6-->P7-->P8
    P3-->|no|SKIP["no trade"]:::skip

    classDef in fill:#0b3d5c,stroke:#2962FF,color:#fff;
    classDef step fill:#374151,stroke:#9E9E9E,color:#fff;
    classDef dec fill:#4a3209,stroke:#F7931A,color:#fff;
    classDef out fill:#14532d,stroke:#26A69A,color:#fff;
    classDef skip fill:#1f2937,stroke:#6b7280,color:#fff;
```

## 2 · Walk-forward timeline (who is trained on what)

```mermaid
timeline
    title Coverage & evaluation windows (1h BTC)
    2017-11 to 2021-12 : LGBM rolling-WFO p(up) (genuine OOS each fold)
    2022-01 to 2022-12 : LGBM + Mamba WFO coverage begins : (TCN/Patch still in-train ≤2022-12-31)
    2023-01 to 2024-05 : All 4 base models emit OOS p(up) : META TRAIN window (meta-classifier fits here)
    2024-05-31 to 2026-05-16 : Unified 2-yr OOS : Chop(→11-05) ▸ Bull(→2025-10) ▸ Bear : META TEST window
```

## 3 · The regime-mismatch problem (why meta AUC < 0.5)

```mermaid
flowchart LR
    TR["META-TRAIN 2022→2024-05\nmostly BEAR + CHOP\nshort signals tend to WIN"]:::bad
    TE["META-TEST 2024-05→2026\nCHOP → big BULL → BEAR\nsame short signals LOSE in bull"]:::bad
    TR -->|"learned mapping\nprob→win inverts"| TE
    TE --> R["OOS meta AUC 0.46 (< 0.5)\nmeta ranks anti-correctly"]:::out
    classDef bad fill:#7f1d1d,stroke:#EF5350,color:#fff;
    classDef out fill:#4a3209,stroke:#F7931A,color:#fff;
```

The base agents' grids were tuned on the **2022→2024-05 grid-val window**, which is
predominantly bearish/choppy → they are **short-biased** (Mamba 244S/70L, TCN 232S/40L,
PatchTST 129S/0L). The meta then learns "shorts pay" from a bearish train window and
mis-ranks them through the 2024-11 → 2025-10 bull.
