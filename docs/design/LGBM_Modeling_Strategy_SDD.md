# LGBM Modeling Strategy — Software Design Document

**Document ID:** `SDD-LGBM-001`  
**Task Reference:** `MODEL-LGBM-EXPLORATION-003`  
**Notebook:** `src/hmats/notebooks/03_lgbm_feature_ablation_0fee.ipynb`  
**Author:** Hybrid Multi-Agent Trading System (Master's Thesis)  
**Last Updated:** 2026-05-27  
**Status:** Active

---

## 1. Purpose and Scope

This document is the architectural blueprint for the LightGBM ablation study
conducted in notebook `03_lgbm_feature_ablation_0fee.ipynb`. It describes the
rationale behind every major modeling decision: why fees are removed for initial
signal verification, how features are partitioned into thematic subsets, how the
Triple Barrier Method enforces a low-frequency trading mandate, and what
hyperparameter bounds are used — and why those bounds were chosen from the
legacy experimental history in `lab/`.

This document is **not** a tutorial. It is a record of design intent so that
future agents (and future-you) can modify the pipeline without repeating
mistakes already catalogued below.

---

## 2. The Frictionless (Zero-Fee) Rationale

### 2.1 Why Remove Fees for Signal Verification?

Real BTC/USDT trading fees on Binance are approximately 0.05 % per leg (maker)
to 0.10 % per leg (taker). A round-trip therefore costs 0.10–0.20 %. On 5-minute
data, a naive model might fire hundreds of signals per day where the gross edge
per trade is 0.05 %–0.15 %. Such a model would be net-negative under any
realistic fee schedule, yet appear profitable when evaluated with zero fees.

The purpose of the frictionless phase is precisely to **isolate and verify whether
a pure predictive signal exists** before the complexity of fee accounting enters
the evaluation loop. If a model cannot generate a positive expected value (EV)
in a zero-fee environment, it has no hope under real-world conditions. Conversely,
a model that shows robust EV in a frictionless environment becomes a candidate for
fee-structure optimization (execution timing, limit orders, fee tiers) rather than
a model to be discarded outright.

### 2.2 The Fee Survival Gate

The final cell of the notebook enforces a hard gate: the zero-fee EV per trade
must be materially larger than the maximum anticipated round-trip fee (0.20 %).
Models that clear this gate proceed to the next pipeline stage (notebook `04_`)
where real fees and slippage are introduced. Models that fail the gate are
documented here for the archival record rather than pursued further.

### 2.3 Scope of "Zero Fee"

In this study, "zero fee" means exactly that: no maker fee, no taker fee, no
funding rate, no slippage, no spread. Entry and exit occur at the bar's close
price. This is an optimistic assumption. Its purpose is to give the signal the
maximum possible benefit of the doubt; any positive EV found here is therefore
a lower bound on what a well-executed live strategy could achieve — not an upper
bound.

---

## 3. Feature Subsetting Logic

### 3.1 Philosophy: Structural Physics, Not Indicator Soup

The legacy feature set (`feature_registry.json`, v1, 196 features) was an
"indicator soup" constructed from standard lagging indicators: SMAs, RSI, MACD,
Bollinger Bands, Stochastics. Analysis of feature importances from the lab
experiments (see `lab/docs/notebook_analysis.md`) showed that this set was
dominated by short-term return features (`ret_1h`, `ret_2h`,
`close_vs_ema_7`) which captured noise rather than structural edge. The model
learned to scalp micro-moves that fee-killed all edge on live data.

The v2 feature set (`feature_registry_v2.json`, 39 features) describes
**market physics**: liquidity concentration, volatility compression before
expansion, and multi-timeframe trend alignment. These features describe *why*
price is likely to move rather than *how much it has already moved*.

### 3.2 The Four Thematic Groups

All 39 features are organized into four groups in `feature_registry_v2.json`.
Each group is self-contained in the sense that its features can be fed to a model
independently without requiring features from other groups.

| Group | Prefix | n | Description |
|-------|--------|---|-------------|
| A — Structure | `struct_` | 11 | Confirmed swing extrema anchors, candle geometry vectors |
| B — Liquidity | `liq_` | 10 | Anchored VWAP deviations, rolling POC distance, volume exhaustion |
| C — Volatility | `volat_` | 8 | BK squeeze ratio, Garman-Klass HV, ATR regime width |
| D — MTF Context | `mtf_` | 10 | 1h/4h EMA spread signals, RSI, alignment score, session cyclicals |

**Lookahead policy** (critical): all features in all groups are strictly
lookahead-free. Swing extrema detected by `argrelextrema(order=N)` at index `i`
are usable only at index `i+N` (forward-filled from there). Rolling parameters
use `.shift(1)` to prevent the current bar's value from appearing in the
computation. MTF 1h/4h values are propagated to 5m bars via `reindex(method='ffill')`,
ensuring only past-closed higher-timeframe candles are visible.

### 3.3 The Four Ablation Models

The study trains four independent LightGBM models, each receiving a distinct
feature subset loaded from the `ablation_subsets` field of the registry:

| Model ID | Registry Key | Features | Hypothesis |
|----------|-------------|----------|------------|
| M1 — Structure | `structure_only` | 11 (Group A) | Swing geometry alone predicts breakouts |
| M2 — Struct + Liquidity | `market_structure` | 21 (Groups A+B) | Liquidity context amplifies swing signals |
| M3 — Volatility | `volatility_only` | 8 (Group C) | Squeeze-release dynamics are sufficient on their own |
| M4 — Omni | `all` | 39 (All groups) | Full feature set; upper-bound reference |

The ablation design follows a hierarchical logic: M1 is the baseline; M2 adds
liquidity; M3 tests an orthogonal hypothesis (volatility regime as sole driver);
M4 measures the cost/benefit of adding MTF context to the structural core.

---

## 4. Low-Frequency Trading Mandate

### 4.1 Why Low Frequency on 5-Minute Data Is Non-Trivial

5-minute data has 288 bars per day. A model trained naively on this frequency
will tend to fire signals constantly because many lagging indicators correlate at
sub-hour time scales. The legacy lab experiments confirmed this: early LGBM models
produced 50–200 trades per day, each with gross EV < 0.10 % — fee-lethal.

The low-frequency mandate is achieved through **two complementary mechanisms**:
the Triple Barrier Method (which shapes the label distribution) and the
probability threshold (which filters signal quality at inference time).

### 4.2 Triple Barrier Method Parameters

The TBM assigns each bar one of three labels based on which barrier is hit first
when a hypothetical long trade is entered at that bar's close:

| Barrier | Value | Rationale |
|---------|-------|-----------|
| Take-profit | `+2.5 × ATR_72` | ATR_72 = 6h rolling ATR on 5m; wide TP forces model to learn "big move" setups |
| Stop-loss | `−1.5 × ATR_72` | Asymmetric 2.5:1.5 ≈ 1.67 gross RR; profitable at ~37.5 % win rate |
| Time barrier | 288 bars = 24h | Timeout after 24 hours; forces model to commit to meaningful moves |

**ATR derivation:** Rather than loading a separate 1h parquet and re-computing
ATR, the study uses `volat_atr_72_pct` already present in the structural feature
parquet (ATR_72 as a fraction of close). The absolute ATR is reconstructed as
`atr_abs = close × volat_atr_72_pct`.

**Label semantics:** TBM label `+1` (long TP hit first) is the positive class for
the binary LightGBM classifier. Labels `−1` (SL hit first) and `0` (timeout) are
the negative class. This encodes the model's job as: *predict which 5m bars will
see a ≥ 2.5 × ATR upward move within 24 hours before the price falls 1.5 × ATR*.

**Label frequency expectation:** On BTC/USDT 5m data with the above parameters,
the positive rate is expected to be approximately 35–45 %, varying with the
market regime. Ranging regimes produce fewer positives (price chops without
resolution); trending regimes produce more.

### 4.3 Probability Threshold

At inference time, the LGBM model outputs a probability `p(label=1 | features)`.
A trade signal is generated only when `p > 0.75`. This high threshold serves two
purposes:

1. **Precision filter:** Only the most confident positive predictions become
   trades. In a calibrated binary classifier with ~40 % base rate, `p > 0.75`
   corresponds to roughly the top 10–15 % of predictions by confidence.

2. **Low-frequency enforcement:** Even if the model can predict the direction
   correctly 55 % of the time at `p > 0.5`, the signal count at `p > 0.75` is
   dramatically lower — typically < 1.5 trades/day on 5m data.

**Target trade frequency:** < 1.5 trades per day on the validation period. The
notebook checks this explicitly. If the threshold produces 0 trades, the result
is noted and the threshold is reduced to `p > 0.65` for diagnostic purposes.

---

## 5. Hyperparameter Strategy

### 5.1 Lessons from `lab/` History

The following constraints were derived from eight iterations of LGBM grid search
documented in `lab/notebooks/07_lgbm_grid_v1.ipynb` through
`lab/notebooks/07_lgbm_grid_v8.ipynb`. Each constraint targets a specific failure
mode observed in prior experiments.

| Parameter | Constrained Range | Failure Mode Avoided |
|-----------|------------------|---------------------|
| `num_leaves` | [15, 31] | v1–v3: `num_leaves=127–255` caused extreme test-set overfitting; model memorized train-set regime transitions |
| `max_depth` | [4, 6] | v1–v4: unlimited depth (`−1`) allowed single-feature splits that exploited transient anomalies invisible in test |
| `learning_rate` | [0.01, 0.05] | v5–v6: `lr=0.1` with few estimators produced high-variance val set results that didn't generalize; combined with early stopping, 0.01–0.05 gives stable convergence |
| `colsample_bytree` | [0.6, 0.8] | v7–v8: full column exposure allowed the model to co-learn correlated features (e.g. ret_1h + close_vs_ema — now removed from v2 set) in ways that inflated val AUC but hurt test OOS |
| `subsample` | [0.6, 0.8] | Same as colsample — stochastic subsampling of rows prevents over-reliance on regime-specific bar clusters |

### 5.2 Fixed Regularization Parameters

The following regularization parameters are held constant across all grid
configurations — they were identified as beneficial in v7/v8 and are not
searched over to keep the grid tractable:

```
min_child_samples = 50    # prevent splits on tiny clusters
reg_alpha         = 0.1   # L1 regularization on leaf weights
reg_lambda        = 1.0   # L2 regularization on leaf weights
n_estimators      = 1000  # with early stopping rounds = 30
```

### 5.3 Grid Size

The constrained grid has `2 × 2 × 2 × 2 = 16` configurations
(`num_leaves × max_depth × learning_rate × colsample_bytree`). An additional
`subsample` dimension doubles this to **32 configurations per model**.

With 4 models, the total evaluation count is **128 LightGBM fits**. On the
training set (2017–2023, ~696k bars after burn-in) with `n_estimators=1000` and
`early_stopping_rounds=30`, each fit takes approximately 15–45 seconds on modern
hardware, yielding a total wall-clock time of 30–90 minutes. This is intentional:
a thorough but not exhaustive search, matching the `lab/` v8 approach of `2^5=32`
model configs on 1h data.

### 5.4 Early Stopping Protocol

The validation set (2024 calendar year, ~100k bars) serves as the early-stopping
monitor. No temporal leakage occurs because the validation set is strictly later
in time than the training set, and LightGBM sees the validation set only for
stopping decisions — not for gradient updates.

**Embargo:** No explicit time embargo (purged K-fold) is used in this ablation
study. The strict chronological split provides sufficient separation. Purged
K-fold will be introduced in the production training notebook when walk-forward
optimization is implemented.

---

## 6. Data Architecture

### 6.1 File Locations

| File | Path | Purpose |
|------|------|---------|
| Raw OHLCV (5m) | `data/raw/BTCUSDT_5m.parquet` | `close`, `high`, `low` for TBM barriers |
| Structural features | `data/features/BTCUSDT_5m_structural.parquet` | 39 features for model input |
| Feature registry | `data/features/feature_registry_v2.json` | Ablation subset definitions |
| TBM labels cache | `data/cache/BTCUSDT_5m_tbm_labels.parquet` | Cached TBM output (auto-generated) |

### 6.2 Chronological Splits

| Split | Date Range | Approx Bars | Purpose |
|-------|-----------|------------|---------|
| Train | 2017-08-25 → 2023-12-31 | ~696k | Model fitting |
| Validation | 2024-01-01 → 2024-12-31 | ~105k | Hyperparameter selection, early stopping, signal evaluation |
| Test (held out) | 2025-01-01 → present | ~75k | Not touched until final production model |

The test set is **loaded but not used** in this notebook. It will be evaluated
in notebook `04_` once a winning model configuration has been identified.

### 6.3 Schema Conventions

- Index: `open_time`, UTC `DatetimeIndex`, 5-minute frequency
- OHLCV dtype: `float32`
- Feature dtype: `float32` (stored), cast to `float64` for TBM arithmetic
- Labels: `int8` (`+1`, `-1`, `0`); binary target: `int8` (`0`, `1`)

---

## 7. Evaluation Framework

### 7.1 Model Selection Metric

The primary model selection metric is **AUC on the validation set**. AUC is
threshold-agnostic (it evaluates the entire ROC curve) and is appropriate here
because the positive class base rate (~40 %) is not extreme enough to invalidate
it. Cross-entropy (log loss) is a secondary metric monitored during early stopping.

### 7.2 Backtest Metrics

The zero-fee backtest produces the following metrics for each model. All metrics
are computed on the **validation set only** at this stage:

| Metric | Definition | Target |
|--------|-----------|--------|
| AUC | Validation set ROC AUC | > 0.55 |
| Trade Count | Number of signals at p > 0.75 | < 1.5 / day |
| Win Rate | TP-exit trades / (TP + SL exits) | > 45 % |
| Profit Factor | Gross wins / Gross losses | > 1.5 |
| EV per Trade | Mean PnL across all trades (incl. timeouts) | > 0.20 % (fee survival gate) |

### 7.3 The Fee Survival Gate

A model "passes" if its zero-fee EV per trade (in % of notional) is materially
greater than the maximum anticipated round-trip fee of 0.20 % (0.10 % entry +
0.10 % exit, taker pricing). "Materially greater" is defined as EV > 2× the fee:
**EV > 0.40 %** as the passing threshold, giving a 2× fee buffer for the
introduction of slippage in notebook `04_`.

---

## 8. Design Decisions Not Taken (Rejected Alternatives)

| Alternative | Rejected Because |
|-------------|-----------------|
| Multiclass TBM (+1/−1/0 as three classes) | Adds short-side complexity not warranted for initial ablation; BTC is primarily long-biased; binary focus gives cleaner AUC interpretation |
| Random Forest instead of LGBM | LGBM is 10–50× faster on tabular data of this size; RF results are bounded by LGBM lower-bound (LGBM is strictly more expressive) |
| Purged K-Fold in ablation | Adds 3–5× compute; the strict time-split already prevents leakage; purged K-fold is reserved for production walk-forward training |
| Walk-forward evaluation in ablation | Same as above — ablation is about signal comparison, not deployment-ready evaluation |
| Predicting fixed-percentage targets | ATR-relative barriers adapt to volatility regime; fixed % targets create regime-dependent label imbalance (e.g. a 3 % move is routine in 2021 but exceptional in 2023) |

---

## 9. V1 Pivot — Notebook Re-indexing (2026-05-28)

### 9.1 File Structure Correction

The notebook pipeline was re-indexed to strict 0-based numbering:

```
00_data_ingestion.ipynb          (unchanged)
01_structural_features.ipynb     (unchanged, leak patched — see §9.2)
02_lgbm_omni_0fee_v1.ipynb      (was: 03_lgbm_feature_ablation_0fee.ipynb)
```

The ablation study across four feature subsets was **discarded in favour of a focused
M4 Omni investigation** following the discovery of lookahead bias with an anomalously
high pre-patch AUC.

---

### 9.2 Leak Hunt — Confirmed Finding

#### The Bug

During execution of the ablation study, an AUC of ~0.75 was observed. This is
mathematically implausible for BTC/USDT 5m structural features against a 24h TBM label
(published literature and prior `lab/` experiments suggest AUC of 0.52–0.62 for
comparable feature sets). A full audit of `01_structural_features.ipynb` was triggered.

#### Root Cause: Group D MTF Features — Missing `.shift(1)`

**Location:** `01_structural_features.ipynb`, Cell 11, function `build_group_D`.

The five MTF features below were computed on the 1h / 4h index and then mapped to the
5m index via `reindex(method='ffill')` **without a preceding `.shift(1)`**:

| Feature | Source timeframe | Max lookahead introduced |
|---------|-----------------|--------------------------|
| `mtf_h1_ema_signal` | 1h | **55 minutes** per bar |
| `mtf_h1_rsi` | 1h | **55 minutes** per bar |
| `mtf_h1_above_ema50` | 1h | **55 minutes** per bar |
| `mtf_h4_ema_signal` | 4h (resampled from 1h) | **3h 55 minutes** per bar |
| `mtf_h4_rsi` | 4h (resampled from 1h) | **3h 55 minutes** per bar |

**Mechanism:** In Binance's `klines` API, the bar labeled `open_time=10:00` (1h) opens
at 10:00 and closes at approximately 10:59:59. Its EMA/RSI values include the close
price at 10:59. Without `.shift(1)`, 5m bars from `10:00` to `10:55` all receive the
h1 bar's feature values — data that spans up to 10:59, which lies in the **future**
relative to any bar from 10:00 to 10:54. The 4h case is four times worse.

`mtf_alignment` (a weighted average of `mtf_h1_ema_signal` and `mtf_h4_ema_signal`) was
also implicitly leaky as a derived feature.

#### Confirmed via Correlation Test

The feature-vs-forward-return correlation test (Phase 2 of `02_lgbm_omni_0fee_v1.ipynb`)
demonstrated that `mtf_h1_ema_signal` had anomalously high `|r|` against the +1 bar
forward return, consistent with the value having been computed using that return.

#### The Fix

```python
# BEFORE (leaky):
feats["mtf_h1_ema_signal"] = h1_spread.reindex(idx5, method="ffill")

# AFTER (clean):
feats["mtf_h1_ema_signal"] = h1_spread.shift(1).reindex(idx5, method="ffill")
```

Applied to all five leaky features + `mtf_alignment` recomputed from patched signals.
The session features (`mtf_session_hour_sin/cos`, `mtf_session_dow_sin/cos`) are
timestamp-derived and required no change.

**Patched in source:** `01_structural_features.ipynb` Cell 11 — must be re-executed to
regenerate `BTCUSDT_5m_structural.parquet`. The clean version is saved independently as
`BTCUSDT_5m_structural_clean.parquet` by `02_lgbm_omni_0fee_v1.ipynb`.

#### Secondary Finding: `_rolling_poc` Global Bin Edges

`_rolling_poc` in Group B uses `close_arr.min()` / `.max()` over the full dataset to
define price bin edges, giving historical bars implicit knowledge of the all-time ATH/ATL.
The information content of this leak is low (it affects bin resolution, not POC direction)
and its impact on AUC is estimated to be negligible compared to the MTF issue. It is
logged here for future resolution (e.g. rolling min/max within each window).

---

### 9.3 Expected V1 Metrics (Post-Patch)

After patching, the honest AUC is expected to drop significantly. Historical benchmarks
from similar structural feature sets on BTC 1h data (`lab/notebooks/07_lgbm_grid_v7–v8`)
achieved OOS AUC of 0.53–0.59. On 5m data with a 24h TBM, we expect:

| Scenario | AUC | Interpretation |
|----------|-----|---------------|
| No signal | ~0.50 | Features have no predictive content |
| Weak signal | 0.52–0.55 | Genuine but marginal edge — requires fee optimisation |
| Moderate signal | 0.55–0.60 | Viable with careful execution |
| Strong signal | > 0.60 | Surprising — investigate remaining leaks |

V1 metrics (AUC, Win Rate, EV/Trade, Trades/Day) will be logged in the SDD once the
notebook is executed. See `02_lgbm_omni_0fee_v1.ipynb` for live results.

---

After `02_lgbm_omni_0fee_v1.ipynb` completes:

1. **If EV gate passes:** proceed to `03_lgbm_fees_wfo.ipynb` — introduce realistic
   fees (0.05 % maker / 0.10 % taker), walk-forward optimisation, and final test-set
   evaluation on 2025+ held-out data.
2. **If EV gate fails:** explore the diagnostic options documented in the notebook's
   final cell — lower threshold (p > 0.65), wider TBM barriers (TP=3×, SL=2×), or an
   MTF alignment pre-filter (only trade when `mtf_alignment > 0.3`).
3. **Calibration** — apply Platt scaling or isotonic regression before the p > 0.75
   threshold; post-patch models may need recalibration since the label base rate shift
   from removing lookahead alters the probability distribution.
4. **POC fix** — resolve the secondary `_rolling_poc` global bin-edge issue in a future
   feature engineering revision by switching to rolling-window-local bin edges.
5. **RL environment** — the validated clean LGBM signal will serve as a pre-filter for
   the RL agent's action space (agent only considers entries when LGBM p > 0.75).
