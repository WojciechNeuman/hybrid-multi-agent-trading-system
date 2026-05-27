# HMATS Trading Strategy — Research Context & Current State

**Date:** 2026-05-26  
**Status:** Active development — no profitable strategy under realistic fees yet

---

## 1. Approach Overview

The goal is a hybrid multi-agent trading system that generates alpha on BTC/USDT perpetuals (and spot) using supervised ML signals — currently LightGBM and TCN — with realistic execution modelling. The pipeline is:

1. Engineer features from OHLCV + funding rate data
2. Train a classifier to predict directional price movement
3. Run a backtester with limit entries, ATR-relative SL/TP, funding costs, taker fees
4. Grid-search model hyperparameters and trading parameters jointly
5. Evaluate on a held-out test set under walk-forward conditions

The research question is whether the ML signal survives realistic transaction costs. The short answer so far: **it does not**, but the investigation into why is producing better architecture with each iteration.

---

## 2. What Was Missing in the Initial Models

### 2.1 Zero-fee baseline inflation

The first profitable versions (LGBM vtrain3, TCN vtrain1) were run at **0% fees**. These results were always understood to be upper bounds, but they set false expectations about signal strength:

- LGBM vtrain3: **+106.13% return, Sharpe 1.397**, 897 trades
- TCN vtrain1: **+11.68% return, Sharpe 0.446**, 75 trades

Adding a flat 0.1% taker fee (both entry and exit) immediately destroyed both:
- LGBM vtrain4: **−18% return, Sharpe −0.67**
- TCN vtrain2: **−34% static, −88.52% WFO**

The LGBM model at 897 trades was paying ~0.2% round-trip × 897 ≈ **179% cumulative fee drag** on a strategy that only made 106% before fees. The signal had no edge once friction was applied.

### 2.2 Close-to-close SL/TP (no intra-candle realism)

vtrain3 (LGBM) and vtrain1 (TCN) used close prices for both entry and exit, with no check of whether H/L wicks crossed SL or TP within the bar. This meant:
- Entries were filled at the close of the signal bar (market order assumption)
- Stops were checked only at close — a bar could gap through SL with no penalty
- All exits got the close price regardless of actual intra-bar path

This is a form of lookahead/latency bias that overstated execution quality significantly.

### 2.3 Naive binary labels

Initial models predicted `next_close > current_close` (binary, 1-bar horizon). Problems:
- Next-bar direction is extremely noisy for hourly data
- No consideration of whether the move was large enough to cover fees
- Training signal included many 0.01%–0.05% moves that are net-negative after any fee
- Class imbalance near 50/50 with no meaningful separability

The label doesn't align with what the trading strategy needs: a sustained directional move large enough to overcome round-trip costs.

### 2.4 K-Fold without purging (early versions)

Early prototypes used standard K-Fold cross-validation on time-series data. This allowed training data from the future to leak into the validation set (future bars' features overlap with past labels through rolling indicators). Even in vtrain7 this required careful implementation of **Purged K-Fold** with 168h embargo on both sides of each fold boundary.

### 2.5 Uniform funding model (all trades as Futures)

vtrain3–vtrain6 LGBM models routed **all trades through Futures**, meaning longs were charged the perpetual funding rate. BTC funding has historically been positive (longs pay shorts) at ~+0.01%/8h in trending markets. With 897 trades averaging ~10h hold time, cumulative funding drag compounds the fee problem further. This was only identified and corrected in vtrain7.

---

## 3. LGBM-Specific Struggles

### 3.1 Fee drag at 897 trades

The primary issue for LGBM is trade frequency. At 897 trades over the test period, even 0.05% one-way taker fee on exits produces substantial drag. Maker-only limit entries (vtrain5/6) reduced the entry fee to 0% but entries that don't fill generate opportunity cost (the signal fires but the limit order expires unfilled).

### 3.2 limit_px calculation bug (vtrain5→vtrain6)

vtrain5 computed the limit entry as `px - entry_atr_mult * atr_pct` (absolute subtraction, mixing a price and a fractional). This meant the entry buffer was negligible for small ATR values. vtrain6 fixed this to `px * (1 - entry_atr_mult * atr_pct)` (multiplicative). The bug meant vtrain5 was essentially entering near-market regardless of `entry_atr_mult` setting.

### 3.3 ATR as percentage vs absolute

`atr_14_pct` is `ATR_14 / close` — a dimensionless fraction. Using it multiplicatively (`px * (1 ± mult * atr_pct)`) keeps SL/TP distances proportional to price level. Mixing absolute and fractional quantities in the early versions produced SL widths that were inconsistent across different price regimes.

### 3.4 Funding drag on Spot longs (wrong routing)

vtrain3–vtrain6 charged long trades the perpetual funding rate (SHORT_FUNDING_H borrowed from the shorts model). Spot longs have no funding obligation. This was systematically overstating the cost of long positions across all pre-vtrain7 experiments.

### 3.5 Current state

**vtrain7 is the active attempt.** It attacks the profitability problem from four angles simultaneously:

| Issue | vtrain7 fix |
|---|---|
| Noisy binary labels | TBM 3-class labels (±2×ATR barriers, 24h vertical barrier) |
| Single model config | 48-model grid search (top_n, corr_thresh, num_leaves, lr) |
| All-Futures routing | Spot longs (0% funding) / Futures shorts (funding received) |
| Binary signal | Multiclass P_up / P_down with independent thresholds |

vtrain7 has not yet been run to completion. Results are pending.

---

## 4. TCN-Specific Struggles

### 4.1 Long bias in signal output

TCN vtrain1 produced 72 long trades vs 3 short trades on the test set. The model learned an upward-biased prior from the training period (BTC bull run), making it almost entirely useless for shorts. Feature engineering for the TCN didn't include sufficient regime-agnostic inputs to prevent this.

### 4.2 Sharp OOS → test decay

TCN shows much steeper OOS-to-test Sharpe degradation than LGBM:
- TCN vtrain1: OOS Sharpe ~1.15 → test Sharpe 0.446
- LGBM vtrain3: OOS Sharpe ~1.76 → test Sharpe 1.40 (more stable)

This suggests the TCN is overfitting to the local regime of each validation fold. Tree models with explicit feature selection generalise better on tabular financial data.

### 4.3 WFO warm-start catastrophe

TCN vtrain2 introduced Walk-Forward Optimization with 5-epoch warm-start retraining at each WFO step. This caused catastrophic over-trading: **933 trades in the WFO run vs 75 in the static model**. At 0.1% fees, this produced −88.52% return and Sharpe −10.703 — a complete strategy failure. The warm-start shifted the model's confidence distribution, lowering effective thresholds drastically.

### 4.4 TBM barrier width too tight (vtrain2 static)

TCN vtrain2 applied TBM labels with barriers at 1×σ (approximately 0.2–0.5% for hourly BTC). The round-trip fee alone was 0.1%–0.2%, so the model was being trained to predict moves that only marginally exceeded trading costs. This produced many small, unprofitable round-trips even when the directional prediction was correct. Result: −34% return, Sharpe −4.006.

### 4.5 vtrain4 aligns TCN with LGBM vtrain7

TCN vtrain4 (`12_tcn_grid_search_vtrain4.ipynb`) closes the gap with LGBM vtrain7 in two steps:

| Fix | vtrain3 | vtrain4 |
|-----|---------|---------|
| TBM label width | 1×rolling_σ (0.2–0.5%) — too tight | **±2×ATR (1–3%), 24h vertical** |
| Routing | All Futures (longs charged funding) | **Spot longs (0% funding) / Futures shorts (receive funding)** |
| Confidence exit | `max(0.28, threshold − 0.10)` | **`1 − threshold`** (vtrain7 convention) |
| Architecture | Grid-searched | **Fixed at vtrain3 best: [64,64,64,64] kernel=3** |
| Trading grid ceiling | threshold ≤ 0.50 | **threshold ≤ 0.55** |

The root failure of vtrain3 was TBM labels at 1×σ — every grid config produced ≥2 366 OOS trades
because the narrow barriers fire on noise reversions. Wider TBM labels (same as vtrain7 LGBM)
should reduce trade count and improve signal quality. Model is retrained from scratch; no architecture
search is performed (architecture fixed from vtrain3's proven config).

---

## 5. Current State

**Neither model has produced a profitable strategy under realistic execution costs.**

| Version | Model | Fee model | Return | Sharpe | Trades | Status |
|---|---|---|---|---|---|---|
| LGBM vtrain3 | Binary LGBM | 0% | +106.13% | 1.397 | 897 | 0% fees — not realistic |
| LGBM vtrain4 | Binary LGBM | 0.1% taker | −18% | −0.67 | 897 | Fees destroyed edge |
| LGBM vtrain5/6 | Binary LGBM | Maker entry / 0.05% exit | TBD | < 0 | TBD | Still unprofitable |
| LGBM vtrain7 | TBM 3-class | Spot/Futures asymmetric | **pending** | **pending** | — | Active attempt |
| TCN vtrain1 | Binary TCN | 0% | +11.68% | 0.446 | 75 | 0% fees — not realistic |
| TCN vtrain2 static | TBM TCN | 0.1% taker | −34% | −4.006 | — | TBM barriers too tight |
| TCN vtrain2 WFO | TBM TCN | 0.1% taker | −88.52% | −10.703 | 933 | WFO warm-start failure |
| TCN vtrain3 | TBM TCN (1×σ) | Maker entry / 0.05% exit | −71.55% | −2.602 | 620 | Maker entry helps; still too many trades |
| TCN vtrain4 | TBM TCN (±2×ATR) | Spot/Fut routing, maker entry | **pending** | **pending** | — | Active attempt |

The fundamental question remains open: **is there sufficient directional signal in the engineered feature set to generate alpha after 0.05%–0.1% round-trip costs?**

The evidence so far suggests the raw signal quality (OOS cross-validation accuracy) is in the range of 52–55% on 3-class predictions — marginally above random but potentially enough to be profitable at low trade frequency with wide SL/TP and good position sizing. vtrain7's wider TBM barriers (±2×ATR) and selective 48-model search are the primary levers targeting this.

---

## 6. What vtrain7 Is Designed to Test

1. **Label quality hypothesis:** TBM labels aligned with the actual trade barriers give the model a training signal that directly corresponds to what the backtest rewards. If the model learns to predict these correctly, the predictions should translate to real backtested edge.

2. **Spot routing eliminates funding drag on longs:** Historically the largest hidden cost. Removing it from long positions gives the model a structural advantage on the long side.

3. **48-model search:** A single fixed hyperparameter config may not be expressive enough. Searching across feature count, correlation threshold, tree depth (num_leaves), and learning rate gives the best-fit model a fair chance.

4. **Independent long/short thresholds:** Asymmetric P_up/P_down thresholds allow tuning selectivity separately for longs and shorts — useful given the documented long-bias in regime-specific data.

If vtrain7 also fails to find edge under realistic fees, the likely next steps are:
- Regime filtering (only trade in low-funding environments, or specific volatility quantiles)
- Alternative feature engineering (order book imbalance, funding rate as signal, macro regime)
- Ensemble with TCN as a meta-signal gating LGBM positions
- Reconsider the alpha source entirely (cross-exchange spread, basis trades)
