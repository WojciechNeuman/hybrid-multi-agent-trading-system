 # Feature Engineering Audit & Redesign

**Date:** 2026-05-28
**Reference Notebook:** `src/hmats/notebooks/04_feature_audit.ipynb`
**Status:** Complete — actionable recommendations ready for implementation

---

## Executive Summary

After seven LGBM iterations and one Mamba experiment, all models fail to generate
profitable trading signals. The best OOS AUC across all experiments is **0.575**
(LGBM v3, expanding WFO) — and that configuration produced zero trades. Both
tree-based (LGBM) and sequence-based (Mamba) models plateau at the same AUC
ceiling (~0.52–0.57), confirming the bottleneck is in **feature quality**, not
model architecture.

This document summarizes the full audit and proposes a redesigned feature set (V3)
plus a repeatable feature selection system.

---

## 1. Feature Set History

The project has iterated through three generations of features:

| Gen | Registry | Features | Base TF | Used In | Result |
|-----|----------|----------|---------|---------|--------|
| V0 | `features.py` | 12 | 1h | NEAT/PPO agents | Never profitable |
| V1 | `feature_registry.json` | 196 | 1h | LGBM v0, Grid Search | -6.87% alpha; model learned to scalp noise |
| V2 | `feature_registry_v2.json` | 39 | 5m | LGBM v1–v4, Mamba v1 | AUC 0.50–0.575; marginal edge at best |

### V0 (12 features)

Minimal textbook indicators: log returns, rolling volatility, SMA ratio, MACD
(3 variants), momentum (2 windows), RSI, volume z-score, price z-score.
Effective unique information: ~7 dimensions due to MACD/momentum redundancy.

### V1 (196 features, 22 groups)

Exhaustive TA library covering returns, volatility, MA ratios, Bollinger,
MACD, oscillators, volume, candle structure, price position, calendar,
MA crosses, Ichimoku, SuperTrend, Fibonacci, long-cycle MAs, divergences,
candlestick patterns, volume profile, support/resistance, volatility regime,
statistical, and composite features.

**Problem:** Massive internal redundancy. Estimated effective rank: ~40 features.
Top LGBM importance features were all short-term noise (`ret_1h`, `close_vs_ema_7`),
causing the model to scalp micro-moves that were fee-killed in practice.

### V2 (39 features, 4 groups)

Physics-based redesign: structure (swing extrema), liquidity (VWAP, POC),
volatility (BK squeeze, Garman-Klass), and MTF context (1h/4h trend alignment).

**Problem:** Confirmed lookahead leak in Group D (MTF features, missing `.shift(1)`).
After patching, AUC dropped to honest levels (~0.52). POC features have a minor
leak (global bin edges). Effective rank: ~25 features.

---

## 2. Model Performance Summary

| Experiment | AUC | Trades/Day | EV/Trade | Fee Gate |
|-----------|-----|------------|----------|----------|
| LGBM v2 — Macro TBM | 0.527 | 0.04 | -0.004% | FAIL |
| LGBM v2 — Fixed Horizon | 0.571 | 0.29 | 0.193% | FAIL |
| LGBM v3 — Expanding WFO | 0.575 | 0.00 | — | N/A (no trades) |
| LGBM v3 — 1yr Sliding | 0.552 | 0.15 | 0.461% | PASS |
| LGBM v3 — 3mo Sliding | 0.529 | 0.67 | 0.139% | FAIL |
| Mamba — M1 Struct (Fixed) | 0.555 | 1.74 | 0.119% | FAIL |
| Mamba — M4 Omni (Macro) | 0.507 | 8.47 | 0.003% | FAIL |

**Key insight:** Mamba with full 288-bar sequence context achieves the same AUC
ceiling as single-row LGBM. The features, not the model, are the bottleneck.

---

## 3. Root Cause Analysis

### Why features are weak:

1. **Indicator redundancy** — V1 had 196→40 effective features; V2 has 39→25.
   No generation introduces genuinely new information sources.

2. **Lagging by construction** — All features describe what already happened.
   None capture forward-looking market microstructure phenomena.

3. **Single-asset myopia** — All features are from BTCUSDT spot OHLCV. No
   cross-asset (ETH/BTC, DXY), derivatives (funding, OI), or on-chain data.

4. **No feature-target alignment validation** — Features selected by intuition,
   never validated with mutual information against the specific target.

5. **No walk-forward stability testing** — Features stable on full history may
   be noise within any specific regime window.

---

## 4. Feature Quality Ratings

| V2 Group | Rating | Key Strength | Key Weakness |
|----------|--------|-------------|--------------|
| A — Structure | Moderate | Genuine S/R levels | Distances not ATR-normalized |
| B — Liquidity | Weak-Moderate | VWAP has economic meaning | POC leak; exhaustion features too sparse |
| C — Volatility | Strong | BK squeeze well-documented | Missing vol term structure |
| D — MTF Context | Weak | Theoretically sound | Had confirmed leak; no value post-patch |

---

## 5. Proposed V3 Feature Set (32 features)

Redesigned around six principles: information diversity, stationarity by
construction, ATR normalization, minimal redundancy, target alignment, and
cross-asset context.

### Categories

| Category | Count | Status | Key Features |
|----------|-------|--------|-------------|
| Volatility Regime | 6 | Retain & Enhance | `vol_term_structure`, `vol_bk_squeeze_ratio`, `vol_gk_72_zscore`, `vol_atr_regime`, `vol_parkinson_ratio`, `vol_realized_vs_implied` |
| Momentum & Mean Rev | 6 | Redesign | `mom_norm_24h`, `mom_norm_72h`, `mom_acceleration`, `mom_coherence`, `mean_rev_zscore`, `mean_rev_hurst` |
| Microstructure | 6 | **NEW** | `micro_kyle_lambda`, `micro_amihud_illiq`, `micro_volume_clock`, `micro_trade_intensity`, `micro_vpin`, `micro_roll_spread` |
| Structural | 4 | Simplify | `struct_dist_resistance_atr`, `struct_dist_support_atr`, `struct_range_position`, `struct_wick_rejection` |
| Volume & Liquidity | 4 | Simplify | `liq_vwap_dev_atr`, `liq_volume_anomaly`, `liq_obv_divergence`, `liq_absorption_ratio` |
| Cross-Asset | 4 | **NEW** | `cross_eth_btc_mom`, `cross_altcoin_breadth`, `cross_btc_dominance_chg`, `cross_usdt_dom_chg` |
| Calendar | 2 | Minimal | `cal_session_phase`, `cal_weekend` |

### Design Improvements Over V2

- **Microstructure features** (Kyle's lambda, Amihud illiquidity, VPIN) capture
  order flow dynamics invisible to standard TA
- **Cross-asset features** break the single-asset information ceiling using
  already-available multi-coin OHLCV data
- **ATR normalization** on all distance features makes them regime-invariant
- **Volatility term structure** (short/long vol ratio) captures compression →
  expansion dynamics better than raw squeeze indicators
- **Normalized momentum** (return/volatility) converts raw returns to
  realized Sharpe ratios, which are stationary and comparable across regimes

---

## 6. Feature Selection System

A repeatable 4-stage pipeline to run before every modeling iteration:

```
Stage 1: Statistical Filter
    - Variance filter (remove near-constant)
    - Pairwise correlation > 0.85 → keep higher MI feature
    - Expected reduction: ~30-40%

Stage 2: Univariate Ranking
    - Mutual Information with target
    - Spearman rank correlation
    - Keep top-K by MI (K ≈ 25)

Stage 3: Walk-Forward Stability
    - Rolling 3-month MI windows
    - Keep features in top-K across >= 60% of windows
    - Removes regime-specific features

Stage 4: Model-Based Pruning
    - Train LGBM, compute permutation importance
    - Keep only features whose removal hurts AUC > 0.001
    - Target: 15-25 final features
```

Full implementation code is in `04_feature_audit.ipynb` (Sections 6.2–6.6).

---

## 7. Implementation Priority

| Priority | Action | Impact | Effort |
|----------|--------|--------|--------|
| **P0** | Implement V3 volatility + momentum features | HIGH | LOW |
| **P0** | Run selection pipeline (Stages 1–4) on V3 | HIGH | LOW |
| **P1** | Add microstructure features (Kyle λ, Amihud) | HIGH | MEDIUM |
| **P1** | Build cross-asset pipeline from existing data | MEDIUM-HIGH | MEDIUM |
| **P2** | ATR-normalize structural features | MEDIUM | LOW |
| **P2** | Simplify volume/liquidity group | MEDIUM | LOW |
| **P3** | Investigate derivatives data (funding, OI) | HIGH potential | HIGH |

---

## 8. Success Criteria

If the root cause diagnosis is correct:

| Metric | Current Best | V3 Target |
|--------|-------------|-----------|
| OOS AUC | 0.575 (0 trades) | > 0.58 with tradeable signals |
| EV per Trade | 0.48% (35 trades) | > 0.40% with > 50 trades |
| Feature Stability | Not measured | > 60% stable across 3-month windows |
| Effective Feature Rank | ~25/39 | > 28/32 |

**Pivot criteria:** If V3 also plateaus at ~0.55 AUC, the conclusion is that
single-exchange OHLCV data does not contain sufficient information for 24h
price prediction, and the project should pivot to derivatives/on-chain/alternative
data sources or shorter prediction horizons (1–4h).

---

## References

- Kyle, A.S. (1985). "Continuous Auctions and Insider Trading." *Econometrica*.
- Amihud, Y. (2002). "Illiquidity and Stock Returns." *Journal of Financial Markets*.
- Easley, D., Lopez de Prado, M., & O'Hara, M. (2012). "Flow Toxicity and
  Liquidity in a High-Frequency World." *Review of Financial Studies*.
- Roll, R. (1984). "A Simple Implicit Measure of the Effective Bid-Ask Spread."
  *Journal of Finance*.
- De Prado, M.L. (2018). *Advances in Financial Machine Learning*. Wiley.
  Chapters 3–8 on feature engineering, triple barrier method, and feature importance.
