# Feature Engineering Reference

> **Important for model development.** This document describes all feature sets
> used in the project, the selection pipeline, and the features chosen in the
> most recent experiment. Review this file before training any new model — it
> captures decisions that are not visible in the code alone.

---

## Feature Set V1 — 196 OHLCV 1h Features

**File:** `data/features/BTCUSDT_1h_features.parquet`  
**Registry:** `data/features/feature_registry.json`  
**Source data:** OHLCV (Open, High, Low, Close, Volume) on 1h BTCUSDT candles only  
**Generated:** 2026-05-16

All features are purely quantitative, with no external data. Three columns
(`close`, `sma_200`, `atr_14_pct`) are backtest-only — they are not passed
to the model as inputs.

### Feature Groups

| Group | Count | Key Features |
|-------|-------|-------------|
| `returns` | 18 | `ret_1h`, `log_ret_1h`, `ret_6h` … `ret_168h` (1h–7d) |
| `volatility` | 10 | `vol_6h`–`vol_168h`, `atr_14`, `atr_14_pct`, `atr_24` |
| `volatility_regime` | 11 | `gk_vol_24h` (Garman-Klass), `park_vol_24h` (Parkinson), `vol_of_vol_72h`, `bb_squeeze_20/50`, `atr_14_pct_rank`, `range_vs_atr` |
| `ma_ratios` | 12 | `close_vs_sma/ema_7/14/20/50/100/200` (% deviation from MA) |
| `long_cycle_ma` | 11 | `close_vs_sma/ema_336/504/720/2160/4320` (2wk–6mo), `weekly_mom_accel` |
| `ma_crosses` | 9 | `sma50_vs_sma200`, `golden_cross`, `candles_since_cross`, `ma_bull_score`, `ma_ribbon_width` |
| `bollinger` | 4 | `bb_width_20/50`, `bb_position_20/50` |
| `macd` | 4 | `macd_12_26`, `macd_hist_12_26`, `macd_5_13`, `macd_hist_5_13` |
| `oscillators` | 6 | `rsi_7/14/21`, `stoch_k_14/21`, `williams_r` |
| `volume` | 9 | `vol_z_12/24/72/168h`, `vol_ratio_12/24/72/168h`, `obv_z_72` |
| `volume_profile` | 10 | `close_vs_vwap_24/168h`, `mfi_14/21`, `cmf_20`, `ad_z_48/168h`, `vol_spike_2x/3x`, `vw_rsi_14` |
| `candle_structure` | 4 | `candle_body`, `upper_wick`, `lower_wick`, `is_bullish` |
| `candlestick_patterns` | 8 | `bull_engulf`, `bear_engulf`, `doji`, `hammer`, `shooting_star`, `bull/bear_streak`, `marubozu` |
| `price_position` | 3 | `hl_position_24/48/168h` |
| `support_resistance` | 13 | `close_vs_pivot/r1/r2/s1/s2`, `dist_round_1000/10000`, `breakout_up/down_24/48/168h` |
| `fibonacci` | 10 | `fib_position_48/168h`, `fib_nearest_dist/level/dist_618/below_618` × 2 lookbacks |
| `ichimoku` | 10 | `tk_ratio`, `tk_cross_bull/bear`, `close_vs_cloud_top/bottom`, `inside_cloud`, `cloud_thickness/bullish/flip_recency`, `close_vs_kijun` |
| `supertrend` | 8 | `supertrend_dir/dist_15/20/30`, `supertrend_consensus`, `supertrend_flip_recency` |
| `divergences` | 5 | `rsi_divergence`, `macd_divergence`, `obv_divergence`, `vol_price_div_12/24h` |
| `statistical` | 13 | `hurst_168h`, `skew/kurt_24/72/168h`, `autocorr_ret_1/6/12/24h`, `var_ratio_6/24h` |
| `composite` | 7 | `trend_score`, `rsi_vol_confirm`, `mom_coherence`, `sharpe_ratio_24/72/168h`, `regime_composite` |
| `calendar` | 11 | `hour_sin/cos`, `dow_sin/cos`, `halving_cycle_sin/cos/pos`, `dom_sin/cos`, `quarter_sin/cos` |

**Total: 196 features** (193 model inputs + 3 backtest-only)

---

## Feature Set V3 — 21 External Features

**File:** `data/features/BTCUSDT_1h_v3_features.parquet`  
**Generator:** `src/hmats/notebooks/05_feature_selection_lgbm_v1.py`  
**External sources:** multi-coin OHLCV (9 altcoins 1h), CoinGecko market caps, Fear & Greed Index

These features carry partial causal justification beyond pure OHLCV statistics —
they are derived from data external to BTC price history, making them more
defensible against the correlation-vs-causation critique.

| Group | Prefix | Features | Description |
|-------|--------|---------|-------------|
| Cross-asset | `cross_` | `cross_eth_btc_ratio`, `cross_eth_btc_mom_24h`, `cross_eth_btc_mom_72h`, `cross_altcoin_breadth_24h`, `cross_btc_relative_strength`, `cross_alt_correlation_24h` | ETH/BTC ratio and momentum; % of altcoins with positive 24h return; BTC relative strength vs altcoin basket |
| Market structure | `mkt_` | `mkt_btc_dominance`, `mkt_btc_dominance_chg_7d`, `mkt_eth_dominance`, `mkt_total_mcap_chg_24h`, `mkt_stablecoin_pct` | BTC/ETH market cap dominance; total market cap change; stablecoin share |
| Sentiment | `sent_` | `sent_fear_greed`, `sent_fear_greed_ma7`, `sent_fear_greed_chg_7d` | Fear & Greed Index (0–1 scaled), 7-day MA, 7-day change |
| Microstructure | `micro_` | `micro_amihud_illiq`, `micro_kyle_lambda`, `micro_roll_spread`, `micro_volume_clock` | Amihud illiquidity; Kyle lambda (price impact per unit volume); Roll bid-ask spread estimate; volume regularity |
| Enhanced OHLCV | — | `vol_term_structure`, `mom_normalized_24h`, `mom_normalized_72h` | Volatility term structure (24h/168h ratio); volatility-normalized momentum |

**Total: 21 V3 features**

---

## Feature Set V2-1h — 39 Structural Features on 1h Bars

**File:** `data/features/BTCUSDT_1h_structural.parquet`  
**Registry:** `data/features/feature_registry_v2_1h.json`  
**Generator:** `src/hmats/notebooks/01_structural_features_1h.py`  
**Timeframe:** 1h candles  
**Generated:** 2026-05-30

Adapted from the 5m structural pipeline (`01_structural_features.ipynb`).
Window parameters are remapped so each group captures the same economic time
horizons as the original 5m version.

| Group | Prefix | Count | Description |
|-------|--------|-------|-------------|
| `A_structure` | `struct_` | 11 | Confirmed swing high/low (±12h minor, ±48h major), proximity flags, wick rejection ratios, body ratio |
| `B_liquidity` | `liq_` | 10 | Anchored VWAP deviations (daily, weekly, 24h, 168h rolling), POC distance (24h, 168h), volume z-score, exhaustion spikes |
| `C_volatility` | `volat_` | 8 | ATR 20h/72h (% of close), BB width/position (20h), Bollinger-Keltner squeeze ratio, squeeze flag, Garman-Klass volatility (20h, 72h) |
| `D_mtf` | `mtf_` | 10 | 4h EMA spread, 4h RSI, 4h above-EMA50 flag; daily EMA spread, daily RSI; composite MTF alignment score; UTC hour and day-of-week encoding |

### Window Parameter Mapping (5m → 1h)

| Parameter | 5m version | 1h version | Economic horizon |
|-----------|-----------|-----------|-----------------|
| `SWING_ORDER_S` | 12 bars | 12 bars | ±1h → ±12h minor structure |
| `SWING_ORDER_L` | 48 bars | 48 bars | ±4h → ±48h major structure |
| `VOC_WIN_S` | 72 bars | 24 bars | 6h → 24h short liquidity |
| `VOC_WIN_L` | 288 bars | 168 bars | 24h → 168h weekly liquidity |
| `ATR_WIN_S` | 20 bars | 20 bars | ~1.7h → 20h |
| `ATR_WIN_L` | 72 bars | 72 bars | 6h → 72h (3 days) |
| `BB_WIN` | 20 bars | 20 bars | ~1.7h → 20h |
| MTF context | 1h + 4h | 4h + 1D | higher TF for 1h bars |
| `BURN_IN` | 2500 bars | 200 bars | warm-up period |

**Total: 39 V2-1h features**

---

## Feature Set V2-5m — 39 Structural Features on 5m Bars (experimental)

**File:** `data/features/BTCUSDT_5m_structural.parquet`  
**Registry:** `data/features/feature_registry_v2.json`  
**Generator:** `src/hmats/notebooks/01_structural_features.ipynb`  
**Timeframe:** 5m candles (with 1h and 4h MTF context)  
**Generated:** 2026-05-28

Same architecture as V2-1h but at 5-minute granularity. Not yet used in the
main 1h modelling pipeline.

| Group | Prefix | Count | Description |
|-------|--------|-------|-------------|
| `A_structure` | `struct_` | 11 | Swing extrema (±1h minor, ±4h major), wick/body ratios |
| `B_liquidity` | `liq_` | 10 | Anchored VWAP (daily/weekly/rolling), POC distance, volume z-score, exhaustion |
| `C_volatility` | `volat_` | 8 | ATR 20/72 bars pct, BB, Bollinger-Keltner squeeze, Garman-Klass |
| `D_mtf` | `mtf_` | 10 | 1h + 4h EMA alignment, RSI, composite score, session timing |

---

## Early Agent Features — 12 Features (archived)

**File:** `src/hmats/data/features.py` → `FEATURE_COLS`  
**Used in:** early NEAT/PPO RL agents (before the LGBM/TCN phase)

Simple 12-feature set from first experiments:

```
log_ret_1, vol_24, vol_72, sma_ratio_24_72, macd, macd_signal, macd_hist,
mom_24, mom_72, rsi_14, volu_z_72, z_close_72
```

Features are clipped to `[-10, 10]` and standardised using training-set statistics.

---

## Feature Selection Pipeline (current — v5/v6/v7)

Applied in: `src/hmats/notebooks/02_lgbm_omni_0fee_v5.ipynb` and later versions.

Selection runs **exclusively on pre-OOS data** (before 2024-01-01) to prevent
data leakage into the OOS evaluation period.

```
Pool: V1 + V3 [+ V2-1h in v7]   (216 → 255 features)
         │
         ▼
Stage 1 — Variance + Spearman filter
  • Remove features with variance < 1e-6
  • Remove features with pairwise Spearman ρ > 0.85
    (keep whichever of the pair has higher correlation with the label)
  • Typical survivors: ~120
         │
         ▼
Stage 2 — Mutual Information ranking (top 60)
  • sklearn mutual_info_classif, n_neighbors=5
  • Keep top 60 features by MI score against the label
         │
         ▼
Stage 3 — Walk-forward stability (≥50% of windows)
  • Split training data into overlapping windows
  • In each window compute MI and record the top-K features
  • Keep features that appear in ≥50% of all windows
  • Key stage: spurious correlations are time-unstable; genuine signals recur
  • Typical survivors: ~25–40
         │
         ▼
Stage 4 — Permutation importance pruning
  • Train LGBM on split 2021–2022 / val 2023
  • Permutation importance (10 repeats, scoring=roc_auc)
  • Keep features with importance_mean > 0.0005
  • Typical survivors: ~13–20
```

### Pipeline Hyperparameters

| Parameter | Value |
|-----------|-------|
| `VAR_THRESHOLD` | `1e-6` |
| `CORR_THRESHOLD` | `0.85` (Spearman) |
| `TOP_K_MI` | `60` |
| `MIN_STABLE_FRAC` | `0.50` |
| `PERM_THRESHOLD` | `0.0005` |
| Stage 4 train | 2021-01-01 – 2022-12-31 |
| Stage 4 val | 2023-01-01 – 2023-12-31 |
| OOS start | 2024-01-01 |

---

## Selected Features — v5 Experiment (2026-05-29)

**Notebook:** `src/hmats/notebooks/02_lgbm_omni_0fee_v5.ipynb`  
**Artifacts:** `artifacts/02_lgbm_omni_0fee_v5/results.json`  
**Pool:** V1 (195) + V3 (21) = 216

### Selection Funnel

| Stage | Features | Notes |
|-------|----------|-------|
| Pool | 216 | 195 V1 + 21 V3 |
| Stage 1 (Variance+Corr) | 119 | |
| Stage 2 (MI Top-60) | 60 | |
| Stage 3 (Stability) | 41 | |
| Stage 4 (Perm prune) | **20** | final set |

### 20 Selected Features

| # | Feature | V1 Group | External? |
|---|---------|----------|-----------|
| 1 | `stoch_k_14` | oscillators | |
| 2 | `cross_altcoin_breadth_24h` | — | **V3** |
| 3 | `ret_2h` | returns | |
| 4 | `ad_z_48h` | volume_profile | |
| 5 | `macd_hist_5_13` | macd | |
| 6 | `bear_streak` | candlestick_patterns | |
| 7 | `ret_1h` | returns | |
| 8 | `close_vs_sma_7` | ma_ratios | |
| 9 | `ret_3h` | returns | |
| 10 | `rsi_divergence` | divergences | |
| 11 | `hour_cos` | calendar | |
| 12 | `candle_body` | candle_structure | |
| 13 | `close_vs_sma_720` | long_cycle_ma | |
| 14 | `vw_rsi_14` | volume_profile | |
| 15 | `atr_14_pct_rank` | volatility_regime | |
| 16 | `macd_divergence` | divergences | |
| 17 | `hl_position_48h` | price_position | |
| 18 | `vol_ratio_72h` | volume | |
| 19 | `close_vs_s1` | support_resistance | |
| 20 | `skew_24h` | statistical | |

Only one V3 feature survived all four stages: `cross_altcoin_breadth_24h`
(fraction of altcoins with a positive 24h return — an altcoin market breadth indicator).

### Backtest Results (WFO EXP scheme, OOS 2024-01-01+)

| Scenario | Threshold | Trades | Win Rate | Total Return | Sharpe | Max DD |
|----------|-----------|--------|----------|--------------|--------|--------|
| No fees | 0.55 | 1 021 | 45.2% | +54.6% | 0.63 | −23.9% |
| No fees | 0.60 | 507 | 47.3% | +70.0% | 1.11 | −17.6% |
| With fees (maker 0.02%, taker 0.05%) | 0.55 | 1 021 | 45.1% | **−25.1%** | −0.42 | −43.7% |
| With fees | 0.60 | 507 | 46.7% | **+19.4%** | 0.37 | −22.8% |

**Key finding:** the model is profitable before fees at both thresholds. At
threshold 0.60 with fees the result is still positive (+19.4%), but Sharpe
is low (0.37). Transaction costs are the dominant performance driver.

---

## Selected Features — v6 Experiment (2026-05-30)

**Notebook:** `src/hmats/notebooks/02_lgbm_omni_0fee_v6.ipynb`  
**Artifacts:** `artifacts/02_lgbm_omni_0fee_v6/results.json`  
**Pool:** V1 (195) + V3 (21) = 216 — same as v5  
**Key addition vs v5:** conviction-weighted leverage (1×–20× linear in prob), Futures fee model

Identical feature set to v5 (same 20 features). Key backtest change: ATH-window
evaluation starts 2024-11-10 (first bar BTC reached current ATH levels).

### Backtest Results (ATH window, threshold 0.60, max leverage 20×)

| Scenario | Return | Sharpe | Max DD | Avg Leverage |
|----------|--------|--------|--------|--------------|
| 0.60 threshold, 20× cap | +30.8% | 0.41 | −31.3% | 2.06× |
| 0.55 threshold, 20× cap | −62.9% | −0.85 | −78.5% | 2.43× |

Leverage hurt at threshold 0.55 (too many losing trades amplified). At 0.60
it added modest gains but with higher drawdown than the 1× equivalent.

---

## Version History

| Version | Notebook | Feature Pool | Notes |
|---------|----------|-------------|-------|
| Early agents | `src/hmats/data/features.py` | 12 OHLCV features | NEAT/PPO, no WFO |
| V1 | `data/features/feature_registry.json` | 196 OHLCV 1h | Full technical indicator set |
| V3 | `05_feature_selection_lgbm_v1.py` | V1 + 21 external | Cross-asset, sentiment, microstructure |
| V2-5m (exp.) | `01_structural_features.ipynb` | 39 structural 5m | MTF 1h+4h, swing/liquidity/squeeze |
| V2-1h | `01_structural_features_1h.py` | 39 structural 1h | MTF 4h+daily, adapted for 1h pipeline |
| **v7 (current)** | `02_lgbm_omni_0fee_v7.ipynb` | **V1+V3+V2-1h = 255** | First run including structural features |
