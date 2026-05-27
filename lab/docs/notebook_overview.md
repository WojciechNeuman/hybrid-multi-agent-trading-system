# Notebook Overview: Core Approaches

This document covers the seven foundational notebooks:
- `01_data_exploration.ipynb` — data pipeline and feature engineering
- `07_grid_search_vtrain3.ipynb` — LightGBM agent with walk-forward optimization
- `07_grid_search_vtrain4.ipynb` — LightGBM vtrain3 + three mechanical fixes (intra-candle H/L, 0.1% fees, K-Fold fix)
- `07_grid_search_vtrain5.ipynb` — LightGBM vtrain4 + passive market-making execution (maker entry, buffer, asymmetric fees, funding)
- `12_tcn_grid_search_vtrain.ipynb` — TCN (deep learning) agent with purged K-Fold (vtrain1)
- `12_tcn_grid_search_vtrain2.ipynb` — TCN vtrain1 + five mechanical fixes (fold-specific val, TBM intra-candle, fees, WFO)
- `12_tcn_grid_search_vtrain3.ipynb` — TCN vtrain2 + passive market-making execution (ATR SL/TP, maker entry, buffer, fees)

---

## 1. `01_data_exploration.ipynb` — Data & Feature Pipeline

### Purpose
End-to-end data preparation: fetch OHLCV data, compute features, save to parquet.

### Data
- **Universe**: 10 cryptocurrencies (BTC, ETH, BNB, XRP, SOL, ADA, DOGE, AVAX, DOT, LINK)
- **Source**: Binance REST API, 1-hour candles
- **Range**: up to ~76,525 rows per asset (BTC/ETH from 2017-08-17, newer coins from later)
- **Storage**: one `.parquet` per symbol in `data/raw/`

### Feature Engineering (BTCUSDT only)
Builds **195 features** across 22 groups, saved to `data/features/BTCUSDT_1h_features.parquet`:

| Group | Count | Examples |
|---|---|---|
| returns | 18 | `ret_1h`, `log_ret_168h` |
| ma_ratios | 12 | `close_vs_sma_50`, `close_vs_ema_200` |
| volatility | 10 | `atr_14`, `vol_24h` |
| volatility_regime | 11 | `gk_vol_24h`, `bb_squeeze_20`, `atr_14_pct_rank` |
| ichimoku | 10 | `tk_ratio`, `cloud_bullish`, `cloud_flip_recency` |
| fibonacci | 10 | `fib_position_168h`, `fib_dist_618_48h` |
| support_resistance | 13 | pivot points, breakout magnitude |
| candlestick_patterns | 8 | `bull_engulf`, `hammer`, `doji`, `bull_streak` |
| ma_crosses | 9 | `golden_cross`, `ma_bull_score`, `ma_ribbon_width` |
| supertrend | 8 | `supertrend_dir_20`, `supertrend_flip_recency` |
| volume_profile | 10 | `mfi_14`, `cmf_20`, `vw_rsi_14`, `close_vs_vwap_24h` |
| statistical | 13 | `hurst_168h`, `skew_72h`, `autocorr_ret_6h`, `var_ratio_6h` |
| composite | 7 | `trend_score`, `mom_coherence`, `sharpe_ratio_24h` |
| calendar | 11 | `hour_sin/cos`, `halving_cycle_sin/cos`, `quarter_sin/cos` |
| divergences | 5 | `rsi_divergence`, `macd_divergence`, `obv_divergence` |
| long_cycle_ma | 11 | `close_vs_sma_720`, `close_vs_ema_4320`, `weekly_mom_accel` |
| bollinger | 4 | `bb_width_20`, `bb_position_50` |
| oscillators | 6 | `rsi_7`, `rsi_14`, `stoch_k_14`, `williams_r` |
| macd | 4 | `macd_12_26`, `macd_hist_5_13` |
| volume | 9 | `vol_z_24h`, `vol_ratio_168h`, `obv_z_72` |
| price_position | 3 | `hl_position_24h`, `hl_position_168h` |
| candle_structure | 4 | `candle_body`, `upper_wick`, `lower_wick`, `is_bullish` |

### Label
Binary: `label = 1` if next candle closes higher, `0` otherwise.

### Output
- `data/features/BTCUSDT_1h_features.parquet` — 74,366 rows × 199 cols (195 features + 3 backtest cols + label)
- `data/features/feature_registry.json` — machine-readable registry of all feature names/groups

---

## 2. `07_grid_search_vtrain3.ipynb` — LightGBM with Walk-Forward Optimization

### Purpose
Train a LightGBM binary classifier on BTCUSDT, find optimal trading parameters via purged K-Fold, and evaluate with monthly walk-forward retraining on the test set.

### Data Splits
| Split | Range | Rows |
|---|---|---|
| Train+Val | 2017-11-15 → 2024-11-10 | 61,118 |
| Test | 2024-11-10 → 2026-05-16 | 13,248 |

**Features**: 50 selected features from `lgbm_features.csv` (subset of the 195 from nb01).

### Model
**LightGBM** binary classifier:
- `objective: binary`, `metric: binary_logloss`
- `n_estimators=1000`, `learning_rate=0.02`, `num_leaves=31`
- `subsample=0.8`, `colsample_bytree=0.8`, `reg_alpha=0.1`, `reg_lambda=1.0`
- Label: binary (next candle up/down)
- No explicit preprocessing (tree models are invariant to feature scaling)

### Training: Two-Phase Methodology

#### Phase 1 — Purged K-Fold (K=5) on TrainVal
Goal: generate honest OOS probabilities on the 61k-bar TrainVal set.

- Chronological K-Fold: 5 folds × ~12,223 rows each
- **Embargo**: 168 bars each side of each fold boundary
- **Internal validation**: last 2,500 bars of the training window for early stopping
- Early stopping rounds: 50
- OOS AUC: reported per fold
- Output: OOS `P(up)` for every bar in TrainVal

Fold results (best_iterations): 64, 90, 126, 63, 372 → stable, no premature stopping

#### Phase 2 — Walk-Forward Optimization (WFO) on Test
- Monthly retraining: step = 720h
- 19 retraining steps over 13,248 test bars
- Each step expands the training window (expanding-window WFO)
- Same LGB hyperparameters, same 2,500-bar internal val
- Trading parameters come **only** from Phase 1 grid search — never re-optimized on test

### Grid Search (on OOS probs from Phase 1)
1,944 valid combinations; optimized for Sharpe ratio.

Grid parameters:
```
long_threshold:    [0.54, 0.55, 0.57, 0.59, 0.61]
short_threshold:   [0.39, 0.41, 0.43, 0.45, 0.46]
sl_atr_multiplier: [1.5, 2.0, 2.5]
tp_atr_multiplier: [1.5, 2.0, 3.0]   ← ATR-relative (key fix vs vtrain2)
min_sl:            [0.010, 0.015, 0.020]
min_hold:          [4, 6, 8]
max_hold:          [24, 48]
cooldown:          [2, 3]
```

Both SL and TP are **ATR-relative** (TP = `tp_atr_mult × ATR at entry`). This prevents curve-fitting to a specific volatility regime.

Exit logic:
- Stop-loss: `pnl < -max(sl_atr_mult × ATR, min_sl)`
- Take-profit: `pnl > tp_atr_mult × ATR`
- Max hold: exit after `max_hold` bars
- Confidence exit: if `hold_count >= min_hold` and signal drops below threshold
- Cooldown: `cooldown` bars before next trade

### Best Parameters
```
long_threshold    = 0.57
short_threshold   = 0.46
sl_atr_multiplier = 2.5
tp_atr_multiplier = 1.5
min_sl            = 0.010
min_hold          = 4
max_hold          = 48
cooldown          = 3
```

### Results

| Strategy | Total Return | Sharpe | Max DD | Trades | Win Rate |
|---|---|---|---|---|---|
| WFO monthly retrain | **+106.13%** | **1.397** | -21.60% | 897 | 65.2% |
| Static model (nb06) | +54.35% | 0.839 | -25.46% | 971 | 64.4% |
| Buy & Hold | +3.14% | 0.044 | -50.08% | — | — |

OOS K-Fold (TrainVal sanity check): Sharpe=1.764, Return=+1321% — note this is inflated due to long OOS history 2017–2024; main signal is the test Sharpe.

---

## 3. `12_tcn_grid_search_vtrain.ipynb` — TCN with Purged K-Fold

### Purpose
Train a Temporal Convolutional Network (TCN) as a multi-task model on BTCUSDT, select trading parameters via purged K-Fold on training data, and evaluate once on test.

### Data Splits
| Split | Range | Rows |
|---|---|---|
| Train | 2017-11-27 → 2024-06-01 | 56,949 |
| Val | 2024-06-01 → 2024-11-10 | 3,888 |
| Test | 2024-11-10 → 2026-05-15 | 13,236 |

**Features**: 51 inputs = 50 selected LGBM features + `fracdiff_close` (fractionally differenced log-price with d=0.4, threshold=1e-4).

### Preprocessing
- **QuantileTransformer** (output_distribution='normal', n_quantiles=1000): fitted once on `train_df`, applied to all splits
- **Fractional differentiation** (FFD, d=0.4): applied to log-close to achieve stationarity while preserving memory

### Labels: Triple Barrier Method (TBM)
3-class labels (not binary like LGBM):
- `1` (up): price hits upper barrier = `entry × (1 + PT × σ)` within 12h
- `0` (down): price hits lower barrier = `entry × (1 - SL × σ)` within 12h
- `2` (neutral): neither barrier hit before 12h vertical stop

Parameters: `TBM_VOL_WINDOW=24h`, `TBM_PT=1.0`, `TBM_SL=1.0`, `TBM_VERT_H=12`

### Model: TCNMultiTask
103,652 parameters; runs on Apple MPS.

Architecture:
- 4 causal TCN blocks: channels=[64,64,64,64], kernel=3, dilation=2^i per block
- Weight-normalization on conv layers
- Two heads sharing the TCN backbone:
  - **Direction head**: LayerNorm → Linear(64→32) → GELU → Dropout → Linear(32→3) → 3-class logits
  - **Volatility head**: LayerNorm → Linear(64→16) → GELU → Linear(16→1) → Softplus → scalar vol prediction

Sequence length: `SEQ_LEN=24` (24-hour lookback window).

### Training
- **Optimizer**: AdamW, lr=2e-3, weight_decay=1e-3
- **Scheduler**: warmup 5 epochs + cosine annealing
- **Loss**: `L = CE(direction) + 0.5 × Huber(volatility)`, weighted by `sample_weight` (proportional to current volatility)
- **Class weights**: inverse-frequency weighting (TBM labels are imbalanced)
- **Gradient clipping**: max_norm=1.0
- Batch size: 256

Fold training: 60 epochs max, patience=10 (validation uses the held-out `val_df` for all folds).

### Training: Purged K-Fold (K=5, train split only)

Unlike LGBM which used TrainVal together, TCN's K-Fold runs **only on `train_df`** (56,949 rows).
Val split is used as shared early-stopping validation for all folds.

- Fold size: ~11,389 rows
- **Embargo**: 168 bars each side
- Sequences are built only from **contiguous segments** (no sequence may straddle the embargo gap)
- Lookback context: up to `SEQ_LEN-1=23` rows from before the fold boundary
- Each fold trains a fresh model → predicts on fold rows → OOS probs stored

Fold results (all converged ~11 epochs due to patience=10):
```
Fold 1: best_val_loss=0.972, early stop ep 11, 11,365 OOS predictions
Fold 2: best_val_loss=0.966, early stop ep 11, 11,388 OOS predictions
Fold 3: best_val_loss=0.962, early stop ep 12, 11,388 OOS predictions
Fold 4: best_val_loss=0.975, early stop ep 12, 11,388 OOS predictions
Fold 5: best_val_loss=0.982, early stop ep 11, 11,392 OOS predictions
Total OOS coverage: 56,921 / 56,949 (100%)
```

### Grid Search (on OOS probs from K-Fold)
10,800 combinations; optimized for Sharpe.

Grid parameters:
```
long_threshold:  [0.44, 0.47, 0.50, 0.53, 0.56]
short_threshold: [0.44, 0.47, 0.50, 0.53, 0.56]
atr_multiplier:  [1.5, 2.0, 2.5, 3.0]
min_sl:          [0.010, 0.015, 0.020]
take_profit:     [0.025, 0.035, 0.045]   ← static %, not ATR-relative
min_hold:        [4, 6, 8]
max_hold:        [24, 48]
cooldown:        [2, 3]
```

Note: TCN grid uses **static % take-profit** (not ATR-relative like LGBM vtrain3).

Signal logic:
- Long: `P(up) >= long_threshold`
- Short: `P(down) >= short_threshold AND P(up) < long_threshold`
- Exit: ATR-based SL → static % TP → max hold → confidence exit (after min hold)

### Best Parameters
```
long_threshold  = 0.53
short_threshold = 0.56
atr_multiplier  = 1.5
min_sl          = 0.020
take_profit     = 0.045
min_hold        = 8
max_hold        = 24
cooldown        = 3
exit_threshold_long  = 0.43  (derived: long_threshold - 0.10)
exit_threshold_short = 0.46  (derived: short_threshold - 0.10)
```

### Results

| Split | Total Return | Sharpe | Max DD | Trades | Win Rate |
|---|---|---|---|---|---|
| OOS K-Fold (train) | +329.79% | 1.150 | -37.7% | 388 | — |
| **Test** | **+11.68%** | **0.446** | -19.08% | 75 | 60.0% |
| Buy & Hold (test) | -1.02% | — | — | — | — |

Test exit breakdown:
- Confidence exits: 51 trades, 80.4% win rate, avg +0.89%
- Stop-loss hits: 17 trades, 0% win rate, avg -3.04%
- Take-profit hits: 4 trades, 100% win rate, avg +5.70%
- Max hold exits: 3 trades, 0% win rate, avg -1.10%

Very long/short asymmetry on test: 72 longs vs only 3 shorts (model output skewed toward P(up)).

---

## Comparison

### Similarities
1. **Same base data**: BTCUSDT 1h, same 195-feature parquet from nb01
2. **Same feature subset**: both use the 50 `lgbm_features.csv` features (TCN adds `fracdiff_close`)
3. **Same evaluation philosophy**: purged K-Fold with 168-bar embargo to prevent label leakage
4. **Same backtest logic**: ATR stop-loss, min/max hold, cooldown, confidence exit
5. **Same test split**: 2024-11-10 → 2026-05-15/16
6. **Same optimization target**: Sharpe ratio

### Differences

| Aspect | LGBM (nb07) | TCN (nb12) |
|---|---|---|
| Model type | Gradient boosting (LightGBM) | Deep learning (TCN, 4-block, causal conv) |
| Parameters | ~1000 trees × 31 leaves | 103,652 neural network weights |
| Label type | Binary: next candle up/down | TBM 3-class: up / down / neutral |
| Preprocessing | None (trees scale-invariant) | QuantileTransformer → Gaussian |
| Extra feature | — | `fracdiff_close` (d=0.4) |
| Sequence structure | No temporal window (tabular) | 24-hour lookback sequences |
| K-Fold data | Full TrainVal (61k rows) | Train only (57k rows) |
| WFO phase | Yes — monthly retrain on test | No — production model applied once |
| TP definition | ATR-relative (`tp_atr_mult × ATR`) | Static % (`take_profit = 0.045`) |
| Grid size | 1,944 valid combos | 10,800 combos |
| Train time | ~7s (K-Fold), ~26s (WFO) | ~30–90 min (5 folds × neural net) |
| Test trades | 897 | 75 |
| Test Sharpe | **1.397** | 0.446 |
| Test return | **+106.13%** | +11.68% |
| Test max DD | -21.60% | -19.08% |
| OOS Sharpe | 1.764 | 1.150 |

### Key takeaways
- **LGBM significantly outperforms TCN on the test set** (Sharpe 1.40 vs 0.45, Return 106% vs 12%). LGBM benefits from monthly retraining (WFO) while TCN uses a frozen production model.
- **TCN is far more selective** (75 trades vs 897). Its TBM labels filter for high-conviction moves, but the model outputs are heavily biased toward longs (72 vs 3 shorts), suggesting the model hasn't learned the short side well.
- **OOS → test decay is much sharper for TCN** (Sharpe 1.15 → 0.45) than LGBM (Sharpe 1.76 → 1.40), suggesting the TCN overfit more within the K-Fold training period or that the static TP hurts in production.
- **LGBM's ATR-relative TP** (vtrain3 fix) is likely one reason for its more robust test performance — it adapts to changing volatility regimes rather than using a fixed threshold.
- Both models beat buy-and-hold on the test period, which was relatively flat for BTC (+3.14% / -1.02%).

---

## 4. `07_grid_search_vtrain4.ipynb` — LightGBM + Three Mechanical Fixes

### Purpose
Fix three critical backtesting/training flaws in vtrain3 that inflate reported performance before any capital allocation.

### Fixes Applied

#### Fix 1 — Intra-candle SL/TP (Survivorship Bias Removal)
vtrain3 checked stop-loss and take-profit only against the **hourly close price**, ignoring intra-candle wicks. On 1h BTC candles, a 3% wick down followed by a +2% close would be recorded as a winning trade even though the SL would have been triggered live.

vtrain4 passes `high` and `low` arrays (loaded from `data/raw/BTCUSDT_1h.parquet` and joined to the feature parquet) into `run_backtest`:
- **Long:** check `low ≤ sl_price` → SL hit first; then `high ≥ tp_price` → TP hit
- **Short:** check `high ≥ sl_price` → SL hit first; then `low ≤ tp_price` → TP hit
- If both SL and TP are breached in the same candle → **SL assumed** (cannot know tick order)
- SL/TP exits use the exact SL/TP price (not close); confidence/max-hold exits still use close

#### Fix 2 — Transaction Fees (0.1% per side)
vtrain3 applied no fees. With 897 test trades, even 0.1%/side (0.2% round-trip, conservative Binance taker) creates substantial drag. The compound fee drag over N trades = `(1 - 0.002)^N`. For 897 trades: `0.998^897 ≈ 0.165`, meaning 83.5% of gross profit is consumed by fees alone.

Applied in `run_backtest`:
```python
# Long entry:  units = cash * (1 - fee) / entry_px
# Long exit:   cash  = units * exit_px * (1 - fee)
# Short entry: entry_cash = cash * (1 - fee)
# Short exit:  cash = entry_cash * (1 + pnl_gross) * (1 - fee)
```
The grid search on fee-inclusive OOS equity naturally selects **fewer, longer-held trades** to amortise costs.

#### Fix 3 — K-Fold Internal Validation Temporal Alignment
vtrain3 used `Xtr_k[-n_int:]` (last 2,500 bars of the masked training set) as the internal val for LightGBM early stopping. Because the mask includes all data outside the fold+embargo, this always selects **late-2024 bars** regardless of which fold is being evaluated. Early stopping for the 2018 fold was therefore guided by 2024 market dynamics.

vtrain4 carves the internal val from the **`n_int` bars immediately preceding the fold's embargo zone**:
```
int_end   = emb_start           # bar just before embargo
int_start = emb_start - n_int   # n_int bars back
```
These bars are excluded from both training and the fold test, and closely match the epoch being predicted. Edge case (fold 1, no prior data): falls back to first `min(2500, 1000)` bars after the embargo.

### Architecture (unchanged from vtrain3)
- K=5 purged K-Fold, embargo=168h, 2500-bar internal val, early stopping=50 rounds
- WFO monthly retraining (720h steps, 19 steps)
- ATR-relative SL and TP
- Same LightGBM hyperparameters

### Grid Changes vs vtrain3
- `long_threshold` range extended upward: `[0.54, 0.56, 0.58, 0.60, 0.62]` (higher conviction needed to justify fees)
- `min_hold` extended: `[4, 6, 8, 12]` (12h added to amortise round-trip cost)
- TP definition: ATR-relative (inherited from vtrain3)

### Expected Performance Impact
| Effect | Estimated Magnitude |
|---|---|
| Intra-candle SL (more SL hits) | Moderate negative |
| Intra-candle TP (exact TP price captured) | Small positive |
| Fees on 897 trades (vtrain3 params) | `(1-0.002)^897 − 1 ≈ −83.5%` of gross equity |
| K-Fold fix (better calibrated early stopping) | Better OOS→test generalisation |
| Grid re-optimisation with fees | Recovers some performance via fewer trades |

The actual test Sharpe and return are only known after running the notebook. The comparison cell shows vtrain3-style backtest (close-only, no fee) vs vtrain4 (H/L + 0.1% fee) on the same WFO probabilities.
