# Notebook Analysis — Hybrid Multi-Agent Trading System

**Generated:** 2026-05-27  
**Scope:** All notebooks in `src/hmats/notebooks/`  
**Purpose:** Research trajectory audit, methodology issue tracking, delta analysis between versions

---

## 1. Executive Summary

The project started as a traditional ML pipeline (binary direction classification) and evolved through five major phases:

1. **Early exploration (01–06):** Data pipeline, rule-based agents, NEAT/PPO reinforcement learning, LSTM/GRU with naive binary labels, and a first LGBM model — all without realistic fees and with varying degrees of data leakage.

2. **LGBM grid search (07 series, vtrain1–7):** A long iterative sequence fixing one assumption at a time. The original grid search evaluated parameters *on the test set* (direct leakage). vtrain1 introduced purged K-Fold. vtrain2 introduced WFO. vtrain3 switched to ATR-relative TP. vtrain4 added intra-candle SL/TP and fees (0.1%/side) — **which killed all profit** (best OOS Sharpe = -0.015). vtrain5-6 tried passive maker-entry to reduce fees, vtrain7 switched to TBM labels and multiclass output. All versions are negative on the actual test set after fees.

3. **TCN series (11–12, vtrain1–4):** A Temporal Convolutional Network with TBM labels and fractional differentiation. vtrain1 showed Sharpe 0.446 on test (no fees, close-only). vtrain2 added TBM execution logic and fees → disaster (-34% to -88%). vtrain3 switched to maker entry → still negative (-72%). vtrain4 reworked TBM labels (±2×ATR) and Spot/Futures routing → best OOS Sharpe 0.781 but **test results not yet shown** in the notebook (training was ongoing when captured).

4. **Mamba (13):** Selective SSM with O(N) scan, TBM labels, Spot/Futures routing. OOS Sharpe 0.602, **test Sharpe -0.337**, total return -8.10% — the least negative of the realistic approaches, and smallest max drawdown (-16.94%).

5. **GRU (10):** Complete collapse — output range [0.487, 0.524], zero trades generated.

**Key theme:** Every time a real execution constraint was added (fees, intra-candle SL/TP, realistic fill), the strategies went from profitable to negative. The OOS K-Fold Sharpe consistently overstates test performance by 1–2 Sharpe units, pointing to a regime shift between 2024 training and the Nov 2024–May 2026 test period (bitcoin-range-bound → bull-to-correction) combined with remaining methodology issues.

**Current state:** All three most recent architectures (LGBM vtrain7, TCN vtrain4, Mamba vtrain8) are negative on the honest test set. The "signal" detected by AUC (0.55–0.58) is real but small, and current trading execution cannot extract profit net of fees.

---

## 2. Early Notebooks (01–06)

### Data Pipeline

| Notebook | Purpose | Key Output |
|---|---|---|
| `01_data_exploration.ipynb` | Fetch OHLCV from Binance (10 symbols), compute indicators, save parquet | Dataset: 76k rows of BTC 1h from 2017-08-17; indicators via `compute_indicators()` |
| `01_data_exploration_v1.ipynb` | Earlier version, same purpose | Similar |
| `02_first_run.ipynb` | Wire up RSIAgent + SMACrossoverAgent + Supervisor | Tests rule-based multi-agent framework; no backtesting |
| `03_neat_training.ipynb` | NEAT + PPO reinforcement learning | Fee=0.05%, PPO 200k steps, NEAT 20 generations; TradingEnv with `buy_and_hold_metrics` |
| `04_rolling_test.ipynb` | Rolling test of NEAT/PPO agents | Not listed in primary analysis scope |

### LSTM/GRU/LGBM (05–06 series)

| Notebook | Model | Labels | Split | Fees | Key Result | Problems |
|---|---|---|---|---|---|---|
| `05_lstm_agent.ipynb` | 2-layer LSTM (128 units) | Binary next-close direction | Train ≤2024-11-10, test after | None | Train acc 0.59, test acc 0.51 | No fees; no val set; no early stopping; test accuracy near chance |
| `05_lstm_agent_v2.ipynb` | 2-layer LSTM (64 units) + LayerNorm | Binary next-close direction | Train/Val/Test split (val added) | None | Early stop at epoch 14, val acc 0.50 | No fees; model collapsed — output range ~0.49–0.51 |
| `05_lstm_agent_v3.ipynb` | LSTM variant | Similar | Similar | None | Not shown (no output cells reviewed) | Likely similar collapse |
| `05_lstm_agent_v4.ipynb` | LSTM variant | Similar | Similar | None | Not shown | Same |
| `06_lgbm_agent.ipynb` | LightGBM binary + RF importance | Binary next-close; 195 features, top 50 selected | Train 2017–2024-06, Val 2024-06–2024-11, Test 2024-11→ | 0.05%/side (modeled in backtest) | Val AUC 0.556, Test AUC 0.549; total return **-3.73%**, Sharpe not clearly printed | Grid search parameters evaluated on test set directly; 852 trades in 6 months is very high frequency; SL too tight |
| `06_lgbm_agent_v1.ipynb` | LightGBM binary; 74 features, top 20 | Same | Same | 0.05%/side | Val acc 0.539, Val AUC ≈ 0.556 | Same leakage |
| `06_lgbm_agent_v2.ipynb` | Extended version | Similar | Similar | Not visible | Not shown | Similar |
| `06_lgbm_agent_v3.ipynb` | Extended version | Similar | Similar | Not visible | Not shown | Similar |

**Feature engineering in 06:** 195 features spanning bollinger, calendar, candle structure, candlestick patterns, composite, divergences, fibonacci, ichimoku, long-cycle MA, MA crosses, MA ratios, MACD, oscillators, price position, returns, statistical, supertrend, support/resistance, volatility, volatility regime, volume, volume profile. Label = `close.shift(-1) > close` (next-bar binary).

**Critical problems in 06 series:**
- Label computed as `close.shift(-1) > close` — straightforward next-bar classification, but the label references prices that are not available at prediction time when combined with a window of features. This is borderline: the label itself is correct (we predict whether bar t+1 goes up), but using `shift(-1)` on the **same DataFrame** that contains features computed at bar t is correct only if the label is removed before fitting. This appears to be done correctly.
- The grid search in `07_grid_search.ipynb` and `07_grid_search_1.ipynb` directly evaluates on the test set. All the best Sharpe values (1.66, 1.44, etc.) are post-hoc overfit parameters.

---

## 3. LGBM Grid Search Evolution (07 series)

### Summary Table

| Version | Key Change | OOS Eval Method | Best OOS Sharpe | Test Sharpe | Test Return | Notes |
|---|---|---|---|---|---|---|
| `07_grid_search.ipynb` | Baseline grid on test set | **Test set directly** | 1.661 (test, not OOS!) | N/A (IS leakage) | +149.6% | Critical leakage |
| `07_grid_search_1.ipynb` | Same as above (duplicate?) | **Test set directly** | ~1.6 | N/A | Similar | Critical leakage |
| `07_grid_search_vtrain.ipynb` (vtrain1) | Purged K-Fold on train only (K=5, embargo=168h); grid on OOS probs | Purged K-Fold OOS (57k bars) | 1.150 (OOS, train period) | Not reported | Not reported | Early stopping uses global val_df → look-ahead for some folds |
| `07_grid_search_vtrain2.ipynb` (vtrain2) | Merged Train+Val for K-Fold; Walk-Forward (WFO) on test at 168h steps | K-Fold OOS on TrainVal (61k bars) | OOS AUC 0.5743 | Not directly reported | Not directly reported | K-Fold internal val = last 1000 bars of full dataset (temporal mismatch bug) |
| `07_grid_search_vtrain3.ipynb` (vtrain3) | Larger internal val (2500 bars); ATR-relative TP; monthly WFO (720h) | K-Fold OOS on TrainVal | Best OOS Sharpe 1.764 | WFO Sharpe +1.397 (but 0% fees) | +106.1% (no fees) | Still no fees; massive OOS return of +132,111% (leakage artifact from ATR-TP on 2017–2024 bull) |
| `07_grid_search_vtrain4.ipynb` (vtrain4) | **Intra-candle SL/TP (high/low)** + **0.1% fee/side** + temporal K-Fold val | K-Fold OOS on TrainVal | Best OOS Sharpe **-0.015** | WFO Sharpe **-0.674** | **-18.15%** | Fees destroyed edge; 897 trades × 0.2% RT ≈ 179pp drag |
| `07_grid_search_vtrain5.ipynb` (vtrain5) | Passive maker entry (limit pullback), asymmetric fees (Maker=0%, Taker=0.05%), funding | K-Fold OOS on TrainVal | OOS Sharpe 0.520 | WFO Sharpe **-1.096** | **-32.96%** | Bug in limit price formula; WFO worse than static |
| `07_grid_search_vtrain6.ipynb` (vtrain6) | Fixed limit price formula (`close*(1-mult*ATR)` not `close-mult*ATR`) | K-Fold OOS on TrainVal | OOS Sharpe 1.012 | WFO Sharpe **-0.933** | **-19.06%** | OOS-test gap still large; ~18% fill rate (most orders never fill) |
| `07_grid_search_vtrain7.ipynb` (vtrain7) | TBM labels (±2.0×ATR, 24h horizon), multiclass LGBM (3 classes), Spot/Futures routing | Combined model+trading grid on TrainVal K-Fold | OOS Sharpe 1.157 | WFO Sharpe **-1.925** | **-48.85%** | Test worse than any previous; TBM+multiclass OOS overfit |

### Delta Analysis: vtrain1 → vtrain7

| Version → Version | What Changed | Performance Impact |
|---|---|---|
| Baseline → vtrain1 | Replaced test-set grid with purged K-Fold grid on training OOS probs | Sharpe dropped from ~1.6 (IS) to 1.15 (honest); test not evaluated |
| vtrain1 → vtrain2 | Merged Train+Val into K-Fold; added WFO with weekly retraining | WFO underperformed static model by a large margin (instability from 30-day internal val for early stopping) |
| vtrain2 → vtrain3 | Increased internal val to 2500 bars; switched to ATR-relative TP; monthly WFO | WFO Sharpe improved significantly (but 0% fees — misleading) |
| vtrain3 → vtrain4 | Added intra-candle H/L for SL/TP; added 0.1%/side fees | **Profit completely eliminated.** vtrain3 WFO: +106.1%, vtrain4 WFO: -18.15%. Fee drag ≈ 83pp. |
| vtrain4 → vtrain5 | Switched to limit-order pullback entry (Maker=0%), asymmetric fees, funding drag | Test got *worse* (-32.96%) despite lower fees. Bug in limit price formula inflated OOS. |
| vtrain5 → vtrain6 | Fixed limit price bug (`px*(1-mult*atr)` instead of `px-mult*atr`) | Test improved slightly (-19.06%) but still negative. Fill rate dropped to ~17%. |
| vtrain6 → vtrain7 | TBM multiclass labels; joint model+trading grid (48 model configs × 6912 trading); Spot/Futures routing | Test Sharpe worsened to -1.925. OOS looks good (1.157) but test -48.85%. Regime mismatch. |

### Bad Assumptions Table (LGBM)

| Assumption | First Fixed In | Impact |
|---|---|---|
| Grid search on test set | vtrain1 | Sharpe went from ~1.6 to ~1.1 on OOS |
| No fees | vtrain4 | 100+ pp of drag revealed |
| Close-only SL/TP (misses wicks) | vtrain4 | Combined with fees killed edge |
| K-Fold internal val from end of dataset (temporal mismatch) | vtrain4 | Fixed: val carved from pre-fold bars |
| Static TP % (curve-fit to bull vol regime) | vtrain3 | ATR-relative TP introduced |
| Weekly WFO chasing noise | vtrain3 | Monthly WFO adopted |
| Limit price calculated wrong (`px-mult*atr`) | vtrain6 | Fixed: `px*(1-mult*atr)` |

### Final State (vtrain7) Critical Issue Checklist

| Issue | Status |
|---|---|
| Data leakage (test seen during training) | **CLEAN** — K-Fold strictly separates TrainVal from test |
| Transaction fees | **CLEAN** — 0.05% taker, 0% maker, separate Spot/Futures |
| Lookahead bias in features | **CLEAN** — features computed from past OHLCV only |
| Label leakage | **PRESENT** — TBM labels use up to 24 future bars. This is the correct ML approach (labels derived from future outcomes are acceptable for training), but TBM label computation uses intra-bar data (`close_arr[i+j]` loop), which is correct. No contamination of input features. |
| Grid search overfitting | **PARTIALLY CLEAN** — model+trading grid on OOS K-Fold probs, not test set. But 331,776 evaluations on the same 61k-bar dataset creates selection bias. |
| Position sizing | **REALISTIC** — 1x leverage, full-capital per trade |
| Benchmark | **PRESENT** — buy-and-hold comparison shown (+6.28% in test period) |
| OOS-test gap | **CRITICAL CONCERN** — OOS Sharpe 1.157 vs test Sharpe -1.925. Gap = 3.08 Sharpe units. This is large enough to be a methodology problem, not just noise. |

---

## 4. TCN Evolution (11–12 series)

### Summary Table

| Version | Key Change | OOS Method | Best OOS Sharpe | Test Sharpe | Test Return |
|---|---|---|---|---|---|
| `11_tcn_tbm.ipynb` | Initial TCN + TBM + multi-task learning (vol aux); no fees; close-only SL/TP | N/A — single train/val/test | N/A | **-0.082** | -4.21% |
| `12_tcn_grid_search.ipynb` | Grid search on test set directly; model frozen | Test set directly (leakage) | 1.809 (IS) | N/A (IS leakage) | +96.3% |
| `12_tcn_grid_search_vtrain.ipynb` (vtrain1) | Purged K-Fold on train only; grid on OOS probs; no fees; close-only SL/TP | K-Fold OOS (57k bars, train) | 1.150 | **+0.446** | +11.68% |
| `12_tcn_grid_search_vtrain2.ipynb` (vtrain2) | TBM barrier execution (not ATR SL/TP); 0.1%/side fees; WFO with 5-epoch warm-start | K-Fold OOS on train | -9.261 (OOS collapsed) | Static: **-4.006** / WFO: **-10.703** | Static: -34% / WFO: -88.52% |
| `12_tcn_grid_search_vtrain3.ipynb` (vtrain3) | Maker-only pullback entry; ATR SL/TP (replaces TBM barriers); asymmetric fees; no WFO | K-Fold OOS on train | 0.166 | **-2.602** | -71.55% |
| `12_tcn_grid_search_vtrain4.ipynb` (vtrain4) | TBM labels changed to ±2.0×ATR (wider); Spot/Futures routing; new model trained | K-Fold OOS on train | 0.781 | Not visible (training ongoing) | Not yet shown |

### Delta Analysis: TCN vtrain1 → vtrain4

| Version → Version | What Changed | Performance Impact |
|---|---|---|
| 12_tcn → vtrain1 | Moved grid from test set to K-Fold OOS on training data | Test Sharpe 1.809 (IS) → +0.446 (honest). Significant degradation, but still positive! |
| vtrain1 → vtrain2 | Added TBM-barrier execution + 0.1% flat fees + WFO warm-start | Catastrophic: OOS Sharpe -9.26, test static -34%, WFO -88.5%. TBM barriers at 1×σ ≈ 0.2–0.5% generate excessive trades. 5-epoch WFO warm-start destabilized model. |
| vtrain2 → vtrain3 | Reverted to ATR SL/TP execution; maker entry (0% entry fee); removed WFO | OOS Sharpe improved from -9.26 to +0.17. Test still -71.55%. Fill rate 85–91% (pullback too shallow). |
| vtrain3 → vtrain4 | Changed TBM label parameters (1σ/12h → 2.0×ATR/24h); Spot/Futures routing; retrained model from scratch | OOS Sharpe improved to 0.781; TBM label change shifts class distribution — more neutral (15% → same); fewer, larger moves. Test pending. |

### Key TCN Findings

- The TCN architecture itself shows genuine discriminative ability (AUC 0.60 in 11_tcn_tbm).
- Early win rate of 60% (vtrain1, no fees) shows model has real signal on close-to-close moves.
- The model collapses when asked to predict TBM outcomes with 1×σ barriers (too noisy at 12h horizon).
- vtrain4's wider barriers (2.0×ATR) produce a cleaner signal. OOS AUC (long) = 0.54, AUC (short) = 0.57.
- Short predictions are consistently stronger than long predictions across all TCN versions.
- WFO with neural nets (5-epoch warm-start) proved catastrophically unstable. Recommendation: avoid WFO for TCN.

### TCN vtrain4 Critical Issue Checklist

| Issue | Status |
|---|---|
| Data leakage | **CLEAN** — K-Fold on train only, test held out |
| Transaction fees | **CLEAN** — 0.05% taker, 0% maker, Spot/Futures routing |
| Lookahead bias | **CLEAN** — fracDiff uses past bars only; QT fitted on train only |
| Label leakage | **CLEAN** — TBM labels use future bars (correct, labels are targets not features) |
| Grid overfitting | **MINOR CONCERN** — 6912 combos on same 57k bars |
| Benchmark | **PRESENT** |

---

## 5. Mamba (13_mamba_tbm.ipynb)

### Architecture

- **Model:** Pure PyTorch selective SSM (Mamba), 2 layers, D_model=64, D_state=8, sequence length=48h
- **Parameters:** 63,107 (smallest of all architectures tested)
- **Scan:** Chunked vectorized SSM scan (no CUDA kernel dependency, runs on MPS)
- **Multi-task:** Not present (unlike TCN) — single 3-class head

### Data Pipeline

| Component | Detail |
|---|---|
| Labels | TBM ±2.0×ATR, 24h vertical barrier (identical to TCN vtrain4) |
| Features | 197 features (195 from parquet + high/low from raw) |
| Normalization | Per-fold QuantileTransformer (fitted on each fold's training rows) |
| Sequence | 48h trailing window |
| Fractal diff | Not used (unlike TCN) |
| Train/Val/Test | TrainVal 61,118 bars (2017-11-15 → 2024-11-10); Test 13,224 bars |

### K-Fold Process

K=5, embargo=168h, internal val=2500 bars (pre-fold). OOS AUC: Long=0.5572, Short=0.5687. Total OOS coverage 60,883/61,118 bars.

### Results

| Metric | OOS K-Fold (TrainVal) | Test Set |
|---|---|---|
| Sharpe | 0.602 | **-0.337** |
| Total Return | +130.62% | **-8.10%** |
| Max Drawdown | -46.38% | **-16.94%** |
| Trades | 1,061 | 305 |
| Win Rate | 47.7% | 46.2% |
| Fill Rate | 53.0% | 50.2% |
| AUC (Long) | 0.557 | 0.569 |
| AUC (Short) | 0.569 | 0.550 |

**Best params:** long_threshold=0.55, short_threshold=0.50, entry_atr_mult=0.3, sl_atr=1.5×, tp_atr=3.0×, min_sl=1.5%, min_hold=8h, max_hold=48h, cooldown=2

### Mamba Critical Issue Checklist

| Issue | Status |
|---|---|
| Data leakage | **CLEAN** — per-fold QT; K-Fold on TrainVal; test touched once |
| Transaction fees | **CLEAN** — Maker=0%, Taker=0.05%, Spot funding=0%, Futures receive +0.00077%/h |
| Lookahead bias | **CLEAN** — all features past-only; QT per-fold |
| Label leakage | **CLEAN** — TBM labels are targets, not features |
| Grid overfitting | **MINOR** — 6912 combos on 61k-bar OOS |
| Unfilled order cooldown | **FIXED** (v8 fix explicitly mentioned) |
| Benchmark | **PRESENT** |
| OOS-test gap | **CONCERN** — Sharpe gap = 0.602 - (-0.337) = 0.939. Better than LGBM vtrain7 (gap=3.08) but still present. |

**Mamba-specific observation:** 72% of trades are shorts (288 short vs 17 long in test). The model has learned a strong short bias. During the test period (Nov 2024–May 2026), BTC was generally in a bull-to-range regime, which partially explains why short-biased strategies underperform. The model's AUC for short prediction (0.55) slightly exceeded long (0.57 → reversed on test), suggesting model generalization is asymmetric.

---

## 6. GRU Agent (10_gru_agent.ipynb) and Full Grid Search (08_full_grid_search.ipynb)

### GRU (10_gru_agent.ipynb)

| Component | Detail |
|---|---|
| Architecture | Bidirectional GRU, 128 hidden, 2 layers, temporal attention |
| Features | 50 RF-selected features from parquet (same as LGBM) |
| Labels | Binary next-close direction |
| Training | Early stopping at epoch 13; val_loss=0.693 (near random) |
| Probabilities | Range [0.487, 0.524], mean=0.513, std=0.004 |
| Trades generated | **0** — model output never crosses 0.57 threshold |
| Conclusion | **Complete collapse.** GRU with binary labels on engineered features failed entirely. Model output is essentially constant. |

**Root cause:** Same as LSTM v2. Binary direction labels are too noisy at 1h frequency. The GRU learned to output the class prior (~0.51) rather than discriminative features.

### Full Grid Search (08_full_grid_search.ipynb)

| Component | Detail |
|---|---|
| Model grid | 48 LGBM configs: top_n_features ∈ {20,35,50}, corr_threshold ∈ {0.85,0.90}, num_leaves ∈ {31,63}, min_child_samples ∈ {30,50}, lr ∈ {0.01,0.02} |
| Trading grid | Same 1944-combo grid as 07_grid_search.ipynb |
| OOS method | **Test set directly** — critical leakage |
| Best result | Sharpe 3.125, return +420.09%, max_dd -15.58%, 728 trades |
| Benchmark | Buy & Hold +3.14% (test period) |

**This notebook is unreliable** due to evaluating on the test set. Best config (model_id 33: top_n=50, lr=0.02, leaves=31, mc=30) achieves val AUC 0.5590 which is in line with others, but the trading Sharpe 3.125 is clearly test-set overfitted.

---

## 7. Cross-Cutting Issues Found

### 7.1 Persistent OOS-to-Test Gap

Across all model families, the OOS K-Fold Sharpe consistently exceeds the test Sharpe by 1–3 Sharpe units. This gap is not explained by fees alone. Possible explanations:

- **Regime shift:** The test period (Nov 2024–May 2026) includes a post-election Bitcoin rally, a correction in early 2025, and range-bound behavior — a qualitatively different regime from 2017–2024 training which includes two major bull cycles and two crashes.
- **K-Fold selection bias:** With 1944–6912 combinations evaluated on the same 61k-bar OOS pool, the best combination is cherry-picked. The effective number of independent tests is much smaller than the number of combinations.
- **Non-stationarity in feature importance:** Short-term return features (ret_1h, ret_2h, close_vs_ema_7) consistently top the RF importance ranking. These momentum signals change sign across regimes.

### 7.2 Short-Side Bias in Recent Models

vtrain7 (LGBM) and Mamba both produce overwhelmingly short signals during the test period. In vtrain7, 94% of trades (1624 long vs 565 short from OOS, but on test: primarily short). The test period (Nov 2024) starts right at a Bitcoin all-time-high run, which is the worst possible time to be short-biased.

### 7.3 TBM Label Sensitivity

vtrain2 (TCN) showed that tight TBM barriers (1×σ, 12h) generate 4–10× more trades than ATR-based thresholds because crypto volatility at 1h is high enough that the σ barriers are frequently touched within the 12h window. vtrain4 (±2×ATR, 24h) is much more stable but still shows OOS-test gap.

### 7.4 WFO Instability for Neural Networks

vtrain2 (TCN) demonstrated that 5-epoch WFO warm-start destroyed performance (-88.5%). Neural network weights are sensitive to gradient updates; a 5-epoch fine-tune on 4 months of data introduces noise rather than adaptation. This finding should inform all future neural net WFO designs. For LGBM, monthly WFO was more stable but still showed consistent test underperformance.

### 7.5 Fill Rate Reality

The maker-entry execution model (vtrain5+, TCN vtrain3+) assumes limit orders fill when the wick penetrates the limit price by 5bp. Actual fill rates in the OOS backtest range from 17% (vtrain6 LGBM) to 91% (TCN vtrain3). The extreme variance in fill rate — driven purely by the `entry_atr_mult` parameter — suggests the fill model is highly sensitive and potentially unrealistic. A 0.3×ATR pullback on a 1h bar gives fill rate ~50%; 0.8×ATR gives ~17%.

### 7.6 Feature Importance Dominance by Short-Term Returns

Across all LGBM versions, `ret_2h`, `close_vs_ema_7`, `ret_1h` consistently rank 1–3 in importance. These are essentially momentum indicators at very short horizons. Including these as features while the label is a medium-term move (TBM) creates a potential false edge: the model learns short-term momentum as a proxy for medium-term direction. This signal may be spurious or regime-dependent.

### 7.7 Absence of Walk-Forward on Neural Nets in Latest Versions

vtrain4 (TCN) and Mamba (13) train a **static production model** on the full TrainVal set, then apply it to the entire test period without retraining. Over a 1.5-year test period, market microstructure changes significantly. The static model may be particularly stale in the latter half of the test period.

### 7.8 No Slippage Modeling

None of the backtests include price impact or bid-ask spread beyond the fee structure. For BTC at $80k–$100k on a 1h chart with typical volume, this may not be a major issue, but it remains an unmodeled cost.

---

## 8. Recommendations

The following methodology issues remain that could potentially be fixed:

### High Priority (likely to improve honest results)

| Issue | Recommendation |
|---|---|
| OOS-test Sharpe gap | Analyze which months in the test period are profitable vs losing. If losses are concentrated in a specific regime (e.g. Q1 2025 correction), implement a regime filter that reduces position size or goes flat in trending-against conditions. |
| Short-side bias during bull market | Add a simple market regime filter: e.g., if BTC is above its 200h SMA, disable shorts or reduce short conviction threshold. This is not hindsight bias if derived from training data only. |
| Selection bias in grid search | Reduce grid size. Use information-theoretic criterion (MDL or AIC) to penalize number of parameters. Alternatively, use nested K-Fold: outer fold = test, inner fold = grid search. |
| Static model over 1.5-year test | Implement proper expanding-window WFO for LGBM (where it is stable): retrain monthly, evaluate on next month only. Report month-by-month returns. |

### Medium Priority

| Issue | Recommendation |
|---|---|
| Feature importance dominated by short-term returns | Experiment with a feature set that explicitly excludes very-short-term return features (ret_1h, ret_2h) to test if longer-cycle signal (ichimoku, fibonacci, divergences) alone can drive a trade. |
| Fill rate sensitivity | Test with market-order execution (uniform 0.05% taker both sides) for simpler simulation; separately model limit order fill probability as a function of bar volatility and pullback distance. |
| TBM label sensitivity | The 2×ATR barrier used in vtrain7/vtrain4 generates 42% short, 43% long, 15% neutral. Compare with 1.5× and 2.5× to see if label distribution affects model calibration. |
| Neural net WFO | Instead of weight warm-start, use frozen-backbone + retrained head: freeze all TCN/Mamba layers, retrain only the 3-class classification head on the most recent 3 months. This provides regime adaptation without destabilizing the learned representations. |

### Lower Priority

| Issue | Recommendation |
|---|---|
| Slippage | Add market impact model: assume 0.5× bid-ask spread (≈5–10 bps for BTC at volume) as additional cost on market orders. |
| Calendar features in model | Bitcoin shows documented day-of-week and hour-of-day effects. The feature set includes calendar encodings but these may be drowned out by short-term return features. Try a model trained only on calendar + volatility-regime features as a baseline. |
| Multi-asset signals | All models use BTC/USDT only. ETH and BNB show correlated moves that could provide leading indicators. |

---

## Appendix: Key Numbers Reference

### LGBM vtrain7 Final Test Results

```
vtrain7 WFO (TBM, multiclass, Spot/Futures):
  Total Return: -48.85%  |  Ann. Return: -35.86%
  Sharpe:       -1.925   |  Max DD:      -55.29%
  Win Rate:     45.7%    |  Trades:      346   Fill: 55.7%

Buy & Hold (test period): +6.28%
```

### TCN vtrain1 Final Test Results (best clean TCN result)

```
vtrain1 (close-only, 0% fees):
  Total Return: +11.68%  |  Sharpe: +0.446  |  Max DD: -19.08%
  Trades: 75  |  Win Rate: 60.0%

Buy & Hold: -1.02%
```

### Mamba vtrain8 Final Test Results

```
Mamba WFO (TBM, Spot/Futures):
  Total Return: -8.10%   |  Sharpe: -0.337  |  Max DD: -16.94%
  Trades: 305  |  Fill: 50.2%  |  Win Rate: 46.2%

Buy & Hold: -1.02% (same test period)
```

### Model AUC Comparison (OOS)

| Model | AUC Long | AUC Short |
|---|---|---|
| LGBM vtrain7 (best config) | 0.5509 | 0.5665 |
| TCN vtrain4 | 0.54 | 0.57 |
| Mamba vtrain8 | 0.5572 | 0.5687 |

All models show genuine but small signal: AUC in the range 0.55–0.57. Short prediction is consistently stronger than long prediction.
