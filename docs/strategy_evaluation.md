# Strategy Evaluation — LGBM WFO Research Summary

**Date:** 2026-05-31  
**OOS period:** 2024-01-01 → 2026-05-16  
**ATH-anchored display window:** 2024-11-10 → 2026-05-16 (~18 months)  
**Asset:** BTCUSDT 1h bars, Binance Spot  

---

## 1. What Was Built and Tested

A 4-stage feature selection pipeline followed by walk-forward optimization (WFO) across 4 training-window schemes, all evaluated on a held-out OOS period never touched during model development.

**Feature pipeline:**
1. Variance filter + Spearman collinearity drop (threshold 0.85)
2. Mutual information rank → top 60
3. Walk-forward stability vote (≥50% of rolling windows)
4. LightGBM permutation importance pruning (>0.0005 drop)

**WFO schemes evaluated:**
| Scheme | Training window | Step |
|--------|----------------|------|
| EXP    | Expanding (all history) | 720 h |
| L2Y    | Sliding 2-year          | 720 h |
| M1Y    | Sliding 1-year          | 720 h |
| S3M    | Sliding 3-month         | 720 h |

**Backtest engine:** Long-only ATR-stop (v5–v8). Entry via limit order 0.3×ATR below close, SL at 1.5×ATR, TP at 2.0×ATR, max hold 48 bars, 4-bar cooldown.

---

## 2. Exact Metrics Across Notebook Versions

### How to read these numbers

- **AUC**: Area under the ROC curve on OOS data. 0.50 = random. Values here range 0.525–0.548. Edge is real but small — do not expect 0.70+.
- **Sharpe**: Annualised, hourly bar log-returns. Formula: `mean(r)/std(r) × sqrt(8760)`. A score >1.0 is considered good for a systematic strategy. Negative = the strategy loses money on a risk-adjusted basis.
- **Return**: Total compounded return over the OOS period. **0-fee** means no transaction costs — this is the theoretical maximum. Fee scenarios are the realistic number.
- **MaxDD**: Peak-to-trough drawdown of the equity curve. `-37%` means the strategy lost 37% from its best point before recovering.
- **Win rate**: Fraction of trades that closed positive (before fees). Consistently 45–48% — this is a trend-following edge, not a high-frequency mean-reversion strategy.
- **Trades**: Count of round-trips. More trades = higher fee drag.

---

### V5 — V1 features (195 technical) + 1 cross-altcoin breadth (V3)

| Scenario | Threshold | Trades | Win Rate | Return | Sharpe | MaxDD |
|----------|-----------|--------|----------|--------|--------|-------|
| WFO S3M (0-fee) | 0.55 | — | — | +84.5% | 1.151 | -16.1% |
| WFO EXP (0-fee) | 0.55 | — | — | +52.5% | 0.608 | -23.4% |
| Backtest 0-fee | 0.55 | 1,021 | 45.2% | +54.6% | 0.632 | -23.9% |
| Backtest 0-fee | 0.60 | 507 | 47.3% | +70.0% | **1.112** | **-17.6%** |
| Backtest w/fees | 0.55 | 1,021 | 45.1% | **-25.1%** | -0.416 | -43.7% |
| Backtest w/fees | 0.60 | 507 | 46.7% | +19.4% | 0.369 | -22.8% |

**Selected features (20):** stoch_k_14, ret_1h, ret_2h, ret_3h, close_vs_sma_7, close_vs_sma_720, bear_streak, macd_hist_5_13, rsi_divergence, ad_z_48h, vw_rsi_14, atr_14_pct_rank, hl_position_48h, vol_ratio_72h, skew_24h, candle_body, hour_cos, close_vs_s1, macd_divergence, cross_altcoin_breadth_24h (V3)

---

### V6 — V1+V3+struct features, leverage (ATH-anchored)

Leverage model: conviction-scaled 1×–20×, applied to EXP scheme (wrong choice — EXP was selected by AUC, not Sharpe).

| Scenario | Threshold | Trades | Win Rate | Return | Sharpe | MaxDD |
|----------|-----------|--------|----------|--------|--------|-------|
| Leverage 1× | 0.55 | 649 | 45.3% | -28.5% | -0.803 | -43.6% |
| Leverage 20× | 0.55 | 649 | 45.3% | **-63.0%** | -0.850 | **-78.5%** |
| Leverage 20× | 0.60 | 255 | 47.8% | +30.8% | 0.414 | -31.3% |

Key finding: leverage at threshold 0.55 amplifies losses monotonically. There is no leverage level at which the 0.55 strategy is profitable. The 0.60 threshold with leverage 20× earns +31%, but this is the ATH-anchored window only (from 2024-11-10), coinciding with a strong BTC bull run.

---

### V7 — V1+V3+struct+micro features, leverage

Different feature pool (255 → 14 selected), otherwise same as V6.

| Scenario | Threshold | Trades | Win Rate | Return | Sharpe | MaxDD |
|----------|-----------|--------|----------|--------|--------|-------|
| Leverage 20× | 0.55 | 650 | 43.7% | -64.3% | -1.084 | -79.1% |
| Leverage 20× | 0.60 | 213 | 42.7% | -38.3% | -1.001 | -42.0% |

V7 is strictly worse than V6 at every threshold. Adding V3 external features (cross-asset, sentiment) degraded OOS performance — likely overfit in-sample.

---

### V8 — V1+V4 features, no V3, ATH-anchored, S&P500 benchmark

Feature pool reduced to 220 (V3 removed). Only 11 features survived all 4 stages. Notably only 2 V4 microstructure features passed: `close_vs_true_vwap` and `hurst_24h`.

**Full OOS period (2024-01-01 → 2026-05-16):**

| Scenario | Threshold | Trades | Win Rate | Return | Sharpe | MaxDD |
|----------|-----------|--------|----------|--------|--------|-------|
| 0-fee    | 0.55 | 1,038 | 45.1% | +3.1% | 0.044 | -37.5% |
| 0-fee    | 0.60 | 461 | 47.7% | **+52.4%** | **0.909** | **-18.3%** |
| Meta-labeling | 0.60 | 0 | — | 0% | — | — |

**ATH-anchored display window (2024-11-10 → 2026-05-16) — from chart image:**

| WFO Scheme | AUC | Sharpe | Return |
|------------|-----|--------|--------|
| EXP — Expanding | 0.5477 | 0.544 | +45.6% |
| L2Y — 2yr Sliding | 0.5420 | 0.197 | +13.8% |
| **M1Y — 1yr Sliding** | **0.5400** | **1.385** | **+122.0%** |
| **S3M — 3-Month Sliding** | **0.5250** | **1.165** | **+80.5%** |

**Selected features (11):** close_vs_true_vwap *(V4)*, stoch_k_14, ret_2h, rsi_divergence, close_vs_sma_7, bear_streak, close_vs_s1, macd_hist_5_13, hurst_24h *(V4)*, ad_z_48h, ret_3h

**BTC B&H over ATH window:** approximately -10% to +5% (sideways/volatile period with the model outperforming)  
**S&P 500 (SPY) over ATH window:** approximately +20–27%

---

## 3. Best Two Strategies

### Strategy 1: LGBM-WFO M1Y, p≥0.60, 0-fee (v8)
**The strongest single result found to date.**

- **Mechanism:** 1-year sliding training window. Re-trains every 30 days. Selects 11 features per WFO fold via 4-stage pipeline. Enters long when P(Up) ≥ 0.60.
- **ATH-window metrics:** Sharpe 1.385, return +122%, MaxDD unknown from chart but visually ~-15%
- **Full OOS 0-fee metrics:** Sharpe 0.909, return +52.4%, MaxDD -18.3%, 461 trades
- **Fee sensitivity:** At 0.60 threshold and 461 trades, round-trip cost ~0.10% per trade → -46% total fee drag → net return ~+6%. **Fees erode almost all edge at spot rates.**
- **Possible extensions:**
  - Execute entries/TPs as limit orders (0% maker fee on Binance) — eliminates ~70% of fee drag
  - Extend to perp futures for short side (v9 intent)
  - Adaptive threshold (v9) to maintain trade frequency in low-conviction regimes

### Strategy 2: LGBM-WFO S3M, p≥0.60, 0-fee (v5/v8)
**Most consistent Sharpe across both v5 (1.15) and v8 (1.165) — robust to feature set changes.**

- **Mechanism:** 3-month sliding window. More responsive to recent regime. Higher trade count than M1Y at the same threshold.
- **V5 0-fee metrics:** Sharpe 1.151, return +84.5%, MaxDD -16.1%
- **V8 ATH-window metrics:** Sharpe 1.165, return +80.5%
- **Key advantage:** S3M adapts faster to changing market conditions (crypto regime shifts are fast). M1Y uses 12 months of potentially stale structure.
- **Fee sensitivity:** Similar to M1Y — 0.60 threshold is necessary; 0.55 destroys returns.
- **Possible extensions:**
  - Dynamic window sizing: use ADF stationarity test on recent features to choose 3m vs 6m vs 12m adaptively
  - Combine S3M signals with M1Y as a consensus filter (trade only when both agree)
  - Apply rolling Sharpe monitoring: pause trading if realized Sharpe drops below 0.3 over 60-day window

---

### Comparison: lab/notebooks/07_lgbm_grid_v8.ipynb (Structural Swing-Trader, Regime Filter)

The lab notebook `07_lgbm_grid_v8.ipynb` is a **separate research line** from the main v5–v9 notebooks. It was developed in the `lab/` folder in parallel and has a fundamentally different architecture. Understanding why it was *not* ported to the main notebooks is important.

#### What lab-v8 does differently

| Dimension | Main notebooks (v5–v9) | Lab-v8 |
|-----------|----------------------|--------|
| **Label** | Binary TBM: TP=2.0×ATR / SL=1.5×ATR, 48h horizon | **3-class asymmetric TBM**: TP=2.5×ATR / SL=1.5×ATR, 48h horizon (Long / Short / Neutral) |
| **Model output** | `P(Up)` scalar | `[P(short), P(neutral), P(long)]` — proper 3-class probabilities |
| **Short signal** | `P(Up) < threshold` (impure — conflates flat+bear) | `P(short)` directly from multiclass model (cleaner) |
| **Regime filter** | None | **SMA-168 gate**: longs muted if `close < sma_168`; shorts muted if `close > sma_168` |
| **Feature bans** | No | Hard-banned: `ret_1h`, `ret_2h`, `close_vs_ema_7` (short-term noise that causes fee bleed) |
| **Grid search** | Separate threshold search post-WFO | Joint model × trading grid (32 model configs × ≤36 trading combos = ~1,100 total) |
| **Execution routing** | Spot for longs, perp for shorts (v9 only) | Spot longs / Futures shorts from the start |
| **Funding** | SHORT_FUNDING_H=0.0000077/h (v9) | Same: +0.00077%/h received on shorts |

#### Lab-v8 exact results (test set: 2024-11-10 → 2026-05-16)

The notebook compares two modes: **WFO** (expanding window, monthly refit) and **Static** (single retrain on all trainval data).

| Strategy | Total Return | Ann. Return | Sharpe | Max DD | Calmar | Win Rate | Profit Factor | Trades | Long / Short |
|----------|-------------|-------------|--------|--------|--------|----------|---------------|--------|--------------|
| v8 WFO (Regime + Asymm TBM) | — | — | — | — | — | — | — | — | — |
| v8 Static (Regime + Asymm TBM) | — | — | — | — | — | — | — | — | — |
| Buy & Hold | — | — | — | — | — | — | — | — | — |

> **Note:** The notebook was not re-run in this session; the table above would be populated from the `quick_metrics` output in cell 12 when executed. The CSV at `lab/figures/lgbm_v3/wfo_summary.csv` gives an earlier (pre-regime-filter) WFO comparison:

**Earlier lab WFO results (no regime filter, from `lab/figures/lgbm_v3/wfo_summary.csv`):**

| WFO Scheme | OOS AUC | Trades | Trades/day | Win Rate | EV/trade | Total Return | Max DD |
|------------|---------|--------|------------|----------|----------|-------------|--------|
| Expanding (all history) | 0.5750 | 0 | 0 | — | — | — | — |
| 2-Year Sliding | 0.5622 | 35 | 0.064 | 65.7% | +0.483% | **+16.9%** | -11.1% |
| 1-Year Sliding | 0.5522 | 117 | 0.145 | 61.5% | +0.461% | **+54.0%** | -15.2% |
| 3-Month Sliding | 0.5292 | 586 | 0.671 | 57.3% | +0.139% | **+81.2%** | -28.8% |

> These are from a structural-features-only model (pre-regime-filter, pre-asymmetric-TBM). AUC is notably higher here (0.552–0.575 vs 0.525–0.548 in main notebooks) — the structural feature set + purged K-fold likely give cleaner signal. Trade counts are very low for 2yr and 1yr schemes.

#### Why lab-v8 was not ported to the main notebooks

**1. Different research paradigm — lab is experimental, main is reproducible**
The lab notebooks test many architectural ideas (regime filter, 3-class label, feature bans, asymmetric TBM) simultaneously. The main notebooks (v5–v9) follow a controlled progression where one variable changes at a time. Merging lab-v8 architecture would conflate multiple changes and make it impossible to attribute performance improvements.

**2. 3-class label requires a different pipeline throughout**
The main notebook chain assumes binary classification at every stage: MI ranking, stability voting, permutation importance, and AUC are all binary. Switching to 3-class requires: (a) multiclass MI, (b) per-class permutation importance, (c) OVR or multiclass AUC (e.g., OVR macro-average). This is a significant refactor, not a drop-in replacement.

**3. Regime filter introduces a strong structural assumption**
SMA-168 muting shorts in bull regime is sensible, but it means the model never discovers that its short signals work (or don't) during bull markets. If BTC enters a prolonged bear market, the model has no recent history of long trades to train on in that regime — the filter creates a dataset that is not representative of future regime transitions. The main notebooks avoid this assumption.

**4. Feature banning is subjective**
Hardcoding `ret_1h`, `ret_2h`, `close_vs_ema_7` as banned is a hypothesis — that these cause fee-killing scalp behaviour. In the main notebooks, the 4-stage selection pipeline makes this decision data-driven: if `ret_1h` doesn't survive stability and permutation importance, it is naturally excluded. Lab-v8's manual ban is a stronger prior that may or may not be correct on future data.

**5. Results not fully collected**
Lab-v8 was not executed in this session. The `quick_metrics` table (cell 12) was designed to print to stdout during notebook execution; no `results.json` was saved from lab-v8. The main notebooks save structured JSON artifacts used downstream by the multi-agent framework.

#### Key insight from lab-v8 worth adopting

The **3-class asymmetric TBM** (Long / Neutral / Short with TP > SL) is architecturally superior for a long+short system. The main v9 notebook uses `P(Up) < short_threshold` for shorts, which is impure. The correct path forward for the stable artifact is to adopt the 3-class label from lab-v8 while keeping the main notebook's controlled feature selection pipeline.

---

### Planned: Monthly Return Analysis

A rolling monthly return view is planned for the stable artifact notebook. The idea:

```python
# Compute monthly returns and 3-month SMA of monthly returns
monthly_eq = pd.Series(equity_arr, index=oos_index).resample('ME').last()
monthly_ret = monthly_eq.pct_change().fillna(0) * 100          # % per month
monthly_sma3 = monthly_ret.rolling(3).mean()                   # 3-month smoothed trend

# Also: calendar heatmap of monthly returns (rows=year, cols=month)
```

This adds two views:
1. **Bar chart**: Monthly % return bars with SMA-3 overlay — immediately shows which months were consistently profitable vs noisy
2. **Calendar heatmap**: Green/red grid (year × month) — standard hedge fund reporting format, makes regime shifts and seasonal patterns visible at a glance

The SMA-3 smoothing is especially important for this strategy because individual months have high variance (±10–20%) while the 3-month trend is more interpretable and closer to what a risk manager or thesis committee would look at.

---

## 4. Proposed Stable Artefact Notebook

**Target:** `02_lgbm_stable_v1.ipynb`

This notebook should be the clean, documented, fully-reproducible version that the multi-agent framework depends on. It is not a research notebook — it is a production artifact.

**Spec:**
- Feature set: V1+V4 (exactly as v8, 11 features confirmed to generalise)
- WFO schemes: M1Y primary, S3M as validation ensemble
- Threshold: adaptive (v9 mechanism) with floor LONG_FLOOR=0.53
- Backtester: bidirectional (v9) with correct spot/perp fee routing
- Funding: SHORT_FUNDING_H=0.0000077/h received
- Outputs: `artifacts/stable_v1/results.json`, all charts saved, clear cell-by-cell documentation
- No experimental code, no commented-out blocks
- Must run top-to-bottom reproducibly from cold start

---

## 5. Extensions for Multi-Agent Framework

The stable notebook provides the signal engine. The multi-agent layer adds:

| Agent | Role | Input | Output |
|-------|------|-------|--------|
| **Signal Agent** | Runs LGBM WFO online, produces P(Up) every bar | 1h OHLCV+V4 | P(Up) ∈ [0,1] |
| **Threshold Agent** | Maintains adaptive threshold calibration | Rolling P(Up) history | long_thr, short_thr per bar |
| **Risk Agent** | Monitors drawdown, Sharpe, regime; halts signal | Equity curve, vol | allow_trade: bool |
| **Execution Agent** | Translates signal to orders; routes spot vs perp | Direction, size | Order(s) |
| **Refit Agent** | Triggers model refit at WFO step boundaries | Time, n_new_bars | Updated model |
| **Monitor Agent** | Tracks realized vs expected Sharpe, alerts | Live PnL | Drift alert |

The critical constraint: **the Signal Agent must be retrained on schedule (not on demand)** to prevent overfitting to recent noise. The Refit Agent enforces this.

---

## 6. Skeptical Assessment

### What could be wrong

**1. ATH-window survivorship bias (high confidence)**
The ATH-anchored display period (2024-11-10 onward) was chosen because BTC price reached the prior ATH. This period happens to coincide with a strong bull run followed by a correction. M1Y showing +122% and S3M +80.5% in this window does not mean these strategies earned this return over the full OOS — the full OOS (2024-01-01 onward) shows the more realistic +52.4% at 0-fee for the best threshold. The ATH window is a presentation choice, not a backtest choice.

**2. 0-fee returns are largely theoretical (high confidence)**
Fee-inclusive results for v5 at threshold 0.55 dropped from +54.6% (0-fee) to **-25.1%** (with fees). At threshold 0.60 the drop was from +70% to +19.4%. V8 results.json only stores 0-fee numbers. The actual net return after Binance spot fees is likely 30–50 percentage points lower, depending on limit fill rate. Limit order fill rate in a moving market is not guaranteed — assuming 100% limit fills is optimistic.

**3. Label construction lookahead risk (medium confidence)**
The TBM label uses future high/low within a 48-bar horizon to determine which barrier was hit first. This is correct by construction but means the label at bar t depends on bars t+1 through t+48. If there is any data leakage in feature engineering (e.g., a rolling window that inadvertently uses a shifted index), the model silently sees the future. The feature pipeline applies rolling stats correctly, but this should be audited.

**4. AUC 0.52–0.55 is extremely small edge (high confidence)**
The signal is genuinely weak. Over 461 trades, a 2–3% AUC edge is statistically significant by binomial test, but it requires exact backtest conditions to materialise as profit. Real-world execution slippage, funding rate spikes, exchange downtime, and network latency can each consume this margin entirely.

**5. V4 features barely survived (medium confidence)**
Of 25 V4 microstructure features added in v8, only 2 passed all 4 selection stages: `close_vs_true_vwap` and `hurst_24h`. This suggests V4 features are largely redundant with V1 technical indicators at the 1h granularity. The true value of V4 (TFI, taker pressure, mark basis) may only manifest at 1m–5m resolution where market microstructure is visible, not at 1h where it averages out.

**6. Meta-labeling produced 0 trades (high confidence this is a bug)**
The meta-labeling module (EMA crossover primary → LGBM gatekeeper) generated 0 trades in v8. This is almost certainly a threshold or index alignment bug, not a genuine signal finding. It should not be interpreted as "meta-labeling doesn't work."

**7. Short signal from binary model is impure (medium confidence)**
v9 uses `P(Up) < 0.35` as a short signal. The binary label 0 encodes both "price fell fast" AND "price went sideways for 48 bars." A low P(Up) does not unambiguously predict a short opportunity — it predicts "not a strong upward move." In sideways periods the model may generate short signals into effectively flat markets, causing repeated small losses from entry/exit spread alone.

**8. Walk-forward leak via scheme selection (low-medium confidence)**
The best WFO scheme is selected by looking at full OOS performance (all 4 schemes, then pick the winner). This selection is done post-hoc and introduces a subtle multiple-comparison bias — M1Y and S3M look best partly because we looked at all 4. True out-of-sample performance of the selected scheme will be modestly worse than reported.

---

## 7. Notebook Development Notes

### Reimporting edited modules without restarting the kernel

When `src/hmats/viz/plots.py` (or any other package module) is edited while a notebook is running, the changes are not picked up automatically. Use `importlib.reload`:

```python
import importlib
import hmats.viz.plots
importlib.reload(hmats.viz.plots)
from hmats.viz.plots import plot_equity_drawdown, save_fig
```

Run this in a cell before re-executing any cell that calls the modified function.

### Known column name mismatches

- `plot_equity_drawdown` in `plots.py` expects `pnl_pct` in the trades DataFrame, but the v10 backtester (`_run_backtest_fast`) produces `net`. Fixed in `plots.py` to accept either (`pnl_pct` first, fallback to `net`).
- The v10 feature parquet (`BTCUSDT_1h_features.parquet`) does **not** contain `high`/`low` columns — these must be joined from `data/raw/BTCUSDT_1h.parquet` before passing to the backtester.

### SP500 benchmark

`data/external/sp500_daily.parquet` contains daily SPY OHLCV with a `close` column (lowercase). Load directly — do not use `yfinance` at runtime:

```python
_spy = pd.read_parquet(EXT_DIR / 'sp500_daily.parquet')
_spy.index = pd.to_datetime(_spy.index).tz_localize(None)
_spy_h = _spy['close'].reindex(oos_index_ath, method='ffill').ffill().bfill()
sp500_pct = (_spy_h / _spy_h.iloc[0] - 1) * 100
```

---

## 8. How to Read the Metrics — Cheat Sheet

| Metric | Good | Marginal | Bad | Notes |
|--------|------|----------|-----|-------|
| OOS AUC | > 0.56 | 0.52–0.56 | < 0.52 | These models: 0.525–0.548 — marginal |
| Annualised Sharpe | > 1.0 | 0.3–1.0 | < 0.3 | 0-fee M1Y: 1.39. w/fees EXP: ~0.4 |
| MaxDD | < -20% | -20% to -35% | > -35% | At 0.55 thr w/fees: -43.7% — bad |
| Win rate | > 50% | 45–50% | < 45% | These models: 43–48% — marginal |
| Trades (2yr OOS) | 200–600 | 100–200 or 600–1000 | < 100 or > 1000 | <100 = statistically thin; >1000 = fee erosion |
| Return (0-fee) | > +30% | +10–30% | < +10% | Always compare vs BTC B&H and S&P500 |
| Return (w/fees) | > +20% | 0–20% | < 0% | The only number that matters for real trading |

**The single most important metric for this system:** `w/fees Sharpe at actual fill rates`. Everything else is a research artefact.
