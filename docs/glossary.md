# Glossary

Definitions of acronyms and domain-specific terms appearing in the project notebooks.

---

## Backtesting & Evaluation Framework

**OOS — Out-Of-Sample**
Data held out from training, used only for evaluation. In walk-forward setups the OOS period is the contiguous future slice after the training window. In this project the primary OOS period is 2024-01-01 → present.

**IS — In-Sample**
The training window used to fit each model in a WFO fold. Contrast with OOS.

**WFO — Walk-Forward Optimization**
A backtesting protocol that repeatedly trains on a historical window and evaluates on the immediately following OOS slice, then steps forward in time. Avoids fitting a single model on the entire history and testing it in-sample. Each step produces one fold of results; the folds are chained to form a continuous equity curve.

**WFO Schemes (EXP / L2Y / M1Y / S3M)**
The four training-window strategies compared in Phase 2:

| Code | Name | Training window |
|------|------|----------------|
| EXP | Expanding (All History) | All bars from the start of data up to each fold's cutoff |
| L2Y | 2-Year Sliding | Fixed 17,520-bar (≈2 year) rolling window |
| M1Y | 1-Year Sliding | Fixed 8,760-bar (≈1 year) rolling window |
| S3M | 3-Month Sliding | Fixed 2,160-bar (≈3 month) rolling window |

Expanding grows the training set over time; sliding keeps it fixed so recent data is always represented equally.

**ATH — All-Time High (display window)**
In this project "ATH window" refers to the display period starting from the first bar where BTC price reached the level it sits at at the end of OOS. Everything before that point is below the "current price level" and visually uninteresting, so equity curves are plotted from ATH_START onward.

**BH — Buy & Hold**
Passive baseline: buy at the first OOS bar, hold to the end. Used as a benchmark for strategy returns.

**TBM — Triple Barrier Method**
A labeling technique from Marcos Lopez de Prado. Each bar gets a label based on which of three barriers is hit first: take-profit (TP), stop-loss (SL), or a time barrier (max holding period). Produces binary (up/down) or three-class (up/neutral/down) labels.

**Embargo**
A gap of bars between the IS window and the OOS window that is excluded from both training and evaluation. Prevents data leakage caused by features that look back across the IS/OOS boundary.

---

## Trade Execution Terms

**SL — Stop-Loss**
An exit order placed below entry price (for long positions). Triggered when price falls to `entry × (1 − SL_mult × ATR_pct)`. Limits downside on losing trades.

**TP — Take-Profit**
An exit order placed above entry price. Triggered when price rises to `entry × (1 + TP_mult × ATR_pct)`.

**ATR — Average True Range**
A volatility measure: the rolling average of `max(high−low, |high−prev_close|, |low−prev_close|)`. Used to size SL and TP distances so they adapt to current market volatility.

**RR — Risk-Reward Ratio**
TP distance divided by SL distance. E.g. TP=2×ATR, SL=1.5×ATR → RR=1.33. Sets the minimum win rate needed to break even.

**Cooldown**
A mandatory waiting period (in bars) after closing a trade before the next entry is allowed. Prevents whipsawing into back-to-back losing trades.

**Maker / Taker Fees**
Exchange fee structure. A *maker* order adds liquidity (limit order resting on the book) and pays a lower fee (0.02% Futures, 0% Spot). A *taker* order removes liquidity (market order or limit that crosses immediately) and pays a higher fee (0.05%). Round-trip cost = entry fee + exit fee.

**Funding Rate**
Perpetual futures only. A periodic payment between long and short holders to keep the futures price anchored to spot. Positive funding = longs pay shorts.

---

## Model & ML Terms

**LGBM — LightGBM**
Gradient-boosted decision tree library by Microsoft. Fast, handles high-dimensional tabular data well. Used here as a binary classifier predicting whether price will go up over the next N bars.

**AUC — Area Under the (ROC) Curve**
Primary classification metric. Measures the probability that the model ranks a random positive example higher than a random negative one. 0.5 = random, 1.0 = perfect. Used to select the best WFO scheme and best hyperparameters.

**MI — Mutual Information**
A feature-ranking metric measuring how much knowing a feature reduces uncertainty about the label. Used in Stage 2 of the feature selection pipeline.

**Permutation Importance**
Feature importance estimate: shuffle one feature's values, measure the increase in model error. A large increase → the feature matters. Used in Stage 4 to prune weak features.

**Early Stopping**
Training is halted when validation loss stops improving for N consecutive rounds (e.g. `EARLY_STOP_ROUNDS = 30`). Prevents overfitting.

**Multiclass**
3-class variant of the label: Up / Neutral / Down. Uses `objective='multiclass'`, `num_class=3`, and `metric='multi_logloss'`.

**Meta-Labeling**
A two-stage framework (Lopez de Prado). A primary model generates binary signals (e.g. EMA crossover). A secondary meta-labeler predicts whether each primary signal will be profitable, and filters out low-confidence ones. Only trades approved by both models are taken.

---

## Feature Selection Pipeline (4 Stages)

**Stage 1 — Variance + Correlation Filter**
Remove near-constant features (low variance) and highly correlated pairs (Spearman |ρ| > 0.85, keep the one with higher MI).

**Stage 2 — MI Ranking**
Score remaining features by Mutual Information on the IS period, keep top K (e.g. top 60).

**Stage 3 — Walk-Forward Stability**
Run a mini-WFO and keep only features that appear important (by MI or permutation) in at least 50% of sub-windows. Filters features that are only relevant in specific regimes.

**Stage 4 — Permutation Pruning**
Train a final model, permute each feature, and keep only those whose removal meaningfully hurts AUC (threshold > 0.0005).

---

## Technical Indicators

**EMA — Exponential Moving Average**
A weighted moving average that decays exponentially, giving more weight to recent bars.

**RSI — Relative Strength Index**
Momentum oscillator (0–100). Above 70 = overbought, below 30 = oversold. Computed with Wilder's smoothing over a 14-bar window.

**MACD — Moving Average Convergence Divergence**
Difference between two EMAs (fast and slow). The histogram (`macd_hist`) is the MACD minus its own signal line; used here as a feature.

**BB — Bollinger Bands**
Envelope of SMA ± N×σ. `bb_width_pct` measures how wide the band is relative to price (a volatility proxy). `bb_position` measures where price sits within the band.

**VWAP — Volume-Weighted Average Price**
Average price weighted by volume over a window (or anchored to a reference point like daily open). `close_vs_true_vwap` measures how far price deviates from VWAP.

**ATH (indicator context) — All-Time High**
The highest price ever recorded. Also used loosely for the highest price in a rolling window.

**TFI — Trade Flow Imbalance**
`taker_buy_volume / (taker_buy_volume + taker_sell_volume)` — the fraction of volume that was aggressively bought. Values > 0.5 indicate buy-side pressure. Features: `tfi_pct`, `tfi_z_24h`, `tfi_ema_12`, etc.

**Hurst Exponent**
A statistic characterising the long-term memory of a time series. H < 0.5 → mean-reverting; H = 0.5 → random walk; H > 0.5 → trending. Computed here over rolling 24h and 72h windows.

**FracDiff — Fractional Differentiation**
A way to make price stationary while preserving as much memory as possible, by differencing to a fractional order d (0 < d < 1) rather than a full integer difference. Feature: `fracdiff_close_d0.2`.

**ADF — Augmented Dickey-Fuller test**
Statistical test for stationarity. The t-statistic and p-value are used as features indicating whether price is in a trending or mean-reverting regime at a given lookback window (168h, 336h, 720h).

**POC — Point of Control**
The price level with the highest volume in a volume profile (histogram of volume by price). Acts as a magnet / support-resistance level.

**MTF — Multi-TimeFrame**
Features computed on a higher timeframe (e.g. 4h or daily) and forward-filled onto the base timeframe (1h). `mtf_h4_ema_signal` is the EMA crossover direction computed on 4h bars.

---

## Performance Metrics

**Sharpe Ratio**
Annualised `mean(returns) / std(returns)`. Higher is better; > 1.0 is generally considered good. Annualisation factor: `sqrt(24 × 365)` for 1h bars.

**MDD / MaxDD — Maximum Drawdown**
The largest peak-to-trough percentage decline over the full OOS period. A measure of worst-case loss.

**WR — Win Rate**
Fraction of trades that are profitable: `(pnl > 0).mean()`. Must exceed `1 / (1 + RR)` to break even.

**PF — Profit Factor**
`gross_wins / gross_losses`. PF > 1 → profitable in aggregate.

**EV — Expected Value**
Average PnL per trade. `EV = WR × avg_win − (1 − WR) × avg_loss`.

---

## Data & Infrastructure

**OHLCV**
Open, High, Low, Close, Volume — the standard candlestick data format.

**1h / 5m**
1-hour and 5-minute bar intervals. The primary modelling timeframe in this project is 1h.

**Parquet**
Columnar binary file format (Apache Arrow). Used for all feature and OHLCV storage; roughly 2× smaller than CSV and much faster to read.

**BTCUSDT**
Binance trading pair: BTC priced in USDT. Base asset / quote asset notation.

**Lookahead Bias**
A bug where a feature or label is computed using information from future bars, making backtests unrealistically optimistic. E.g. MTF features in v0 carried lookahead bias; corrected in v1.
