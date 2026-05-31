# Hybrid Multi-Agent Trading System — Project Context

## Project overview

**Name:** `hybrid-multi-agent-trading-system`

**Package:** `hmats` (installed as editable via `uv`)

**Python:** 3.12+ (3.13 in active venv)

**Build system:** Hatchling, `pyproject.toml`

**Task manager:** `make` (Makefile)

**Runner:** `uv run`

### Repository layout

```

hybrid-multi-agent-trading-system/

├── src/hmats/

│   ├── cli.py                  # typer entry point: `trading`

│   ├── data/

│   │   ├── binance_store.py    # raw OHLCV fetch + parquet cache

│   │   └── splits.py           # calendar_split()

│   └── notebooks/

│       ├── 01_data_exploration.ipynb

│       ├── 06_lgbm_agent.ipynb

│       ├── 07_grid_search.ipynb

│       └── data_cache/

├── data/

│   ├── raw/                    # raw OHLCV parquets from Binance

│   └── features/               # engineered feature parquets + registry

├── models/                     # saved LightGBM model, feature list, grid results

├── figures/

├── pyproject.toml

└── Makefile

```

---

## Data

### Raw parquets — `data/raw/`

| Symbol | Rows | Range |

|---|---|---|

| BTCUSDT_1h | 76,237 | 2017-08-17 → 2026-05-04 |

| ETHUSDT_1h | 76,237 | 2017-08-17 → 2026-05-04 |

| BNBUSDT_1h | 74,300 | 2017-11-06 → 2026-05-04 |

| ADAUSDT_1h | 70,445 | 2018-04-17 → 2026-05-04 |

| XRPUSDT_1h | 70,033 | 2018-05-04 → 2026-05-04 |

| LINKUSDT_1h | 63,891 | 2019-01-16 → 2026-05-04 |

| DOGEUSDT_1h | 59,825 | 2019-07-05 → 2026-05-04 |

| DOTUSDT_1h | 49,998 | 2020-08-18 → 2026-05-04 |

| SOLUSDT_1h | 50,183 | 2020-08-11 → 2026-05-04 |

| AVAXUSDT_1h | 49,175 | 2020-09-22 → 2026-05-04 |

| btc_daily | 2,316 | 2020-01-01 → 2026-05-04 |

### Feature parquet — `data/features/BTCUSDT_1h_features.parquet`

Built by `01_data_exploration.ipynb` Section 5. No API calls — derived entirely from `BTCUSDT_1h.parquet`.

**Columns:**

- ~220 ML feature columns

- 3 backtest-only columns: `close`, `sma_200`, `atr_14_pct`

- 1 label column: `label` (1 if next close > current close, else 0)

### Feature registry — `data/features/feature_registry.json`

**Single source of truth** for all feature names. Structure:

```json

{

  "generated_at": "...",

  "source_data": "BTCUSDT_1h.parquet",

  "output_parquet": "BTCUSDT_1h_features.parquet",

  "total_features": 220,

  "backtest_only_cols": ["close", "sma_200", "atr_14_pct"],

  "label_col": "label",

  "groups": {

    "group_name": { "count": N, "features": ["feat1", ...] }

  },

  "features": {

    "feat_name": { "group": "group_name", "description": "..." }

  }

}

```

**Feature groups (13 total):**

| Group | Description |

|---|---|

| `returns` | ret/log_ret over 1,2,3,6,12,24,48,72,168h |

| `volatility` | rolling log-return std; ATR 14/24 raw + pct |

| `ma_ratios` | close vs SMA/EMA 7,14,20,50,100,200 |

| `bollinger` | BB width + position, bands 20 and 50 |

| `macd` | MACD (12,26,9) and (5,13,4), normalised |

| `oscillators` | RSI 7/14/21, Stoch %K 14/21, Williams %R |

| `volume` | volume z-score/ratio 12/24/72/168h, OBV z-score |

| `candle_structure` | body pct, upper/lower wick, is_bullish |

| `price_position` | high/low position 24/48/168h |

| `calendar` | hour/DoW sin-cos, day-of-month, quarter, BTC halving cycle |

| `ma_crosses` | SMA pair distances, golden cross flag, recency, bull score, ribbon width |

| `ichimoku` | TK ratio, cross signals, cloud position/thickness/flip recency |

| `supertrend` | direction + distance for mult 1.5/2.0/3.0, consensus, flip recency |

| `fibonacci` | grid position, nearest level/distance, 0.618 proximity (48h + 168h) |

| `long_cycle_ma` | close vs SMA/EMA 336/504/720/2160/4320h, weekly momentum accel |

| `divergences` | RSI/MACD/OBV divergence (+1 bull, -1 bear, 0 none), vol-price div |

| `candlestick_patterns` | bull/bear engulf, doji, hammer, shooting star, streaks, marubozu |

| `volume_profile` | VWAP 24/168h, MFI 14/21, CMF 20, A/D z-score, vol spikes, vwRSI |

| `support_resistance` | daily pivots P/R1/R2/S1/S2, round levels, breakout magnitude 24/48/168h |

| `volatility_regime` | Garman-Klass/Parkinson vol, vol-of-vol, ATR percentile rank, BB squeeze |

| `statistical` | Hurst 168h, skew/kurt 24/72/168h, autocorr lag 1/6/12/24h, variance ratio |

| `composite` | trend score (0-5), RSI×vol confirm, momentum coherence, Sharpe ratio, regime composite |

**Notable implementation details:**

- `var_ratio_168h` omitted (too slow without Numba — only 6h and 24h computed)

- Hurst is slow (~2 min on 76k rows); rolling 168h window

- Divergences use `scipy.signal.argrelextrema`, window=48, order=5

- `o.copy()` called after Group 3 (SuperTrend) to defrag fragmented DataFrame

- `fibonacci_features`: `idxmin` pre-filtered with `valid_mask` to avoid all-NaN rows error

---

## Notebooks

### `01_data_exploration.ipynb`

**Sections 1–4:** Original data exploration (OHLCV inspection, candlestick plots).

**Section 5 (added):** Extended feature engineering pipeline.

- Loads `BTCUSDT_1h.parquet` (reuses `btc_1h` if in memory)

- Defines all helper functions in one cell

- Builds all feature groups sequentially, registering each into `feature_registry` dict

- Saves `data/features/BTCUSDT_1h_features.parquet`

- Saves `data/features/feature_registry.json`

- Sanity check cell: reloads parquet, verifies zero NaNs

### `06_lgbm_agent.ipynb`

**Pipeline:**

1. Load `BTCUSDT_1h_features.parquet` + registry

2. Feature group bar chart (overview)

3. Train/val/test split via `calendar_split`

4. Random Forest for feature importance (300 trees)

5. Pearson correlation filter (threshold 0.90)

6. Keep top 50 features by RF importance

7. LightGBM training with early stopping on val

8. Evaluation: calibration curve, bucket win rates, classification report

9. Backtest — long + short with ATR-adaptive SL

10. Buy-and-hold benchmark

11. Metrics table

12. Results plot (equity curve + drawdown)

13. Trade log

14. **Strategy summary** (trade counts, exit reasons, return percentiles, final equity)

**Train/val/test split:**

- Train: start → `2024-06-01`

- Val: `2024-06-01` → `2024-11-10`

- Test: `2024-11-10` → present

**LightGBM config:**

```python

LGB_PARAMS = {

    'objective': 'binary',

    'n_estimators': 1000,

    'learning_rate': 0.02,

    'num_leaves': 31,

    'min_child_samples': 50,

    'subsample': 0.8,

    'colsample_bytree': 0.8,

    'reg_alpha': 0.1,

    'reg_lambda': 1.0,

}

EARLY_STOPPING_ROUNDS = 50

TOP_N_FEATURES = 50

CORR_THRESHOLD = 0.90

```

**Trading parameters (current):**

```python

LONG_THRESHOLD        = 0.57   # model output compressed to ~0.35–0.65

SHORT_THRESHOLD       = 0.43

EXIT_THRESHOLD_LONG   = 0.48

EXIT_THRESHOLD_SHORT  = 0.52

MIN_HOLD_CANDLES      = 6

MAX_HOLD_CANDLES      = 48

COOLDOWN_CANDLES      = 3

ATR_MULTIPLIER        = 2.0

MIN_SL                = 0.015  # 1.5% floor

TAKE_PROFIT           = 0.03   # 3%

```

**Threshold rationale:** The model's predicted probabilities are compressed into ~0.35–0.65. Win-rate-by-bucket analysis confirmed:

- `0.55–0.60` bucket: 57.2% long win rate

- `0.60–0.65` bucket: 58.8% long win rate

- `0.40–0.45` bucket: 55.7% short win rate

Thresholds were lowered from original 0.65/0.35 to 0.57/0.43 to generate sufficient signals.

**Saved model artefacts (`models/`):**

- `lgbm_model.txt` — LightGBM booster

- `lgbm_features.csv` — selected feature names (one per line, no header)

**Backtest engine (short position):**

- Uses `entry_cash` snapshotted at entry; `pnl_cash = entry_cash * (1 + pnl)`

- Fixes bug where `cash = 0` after long entry would corrupt short PnL

**Fee assumptions (MEXC futures):**

- Maker fee: **0%** (limit orders)

- Taker fee: 0.02% (not used — strategy assumes limit orders)

- Funding rate: ±0.01% per 8h window — **not modelled** in backtest (removed)

- Strategy hold of 6–22h crosses 1–3 funding windows; at 0.01% per window this is negligible for small conviction trades

**Known issue:** After switching to the expanded 220-feature set, strategy performance regressed vs the simpler feature set:

- Before: Alpha +48.67pp, Sharpe 0.881, MaxDD -31.02%

- After: Alpha -6.87pp, Sharpe -0.080, MaxDD -35.46%

- Root cause: expanded features changed model calibration; trading thresholds/params need re-optimisation → addressed in `07_grid_search.ipynb`

### `07_grid_search.ipynb`

**Purpose:** Grid search over trading parameters only — model is frozen, predictions computed once.

**Flow:**

1. Load `lgbm_model.txt` + `lgbm_features.csv` from `models/`

2. Load feature parquet, split identically to 06

3. Compute `probs_test` once

4. Define `run_backtest()` and `score()` functions

5. Build valid combinations (constraint: `long_thr - (1 - short_thr) ≥ 0.02`)

6. Loop with `tqdm`, filter by `MIN_TRADES = 30`

7. Leaderboard (top 20)

8. Distribution plots (6 metrics)

9. Sensitivity analysis (median metric per param value)

10. Best config equity curve + summary

11. Save `models/grid_search_results.csv` + `models/best_trading_params.json`

**Search grid (current):**

```python

GRID = {

    'long_threshold':  [0.54, 0.55, 0.57, 0.59, 0.61],

    'short_threshold': [0.39, 0.41, 0.43, 0.45, 0.46],

    'atr_multiplier':  [1.5, 2.0, 2.5],

    'min_sl':          [0.010, 0.015, 0.020],

    'take_profit':     [0.025, 0.030, 0.040],

    'min_hold':        [4, 6, 8],

    'max_hold':        [24, 48],

    'cooldown':        [2, 3],

}

```

**Optimisation metric:** `OPTIMISE_METRIC = 'sharpe'` (configurable: `sharpe`, `total_return`, `calmar`, `win_rate`, `profit_factor`)

**Known bugs fixed:**

- `tqdm.notebook` → `tqdm` (ipywidgets not installed)

- `import matplotlib.dates as mdates` moved to main imports cell

---

## Makefile rules

```makefile

notebook:   # Launch Jupyter Lab

run:        # Run CLI entry point (pass ARGS="...")

clean:      # Remove __pycache__, .pytest_cache, .mypy_cache, .ruff_cache, *.pyc, dist/

clean-all:  # clean + remove .venv

nbconvert:  # Convert notebook to .py script

            # Usage: make nbconvert NB=src/hmats/notebooks/06_lgbm_agent.ipynb

```

`nbconvert` requires `nbconvert` package (`uv add --dev nbconvert`, already in `[dependency-groups]` — **should be moved to `[project.optional-dependencies] dev`**).

---

## Dependencies

**Key production deps:** numpy, pandas, scipy, torch, transformers, gymnasium, stable-baselines3, deap, neat-python, lightgbm, scikit-learn, matplotlib, pyarrow, pydantic, loguru, requests, tqdm, typer

**Dev deps (`[project.optional-dependencies] dev`):** ruff, mypy, pytest, pytest-cov, pytest-asyncio, ipykernel, jupyter, nbconvert

**Known issue:** `[dependency-groups] dev` section also exists in `pyproject.toml` with `nbconvert` — this is redundant and inconsistent. Should be merged into `[project.optional-dependencies] dev` and the `[dependency-groups]` section removed.

---

## Common issues & fixes

| Error | Fix |

|---|---|

| `ValueError: Encountered all NA values` in `fibonacci_features` | Pre-filter rows with `valid_mask = distances.notna().any(axis=1)` before calling `idxmin` |

| `PerformanceWarning: DataFrame is highly fragmented` | Call `o = o.copy()` after SuperTrend group (Group 3) |

| `ImportError: IProgress not found` | Use `from tqdm import tqdm` not `from tqdm.notebook import tqdm` |

| `NameError: name 'mdates' is not defined` | Add `import matplotlib.dates as mdates` to main imports cell |

| `Jupyter command jupyter-nbconvert not found` | `uv add --dev nbconvert` or use `uv run --with nbconvert jupyter nbconvert ...` |

| Short position PnL wrong | Use `entry_cash` snapshot at short entry; never use `cash` (which is 0 after long entry) |
