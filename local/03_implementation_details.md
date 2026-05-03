# Implementation Details

## Hybrid Multi-Agent Trading System

---

## 1. Module Reference

### 1.1 agents/base.py — Core Contracts

**Purpose:** Defines the three data contracts and the abstract agent interface.

```python
class BaseAgent(ABC):
    agent_id: str

    @abstractmethod
    def compute(self, snapshot: MarketSnapshot) -> AgentSignal: ...
```

**Design decisions:**
- `MarketSnapshot.ohlcv` is a full DataFrame (not just OHLCV columns) — this is intentional so rule-based agents can access historical indicator values for crossover detection.
- `AgentSignal.metadata` allows each agent to store strategy-specific data (e.g. NEAT stores raw output activations) without polluting the shared contract.
- `TradingDecision.reasoning` preserves all individual signals for auditability and post-hoc analysis.

### 1.2 agents/sma_crossover.py

**Key logic:** Reads `SMA_20` and `SMA_50` from the `ohlcv` DataFrame at positions `iloc[-1]` and `iloc[-2]` to detect crossover events.  Falls back to the `indicators` dict if columns aren't present (but then cannot detect crossovers, only current relative position).

**Edge cases handled:**
- DataFrame with < 2 rows → hold
- NaN in SMA values → hold
- Zero SMA(50) → zero division guard

### 1.3 agents/rsi_agent.py

**Key logic:** Reads `RSI_14` from `snapshot.indicators`.  Thresholds at 30/70.

**Edge cases handled:**
- Missing RSI → hold with confidence 0

### 1.4 agents/neat_agent.py

**Key logic:**
1. Load genome from pickle file + NEAT config from INI file.
2. Construct `neat.nn.FeedForwardNetwork` from genome.
3. On `compute()`: build 14-dim observation from `snapshot.indicators`, run `net.activate()`, apply softmax, argmax for action.

**Important:** `neat-python` is *not* added to `hmats.agents.__init__.__all__` to avoid making it a hard import dependency.

### 1.5 coordinator/supervisor.py

**State management:**
- `_records: dict[str, AgentRecord]` — full signal/outcome history per agent.
- `_last_signals: dict[str, AgentSignal]` — signals from the most recent `run()` call, used by `update_outcome()`.

**Weight recomputation uses numpy softmax** with numerical stability (subtract max before exp).

### 1.6 data/pipeline.py

**Functions:**
- `compute_indicators(df)` — adds 13 indicator columns to a DataFrame with OHLCV columns.
- `build_snapshot(df, ticker, lookback)` — takes the last `lookback` rows and creates a `MarketSnapshot` with the full enriched DataFrame as `ohlcv` and the latest row's indicator values as `indicators`.

### 1.7 data/features.py

**Functions:**
- `make_features(df)` — computes 12 hourly features from lowercase OHLCV columns. Drops rows with NaN. Clips to [-10, 10].
- `standardise(train, *others)` — z-score standardisation using train-set statistics. Returns `(mu, sd, [standardised arrays], train_prices, [other_prices])`.

**RSI implementation:** Uses EWM with `alpha=1/period` (equivalent to Wilder smoothing).

### 1.8 data/trading_env.py

**Classes:**
- `TradingEnvConfig` — fee, start_cash, max_episode_steps.
- `TradingEnv` — Gymnasium-compatible environment.

**Evaluation helpers:**
- `max_drawdown(equity)` — peak-to-trough ratio.
- `annualization_factor(interval)` — supports "1m", "1h", "1d" formats.
- `sharpe_annualized(log_returns, ann_factor)` — annualised Sharpe from log returns.
- `evaluate_policy(env, act_fn)` — full episode rollout with metrics.
- `buy_and_hold_metrics(prices)` — passive baseline.
- `neat_fitness_from_episode(env)` — multi-objective fitness for NEAT training.

### 1.9 data/binance.py

**Function:** `fetch_binance_klines(symbol, interval, start, end)` — paginated download from Binance REST API with local CSV caching.

**Pagination:** Uses `startTime` / `endTime` parameters, advancing `startTime` after each batch.  Sleeps 150ms between requests to respect rate limits.

### 1.10 data/splits.py

**Functions:**
- `calendar_split(df, train_end, val_end)` — three-way split on DatetimeIndex using `pd.Timestamp` comparisons.
- `rolling_test_windows(df, start, window_days)` — non-overlapping windows using `pd.Timedelta`.

---

## 2. Notebooks

### 01_data_exploration.ipynb

1. Fetches 3 years of BTC-USD daily data via `yfinance`.
2. Calls `compute_indicators()` from `pipeline.py`.
3. Plots: close price with SMA overlay, RSI, MACD, Bollinger Bands, Volume Delta.
4. Saves to `data/raw/btc_daily.parquet`.

### 02_first_run.ipynb

1. Loads parquet (or fetches fresh).
2. Calls `build_snapshot()` with lookback=60.
3. Instantiates `SMACrossoverAgent` and `RSIAgent`.
4. Creates `Supervisor` with both agents, calls `run()`.
5. Prints `TradingDecision` and individual signals.

### 03_neat_training.ipynb

1. Fetches BTCUSDT 1h data from Binance (2023-01-01 to 2025-12-31).
2. Computes hourly features via `make_features()`.
3. Splits: train (<2024), val (2024), test (≥2025).
4. Standardises features, saves scaler to `models/feature_scaler.npz`.
5. Trains PPO (200k timesteps) on `TradingEnv`.
6. Trains NEAT (50 pop × 20 gens × 5 episodes/eval) using custom fitness.
7. Saves `ppo_model.zip` and `neat_winner.pkl` to `models/`.
8. Evaluates both on validation set, plots equity curves.

### 04_rolling_test.ipynb

1. Loads saved models and scaler.
2. Generates rolling 140-day windows starting 2025-01-01.
3. Per window: evaluates NEAT, PPO, and buy-and-hold individually.
4. Supervisor weight evolution demo: step-by-step supervisor with NEAT + SMA + RSI agents, calling `update_outcome()` at each bar.
5. Plots: per-window Sharpe comparison, weight evolution over time.

---

## 3. Configuration Files

### configs/neat_trading.ini

NEAT-python configuration.  Key points:
- `num_inputs = 14` (12 features + 2 state variables)
- `num_outputs = 3` (hold, buy, sell)
- `num_hidden = 0` (start minimal, grow through mutation)
- `feed_forward = True` (no recurrent connections)
- `pop_size = 50`
- `fitness_criterion = max`
- `compatibility_threshold = 3.0` (speciation distance)

### pyproject.toml

- Build system: `hatchling`
- Python: `>=3.12`
- Ruff rules: E, W, F, I, B, C4, UP, N, SIM, TID, RUF
- Source layout: `src/hmats/`
- Entry point: `trading = "hmats.cli:app"` (planned)

---

## 4. Dependency Graph

```
hmats.agents.base          ← no internal deps (only pandas, stdlib)
hmats.agents.sma_crossover ← agents.base
hmats.agents.rsi_agent     ← agents.base
hmats.agents.neat_agent    ← agents.base, neat-python, numpy
hmats.coordinator.supervisor ← agents.base, numpy, loguru
hmats.data.pipeline        ← agents.base, pandas, numpy
hmats.data.features        ← pandas, numpy
hmats.data.splits          ← pandas
hmats.data.binance         ← pandas, numpy, requests
hmats.data.trading_env     ← gymnasium, numpy
```

`neat-python` is an optional dependency at the package level — only imported by `neat_agent.py`.

---

## 5. Build and Development

```bash
# Install all dependencies
make install       # runs: uv sync --all-extras

# Lint + type-check
make lint          # runs: uv run ruff check src tests
                   #        uv run ruff format --check src tests

# Auto-format
make format        # runs: uv run ruff format src tests

# Run tests
make test          # runs: uv run pytest

# Single command: format + lint + test
make ci
```

---

## 6. Serialised Artefacts

| File | Format | Contents | Generated by |
|------|--------|----------|-------------|
| `models/neat_winner.pkl` | Python pickle | Best NEAT genome (topology + weights) | 03_neat_training.ipynb |
| `models/ppo_model.zip` | SB3 archive | PPO policy network + optimiser | 03_neat_training.ipynb |
| `models/feature_scaler.npz` | NumPy NPZ | Train-set mean and std arrays | 03_neat_training.ipynb |
| `data/raw/btc_daily.parquet` | Parquet | BTC-USD daily OHLCV + indicators | 01_data_exploration.ipynb |
| `data_cache/*.csv` | CSV | Cached Binance kline downloads | binance.py (automatic) |
