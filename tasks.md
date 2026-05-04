# Hybrid Multi-Agent Trading System — Architecture Summary

## Asset & Timeframe
- **Assets:** BTCUSDT, ETHUSDT (Binance spot)
- **Candle resolution:** 1h (primary); 4h and 24h as derived aggregations for multi-horizon prediction
- **Trading style:** Intraday / short-term (SL/TP managed by execution agent)
- **Data source:** Binance REST API (historical) + WebSocket (live) — yfinance removed

---

## Project Structure (Option A)

```
hybrid-multi-agent-trading-system/
├── pyproject.toml          # uv + ruff + hatchling, requires Python >=3.12
├── Makefile                # install, lint, format, test, run targets
└── src/
    └── hmats/
        ├── __init__.py
        ├── agents/
        │   ├── __init__.py
        │   └── base.py     # shared contracts (see below)
        ├── coordinator/
        │   ├── __init__.py
        │   └── supervisor.py
        ├── data/
        │   ├── __init__.py
        │   └── pipeline.py
        └── notebooks/
            ├── 01_data_exploration.ipynb
            └── 02_agent_prototyping.ipynb
```

---

## The Three Contracts (defined in `agents/base.py`)

### 1. `MarketSnapshot` — input to every agent
```python
@dataclass
class MarketSnapshot:
    ticker:     str
    timestamp:  datetime
    ohlcv:      pd.DataFrame        # last N daily candles
    indicators: dict[str, float]    # precomputed SMA, EMA, RSI, MACD
    sentiment:  float | None        # –1.0 to 1.0, optional
```

### 2. `AgentSignal` — output of every agent
```python
@dataclass
class AgentSignal:
    agent_id:   str
    timestamp:  datetime
    ticker:     str
    action:     Literal["buy", "sell", "hold"]
    confidence: float               # 0.0 – 1.0
    horizon:    str                 # "1d", "1w"
    metadata:   dict                # agent-specific extras
```

### 3. `TradingDecision` — output of the supervisor
```python
@dataclass
class TradingDecision:
    timestamp:     datetime
    ticker:        str
    action:        Literal["buy", "sell", "hold"]
    confidence:    float
    position_size: float            # fraction of portfolio 0.0–1.0
    reasoning:     dict[str, AgentSignal]
```

---

## Agents

| Agent | Technique | Input | Notes |
|---|---|---|---|
| `GRUAgent` | Deep learning (GRU) | OHLCV + indicators | Predicts next-day return direction |
| `RLAgent` | Reinforcement learning (PPO) | OHLCV + indicators | Trained on market simulation via Gymnasium |
| `EvolutionaryAgent` | Genetic algorithm (DEAP) | Indicators | Evolves rules e.g. "buy if RSI < 30 and MACD crosses up" |
| `SentimentAgent` | NLP (FinBERT) | News headlines | FinBERT from HuggingFace, no GPU needed for inference |
| `RuleBasedAgent` | Heuristic | Indicators | Simple baseline e.g. SMA crossover |
| `RandomAgent` | Random | — | Lower baseline |

All agents extend `BaseAgent` and implement a single `compute(snapshot: MarketSnapshot) -> AgentSignal` method.

---

## Coordinator / Supervisor

- Receives `list[AgentSignal]` from all agents
- Aggregates via **weighted voting** — weights derived from each agent's validation-period performance (Sharpe ratio)
- Also accepts user-defined preferences: `risk_tolerance` scalar that shifts position sizing
- Emits one `TradingDecision` per evaluation cycle

Start simple: equal weights → then learned weights. That progression is itself a thesis contribution.

---

## Indicators (precomputed in `data/pipeline.py`)
- SMA (20, 50, 200)
- EMA (12, 26)
- RSI (14)
- MACD (12, 26, 9)
- Bollinger Bands
- Volume delta

---

## Sentiment Pipeline
- **Tier 1 (prototype):** VADER + Fear & Greed Index (free, no GPU)
- **Tier 2 (final):** FinBERT (`ProsusAI/finbert`) on daily news headlines
- Sources: yfinance news feed, Reddit (`praw`), RSS headlines

---

## Evaluation Metrics
- Cumulative return
- Sharpe ratio (primary)
- Maximum drawdown
- Win rate
- Per-agent vs. ensemble comparison (ablation)

---

---

## Task 1 — Foundation: Data Pipeline + Base Interfaces + Two Simple Agents

**Goal:** Establish the skeleton that all future agents will plug into.
No ML yet — just the contracts, two simple technical agents, and a working data notebook.

### 1.1 — Notebook: `01_data_exploration.ipynb`

Fetch BTC/USD daily OHLCV data via `yfinance` and compute basic features.

**Steps:**
- Fetch `BTC-USD` daily candles (e.g. 2020–present) with `yfinance`
- Compute and plot indicators: SMA(20, 50, 200), EMA(12, 26), RSI(14), MACD(12, 26, 9), Bollinger Bands, Volume delta
- Assemble a clean `pd.DataFrame` with all features as columns
- Save to `data/raw/btc_daily.parquet` for use by the pipeline

**Acceptance:** Notebook runs top-to-bottom without errors, produces a feature DataFrame, and saves the parquet file.

---

### 1.2 — `agents/base.py` — Core contracts

Implement the three dataclasses and the `BaseAgent` abstract class:

```
AgentSignal      — what every agent returns
MarketSnapshot   — what every agent receives
TradingDecision  — what the supervisor returns
BaseAgent        — ABC with abstract method compute()
```

**Acceptance:** All dataclasses are importable, `BaseAgent` cannot be instantiated directly.

---

### 1.3 — `agents/sma_crossover.py` — Agent 1

A simple SMA crossover agent. Implements `BaseAgent`.

**Logic:**
- `buy`  if SMA(20) crosses above SMA(50) in the last candle
- `sell` if SMA(20) crosses below SMA(50)
- `hold` otherwise
- `confidence` = absolute percentage gap between the two SMAs, clamped to [0, 1]

---

### 1.4 — `agents/rsi_agent.py` — Agent 2

An RSI mean-reversion agent. Implements `BaseAgent`.

**Logic:**
- `buy`  if RSI(14) < 30  (oversold)
- `sell` if RSI(14) > 70  (overbought)
- `hold` otherwise
- `confidence` = how far RSI is from 50, normalised to [0, 1]

---

### 1.5 — `coordinator/supervisor.py` — Orchestration skeleton

Implement a `Supervisor` class that:
- Accepts a list of `BaseAgent` instances at init
- Exposes a `run(snapshot: MarketSnapshot) -> TradingDecision` method
- Calls `compute()` on each agent, collects `AgentSignal` list
- Aggregates via simple majority vote (equal weights for now)
- Logs each agent's signal to stdout via `loguru`
- Returns a `TradingDecision` with `reasoning` populated

---

### 1.6 — `data/pipeline.py` — MarketSnapshot factory

A `build_snapshot(ticker, df) -> MarketSnapshot` function that:
- Takes a ticker string and the feature DataFrame from the notebook
- Returns a `MarketSnapshot` with the latest row of indicators populated
- This is what gets passed to every agent

---

### 1.7 — Wiring notebook: `02_first_run.ipynb`

A short notebook that runs the full stack end-to-end:
```python
snapshot   = build_snapshot("BTC-USD", df)
supervisor = Supervisor(agents=[SMACrossoverAgent(), RSIAgent()])
decision   = supervisor.run(snapshot)
print(decision)
```

**Acceptance:** Prints a valid `TradingDecision` with both agent signals visible in `reasoning`.

---

### Definition of Done for Task 1
- [ ] `01_data_exploration.ipynb` runs cleanly, parquet saved
- [ ] `base.py` contracts in place, importable
- [ ] `SMACrossoverAgent` and `RSIAgent` return valid `AgentSignal`
- [ ] `Supervisor.run()` returns a valid `TradingDecision`
- [ ] `02_first_run.ipynb` prints a decision end-to-end
- [ ] `make lint` passes with no errors

---

## Task 2 — NEAT Agent + Weighted Supervisor + Train/Test Framework

### Background: What the uploaded notebook was doing

The notebook (`Neuroevolution_of_Small_Neural_Networks_Trading.ipynb`) implemented and compared two learned agents on BTC/USDT hourly data from Binance:

**Data & features:**
- Fetched `BTCUSDT` 1h candles from Binance API (2021–2026) with local CSV caching
- Computed 12 features: log return, rolling volatility (24h, 72h), SMA ratio (24/72), MACD line/signal/histogram, momentum (24h, 72h), RSI(14), volume z-score (72h), price z-score (72h)
- Features were clipped to [−10, 10] and standardised using train-set mean/std only (no leakage)
- Split: 2021–2024 train | 2024–2025 val | 2025–2026 test (calendar-based, not fractional)

**Trading environment (`TradingEnv`, Gymnasium):**
- Actions: 0=hold, 1=buy (all in), 2=sell (all out) — binary position, no partial sizing
- Reward: log equity return per step
- Observation: 12 features + position flag + cash fraction (14 dims total)
- Transaction fee: 0.05% per trade

**Agent 1 — PPO (Stable-Baselines3):**
- Standard MLP policy, trained for 200k timesteps on the training environment
- Deterministic inference at test time

**Agent 2 — NEAT (neuroevolution):**
- Evolves small feedforward neural networks using the `neat-python` library
- Population of 50, 20 generations
- Fitness function: `400×mean_log_return − 80×volatility − 2×max_drawdown + activity_score + exposure_score`
  - Activity score: Gaussian reward for ~30 trades per 1000 steps (avoids degenerate do-nothing policies)
  - Exposure score: Gaussian reward for ~50% time-in-market
- Each genome evaluated over 5 random episodes on the training data
- Winner genome serialised to a feedforward network for test inference

**Evaluation:**
- Both agents tested on the held-out 2025 period
- Metrics: final equity, total return, annualised Sharpe, max drawdown, number of trades, wall-clock training time
- Compared against a buy-and-hold baseline
- Equity curves plotted for visual comparison

---

### What Task 2 builds

#### 2.1 — `agents/neat_agent.py` — NEAT Agent

Wrap the NEAT logic from the notebook into a `BaseAgent` subclass.

**Key design decisions vs. the notebook:**
- The genome is trained externally (in a notebook or CLI) and the winner is saved to `models/neat_winner.pkl`
- `NEATAgent` loads the serialised winner at init — `compute()` runs a single forward pass, no training
- Action mapping: `argmax(net.activate(obs))` → `{0: hold, 1: buy, 2: sell}` → `AgentSignal`
- Confidence = softmax of NEAT output activations, max value

```python
class NEATAgent(BaseAgent):
    def __init__(self, genome_path: str, config_path: str): ...
    def compute(self, snapshot: MarketSnapshot) -> AgentSignal: ...
```

**New files:**
- `agents/neat_agent.py`
- `models/` directory (gitignored except `.gitkeep`)
- `configs/neat_trading.ini` — the NEAT config (extracted from notebook)

---

#### 2.2 — `coordinator/supervisor.py` — Weighted memory

Extend the existing `Supervisor` to track per-agent historical performance and compute dynamic weights.

**New internal state:**
```python
@dataclass
class AgentRecord:
    agent_id:    str
    signals:     list[AgentSignal]   # full history
    outcomes:    list[float]         # realised PnL for each signal, filled in retrospectively
    sharpe:      float               # rolling annualised Sharpe, updated periodically
    weight:      float               # current voting weight
```

**Weight update logic:**
- After each `run()` call, store the emitted signal
- When price outcome is observed (next candle close), retrospectively score the signal: correct direction = +1, wrong = −1, scaled by confidence
- Recompute weights as softmax of rolling Sharpe ratios (window = last 30 decisions)
- Equal weights at init (cold start), weights diverge as history accumulates
- Expose `supervisor.weights` property for inspection and logging

**New method:**
```python
def update_outcome(self, ticker: str, realised_return: float) -> None:
    """Call this after the next candle closes to score the previous decision."""
```

---

#### 2.3 — Train/test split framework

Formalise the calendar-based train/test approach from the notebook into a reusable utility in `data/splits.py`.

**Design:**

```
|←————— TRAIN ——————————————→|←— VAL —→|←—— TEST WINDOWS ——→|
  2020-01-01        2023-12-31  2024-06-30   140d  140d  140d ...
```

- **Train period:** 4 years (first full BTC cycle: 2020–2023). Used to fit all learned agents (NEAT, PPO, GRU later).
- **Validation period:** 6 months (2024-H1). Used for hyperparameter tuning and weight initialisation.
- **Test windows:** Rolling 140-day windows starting from 2024-07-01. Each window is fully out-of-sample. The supervisor's weighted voting is evaluated independently per window.

**`data/splits.py`:**
```python
def calendar_split(
    df: pd.DataFrame,
    train_end: str,       # e.g. "2023-12-31"
    val_end: str,         # e.g. "2024-06-30"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: ...

def rolling_test_windows(
    df: pd.DataFrame,
    start: str,           # e.g. "2024-07-01"
    window_days: int = 140,
) -> list[pd.DataFrame]: ...
```

---

#### 2.4 — Training notebook: `03_neat_training.ipynb`

A notebook that:
- Fetches `BTCUSDT` 1h data from Binance API (reuse the `fetch_binance_klines` function from the original notebook)
- Applies `make_features()` and `calendar_split()` to get train/val/test sets
- Standardises features using train-set stats only (save `mu`, `sd` to `models/feature_scaler.npz`)
- Trains NEAT for 20 generations on training data using the fitness function from the notebook
- Trains PPO for 200k timesteps on the same training data (for comparison)
- Saves winner genome to `models/neat_winner.pkl` and PPO model to `models/ppo_model.zip`
- Evaluates both on the validation set, prints metrics table (Sharpe, return, drawdown, trades)
- Plots equity curves on validation set

---

#### 2.5 — Testing notebook: `04_rolling_test.ipynb`

A notebook that:
- Loads saved models (`neat_winner.pkl`, `ppo_model.zip`, `feature_scaler.npz`)
- Runs all 140-day test windows using `rolling_test_windows()`
- For each window: creates a `Supervisor` with `NEATAgent` + `SMACrossoverAgent` + `RSIAgent`, runs it step-by-step, calls `update_outcome()` after each candle, collects `TradingDecision` history
- Aggregates metrics per window and overall
- Plots per-window Sharpe and equity curves
- Compares ensemble vs. individual agents vs. buy-and-hold

---

### New files summary

```
src/hmats/
├── agents/
│   └── neat_agent.py           # NEW
├── coordinator/
│   └── supervisor.py           # MODIFIED — adds AgentRecord + weighted voting
├── data/
│   └── splits.py               # NEW — calendar_split + rolling_test_windows
configs/
│   └── neat_trading.ini        # NEW — NEAT hyperparameters
models/
│   └── .gitkeep                # NEW — saved genomes/models go here, gitignored
notebooks/
│   ├── 03_neat_training.ipynb  # NEW
│   └── 04_rolling_test.ipynb   # NEW
```

**Dependencies to add to `pyproject.toml`:**
```toml
"neat-python>=0.92",
"stable-baselines3>=2.3",   # already present
"gymnasium>=0.29",           # already present
```

---

### Definition of Done for Task 2
- [ ] `NEATAgent` loads a saved genome and returns valid `AgentSignal` via `compute()`
- [ ] `Supervisor` tracks per-agent history and updates weights after `update_outcome()` is called
- [ ] `supervisor.weights` reflects diverged weights after 30+ decisions (not all equal)
- [ ] `data/splits.py` — `calendar_split()` and `rolling_test_windows()` work correctly, no lookahead
- [ ] `03_neat_training.ipynb` runs end-to-end, saves models to `models/`
- [ ] `04_rolling_test.ipynb` loads saved models, runs all 140-day windows, prints metrics table
- [ ] `make lint` passes with no errors

---

## Task 3 — Data Layer Overhaul: Binance-only, Parquet store, multi-symbol

### Context & motivation

The current data layer has two problems:
- `yfinance` is used in early notebooks but only provides ~2 years of hourly history, which is insufficient for a 4-year training window
- The `fetch_binance_klines` cache writes one CSV file per `(symbol, interval, start, end)` tuple — fetching a new date range creates a new file rather than extending existing data, making incremental updates wasteful and fragile

This task replaces both with a single Binance-native, append-capable Parquet store. yfinance is removed from the codebase entirely.

---

### 3.1 — New module: `data/binance_store.py`

Replaces the old `fetch_binance_klines` CSV cache. Responsibilities:

**Storage layout:**
```
data/
└── raw/
    ├── BTCUSDT_1h.parquet
    ├── ETHUSDT_1h.parquet
    └── ...
```

One Parquet file per `(symbol, interval)` pair. The file grows by appending new rows — no date range baked into the filename.

**Core function: `fetch_and_store`**

```python
def fetch_and_store(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    start: str = "2020-01-01",       # full history default
    end: str | None = None,           # None = now
    store_dir: str = "data/raw",
    limit: int = 1000,
    sleep_s: float = 0.15,
) -> pd.DataFrame:
    """
    Fetches Binance klines for the full requested range using pagination
    (same logic as fetch_binance_klines). Writes/appends to a Parquet file.
    If the file already exists, only fetches rows newer than the last
    stored timestamp (incremental update). Returns the full stored DataFrame.
    """
```

**Incremental update logic:**
1. If `data/raw/BTCUSDT_1h.parquet` exists, read the max `open_time` from it
2. Set `start_ms = max_stored_timestamp + 1ms`
3. Fetch only the new rows from Binance
4. Append to existing Parquet and deduplicate on `open_time`
5. If file does not exist, fetch the full range from `start`

**`load(symbol, interval, start, end) -> pd.DataFrame`** — reads the parquet, filters by date range, returns clean DataFrame with `open_time` as DatetimeIndex.

---

### 3.2 — Initial bulk fetch

A one-time script / notebook cell that populates the store with full history:

```python
# Fetch full hourly history for both assets
for symbol in ["BTCUSDT", "ETHUSDT"]:
    fetch_and_store(symbol=symbol, interval="1h", start="2020-01-01")
```

This gives ~43,000 rows per symbol (2020–2026). Runtime: ~5–10 minutes per symbol due to Binance rate limiting and the `sleep_s=0.15` between pages.

---

### 3.3 — Deprecate and remove old data code

- Delete or archive `data/binance.py` (the old `fetch_binance_klines` with CSV cache)
- Remove `data/raw/*.csv` and `data_cache/` from the repo, add both to `.gitignore`
- Update `data/pipeline.py` — replace any `fetch_binance_klines` calls with `binance_store.load()`
- Update notebooks 01–04 to import from `binance_store` instead
- Remove `yfinance` from `pyproject.toml` dependencies

---

### 3.4 — Update notebook `01_data_exploration.ipynb`

Replace the yfinance fetch cell with:
```python
from hmats.data.binance_store import fetch_and_store, load

# Incremental update (no-op if already current)
fetch_and_store("BTCUSDT", "1h", start="2020-01-01")

# Load for analysis
df = load("BTCUSDT", "1h", start="2020-01-01", end="2026-01-01")
print(f"Loaded {len(df):,} rows: {df.index.min()} → {df.index.max()}")
```

---

### New files summary

```
src/hmats/data/
├── binance_store.py    # NEW — replaces binance.py + CSV cache
└── binance.py          # DELETED (or moved to archive/)

data/raw/
├── BTCUSDT_1h.parquet  # populated by fetch_and_store (gitignored)
└── ETHUSDT_1h.parquet  # populated by fetch_and_store (gitignored)
```

**`.gitignore` additions:**
```
data/raw/*.parquet
data/raw/*.csv
data_cache/
```

**`pyproject.toml` changes:**
- Remove `yfinance`
- Add `pyarrow>=16.0` (Parquet backend, likely already present as a transitive dep)

---

### Definition of Done for Task 3
- [ ] `binance_store.fetch_and_store()` fetches full BTCUSDT 1h history from 2020-01-01 and writes to Parquet
- [ ] Re-running `fetch_and_store()` only fetches rows newer than the last stored timestamp (incremental, no duplicates)
- [ ] `binance_store.load()` returns a clean DataFrame with DatetimeIndex, correct dtypes
- [ ] ETHUSDT 1h populated with the same pipeline
- [ ] `yfinance` removed from `pyproject.toml`, `make install` still resolves cleanly
- [ ] All notebooks updated — no remaining imports of `yfinance` or old `fetch_binance_klines`
- [ ] `data_cache/` and `data/raw/*.parquet` added to `.gitignore`
- [ ] `make lint` passes with no errors

---

## Task 4 — Multi-asset data fetch into `01_data_exploration.ipynb`

### Context

`check_minimum_availability_date.py` currently lives in `src/hmats/data/` as a standalone script. It needs to be absorbed into notebook 01 and extended to fetch OHLCV data for all tickers. Notebooks 02, 03, 04 are not touched.

---

### 4.1 — Define `COIN_IDS` at the top of notebook 01

Move the symbol/coin-id mapping to the very first cell of `01_data_exploration.ipynb` so it acts as the single configuration point for the whole notebook:

```python
COIN_IDS = {
    "BTCUSDT":  "bitcoin",
    "ETHUSDT":  "ethereum",
    "BNBUSDT":  "binancecoin",
    "XRPUSDT":  "ripple",
    "SOLUSDT":  "solana",
    "ADAUSDT":  "cardano",
    "DOGEUSDT": "dogecoin",
    "AVAXUSDT": "avalanche-2",
    "DOTUSDT":  "polkadot",
    "LINKUSDT": "chainlink",
}
```

---

### 4.2 — Absorb `check_minimum_availability_date.py` into notebook 01

Replace the standalone script with a notebook cell that:
- Checks if `data/raw/crypto_market_caps.csv` already exists — if yes, loads it and skips the API calls entirely
- If not, runs the full CoinGecko + Binance earliest-candle logic and saves the CSV

```python
MARKET_CAPS_PATH = Path("data/raw/crypto_market_caps.csv")

if MARKET_CAPS_PATH.exists():
    print("crypto_market_caps.csv already exists — skipping fetch")
    caps_df = pd.read_csv(MARKET_CAPS_PATH)
else:
    # run fetch logic, save to MARKET_CAPS_PATH
    ...
```

After this cell, `caps_df` is always available regardless of which branch was taken.

---

### 4.3 — Fetch OHLCV for all tickers from minimum availability date

Add a cell that iterates over `COIN_IDS`, reads `earliest_1h_candle` from `caps_df`, and calls `fetch_and_store` for each symbol:

```python
from hmats.data.binance_store import fetch_and_store

for symbol in COIN_IDS:
    row = caps_df[caps_df["symbol"] == symbol].iloc[0]
    start_date = row["earliest_1h_candle"]  # already a YYYY-MM-DD string
    print(f"Fetching {symbol} from {start_date}...")
    fetch_and_store(symbol=symbol, interval="1h", start=start_date)
    print(f"  done")
```

`fetch_and_store` is already incremental — re-running the notebook only fetches new candles, existing parquet files are extended not rewritten.

---

### 4.4 — Cleanup

- Delete `src/hmats/data/check_minimum_availability_date.py`
- Move `crypto_market_caps.csv` target path from `notebooks/data_cache/` to `data/raw/` so it sits alongside the parquet files
- Add `data/raw/crypto_market_caps.csv` to `.gitignore`

---

### Definition of Done for Task 4
- [ ] `COIN_IDS` defined in the first cell of notebook 01, used throughout
- [ ] Market caps cell skips fetch if `data/raw/crypto_market_caps.csv` exists
- [ ] Running notebook 01 produces `data/raw/{SYMBOL}_1h.parquet` for all 10 symbols
- [ ] Re-running notebook 01 is a no-op for already-fetched data (incremental)
- [ ] `check_minimum_availability_date.py` deleted from `src/hmats/data/`
- [ ] Notebooks 02, 03, 04 untouched and still runnable

---

## Future Phases (not in scope now)

### Phase 2 — CLI framework
- Backtest runner via `make run ARGS="backtest --ticker BTC-USD"`
- Per-agent and ensemble results exported to CSV / plots

### Phase 3 — Live service
- FastAPI endpoints:
  - `GET /signals/{ticker}` → latest `TradingDecision`
  - `GET /signals/{ticker}/agents` → individual `AgentSignal` list
  - `POST /orders/buy` / `POST /orders/sell` → broker API integration
- APScheduler: data fetch + agent recalculation on a configurable interval (e.g. every 1h)
- Broker API key stored in `.env`, never committed

---

## Tooling
- **Python** ≥ 3.12
- **uv** — dependency management
- **ruff** — linting + formatting
- **pytest** — testing
- **hatchling** — build backend
- `make check` / `make test` for CI-style local validation
