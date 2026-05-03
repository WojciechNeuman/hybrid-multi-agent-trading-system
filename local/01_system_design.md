# System Design Document (SDD)

## Hybrid Multi-Agent Trading System (HMATS)

**Author:** Wojciech Neuman
**Date:** May 2026
**Version:** 0.2.0 (Tasks 1–2 complete)

---

## 1. Problem Statement

Can a heterogeneous ensemble of AI trading agents — combining deep learning, neuroevolution, reinforcement learning, NLP, and rule-based heuristics — outperform individual agents and passive strategies (buy-and-hold) on cryptocurrency markets?

The system targets **BTC/USD swing trading on daily candles** (rule-based agents) and **BTC/USDT intraday trading on 1-hour candles** (learned agents), with a unified supervisor that dynamically re-weights agents based on their rolling out-of-sample performance.

---

## 2. Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                    MarketSnapshot                        │
│  (ticker, timestamp, OHLCV DataFrame, indicators dict)  │
└────────────────────────┬─────────────────────────────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
   ┌──────────┐   ┌──────────┐   ┌──────────┐
   │ Agent 1  │   │ Agent 2  │   │ Agent N  │
   │(SMA, RSI,│   │ (NEAT)   │   │(PPO, GRU,│
   │ Random…) │   │          │   │FinBERT…) │
   └────┬─────┘   └────┬─────┘   └────┬─────┘
        │              │              │
        ▼              ▼              ▼
   AgentSignal    AgentSignal    AgentSignal
   (action, confidence, horizon, metadata)
        │              │              │
        └──────────────┼──────────────┘
                       ▼
              ┌────────────────┐
              │   Supervisor   │
              │ (weighted vote │
              │  + outcome     │
              │  tracking)     │
              └───────┬────────┘
                      ▼
              TradingDecision
   (action, confidence, position_size, reasoning)
```

### Design Principles

1. **Contract-driven:** Three dataclasses (`MarketSnapshot`, `AgentSignal`, `TradingDecision`) define the interface between all components.  Adding a new agent requires only implementing `BaseAgent.compute()`.
2. **Heterogeneous ensemble:** Agents use fundamentally different AI paradigms (genetic algorithms, RL, deep learning, NLP, heuristics).  This is the core thesis contribution.
3. **Retrospective scoring:** The supervisor doesn't just aggregate — it *learns* which agents perform well by scoring past signals against realised returns and adapting weights via softmax over rolling Sharpe ratios.
4. **No lookahead:** All train/val/test splits are calendar-based.  Feature standardisation uses train-set statistics only.  Trading environment executes at current-bar prices.

---

## 3. Contracts (API)

### 3.1 MarketSnapshot

```python
@dataclass
class MarketSnapshot:
    ticker:     str                  # e.g. "BTC-USD", "BTCUSDT"
    timestamp:  datetime
    ohlcv:      pd.DataFrame         # last N rows, may include indicator columns
    indicators: dict[str, float]     # precomputed scalars (SMA, RSI, MACD, etc.)
    sentiment:  float | None = None  # −1.0 to 1.0, optional
```

Two usage modes:
- **Daily agents** (SMA, RSI): read from `ohlcv` DataFrame columns.
- **Learned agents** (NEAT, PPO): read from `indicators` dict (standardised features).

### 3.2 AgentSignal

```python
@dataclass
class AgentSignal:
    agent_id:   str
    timestamp:  datetime
    ticker:     str
    action:     Literal["buy", "sell", "hold"]
    confidence: float                # 0.0 – 1.0
    horizon:    str = "1d"           # "1d", "1h", "1w"
    metadata:   dict = {}            # agent-specific (e.g. raw network outputs)
```

### 3.3 TradingDecision

```python
@dataclass
class TradingDecision:
    timestamp:     datetime
    ticker:        str
    action:        Literal["buy", "sell", "hold"]
    confidence:    float
    position_size: float             # 0.0–1.0, fraction of portfolio
    reasoning:     dict[str, AgentSignal]  # keyed by agent_id
```

---

## 4. Agents Implemented

| Agent | Class | Technique | Input | Status |
|-------|-------|-----------|-------|--------|
| SMA Crossover | `SMACrossoverAgent` | Heuristic | SMA(20), SMA(50) from OHLCV | Done (Task 1) |
| RSI Mean-Reversion | `RSIAgent` | Heuristic | RSI(14) from indicators | Done (Task 1) |
| NEAT | `NEATAgent` | Neuroevolution (NEAT) | 12 standardised features + position/cash | Done (Task 2) |
| PPO | — | Reinforcement Learning | Same 14-dim observation | Trained (Task 2), not yet wrapped as BaseAgent |
| GRU | — | Deep Learning | OHLCV + indicators | Planned (Task 3+) |
| Sentiment | — | NLP (FinBERT) | News headlines | Planned (Task 3+) |
| Evolutionary | — | Genetic Algorithm (DEAP) | Indicators | Planned (Task 3+) |
| Random | — | Random baseline | — | Planned |

### 4.1 SMA Crossover Agent

**Logic:** Buy when SMA(20) crosses above SMA(50); sell on downward crossover; hold otherwise.
**Confidence:** `|SMA20 − SMA50| / SMA50`, clamped to [0, 1].
**File:** `src/hmats/agents/sma_crossover.py`

### 4.2 RSI Agent

**Logic:** Buy if RSI(14) < 30 (oversold); sell if RSI(14) > 70 (overbought); hold otherwise.
**Confidence:** `|RSI − 50| / 50`, clamped to [0, 1].
**File:** `src/hmats/agents/rsi_agent.py`

### 4.3 NEAT Agent

**Architecture:** Small feedforward neural network evolved via the NEAT algorithm (NeuroEvolution of Augmenting Topologies).
**Training:** Population of 50 genomes, 20 generations.  Each genome evaluated over 5 random episodes on training data.
**Fitness function:**
```
fitness = 400 × mean_log_return
        −  80 × volatility
        −   2 × |max_drawdown|
        +   2 × activity_score     (Gaussian, target ≈ 30 trades/1000 steps)
        +   1 × exposure_score     (Gaussian, target ≈ 50% time in market)
```
**Inference:** `argmax(net.activate(obs))` → {0: hold, 1: buy, 2: sell}.  Confidence = softmax of raw output activations.
**File:** `src/hmats/agents/neat_agent.py`
**Config:** `configs/neat_trading.ini`

---

## 5. Supervisor (Coordinator)

**File:** `src/hmats/coordinator/supervisor.py`

### 5.1 Cold Start (Equal Weights)

At init all agents receive weight `1/N`.  Decision = weighted vote where each agent's vote has weight `w_i × confidence_i`.

### 5.2 Performance-Weighted Voting

After each decision, the caller invokes `supervisor.update_outcome(ticker, realised_return)`.

**Scoring:** For each agent's signal:
- `action = hold` → score = 0
- `action = buy/sell`, correct direction → score = +confidence
- `action = buy/sell`, wrong direction → score = −confidence

**Weight update:** Rolling Sharpe ratio computed over last `sharpe_window` scores (default 30).  Weights = softmax over agent Sharpe ratios.

**Position sizing:** `min(confidence × risk_tolerance, 1.0)`.  `risk_tolerance` is a user-configurable scalar (default 1.0).

### 5.3 Internal State

```python
@dataclass
class AgentRecord:
    agent_id:  str
    signals:   list[AgentSignal]    # full history
    outcomes:  list[float]          # retrospective scores
    sharpe:    float                # rolling Sharpe (updated on each outcome)
    weight:    float                # current voting weight
```

---

## 6. Data Pipeline

### 6.1 Daily Indicators (Task 1)

**File:** `src/hmats/data/pipeline.py`

Computes 13 technical indicators from OHLCV data:
- SMA (20, 50, 200)
- EMA (12, 26)
- RSI (14)
- MACD (12, 26, 9) — line, signal, histogram
- Bollinger Bands (20, 2σ) — upper, mid, lower
- Volume delta

Used by: `SMACrossoverAgent`, `RSIAgent`, daily notebooks.

### 6.2 Hourly Features (Task 2)

**File:** `src/hmats/data/features.py`

Computes 12 features from 1-hour Binance OHLCV data:

| # | Feature | Description |
|---|---------|-------------|
| 1 | `log_ret_1` | 1-bar log return |
| 2 | `vol_24` | 24-bar rolling volatility of log returns |
| 3 | `vol_72` | 72-bar rolling volatility |
| 4 | `sma_ratio_24_72` | SMA(24)/SMA(72) − 1 |
| 5 | `macd` | EMA(12) − EMA(26) |
| 6 | `macd_signal` | 9-bar EMA of MACD line |
| 7 | `macd_hist` | MACD − signal |
| 8 | `mom_24` | 24-bar price momentum |
| 9 | `mom_72` | 72-bar price momentum |
| 10 | `rsi_14` | RSI(14) |
| 11 | `volu_z_72` | 72-bar z-score of volume |
| 12 | `z_close_72` | 72-bar z-score of close price |

All features clipped to [−10, 10] before standardisation.
Standardisation: z-score using **train-set mean/std only** (no data leakage).

Used by: `NEATAgent`, PPO, `TradingEnv`.

### 6.3 Binance Data Fetcher

**File:** `src/hmats/data/binance.py`

Paginated download of Binance kline data via REST API.  Results cached as local CSV files in `./data_cache/`.

### 6.4 Train/Test Splits

**File:** `src/hmats/data/splits.py`

- `calendar_split(df, train_end, val_end)` — deterministic calendar-based split.
- `rolling_test_windows(df, start, window_days=140)` — non-overlapping rolling windows for out-of-sample evaluation.

---

## 7. Trading Environment

**File:** `src/hmats/data/trading_env.py`
**Type:** Gymnasium environment (`gym.Env`)

| Property | Value |
|----------|-------|
| Action space | Discrete(3): 0=hold, 1=buy, 2=sell |
| Position model | Binary: all-in or all-out |
| Observation | 12 features + position_flag + cash_fraction = 14 dims |
| Reward | Log equity return per step |
| Transaction fee | 0.05% per trade |
| Starting equity | 1.0 |

Used for training both NEAT and PPO agents.

---

## 8. Evaluation Framework

### 8.1 Metrics

| Metric | Description |
|--------|-------------|
| Cumulative return | Final equity − 1 |
| Annualised Sharpe ratio | `(mean / std) × √(steps_per_year)` on log returns |
| Maximum drawdown | Worst peak-to-trough decline |
| Number of trades | Total buy+sell actions |
| Win rate | Planned |

### 8.2 Baselines

- **Buy and hold:** Buy at first price, sell at last, with fees.
- **Random agent:** Uniform random actions (planned).

### 8.3 Evaluation Protocol

1. Train agents on **train set** (< 2024-01-01).
2. Tune hyperparameters on **validation set** (2024).
3. Evaluate on **rolling 140-day test windows** starting 2025-01-01.
4. Compare: individual agents, ensemble (equal weight), ensemble (learned weight), buy-and-hold.

---

## 9. Project Structure

```
hybrid-multi-agent-trading-system/
├── pyproject.toml                     # uv + ruff + hatchling, Python >=3.12
├── Makefile                           # install, lint, format, test, run targets
├── configs/
│   └── neat_trading.ini               # NEAT hyperparameters
├── models/                            # saved models (gitignored)
│   └── .gitkeep
├── data/                              # raw/cached data (gitignored)
├── tests/
│   └── __init__.py
└── src/hmats/
    ├── __init__.py
    ├── agents/
    │   ├── __init__.py
    │   ├── base.py                    # MarketSnapshot, AgentSignal, TradingDecision, BaseAgent
    │   ├── sma_crossover.py           # SMACrossoverAgent
    │   ├── rsi_agent.py               # RSIAgent
    │   └── neat_agent.py              # NEATAgent
    ├── coordinator/
    │   ├── __init__.py
    │   └── supervisor.py              # Supervisor with weighted voting
    ├── data/
    │   ├── __init__.py
    │   ├── pipeline.py                # Daily indicators + build_snapshot()
    │   ├── features.py                # Hourly features for NEAT/PPO
    │   ├── trading_env.py             # Gymnasium TradingEnv + eval helpers
    │   ├── binance.py                 # Binance kline data fetcher
    │   └── splits.py                  # Calendar split + rolling windows
    └── notebooks/
        ├── 01_data_exploration.ipynb   # BTC-USD daily data + indicator plots
        ├── 02_first_run.ipynb          # End-to-end: snapshot → supervisor → decision
        ├── 03_neat_training.ipynb      # Train NEAT + PPO on hourly data
        └── 04_rolling_test.ipynb       # Rolling 140-day test windows + weight demo
```

---

## 10. Technology Stack

| Component | Tool | Version |
|-----------|------|---------|
| Language | Python | ≥ 3.12 |
| Package manager | uv | latest |
| Linter/formatter | ruff | ≥ 0.4 |
| Build backend | hatchling | latest |
| Testing | pytest + pytest-cov | ≥ 8.2 |
| Deep learning | PyTorch | ≥ 2.3 |
| RL framework | Stable-Baselines3 | ≥ 2.3 |
| RL environment | Gymnasium | ≥ 0.29 |
| Neuroevolution | neat-python | ≥ 0.92 |
| Evolutionary algorithms | DEAP | ≥ 1.4 |
| NLP (planned) | HuggingFace Transformers | ≥ 4.40 |
| Financial data | yfinance, Binance API | latest |
| Visualization | matplotlib | ≥ 3.8 |
| Logging | loguru | ≥ 0.7 |
