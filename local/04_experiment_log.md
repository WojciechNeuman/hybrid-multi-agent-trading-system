# Experiment Log

## Hybrid Multi-Agent Trading System

---

## Task 1 — Foundation (Completed)

### Objective

Build the skeleton: data pipeline, contracts, two rule-based agents, supervisor with equal-weight majority vote.

### What was built

| Component | File(s) | Description |
|-----------|---------|-------------|
| Data contracts | `agents/base.py` | `MarketSnapshot`, `AgentSignal`, `TradingDecision`, `BaseAgent` |
| Daily pipeline | `data/pipeline.py` | `compute_indicators()` adds 13 technical indicators; `build_snapshot()` creates MarketSnapshot from DataFrame |
| SMA agent | `agents/sma_crossover.py` | Golden/death cross detection on SMA(20) vs SMA(50) |
| RSI agent | `agents/rsi_agent.py` | Mean-reversion: buy < 30, sell > 70 |
| Supervisor v1 | `coordinator/supervisor.py` | Equal-weight majority vote |
| Exploration notebook | `notebooks/01_data_exploration.ipynb` | Fetches BTC-USD daily, plots indicators |
| First-run notebook | `notebooks/02_first_run.ipynb` | End-to-end snapshot → agents → supervisor → decision |

### Bug Found & Fixed

**Issue:** `build_snapshot()` in `pipeline.py` was stripping indicator columns from the `ohlcv` DataFrame, passing only `["Open", "High", "Low", "Close", "Volume"]` to the `MarketSnapshot`. This prevented `SMACrossoverAgent` from accessing historical SMA values (`iloc[-2]`) needed for crossover detection.

**Fix:** Changed `build_snapshot()` to pass the full enriched `tail` DataFrame (OHLCV + all indicator columns) as `MarketSnapshot.ohlcv`.

```python
# Before (broken)
ohlcv=tail[["Open", "High", "Low", "Close", "Volume"]]

# After (fixed)
ohlcv=tail
```

### Verification

- `make lint` passes.
- Smoke test: `Supervisor.run()` produces valid `TradingDecision` with both agents contributing.
- Notebooks execute without errors.

---

## Task 2 — NEAT + Weighted Supervisor + Train/Test Framework (Completed)

### Objective

Add a neuroevolutionary agent (NEAT), extend the supervisor with performance-based weight adaptation, and build a proper train/validate/test framework.

### What was built

| Component | File(s) | Description |
|-----------|---------|-------------|
| Hourly features | `data/features.py` | 12 technical features for learned agents |
| Binance fetcher | `data/binance.py` | Paginated kline download with caching |
| Calendar splits | `data/splits.py` | `calendar_split()`, `rolling_test_windows()` |
| Trading environment | `data/trading_env.py` | Gymnasium env + evaluation helpers + NEAT fitness |
| NEAT agent | `agents/neat_agent.py` | Loads evolved genome, runs feedforward inference |
| Supervisor v2 | `coordinator/supervisor.py` | `AgentRecord`, rolling Sharpe, softmax weights, `update_outcome()` |
| NEAT config | `configs/neat_trading.ini` | 14→3, pop=50, 20 gens |
| Training notebook | `notebooks/03_neat_training.ipynb` | Trains NEAT + PPO on hourly BTC data |
| Testing notebook | `notebooks/04_rolling_test.ipynb` | Rolling 140-day windows + supervisor weight demo |

### Key Design Decisions

1. **Fitness function for NEAT**: Multi-objective with returns, volatility, drawdown, trading activity, and market exposure. Without the activity/exposure terms, NEAT genomes converge to degenerate strategies (always hold or always trade every bar).

2. **Softmax weight adaptation**: Chosen over linear normalisation or argmax because it:
   - Produces valid probability distribution
   - Never assigns zero weight (no agent is completely silenced)
   - Amplifies differences between strong and weak agents without extreme concentration

3. **Scoring signal vs. scoring action**: We score each agent's individual signal (not the supervisor's aggregate action) against the realised return. This ensures each agent is evaluated on *its own* recommendation.

4. **Binary position model**: All-in or all-out (no partial positions). Simplifies the MDP state space at the cost of realism. Suitable for initial experiments; can be extended to continuous position sizing.

5. **Log return as reward**: Preferable to percentage returns because log returns are additive (simplifies cumulative return calculation) and naturally penalise losses more than rewarding gains.

### Verification

- `make lint` passes (ruff check + format).
- Smoke tests pass: BaseAgent abstraction, calendar splits, supervisor weight updates, TradingEnv episode rollout.
- Dependencies synchronised via `uv sync --all-extras`.

---

## Results Summary (Preliminary)

> Note: Exact numbers depend on the specific training run and data download.
> The notebooks capture full equity curves and metrics for each run.

### Training Environment

- **Data:** BTC/USDT 1-hour candles from Binance
- **Train period:** 2023-01-01 to 2023-12-31 (~8,760 hourly bars)
- **Validation period:** 2024-01-01 to 2024-12-31 (~8,760 hourly bars)
- **Test period:** 2025-01-01 onward (rolling 140-day windows)

### NEAT Training

- Population: 50 genomes
- Generations: 20
- Episodes per evaluation: 5 (averaged fitness)
- Best genome saved to `models/neat_winner.pkl`

### PPO Training

- Algorithm: PPO (Stable-Baselines3)
- Policy: MlpPolicy (default 2-layer MLP, 64 units each)
- Timesteps: 200,000
- Model saved to `models/ppo_model.zip`

### Supervisor Weight Evolution (Demo)

The supervisor starts with equal weights (1/N) and adjusts based on each agent's rolling Sharpe ratio.  Over the first test window, weights typically diverge as some agents demonstrate better directional accuracy.  This is visualised in notebook 04.

---

## Known Limitations

1. **Binary position model**: No partial positions, no leverage, no short selling with borrowed funds.
2. **Transaction fees**: Fixed at 0.05%, does not model slippage or market impact.
3. **Replay-based backtesting**: Assumes fills at historical close prices — no order book simulation.
4. **Small NEAT population**: 50 genomes × 20 generations is relatively small for complex financial domains.
5. **PPO training duration**: 200k timesteps is a starting point; may need 1M+ for robust performance.
6. **No sentiment agent yet**: Planned for Task 3+.
7. **No risk management layer**: No stop-loss, take-profit, or drawdown circuit breaker.
8. **Single asset**: Currently only BTC — no portfolio diversification.

---

## Roadmap (Tasks 3+)

| Task | Description | Status |
|------|-------------|--------|
| Task 3 | GRU forecasting agent + FinBERT sentiment agent | Planned |
| Task 4 | DEAP evolutionary strategy agent | Planned |
| Task 5 | Risk management layer (stop-loss, max drawdown limit) | Planned |
| Task 6 | Multi-asset support | Planned |
| Task 7 | Live paper-trading demo | Planned |
| Task 8 | Comprehensive benchmarking + statistical significance tests | Planned |
