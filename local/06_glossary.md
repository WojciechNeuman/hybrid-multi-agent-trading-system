# Glossary

## Hybrid Multi-Agent Trading System

---

| Term | Definition |
|------|-----------|
| **Agent** | An autonomous software component that analyses a `MarketSnapshot` and produces an `AgentSignal`. All agents extend `BaseAgent`. |
| **AgentRecord** | Internal supervisor data structure tracking one agent's signal history, retrospective outcome scores, rolling Sharpe ratio, and current voting weight. |
| **AgentSignal** | The output of an agent: `(agent_id, timestamp, ticker, action, confidence, horizon, metadata)`. |
| **Bollinger Bands** | Volatility bands placed 2 standard deviations above and below a 20-period SMA. Used as daily indicators. |
| **Calendar Split** | A train/validate/test split based on fixed date boundaries (e.g. train < 2024, val = 2024, test ≥ 2025) to prevent lookahead bias. |
| **Confidence** | A float in [0, 1] expressing how strongly an agent believes in its action. Used in weighted voting and position sizing. |
| **Crossover (golden/death)** | A golden cross occurs when a short-term moving average crosses above a long-term one (bullish). A death cross is the opposite (bearish). |
| **DEAP** | Distributed Evolutionary Algorithms in Python. Library for genetic algorithms. Planned for a future evolutionary strategy agent. |
| **EMA** | Exponential Moving Average. Gives more weight to recent prices than SMA. Used in MACD calculation. |
| **Fitness Function** | A scalar score used by evolutionary algorithms to rank candidate solutions (genomes). In HMATS: multi-objective combining returns, risk, and trading activity. |
| **Genome** | In NEAT: a genotype encoding a neural network's topology and weights. Evolved through mutation and crossover. |
| **Gymnasium** | OpenAI's successor to Gym. Provides standardised `Env` interface (`reset`, `step`, observation/action spaces). |
| **HMATS** | Hybrid Multi-Agent Trading System — the name of this project. |
| **Kline** | Binance term for candlestick/OHLCV bar. |
| **Log Return** | `ln(P_t / P_{t-1})`. Additive over time; used as reward in `TradingEnv` and for Sharpe calculation. |
| **Lookahead Bias** | Using future information to make past decisions. Avoided via calendar splits and train-only standardisation. |
| **MACD** | Moving Average Convergence/Divergence. MACD line = EMA(12) − EMA(26). Signal = EMA(9) of MACD line. Histogram = MACD − Signal. |
| **MarketSnapshot** | Input consumed by every agent: `(ticker, timestamp, ohlcv DataFrame, indicators dict, sentiment)`. |
| **Maximum Drawdown (MDD)** | The worst peak-to-trough decline in portfolio equity during a period. `min(equity / running_max − 1)`. |
| **MDP** | Markov Decision Process. Formal framework for sequential decision-making: `(S, A, T, R, γ)`. |
| **NEAT** | Neuroevolution of Augmenting Topologies (Stanley & Miikkulainen, 2002). Evolves both neural network topology and weights. |
| **OHLCV** | Open, High, Low, Close, Volume — the five standard fields of a candlestick/price bar. |
| **PPO** | Proximal Policy Optimization (Schulman et al., 2017). A policy gradient RL algorithm with clipped surrogate objective. |
| **Position Flag** | Binary indicator (0 or 1) in the observation vector: 0 = flat (no position), 1 = long. |
| **Rolling Window** | A fixed-size time window that slides forward through the data. Used for test evaluation (140 days, non-overlapping). |
| **RSI** | Relative Strength Index (Wilder, 1978). Oscillator in [0, 100]; < 30 = oversold, > 70 = overbought. |
| **Sharpe Ratio** | Risk-adjusted return: `(mean return / std of returns) × √(annualisation_factor)`. Higher = better. |
| **SMA** | Simple Moving Average. `SMA_w(t) = (1/w) × Σ C_{t-i}` for `i = 0..w-1`. |
| **Softmax** | `softmax(x_i) = exp(x_i) / Σ exp(x_j)`. Maps real-valued scores to a probability distribution. Used for weight normalisation. |
| **Speciation** | In NEAT: grouping genomes by topological similarity. Protects novel structures from being outcompeted before they mature. |
| **Supervisor** | Central coordinator that collects `AgentSignal`s from all agents and produces a single `TradingDecision` via weighted voting. |
| **TradingDecision** | Output of the supervisor: `(timestamp, ticker, action, confidence, position_size, reasoning)`. |
| **TradingEnv** | Custom Gymnasium environment for backtesting. Binary position, log-return reward, 0.05% transaction fee. |
| **Transaction Fee** | 0.05% of trade value, applied on both buy and sell sides. Models exchange fees. |
| **Volume Delta** | Absolute change in trading volume between consecutive bars. |
| **Z-Score** | `(x − μ) / σ`. Standardised value indicating how many standard deviations from the mean. Used for feature standardisation and as features themselves (volume z-score, price z-score). |
