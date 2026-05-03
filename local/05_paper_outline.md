# Paper Outline (Draft)

## Working Title

**Hybrid Multi-Agent Trading System: Combining Neuroevolution, Reinforcement Learning, and Rule-Based Heuristics for Cryptocurrency Trading**

---

## Abstract (sketch)

We present a hybrid multi-agent system for algorithmic cryptocurrency trading that combines heterogeneous AI agents — rule-based heuristics, neuroevolutionary networks (NEAT), and reinforcement learning (PPO) — under a performance-weighted supervisor.  Each agent independently analyses market data and produces directional signals with confidence scores.  The supervisor aggregates these signals through a weighted voting mechanism where weights adapt dynamically based on each agent's rolling out-of-sample Sharpe ratio.  We evaluate the system on BTC/USDT hourly data with a strict calendar-based train/validate/test protocol and rolling out-of-sample windows.  Preliminary results indicate that [TBD — fill after experiments are mature].

---

## 1. Introduction

- Motivation: cryptocurrency markets are volatile, 24/7, and driven by heterogeneous information sources.
- No single model dominates across all market regimes.
- Hypothesis: a heterogeneous ensemble with adaptive weighting can achieve better risk-adjusted returns than individual agents or passive strategies.
- Contributions:
  1. Open-source multi-agent framework with pluggable agent architecture.
  2. Novel combination of NEAT, PPO, and rule-based agents in a single ensemble.
  3. Performance-weighted supervisor using rolling Sharpe with softmax.
  4. Rigorous evaluation protocol with calendar splits and rolling windows.

---

## 2. Related Work

### 2.1 Technical Analysis and Rule-Based Trading

- Moving average strategies (Brock et al., 1992)
- RSI and momentum indicators (Wilder, 1978)

### 2.2 Reinforcement Learning for Trading

- Deep RL for portfolio management (Jiang et al., 2017)
- PPO applications in finance (Yang et al., 2020)
- FinRL framework (Liu et al., 2021)

### 2.3 Neuroevolution

- NEAT (Stanley & Miikkulainen, 2002)
- Neuroevolution for trading (Lohpetch & Corne, 2011)

### 2.4 Ensemble and Multi-Agent Systems

- Ensemble methods for time series forecasting
- Multi-agent systems in finance (Lux & Marchesi, 1999)
- Mixture of experts (Jacobs et al., 1991)

### 2.5 NLP and Sentiment Analysis for Trading (planned)

- FinBERT (Araci, 2019)
- News-driven trading (Ding et al., 2015)

---

## 3. System Design

### 3.1 Architecture Overview

Refer to: `local/01_system_design.md` §2.

### 3.2 Data Contracts

- MarketSnapshot
- AgentSignal
- TradingDecision

### 3.3 Agent Interface

- BaseAgent abstract class
- compute() method signature

---

## 4. Agents

### 4.1 SMA Crossover Agent

- Mathematical formulation of SMA(20) vs SMA(50) crossover
- Confidence metric definition

### 4.2 RSI Mean-Reversion Agent

- RSI formula (Wilder smoothing)
- Oversold/overbought thresholds
- Confidence metric

### 4.3 NEAT Agent

- NEAT algorithm summary
- Network topology: 14 inputs, 3 outputs, evolved hidden layer
- Fitness function (multi-objective: returns, volatility, drawdown, activity, exposure)
- Training procedure (population, generations, episodes)

### 4.4 PPO Agent

- PPO algorithm summary (clipped surrogate objective)
- Policy architecture (MLP)
- Training procedure (timesteps, environment)

### 4.5 Planned Agents

- GRU sequence model
- FinBERT sentiment agent

---

## 5. Supervisor and Ensemble Mechanism

### 5.1 Weighted Voting

- Mathematical formulation of vote aggregation
- Confidence-weighted voting formula

### 5.2 Retrospective Scoring

- Signal scoring: correct direction → +confidence, wrong → -confidence

### 5.3 Adaptive Weight Update

- Rolling Sharpe ratio computation
- Softmax normalisation
- Analysis: why softmax? (compared to linear normalisation, argmax, etc.)

### 5.4 Position Sizing

- Risk-adjusted position sizing formula

---

## 6. Experimental Setup

### 6.1 Data

- BTC/USDT 1-hour candles from Binance
- Period: 2023-01-01 to 2025-12-31 (or latest)
- Feature set: 12 technical features (Table X)

### 6.2 Train/Validate/Test Split

- Calendar-based split (no lookahead)
- Train: 2023, Val: 2024, Test: 2025+
- Feature standardisation on training set only

### 6.3 Evaluation Metrics

- Cumulative return
- Annualised Sharpe ratio
- Maximum drawdown
- Number of trades
- Win rate

### 6.4 Baselines

- Buy and hold
- Random agent
- Individual agents (not ensembled)

---

## 7. Results

### 7.1 Individual Agent Performance

Table comparing NEAT, PPO, SMA, RSI, Buy-and-Hold across rolling windows.

### 7.2 Ensemble Performance

Equal-weight vs. learned-weight supervisor.

### 7.3 Weight Evolution Analysis

How do agent weights change over time?  Do they correlate with market regimes?

### 7.4 Ablation Studies (planned)

- Remove NEAT: does ensemble degrade?
- Remove rule-based agents: same question.
- Fix equal weights: how much does adaptation help?

---

## 8. Discussion

- When does the ensemble outperform individuals?
- Failure modes (all agents wrong simultaneously, extreme regime shifts)
- Computational cost comparison
- Limitations (see `local/04_experiment_log.md` §Known Limitations)

---

## 9. Conclusion and Future Work

- Summary of findings
- Future agents: GRU, FinBERT, DEAP
- Multi-asset portfolio extension
- Live trading demo
- Statistical significance testing (bootstrap confidence intervals, paired t-tests on rolling Sharpe)

---

## Key Figures to Include

1. System architecture diagram (§2 of this paper)
2. NEAT genome topology example (visualise best genome)
3. Equity curves: individual agents vs. ensemble vs. buy-and-hold
4. Sharpe ratio comparison bar chart across rolling windows
5. Supervisor weight evolution over time (line plot)
6. Feature correlation heatmap
7. Drawdown comparison plot
8. Fitness evolution across NEAT generations

---

## Key Tables to Include

1. Feature description table (12 features)
2. NEAT hyperparameter table
3. PPO hyperparameter table
4. Per-window performance metrics (return, Sharpe, MDD, trades)
5. Aggregate performance summary (mean ± std across windows)
6. Agent comparison: final ensemble weights vs. individual Sharpe
