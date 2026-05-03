# Algorithms and Methods

## Hybrid Multi-Agent Trading System

---

## 1. Agent Taxonomy

The system implements a **heterogeneous ensemble** — agents belong to distinct algorithmic families so that their errors are less correlated than in a homogeneous ensemble.

| Family | Agent(s) | Paradigm | Strengths |
|--------|----------|----------|-----------|
| Rule-based | SMA Crossover, RSI | Expert heuristic | Transparent, no training, well-known | 
| Neuroevolution | NEAT | Topology + weight evolution | Evolves architecture; sparse networks; no gradient |
| Reinforcement learning | PPO | Gradient-based policy optimisation | Handles sequential decisions; learns reward shaping |
| Deep learning | GRU (planned) | Supervised sequence model | Captures temporal patterns in price series |
| NLP | FinBERT (planned) | Pre-trained transformer + fine-tuning | Incorporates qualitative market sentiment |

---

## 2. SMA Crossover — Technical Details

### Moving Average Crossover

Two simple moving averages are tracked:
- **Short-term SMA** with window `s = 20` (bars)
- **Long-term SMA** with window `l = 50` (bars)

$$\text{SMA}_w(t) = \frac{1}{w} \sum_{i=0}^{w-1} C_{t-i}$$

where $C_t$ is the closing price at bar $t$.

### Signal Generation

A **crossover event** occurs when the sign of $\text{SMA}_s - \text{SMA}_l$ changes between bars $t-1$ and $t$:

- **Golden cross** (bullish crossover): $\text{SMA}_s(t-1) \leq \text{SMA}_l(t-1) \;\wedge\; \text{SMA}_s(t) > \text{SMA}_l(t)$ → action = `buy`
- **Death cross** (bearish crossover): $\text{SMA}_s(t-1) \geq \text{SMA}_l(t-1) \;\wedge\; \text{SMA}_s(t) < \text{SMA}_l(t)$ → action = `sell`
- No crossover → action = `hold`

### Confidence

$$\text{confidence} = \min\left(\frac{|\text{SMA}_s - \text{SMA}_l|}{\text{SMA}_l},\; 1.0\right)$$

Larger divergence between the two averages ≈ higher conviction.

---

## 3. RSI Mean-Reversion — Technical Details

### Relative Strength Index

$$\text{RSI}(t) = 100 - \frac{100}{1 + RS(t)}$$

where $RS = \frac{\text{avg gain over } n \text{ periods}}{\text{avg loss over } n \text{ periods}}$, $n = 14$.

Implementation uses exponential moving averages (Wilder smoothing) for gain/loss averages.

### Signal Generation

| Condition | Action | Rationale |
|-----------|--------|-----------|
| RSI < 30 | Buy | Oversold — expect mean reversion upward |
| RSI > 70 | Sell | Overbought — expect mean reversion downward |
| 30 ≤ RSI ≤ 70 | Hold | Neutral zone |

### Confidence

$$\text{confidence} = \min\left(\max\left(\frac{|RSI - 50|}{50},\; 0\right),\; 1\right)$$

---

## 4. NEAT (Neuroevolution of Augmenting Topologies)

### Overview

NEAT simultaneously evolves both the **topology** (nodes, connections) and **weights** of small neural networks through genetic operators: mutation (add node, add connection, perturb weight) and crossover of genomes.

Key innovations of NEAT (Stanley & Miikkulainen, 2002):
1. **Historical markings** — each gene has an innovation number for meaningful crossover.
2. **Speciation** — genomes clustered by topological similarity; protects innovation.
3. **Minimal initialization** — starts with no hidden nodes; complexity grows only if beneficial.

### Hyperparameters (configs/neat_trading.ini)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Population size | 50 | Moderate for hourly trading |
| Generations | 20 | Sufficient for convergence on this data |
| Num inputs | 14 | 12 features + position_flag + cash_frac |
| Num outputs | 3 | hold, buy, sell |
| Initial hidden nodes | 0 | Start minimal |
| Activation | tanh | Bounded, smooth |
| Initial connection | full | All inputs connected to all outputs |
| Connection add probability | 0.5 | High topological exploration |
| Node add probability | 0.2 | Moderate complexity growth |
| Weight mutation rate | 0.8 | Aggressive weight search |
| Weight mutation power | 0.5 | ±0.5 standard Gaussian perturbation |
| Compatibility threshold | 3.0 | Species formation |
| Max stagnation | 15 | Kill stale species |
| Elitism | 2 | Preserve best genomes |

### Fitness Function

Each genome is evaluated over 5 random episodes in `TradingEnv`.  The fitness function balances multiple objectives:

$$F = 400 \cdot \bar{r} - 80 \cdot \sigma_r - 2 \cdot |\text{MDD}| + 2 \cdot A + 1 \cdot E$$

where:
- $\bar{r}$: mean per-step log return
- $\sigma_r$: standard deviation of per-step log returns
- MDD: maximum drawdown
- $A$: trading activity score (Gaussian, peak at 30 trades per 1000 steps)
- $E$: market exposure score (Gaussian, peak at 50% time in position)

$$A = \exp\left(-\left(\frac{\text{trades}/1000 - 30}{25}\right)^2\right)$$

$$E = \exp\left(-\left(\frac{\text{exposure\_ratio} - 0.5}{0.4}\right)^2\right)$$

### Penalties

- No trades → fitness −0.5
- Exposure < 1% → fitness −0.5

### Inference

```
raw_output = net.activate(observation)
probabilities = softmax(raw_output)
action = argmax(probabilities)
confidence = max(probabilities)
```

---

## 5. PPO (Proximal Policy Optimization)

### Overview

PPO (Schulman et al., 2017) is a policy gradient algorithm that constrains the policy update to prevent destructive large steps.  The clipped surrogate objective:

$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t\left[\min\left(r_t(\theta)\hat{A}_t,\; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}$ and $\hat{A}_t$ is the GAE advantage estimate.

### Implementation

Uses Stable-Baselines3 `PPO` with `MlpPolicy`.  Trained for 200,000 timesteps on the same `TradingEnv` (Gymnasium environment) as NEAT.

### Observation Space

Same 14-dimensional vector as NEAT:

$$\mathbf{o}_t = [\text{log\_ret\_1},\; \text{vol\_24},\; \ldots,\; \text{z\_close\_72},\; \text{position\_flag},\; \text{cash\_frac}]$$

---

## 6. Trading Environment (TradingEnv)

### MDP Formulation

- **State space** $\mathcal{S}$: 14-dimensional continuous — 12 standardised market features + current position indicator + cash fraction.
- **Action space** $\mathcal{A} = \{0, 1, 2\}$ — hold, buy (all-in), sell (all-out).
- **Transition:** deterministic replay of historical prices.
- **Reward:** log equity return: $r_t = \ln\frac{E_{t+1}}{E_t}$, where $E_t$ = portfolio equity at time $t$.
- **Fee:** 0.05% per trade (applied on both buy and sell).

### Why Log Returns?

- Additive: total return = sum of per-step log returns.
- Better-conditioned for neural network training than multiplicative returns.
- Penalises losses more than it rewards equivalent gains (risk aversion).

---

## 7. Supervisor: Weighted Ensemble Aggregation

### Signal Aggregation

The supervisor collects $N$ agent signals and produces a single decision via weighted voting.

For each action $a \in \{\text{buy}, \text{sell}, \text{hold}\}$:

$$V(a) = \sum_{i: a_i = a} w_i \cdot c_i$$

where $w_i$ is agent $i$'s weight and $c_i$ is its confidence.  Final action = $\arg\max_a V(a)$.

### Retrospective Scoring

After observing the realised return $R$ for the period:

$$\text{score}_i = \begin{cases}
+c_i & \text{if } \text{sign}(R) = \text{sign}(a_i) \\
-c_i & \text{if } \text{sign}(R) \neq \text{sign}(a_i) \\
0 & \text{if } a_i = \text{hold}
\end{cases}$$

### Weight Adaptation

Rolling Sharpe ratio for agent $i$ over the last $W$ outcomes (default $W = 30$):

$$S_i = \frac{\mu_i}{\sigma_i + \epsilon}$$

where $\mu_i$ and $\sigma_i$ are the mean and standard deviation of the last $W$ scores.

Weights from softmax:

$$w_i = \frac{e^{S_i}}{\sum_j e^{S_j}}$$

### Position Sizing

$$\text{position\_size} = \min(\text{confidence} \times \text{risk\_tolerance},\; 1.0)$$

---

## 8. Feature Engineering Pipeline

### Daily Features (pipeline.py)

| Feature | Parameters |
|---------|-----------|
| SMA | 20, 50, 200 |
| EMA | 12, 26 |
| RSI | 14 |
| MACD | (12, 26, 9) |
| Bollinger Bands | (20, 2σ) |
| Volume delta | 1-bar absolute change |

### Hourly Features (features.py)

Designed for learned agents.  All computed from `close` and `volume`:

1. **Log return** (1-bar): $\ln(C_t / C_{t-1})$
2. **Volatility** (24h, 72h): rolling std of log returns
3. **SMA ratio**: $\text{SMA}_{24} / \text{SMA}_{72} - 1$
4. **MACD triple**: line, signal, histogram
5. **Momentum** (24h, 72h): $C_t / C_{t-k} - 1$
6. **RSI** (14 bars)
7. **Volume z-score** (72-bar): standardised volume
8. **Price z-score** (72-bar): standardised close price

All features clipped to $[-10, 10]$ before z-score standardisation.

### Standardisation

Using training set statistics only:

$$\hat{x} = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}} + 10^{-8}}$$

Scaler statistics ($\mu$, $\sigma$) saved to `models/feature_scaler.npz` for inference.

---

## 9. Train/Validate/Test Protocol

### Calendar Split

| Period | Usage | Date Range |
|--------|-------|------------|
| Train | Model training | 2023-01-01 to 2023-12-31 |
| Validation | Hyperparameter tuning, model selection | 2024-01-01 to 2024-12-31 |
| Test | Final evaluation (rolling windows) | 2025-01-01 onward |

### Rolling Test Windows

Non-overlapping 140-day windows from the test start date.
Prevents cherry-picking a single test period.
Last window may be shorter if data runs out.

### Avoiding Lookahead Bias

1. Features standardised using **train-set** statistics only.
2. Calendar split ensures no future data leaks into training.
3. Trained model artefacts serialised to disk and loaded for test-time inference.
4. Trading environment replays prices sequentially — no future prices visible.

---

## 10. References

1. K. O. Stanley and R. Miikkulainen, "Evolving Neural Networks through Augmenting Topologies," *Evolutionary Computation*, vol. 10, no. 2, pp. 99–127, 2002.
2. J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal Policy Optimization Algorithms," *arXiv:1707.06347*, 2017.
3. A. Raffin, A. Hill, A. Gleave, A. Kanervisto, M. Ernestus, and N. Dormann, "Stable-Baselines3: Reliable Reinforcement Learning Implementations," *JMLR*, vol. 22, no. 268, pp. 1–8, 2021.
4. J. W. Wilder, *New Concepts in Technical Trading Systems*. Trend Research, 1978. (RSI, ATR)
5. J. Bollinger, *Bollinger on Bollinger Bands*. McGraw-Hill, 2002.
