# 📄 README — Hybrid Multi-Agent Trading System Thesis

## Overview

This project aims to design and evaluate a **hybrid multi-agent trading system**, where each agent represents a different AI method (e.g., GRU, Transformer, RL, evolutionary algorithms), and a supervisory agent combines their signals into a final trading decision.

The system focuses on **financial time series forecasting**, with a primary application to **cryptocurrency markets**, particularly Bitcoin.

---

# 🧱 Thesis Structure

## 1. Introduction

### Purpose

Introduce the problem and motivate the use of AI and multi-agent systems in financial markets.

### Content

* Overview of financial markets as time series systems
* Differences between:

  * long-term investing vs short-term trading
* Limitations of:

  * fundamental analysis (slow, hard to automate)
  * technical analysis (simplified assumptions)
* Motivation for:

  * AI-based approaches
  * combining multiple strategies

### Key Idea

Most existing approaches use **a single method**, while this thesis explores **combining heterogeneous AI methods**.

---

## 2. Theoretical Background

### Goal

Provide the theoretical foundation needed to understand:

* financial data
* AI methods
* multi-agent systems

---

### 2.1 Financial Markets and Time Series Characteristics

### Content

* Definition of time series
* Financial time series (OHLCV, timeframes)
* Limit Order Book (LOB) as advanced data
* Stylized facts:

  * non-stationarity
  * volatility clustering
  * weak autocorrelation
  * heavy tails
  * nonlinearity
  * long memory

### Conclusion

Financial time series are complex and require **advanced modeling and feature engineering**.

---

### 2.2 Feature Engineering in Financial Data

### Content

* OHLCV as base input
* Technical indicators:

  * trend (SMA, EMA)
  * momentum (RSI, MACD)
  * volatility (ATR, Bollinger Bands)
* Sentiment features:

  * news
  * social media
* Multi-timeframe features:

  * 15m, 1h, 1d
* Feature limitations:

  * noise
  * redundancy
  * overfitting

### Key Idea

Feature engineering bridges raw data and model performance.

---

### 2.3 Artificial Intelligence Methods in Algorithmic Trading

### Content

#### Classical approaches

* rule-based systems
* ARIMA / statistical models

#### Machine learning

* regression
* tree-based methods

#### Deep learning

* GRU / LSTM → sequential dependencies
* TCN → local temporal patterns
* Transformers → long-range dependencies

#### Reinforcement learning

* trading as decision-making process
* reward optimization

#### Evolutionary methods

* genetic algorithms
* neuroevolution

### Key Idea

Different methods capture different aspects of the market.

---

### 2.4 Ensemble Methods and Decision Fusion

### Content

* ensemble learning (bagging, boosting)
* weighted voting
* stacking / meta-learning
* mixture of experts

### Key Idea

Combining models can outperform individual methods.

---

### 2.5 Multi-Agent Systems

### Content

* definition of agents:

  * autonomy
  * reactivity
  * proactiveness
* types:

  * centralized
  * decentralized
  * hierarchical

### Context

* renewed interest due to LLM-based agents
* relevance to decision systems

### Key Idea

Each agent specializes in a different task → system combines them.

---

## 3. System Architecture

### Goal

Define the structure of the proposed system.

---

### 3.1 Overall Architecture

* data flow diagram
* pipeline:

  ```
  data → features → agents → supervisor → decision
  ```

---

### 3.2 Individual Agents

Each agent:

* receives input features
* produces signal (buy / sell / hold)

Types:

* forecasting agents (GRU, TCN, Transformer)
* reinforcement learning agent
* evolutionary agent
* sentiment (NLP) agent
* baseline agents

---

### 3.3 Supervisory Agent (CORE CONTRIBUTION)

### Content

* signal aggregation:

  * weighted voting
  * meta-model
* handling conflicts
* risk preferences

### Key Idea

System is:

* **heterogeneous** → different AI paradigms
* **hybrid** → integrated into one decision

---

## 4. Methodology

### Goal

Describe how the system is built and evaluated.

---

### 4.1 Data Selection

### Decision: Cryptocurrency market

#### Why Bitcoin:

* continuous trading (no session gaps)
* high liquidity and volatility
* fewer fundamental constraints
* widely used in literature 

#### Extension:

* ETH
* SOL
* other high-volume assets

---

### 4.2 Dataset Setup

* BTC/USDT
* timeframe: **1h candles**
* prediction horizons:

  * 1h
  * 4h
  * 24h

### Training strategy:

* use long history (~10 years)
* apply **recency weighting**

---

### 4.3 Data Pipeline

* REST → historical data
* WebSocket → live data
* feature generation
* agent input preparation

---

### 4.4 Feature Engineering

* technical indicators
* sentiment signals (initially simple, then improved)
* multi-timeframe features

---

### 4.5 Training and Evaluation

* time-based split (IMPORTANT)
* walk-forward validation
* comparison:

  * single agents
  * multi-agent system

---

## 5. Implementation

### Content

* tools:

  * Python
  * PyTorch
* architecture decisions
* integration of agents
* limitations:

  * compute
  * latency

---

## 6. Experiments and Results

### Sections

* Individual agent performance
* Multi-agent performance
* Ablation study
* Robustness across market regimes

---

## 7. Financial Evaluation

### Goal

Evaluate real trading performance.

### Metrics:

* return
* Sharpe ratio
* max drawdown

---

### Trading Assumptions (your notes — best place HERE)

#### Execution setup:

* market/limit orders
* immediate TP/SL:

  * TP: +1%
  * SL: -0.5%

#### Costs:

* fees: ~0.02–0.04%
* main risk: **slippage**, not fees

#### Insight:

* leverage does not change fees
* execution quality matters more than model accuracy

---

## 8. Discussion

### Content

* interpretation of results
* which agents contribute most
* limitations:

  * overfitting
  * data leakage
  * missing information (e.g. macro events)

---

## 9. Conclusion

### Content

* summary of findings
* whether multi-agent approach improves performance
* future work:

  * better aggregation
  * more data
  * live deployment

---

# 🧠 Core Research Idea

From your abstract:

> Does combining heterogeneous AI methods produce more stable and reliable trading signals than any single method? 

---

# 🔥 Key Strength of Your Approach

* Not searching for “best model”
* Instead:

  > combine models with different strengths

This directly addresses a known limitation:

> no single model consistently performs best in financial prediction

---

# 🚀 Next Steps (practical)

1. Finish Feature Engineering subsection
2. Implement first agent (GRU or evolutionary — you already started)
3. Define:

   * `AgentInput`
   * `AgentOutput`
4. Build simple **supervisor (weighted voting)**
