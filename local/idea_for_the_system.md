



1. Bitcoin as a main source of the analysis

Bitcoin is a strong main asset for your thesis because:

- It trades continuously, so you avoid the artificial discontinuity of stock-market sessions.

- It has high liquidity and high volatility, so there are enough movements to test short-horizon strategies.

- It has weaker “classical fundamentals” than stocks, so the system can focus on price, volume, technical indicators, volatility, and sentiment.

- The literature also supports this direction. The 2020–2022 review says that interest in cryptocurrency and forex forecasting increased, and that Bitcoin and Ether are the dominant cryptocurrency assets studied.

Extending the analysis for future pairs:
- ETH
- SOL
- Other high volume / high-volatility 


2. Training period

Weighted training:
Use 10 years, but weight recent samples more strongly.


## Proposals

### Proposal 1

Main thesis system:
BTC/USDT
1h candles
prediction horizons: 1h, 4h, 24h
exchange API data
WebSocket for live candles
REST for historical candles
SL/TP handled by execution/risk agent
no LOB in main version


## Important Issues

a) Issue 1
1. Open position (limit or market)
2. Immediately set:
   - TP (e.g. +1%)
   - SL (e.g. -0.5%)

b) Issue 2
Transaction fees are low and predictable,
but slippage and execution uncertainty dominate costs.

c) Issue 3
Cost per trade (your setup):
~0.02%–0.04% + funding

On $1000:
~$0.20–$0.70 total

Leverage:
does NOT change fees

Biggest risk:
slippage, not fees