Symbol       Earliest 1h         ATH Price ATH Date           Circulating Supply     Peak Cap
-----------------------------------------------------------------------------------------------
BTCUSDT      2017-08-17     $   126,080.00 2025-10-06                 20,023,521     $2524.6B
ETHUSDT      2017-08-17     $     4,946.05 2025-08-24                120,687,385      $596.9B
BNBUSDT      2017-11-06     $     1,369.99 2025-10-13                134,785,930      $184.7B
XRPUSDT      2018-05-04     $         3.65 2025-07-18             61,796,225,236      $225.6B
SOLUSDT      2020-08-11     $       293.31 2025-01-19                576,326,292      $169.0B
ADAUSDT      2018-04-17     $         3.09 2021-09-02             36,974,724,813      $114.3B
DOGEUSDT     2019-07-05     $         0.73 2021-05-08            154,100,446,384      $112.7B
AVAXUSDT     2020-09-22     $       144.96 2021-11-21                431,771,961       $62.6B
DOTUSDT      2020-08-18     $        54.98 2021-11-04              1,682,240,546       $92.5B
LINKUSDT     2019-01-16     $        52.70 2021-05-10                727,099,970       $38.3B

\label{04_Methodology}

\thispagestyle{empty}

datasets (markets, timeframe)

feature engineering

training procedures

limitations







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

write me initial state for this chapter. Write what source I use and 


describe the data source I'm using. Say that the most of the analysis is performed on BTC and ETH since they reflect the state of the cryptocurrency market the best and they have huge market caps







mention that the market is huge. 