import requests
from datetime import datetime, timezone

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


def earliest_candle(symbol: str) -> datetime | None:
    r = requests.get(
        "https://api.binance.com/api/v3/klines",
        params={"symbol": symbol, "interval": "1h", "startTime": 0, "limit": 1},
        timeout=10,
    )
    if r.status_code != 200:
        return None
    data = r.json()
    if not data:
        return None
    return datetime.fromtimestamp(data[0][0] / 1000, tz=timezone.utc)


def fetch_coingecko_markets() -> dict[str, dict]:
    r = requests.get(
        "https://api.coingecko.com/api/v3/coins/markets",
        params={
            "vs_currency": "usd",
            "ids": ",".join(COIN_IDS.values()),
            "order": "market_cap_desc",
            "per_page": 20,
            "sparkline": "false",
        },
        timeout=15,
    )
    r.raise_for_status()

    # key by coingecko id for easy lookup
    return {c["id"]: c for c in r.json()}


def main() -> None:
    print("Fetching CoinGecko market data...")
    cg = fetch_coingecko_markets()

    rows = []

    print(f"\n{'Symbol':<12} {'Earliest 1h':<14} {'ATH Price':>14} {'ATH Date':<14} {'Circulating Supply':>22} {'Peak Cap':>12}")
    print("-" * 95)

    for symbol, coin_id in COIN_IDS.items():
        earliest = earliest_candle(symbol)
        earliest_str = earliest.strftime("%Y-%m-%d") if earliest else "NOT FOUND"

        data = cg.get(coin_id)
        if data:
            ath        = data["ath"]
            supply     = data["circulating_supply"]
            ath_date   = data["ath_date"][:10]
            peak       = ath * supply if ath and supply else None
            ath_str    = f"${ath:>13,.2f}"
            supply_str = f"{supply:>22,.0f}"
            peak_str   = f"${peak/1e9:.1f}B" if peak else "N/A"
        else:
            ath = supply = peak = None
            ath_str = ath_date = supply_str = peak_str = "N/A"

        print(f"{symbol:<12} {earliest_str:<14} {ath_str} {ath_date:<14} {supply_str} {peak_str:>12}")

        rows.append({
            "symbol":             symbol,
            "earliest_1h_candle": earliest_str,
            "ath_price_usd":      ath,
            "ath_date":           ath_date if data else "N/A",
            "circulating_supply": supply,
            "peak_market_cap_usd": peak,
        })

    import pandas as pd
    pd.DataFrame(rows).to_csv("crypto_market_caps.csv", index=False)
    print("\nSaved to crypto_market_caps.csv")


"""
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
"""


if __name__ == "__main__":
    main()
