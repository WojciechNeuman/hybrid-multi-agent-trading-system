"""00 — Data Ingestion v1: External Data Sources

Downloads non-OHLCV data that cannot be derived from Binance candles:
  Phase A: Fear & Greed Index (alternative.me — free, daily, from Feb 2018)
  Phase B: Per-coin market caps from CoinGecko (free tier, ~1yr daily)
  Phase C: Approximate historical market caps from OHLCV × circulating supply

Requires existing Binance OHLCV parquets in data/raw/ (from v0 ingestion).
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[3]
RAW_DIR = REPO / "data" / "raw"
EXT_DIR = REPO / "data" / "external"
STATIC_DIR = REPO / "data" / "static"
EXT_DIR.mkdir(parents=True, exist_ok=True)

SYMBOLS_1H = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
]

COINGECKO_IDS = {
    "BTCUSDT": "bitcoin",
    "ETHUSDT": "ethereum",
    "BNBUSDT": "binancecoin",
    "XRPUSDT": "ripple",
    "SOLUSDT": "solana",
    "ADAUSDT": "cardano",
    "DOGEUSDT": "dogecoin",
    "AVAXUSDT": "avalanche-2",
    "DOTUSDT": "polkadot",
    "LINKUSDT": "chainlink",
    "USDTUSDT": "tether",
    "USDCUSDT": "usd-coin",
}


print("=" * 70)
print("00 — Data Ingestion v1: External Data Sources")
print("=" * 70)


# ══════════════════════════════════════════════════════════════════════
# PHASE A: Fear & Greed Index
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE A: Fear & Greed Index (alternative.me)")
print("=" * 70)

FNG_URL = "https://api.alternative.me/fng/"
FNG_OUTPUT = EXT_DIR / "fear_greed_index.parquet"

try:
    resp = requests.get(FNG_URL, params={"limit": "0"}, timeout=30)
    resp.raise_for_status()
    fng_data = resp.json()["data"]

    fng_df = pd.DataFrame(fng_data)
    fng_df["value"] = fng_df["value"].astype(int)
    fng_df["timestamp"] = pd.to_datetime(fng_df["timestamp"].astype(int), unit="s", utc=True)
    fng_df = fng_df.rename(columns={"timestamp": "date", "value_classification": "classification"})
    fng_df = fng_df[["date", "value", "classification"]].sort_values("date").reset_index(drop=True)
    fng_df = fng_df.set_index("date")
    if "time_until_update" in fng_df.columns:
        fng_df = fng_df.drop(columns=["time_until_update"])

    fng_df.to_parquet(FNG_OUTPUT)
    print(f"  Downloaded {len(fng_df)} daily records")
    print(f"  Range: {fng_df.index.min().date()} → {fng_df.index.max().date()}")
    print(f"  Value range: {fng_df['value'].min()} – {fng_df['value'].max()}")
    print(f"  Saved: {FNG_OUTPUT}")
except Exception as e:
    print(f"  ERROR: {e}")
    print("  Fear & Greed download failed. Continuing...")


# ══════════════════════════════════════════════════════════════════════
# PHASE B: CoinGecko Market Caps (~1 year daily)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE B: CoinGecko Market Caps (free tier, ~1yr)")
print("=" * 70)

CG_BASE = "https://api.coingecko.com/api/v3"
CG_OUTPUT = EXT_DIR / "coingecko_market_caps.parquet"
CG_DAYS = 365

all_mcap_records = []

for symbol, cg_id in COINGECKO_IDS.items():
    print(f"  Fetching {cg_id} ({symbol})...", end=" ", flush=True)
    try:
        resp = requests.get(
            f"{CG_BASE}/coins/{cg_id}/market_chart",
            params={"vs_currency": "usd", "days": str(CG_DAYS)},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        prices = data.get("prices", [])
        mcaps = data.get("market_caps", [])
        volumes = data.get("total_volumes", [])

        n = min(len(prices), len(mcaps), len(volumes))
        for i in range(n):
            ts = pd.Timestamp(prices[i][0], unit="ms", tz="UTC")
            all_mcap_records.append({
                "date": ts,
                "symbol": symbol,
                "cg_id": cg_id,
                "price": prices[i][1],
                "market_cap": mcaps[i][1] if mcaps[i][1] else np.nan,
                "total_volume": volumes[i][1] if volumes[i][1] else np.nan,
            })
        print(f"{n} records")
        time.sleep(6.5)
    except Exception as e:
        print(f"ERROR: {e}")
        time.sleep(10)

if all_mcap_records:
    cg_df = pd.DataFrame(all_mcap_records)
    cg_df.to_parquet(CG_OUTPUT, index=False)
    print(f"\n  Total records: {len(cg_df)}")
    print(f"  Coins: {cg_df['symbol'].nunique()}")
    print(f"  Date range: {cg_df['date'].min().date()} → {cg_df['date'].max().date()}")
    print(f"  Saved: {CG_OUTPUT}")
else:
    print("  No CoinGecko data downloaded.")


# ══════════════════════════════════════════════════════════════════════
# PHASE C: Approximate Historical Market Caps (from OHLCV × supply)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE C: Approximate Historical Market Caps")
print("=" * 70)

APPROX_OUTPUT = EXT_DIR / "approx_market_caps.parquet"
SUPPLY_PATH = STATIC_DIR / "crypto_market_caps.csv"

supply_df = pd.read_csv(SUPPLY_PATH)
supply_map = dict(zip(supply_df["symbol"], supply_df["circulating_supply"]))
print(f"  Loaded circulating supply for {len(supply_map)} coins")

approx_records = []

for symbol in SYMBOLS_1H:
    parquet_path = RAW_DIR / f"{symbol}_1h.parquet"
    if not parquet_path.exists():
        print(f"  {symbol}: OHLCV not found, skipping")
        continue

    df = pd.read_parquet(parquet_path)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    supply = supply_map.get(symbol, np.nan)
    if np.isnan(supply):
        print(f"  {symbol}: no supply data, skipping")
        continue

    daily = df["close"].resample("1D").last().dropna()
    daily_mcap = daily * supply

    for ts, mcap in daily_mcap.items():
        approx_records.append({
            "date": ts,
            "symbol": symbol,
            "close_price": daily.loc[ts],
            "circulating_supply": supply,
            "approx_market_cap": mcap,
        })
    print(f"  {symbol}: {len(daily)} daily records, supply={supply:,.0f}")

if approx_records:
    approx_df = pd.DataFrame(approx_records)
    approx_df.to_parquet(APPROX_OUTPUT, index=False)
    print(f"\n  Total records: {len(approx_df)}")
    print(f"  Date range: {approx_df['date'].min().date()} → {approx_df['date'].max().date()}")
    print(f"  Saved: {APPROX_OUTPUT}")


# ══════════════════════════════════════════════════════════════════════
# VALIDATION: Check overlap between CoinGecko and approximation
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("VALIDATION: CoinGecko vs Approximate Market Caps")
print("=" * 70)

if all_mcap_records and approx_records:
    cg = pd.read_parquet(CG_OUTPUT)
    ap = pd.read_parquet(APPROX_OUTPUT)

    cg["date_day"] = cg["date"].dt.normalize()
    ap["date_day"] = ap["date"].dt.normalize()

    for sym in ["BTCUSDT", "ETHUSDT"]:
        cg_sym = cg[cg["symbol"] == sym].set_index("date_day")["market_cap"].dropna()
        ap_sym = ap[ap["symbol"] == sym].set_index("date_day")["approx_market_cap"].dropna()

        overlap = cg_sym.index.intersection(ap_sym.index)
        if len(overlap) > 10:
            ratio = (ap_sym.loc[overlap] / cg_sym.loc[overlap]).dropna()
            print(f"  {sym}: {len(overlap)} overlap days, approx/actual ratio = {ratio.mean():.3f} ± {ratio.std():.3f}")
        else:
            print(f"  {sym}: insufficient overlap ({len(overlap)} days)")

print("\n" + "=" * 70)
print("DONE — All external data downloaded")
print("=" * 70)
print(f"\nOutput files:")
for f in sorted(EXT_DIR.glob("*.parquet")):
    size = f.stat().st_size / 1e6
    print(f"  {f.name:>40s}  {size:.2f} MB")
