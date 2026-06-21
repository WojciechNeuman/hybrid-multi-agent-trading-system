"""Rule-based trading agents (notebook 08) — classical, parameter-free strategy agents.

The four learned agents (LGBM, Mamba, TCN, PatchTST) are all *nonlinear models on the same
feature panel*, so they share failure modes and the coordinator has nothing orthogonal to rotate
into when they all lose together. This module adds three **economically-motivated rule agents**
whose edge comes from *strategy logic*, not from fitting the feature panel:

* ``trend``    — trend-following: long when the moving-average / SuperTrend / MACD structure is
                 bullish, short when bearish. Designed to earn in directional (bull/bear) regimes.
* ``meanrev``  — mean-reversion: fade oscillator extremes (RSI / Stoch / Williams / MFI / Bollinger
                 position). Designed to earn in choppy / range-bound regimes.
* ``volbreak`` — volatility breakout: trade the direction of a range break (24h breakout flags).
* ``sentiment_regime`` — contrarian Fear & Greed: fade sentiment extremes (extreme fear -> long,
                 extreme greed -> short), confirmed by the 7-day sentiment trend. Its only inputs are
                 the crowd-sentiment columns, so it carries information no price-feature model holds.
* ``dominance_rotation`` — cross-asset capital rotation: long BTC when capital is rotating *into* it
                 (BTC dominance rising, ETH/BTC weakening), short/neutral when alts lead (dominance
                 falling, ETH/BTC strengthening). Edge — if any — comes from cross-asset flow, not BTC
                 price structure.

Design discipline (why this is leak-free):

1. **Signals are causal and parameter-free.** Each signal is a vote fraction over *already-causal*
   features using *fixed economic thresholds* (e.g. RSI<0.30 = oversold). No statistic is fitted on
   the data, so there is no scaler/label that can leak future information.
2. **Strategy params are tuned pre-OOS only.** The ATR-bracket parameters (entry/SL/TP, holds,
   thresholds) are grid-searched on the 2022-01 → 2024-05 validation window and *frozen* before the
   OOS window is touched — identical protocol to the learned agents.
3. **Same execution engine as every other agent.** Backtests use ``mas07.bracket_run`` verbatim, so
   the saved ``best_params`` reproduce bit-for-bit when the coordinator rebuilds the agent.

Artifacts are written in the standard agent format
(``oos_probs.npy`` / ``oos_index.npy`` / ``wfo_probs.npy`` / ``wfo_index.npy`` / ``results.json``)
to ``artifacts/notebooks_v2/05_<name>/`` so each agent is a drop-in for ``mas07``.
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .mas07 import (ANN, OOS_END, OOS_START, bracket_run, maxdd, repo_root,
                    sharpe, sortino)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RULE_AGENTS = ["trend", "meanrev", "volbreak", "sentiment_regime", "dominance_rotation"]
RULE_DIR = {"trend": "05_trend", "meanrev": "05_meanrev", "volbreak": "05_volbreak",
            "sentiment_regime": "05_sentiment_regime", "dominance_rotation": "05_dominance_rotation"}
RULE_PARADIGM = {
    "trend": "rule: trend-following",
    "meanrev": "rule: mean-reversion",
    "volbreak": "rule: volatility breakout",
    "sentiment_regime": "rule: contrarian sentiment (Fear & Greed)",
    "dominance_rotation": "rule: cross-asset dominance rotation",
}

GRID_VAL_START = pd.Timestamp("2022-01-01")
GRID_VAL_END = pd.Timestamp("2024-05-31")   # ends just before OOS_START — frozen thereafter

# Trade-count guards on the validation window (mirror the learned agents): reject degenerate
# (too few) or over-trading (overfit) configurations.
MIN_TRADES, MAX_TRADES = 15, 350

# A-priori bracket grid (NOT tuned on OOS). Symmetric thresholds: long fires at p>0.5+t,
# short at p<0.5-t — the same convention used by the learned binary agents.
TRADING_GRID = {
    "signal_threshold": [0.05, 0.10, 0.15, 0.20],
    "entry_atr_mult": [0.0, 0.3, 0.6],
    "sl_atr_mult": [1.5, 2.5],
    "tp_atr_mult": [2.0, 3.0],
    "min_hold": [4, 12],
    "max_hold": [24, 48],
    "cooldown": [3],
    "min_sl": [0.01],
}
_GRID_KEYS = list(TRADING_GRID)
_GRID_COMBOS = list(itertools.product(*TRADING_GRID.values()))


# ---------------------------------------------------------------------------
# Signals — causal, parameter-free, mapped to a [0, 1] conviction (0.5 = neutral)
# ---------------------------------------------------------------------------

def _vote_mean(frames: list[pd.Series]) -> pd.Series:
    """Mean of boolean votes -> [0, 1]; NaN where any constituent feature is missing."""
    M = pd.concat([f.astype("float64") for f in frames], axis=1)
    return M.mean(axis=1)


def trend_signal(df: pd.DataFrame) -> pd.Series:
    """Bullish-structure vote fraction. 1.0 = every trend filter bullish, 0.0 = all bearish."""
    votes = [
        df["close_vs_sma_200"] > 0,        # price above the long MA
        df["sma50_vs_sma200"] > 0,         # golden-cross structure
        df["supertrend_consensus"] > 0,    # SuperTrend multi-timeframe agreement
        df["macd_hist_12_26"] > 0,         # MACD momentum positive
        df["trend_score"] >= 3,            # composite trend score (0..5) majority-bullish
        df["ma_bull_score"] >= 2,          # MA-ribbon score (0..4) majority-bullish
    ]
    return _vote_mean(votes).rename("trend")


def meanrev_signal(df: pd.DataFrame) -> pd.Series:
    """Fade oscillator extremes: oversold -> long (p>0.5), overbought -> short (p<0.5).

    Neutral (0.5) when no oscillator is at an extreme — the agent only acts on dislocations.
    """
    osc = {"rsi": df["rsi_14"], "stoch": df["stoch_k_14"], "williams": df["williams_r"],
           "mfi": df["mfi_14"], "bb": df["bb_position_20"]}
    lo = {"rsi": 0.30, "stoch": 0.20, "williams": 0.20, "mfi": 0.20, "bb": 0.10}
    hi = {"rsi": 0.70, "stoch": 0.80, "williams": 0.80, "mfi": 0.80, "bb": 0.90}
    long_votes = _vote_mean([osc[k] < lo[k] for k in osc])    # oversold fraction
    short_votes = _vote_mean([osc[k] > hi[k] for k in osc])   # overbought fraction
    return (0.5 + 0.5 * (long_votes - short_votes)).clip(0, 1).rename("meanrev")


def volbreak_signal(df: pd.DataFrame) -> pd.Series:
    """Directional range-break: upside break -> long, downside break -> short.

    Conviction is reinforced when the break follows a volatility squeeze (compression -> expansion).
    """
    up = (df["breakout_up_24h"] > 0).astype("float64")
    dn = (df["breakout_down_24h"] > 0).astype("float64")
    squeeze = df.get("volat_squeeze_on", pd.Series(0.0, index=df.index)).fillna(0.0)
    # squeeze active in the last 6h reinforces the break (full conviction vs. half)
    sq_recent = squeeze.rolling(6, min_periods=1).max().fillna(0.0)
    strength = 0.5 + 0.5 * sq_recent          # 0.5 (no squeeze) .. 1.0 (squeeze-confirmed)
    return (0.5 + 0.5 * strength * (up - dn)).clip(0, 1).rename("volbreak")


def sentiment_regime_signal(df: pd.DataFrame) -> pd.Series:
    """Contrarian Fear & Greed: fade sentiment extremes, confirmed by the 7-day sentiment trend.

    The crowd is most wrong at the extremes. Extreme *fear* (capitulation) is a long opportunity;
    extreme *greed* (euphoria) a short one. A second vote requires the 7-day change to be turning
    *against* the extreme (fear basing out / greed rolling over) so the agent leans into mean-
    reversion of sentiment rather than catching a falling knife. All inputs are the (already-causal)
    crowd-sentiment columns; no BTC price feature enters, so the signal is structurally orthogonal.

    Neutral (0.5) whenever sentiment sits in its normal band — the agent only acts on dislocations.
    """
    fg = df["sent_fear_greed"].astype("float64")          # 0 = extreme fear .. 1 = extreme greed
    ma7 = df["sent_fear_greed_ma7"].astype("float64")      # smoothed level
    chg = df["sent_fear_greed_chg_7d"].astype("float64")   # 7-day change in sentiment
    long_votes = _vote_mean([
        fg < 0.25,            # extreme fear right now
        ma7 < 0.30,           # the week has been fearful (persistent capitulation)
        chg > 0.0,            # ...and sentiment is starting to recover (basing out)
    ])
    short_votes = _vote_mean([
        fg > 0.75,            # extreme greed right now
        ma7 > 0.70,           # the week has been euphoric
        chg < 0.0,            # ...and sentiment is starting to roll over
    ])
    return (0.5 + 0.5 * (long_votes - short_votes)).clip(0, 1).rename("sentiment_regime")


def dominance_rotation_signal(df: pd.DataFrame) -> pd.Series:
    """Cross-asset capital rotation: trade BTC by where crypto capital is flowing.

    Rising BTC dominance with a *weakening* ETH/BTC cross means capital is rotating into BTC (a
    risk-off-within-crypto / BTC-leadership regime) -> long BTC. Falling dominance with a
    *strengthening* ETH/BTC cross is the alt-season rotation out of BTC -> short / neutral BTC.
    Inputs are dominance and ETH/BTC cross-asset momentum only — no BTC price-structure feature —
    so the agent reads a genuinely different information source from every price-feature model.

    Neutral (0.5) when dominance and the cross are flat (no clear rotation).
    """
    # The dominance series is daily and built from daily market-cap snapshots. Lag it by 24h before
    # using it on hourly bars so a same-day close/market-cap value cannot leak into that day.
    dom_chg = df["mkt_btc_dominance_chg_7d"].astype("float64").shift(24)
    ebm24 = df["cross_eth_btc_mom_24h"].astype("float64")        # 24h ETH/BTC momentum
    ebm72 = df["cross_eth_btc_mom_72h"].astype("float64")        # 72h ETH/BTC momentum
    long_votes = _vote_mean([
        dom_chg > 0.005,      # BTC dominance rising
        ebm24 < 0.0,          # ETH losing ground to BTC (short horizon)
        ebm72 < 0.0,          # ...and over the longer horizon (capital favouring BTC)
    ])
    short_votes = _vote_mean([
        dom_chg < -0.005,     # BTC dominance falling
        ebm24 > 0.0,          # ETH outpacing BTC (short horizon)
        ebm72 > 0.0,          # ...and over the longer horizon (alt-season rotation)
    ])
    return (0.5 + 0.5 * (long_votes - short_votes)).clip(0, 1).rename("dominance_rotation")


SIGNALS = {"trend": trend_signal, "meanrev": meanrev_signal, "volbreak": volbreak_signal,
           "sentiment_regime": sentiment_regime_signal,
           "dominance_rotation": dominance_rotation_signal}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_frame() -> pd.DataFrame:
    repo = repo_root()
    df = pd.read_parquet(repo / "data" / "features" / "BTCUSDT_1h_unified.parquet")
    df.index = df.index.tz_localize(None) if df.index.tz else df.index
    return df


def _expand(p: dict) -> dict:
    """Symmetric signal_threshold -> explicit long/short thresholds for bracket_run."""
    t = p["signal_threshold"]
    out = {k: v for k, v in p.items() if k != "signal_threshold"}
    out["long_threshold"] = 0.5 + t
    out["short_threshold"] = 0.5 - t
    return out


# ---------------------------------------------------------------------------
# Grid search (validation window only) + OOS evaluation
# ---------------------------------------------------------------------------

def grid_search(signal: pd.Series, df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Select bracket params by validation-window Sharpe (leak-free, frozen before OOS)."""
    gv = (df.index >= GRID_VAL_START) & (df.index <= GRID_VAL_END)
    sub = df[gv]
    prob = signal[gv].values
    close, high, low = sub["close"].values, sub["high"].values, sub["low"].values
    atr = sub["atr_14_pct"].values

    rows = []
    for vals in _GRID_COMBOS:
        p = dict(zip(_GRID_KEYS, vals))
        if p["max_hold"] < p["min_hold"]:
            continue
        bp = _expand(p)
        eq, pos, _ = bracket_run(prob, close, high, low, atr, with_fees=True, **bp)
        n_trades = int((np.diff(np.sign(pos), prepend=0) != 0).sum())  # position changes ~ trades
        active = np.abs(pos) > 0
        if active.sum() < 24:
            continue
        r = float(eq[-1] - 1)
        rows.append({**bp, "signal_threshold": p["signal_threshold"], "ret": r,
                     "sharpe": sharpe(eq), "maxdd": maxdd(eq), "active_bars": int(active.sum())})
    grid = pd.DataFrame(rows)
    if grid.empty:
        raise RuntimeError("grid produced no valid configurations")
    grid = grid.sort_values("sharpe", ascending=False).reset_index(drop=True)
    INT = {"min_hold", "max_hold", "cooldown"}
    keys = ["long_threshold", "short_threshold", "entry_atr_mult", "sl_atr_mult", "tp_atr_mult",
            "min_hold", "max_hold", "cooldown", "min_sl"]
    best = {k: (int(grid.iloc[0][k]) if k in INT else float(grid.iloc[0][k])) for k in keys}
    return best, grid


def _bt_metrics(eq: np.ndarray, pos: np.ndarray) -> dict:
    flips = np.diff(np.sign(pos), prepend=0) != 0
    nl = int(((np.sign(pos) > 0) & flips).sum())
    ns = int(((np.sign(pos) < 0) & flips).sum())
    return {"n_trades": nl + ns, "n_long": nl, "n_short": ns,
            "total_ret": round(float(eq[-1] - 1), 4), "sharpe": round(sharpe(eq), 4),
            "maxdd": round(maxdd(eq), 4)}


def build_agent(name: str, df: pd.DataFrame, save: bool = True, verbose: bool = True) -> dict:
    signal = SIGNALS[name](df).reindex(df.index)
    best, grid = grid_search(signal, df)

    # Full-history (WFO-equivalent: the rule signal is fixed, only params were tuned pre-OOS)
    eq_full, pos_full, _ = bracket_run(
        signal.values, df["close"].values, df["high"].values, df["low"].values,
        df["atr_14_pct"].values, with_fees=True, **best)

    oos = (df.index >= OOS_START) & (df.index <= OOS_END)
    oos_idx = df.index[oos]
    sub = df[oos]
    eq_oos, pos_oos, _ = bracket_run(
        signal[oos].values, sub["close"].values, sub["high"].values, sub["low"].values,
        sub["atr_14_pct"].values, with_fees=True, **best)
    eq_oos0, _, _ = bracket_run(
        signal[oos].values, sub["close"].values, sub["high"].values, sub["low"].values,
        sub["atr_14_pct"].values, with_fees=False, **best)

    results = {
        "notebook": "05_rule_agents_v1", "agent": name, "paradigm": RULE_PARADIGM[name],
        "created": pd.Timestamp.now().isoformat(),
        "model": f"rule-based ({RULE_PARADIGM[name]}), parameter-free causal signal",
        "grid_val": f"{GRID_VAL_START.date()}→{GRID_VAL_END.date()}",
        "oos_period": f"{OOS_START.date()}→{oos_idx[-1].date()}",
        "best_params": best,
        "backtest_wfees": _bt_metrics(eq_oos, pos_oos),
        "backtest_0fee": _bt_metrics(eq_oos0, pos_oos),
        "artifacts": {"oos_probs": "oos_probs.npy", "oos_index": "oos_index.npy (int64 ns)",
                      "wfo_probs": "wfo_probs.npy", "wfo_index": "wfo_index.npy (int64 ns)"},
    }

    if save:
        arts = repo_root() / "artifacts" / "notebooks_v2" / RULE_DIR[name]
        arts.mkdir(parents=True, exist_ok=True)
        np.save(arts / "oos_probs.npy", signal[oos].values.astype(np.float32))
        np.save(arts / "oos_index.npy", oos_idx.astype("datetime64[ns]").astype(np.int64).values)
        np.save(arts / "wfo_probs.npy", signal.values.astype(np.float32))
        np.save(arts / "wfo_index.npy", df.index.astype("datetime64[ns]").astype(np.int64).values)
        json.dump(results, open(arts / "results.json", "w"), indent=2, default=float)
        grid.head(50).to_csv(arts / "grid_leaderboard.csv", index=False)

    if verbose:
        bt = results["backtest_wfees"]
        print(f"[{name:8}] OOS ret={bt['total_ret']:+.1%}  sharpe={bt['sharpe']:.2f}  "
              f"maxdd={bt['maxdd']:.1%}  trades={bt['n_trades']} (L{bt['n_long']}/S{bt['n_short']})  "
              f"| best t={best['long_threshold']-0.5:.2f} sl={best['sl_atr_mult']} tp={best['tp_atr_mult']}")

    results["_eq_oos"] = pd.Series(eq_oos, index=oos_idx)
    results["_signal"] = signal
    return results


def run_pipeline(save: bool = True, verbose: bool = True) -> dict:
    df = load_frame()
    if verbose:
        print(f"Data: {df.shape[0]:,} bars  {df.index.min().date()} → {df.index.max().date()}")
        print(f"Grid: {len(_GRID_COMBOS)} combos/agent  |  val {GRID_VAL_START.date()}→"
              f"{GRID_VAL_END.date()}  |  OOS {OOS_START.date()}→{OOS_END.date()}\n")
    t0 = time.time()
    out = {a: build_agent(a, df, save=save, verbose=verbose) for a in RULE_AGENTS}
    if verbose:
        print(f"\nDone in {time.time()-t0:.0f}s → artifacts/notebooks_v2/05_*")
    out["_df"] = df
    return out


if __name__ == "__main__":
    run_pipeline()
