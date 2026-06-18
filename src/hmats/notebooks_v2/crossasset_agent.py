"""Cross-asset / sentiment agent (notebook 09) — the structurally-orthogonal learned agent.

The four learned agents and the three rule agents all read the BTC price/feature panel. This agent
deliberately trades on a *different information source*: cross-asset relationships (ETH/BTC, BTC
dominance), market sentiment (Fear & Greed), and order-flow microstructure. Its edge — if any — is
uncorrelated with the price-feature models, which is exactly what a fund-of-agents needs.

Method (leak-free, identical protocol to the learned agents):
* **Features**: only the orthogonal columns. We *exclude* `mkt_total_mcap_chg_24h` (known forward
  leak), `cross_altcoin_breadth_24h` (0.56 corr with the realised 24h move — redundant with
  momentum), `mkt_stablecoin_pct` (11% coverage), and raw price levels.
* **Label**: triple-barrier (2σ / 24h), up=1 / down=0, timeouts dropped — same as the deep agents.
* **Model**: LightGBM, expanding-window walk-forward, retrained quarterly, with a 48h embargo that
  covers the 24h label horizon so no training row sees its own test window.
* **Execution**: the agent's ATR-bracket params are grid-searched on 2022-01→2024-05 only, frozen
  for OOS, and run through `mas07.bracket_run`.

Artifacts -> `artifacts/notebooks_v2/09_crossasset/` in the standard format (drop-in for `mas07`).
"""
from __future__ import annotations

import itertools
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from mas07 import OOS_END, OOS_START, bracket_run, maxdd, repo_root, sharpe

SEED = 42

# Orthogonal information only (no BTC price-feature panel). Leakers/redundant cols excluded.
FEATURES = [
    "cross_eth_btc_ratio", "cross_eth_btc_mom_24h", "cross_eth_btc_mom_72h",
    "cross_btc_relative_strength", "cross_alt_correlation_24h", "cross_recency",
    "mkt_btc_dominance", "mkt_btc_dominance_chg_7d", "mkt_eth_dominance",
    "sent_fear_greed", "sent_fear_greed_ma7", "sent_fear_greed_chg_7d",
    "micro_amihud_illiq", "micro_kyle_lambda", "micro_roll_spread", "micro_volume_clock",
    "taker_price_premium", "avg_trade_size_z24",
]

TRAIN_START = pd.Timestamp("2019-01-01")
GRID_VAL_START = pd.Timestamp("2022-01-01")
GRID_VAL_END = pd.Timestamp("2024-05-30")
RETRAIN_MONTHS = 3
EMBARGO_H = 48          # > 24h TBM label horizon → no train row sees its test window
TBM_VOL_WINDOW, TBM_MULT, TBM_VERT_H = 24, 2.0, 24

LGB_PARAMS = dict(objective="binary", n_estimators=400, learning_rate=0.03, num_leaves=31,
                  max_depth=-1, min_child_samples=80, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbose=-1)

TRADING_GRID = {
    "signal_threshold": [0.04, 0.06, 0.08, 0.10],
    "entry_atr_mult": [0.0, 0.3], "sl_atr_mult": [1.5, 2.5], "tp_atr_mult": [2.0, 3.0],
    "min_hold": [4, 12], "max_hold": [24, 48], "cooldown": [3], "min_sl": [0.01],
}
_GK = list(TRADING_GRID)
_GC = list(itertools.product(*TRADING_GRID.values()))


def tbm_labels(df: pd.DataFrame) -> np.ndarray:
    lr = np.log(df["close"]).diff()
    vol = lr.rolling(TBM_VOL_WINDOW).std().values
    c = df["close"].values
    n = len(df)
    y = np.full(n, np.nan, np.float32)
    for i in range(n):
        if np.isnan(vol[i]) or vol[i] == 0:
            continue
        s = vol[i] * c[i]
        up, dn = c[i] + TBM_MULT * s, c[i] - TBM_MULT * s
        for j in range(i + 1, min(i + TBM_VERT_H, n)):
            if c[j] >= up:
                y[i] = 1.0
                break
            if c[j] <= dn:
                y[i] = 0.0
                break
    return y


def load_frame() -> pd.DataFrame:
    df = pd.read_parquet(repo_root() / "data" / "features" / "BTCUSDT_1h_unified.parquet")
    df.index = df.index.tz_localize(None) if df.index.tz else df.index
    return df


def walk_forward(df: pd.DataFrame, y: np.ndarray, verbose=True) -> np.ndarray:
    """Expanding-window quarterly-retrained LightGBM. Returns dense P(up) over all bars."""
    X = df[FEATURES].astype(np.float32)   # keep column names so fit/predict are consistent
    probs = np.full(len(df), np.nan)
    anchor = GRID_VAL_START
    anchors = []
    while anchor <= df.index[-1]:
        anchors.append(anchor)
        anchor += pd.DateOffset(months=RETRAIN_MONTHS)
    t0 = time.time()
    for a in anchors:
        a_end = a + pd.DateOffset(months=RETRAIN_MONTHS)
        cut = a - pd.Timedelta(hours=EMBARGO_H)
        tr = (df.index >= TRAIN_START) & (df.index < cut) & ~np.isnan(y)
        if tr.sum() < 2000:
            continue
        model = lgb.LGBMClassifier(**LGB_PARAMS)
        model.fit(X[tr], y[tr].astype(int))
        te = (df.index >= a) & (df.index < a_end)
        if te.sum():
            probs[np.where(te)[0]] = model.predict_proba(X[te])[:, 1]  # X[te] keeps names → no warning
    if verbose:
        oos = (df.index >= OOS_START) & ~np.isnan(probs) & ~np.isnan(y)
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y[oos].astype(int), probs[oos])
        print(f"WFO LGBM done in {time.time()-t0:.0f}s | OOS AUC={auc:.4f} | "
              f"{len(anchors)} folds, {len(FEATURES)} orthogonal features")
    return probs


def _expand(p):
    t = p["signal_threshold"]
    o = {k: v for k, v in p.items() if k != "signal_threshold"}
    o["long_threshold"], o["short_threshold"] = 0.5 + t, 0.5 - t
    return o


def grid_search(prob: np.ndarray, df: pd.DataFrame):
    gv = (df.index >= GRID_VAL_START) & (df.index <= GRID_VAL_END)
    sub = df[gv]
    p = prob[gv]
    c, hi, lo, atr = sub["close"].values, sub["high"].values, sub["low"].values, sub["atr_14_pct"].values
    rows = []
    for vals in _GC:
        d = dict(zip(_GK, vals))
        if d["max_hold"] < d["min_hold"]:
            continue
        bp = _expand(d)
        eq, pos, _ = bracket_run(p, c, hi, lo, atr, with_fees=True, **bp)
        if (np.abs(pos) > 0).sum() < 24:
            continue
        rows.append({**bp, "sharpe": sharpe(eq), "ret": float(eq[-1] - 1)})
    g = pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)
    INT = {"min_hold", "max_hold", "cooldown"}
    keys = ["long_threshold", "short_threshold", "entry_atr_mult", "sl_atr_mult", "tp_atr_mult",
            "min_hold", "max_hold", "cooldown", "min_sl"]
    return {k: (int(g.iloc[0][k]) if k in INT else float(g.iloc[0][k])) for k in keys}, g


def run_pipeline(save=True, verbose=True) -> dict:
    np.random.seed(SEED)
    df = load_frame()
    y = tbm_labels(df)
    if verbose:
        print(f"Cross-asset agent | {df.shape[0]:,} bars | label up={int((y==1).sum()):,} "
              f"down={int((y==0).sum()):,} timeout={int(np.isnan(y).sum()):,}")
    probs = walk_forward(df, y, verbose=verbose)
    best, grid = grid_search(probs, df)

    oos = (df.index >= OOS_START) & (df.index <= OOS_END)
    oos_idx = df.index[oos]
    sub = df[oos]
    p_oos = pd.Series(probs, index=df.index)[oos].values
    eqf, posf, _ = bracket_run(p_oos, sub["close"].values, sub["high"].values, sub["low"].values,
                               sub["atr_14_pct"].values, with_fees=True, **best)
    eq0, _, _ = bracket_run(p_oos, sub["close"].values, sub["high"].values, sub["low"].values,
                            sub["atr_14_pct"].values, with_fees=False, **best)

    def flips(pos):
        f = np.diff(np.sign(pos), prepend=0) != 0
        return int(((np.sign(pos) > 0) & f).sum()), int(((np.sign(pos) < 0) & f).sum())
    nl, ns = flips(posf)
    results = {
        "notebook": "09_crossasset_v1", "agent": "crossasset",
        "paradigm": "learned: cross-asset / sentiment / flow", "created": pd.Timestamp.now().isoformat(),
        "model": "LightGBM expanding-window WFO on orthogonal features",
        "features": FEATURES, "seed": SEED,
        "oos_period": f"{OOS_START.date()}→{oos_idx[-1].date()}", "best_params": best,
        "backtest_wfees": {"n_trades": nl + ns, "n_long": nl, "n_short": ns,
                           "total_ret": round(float(eqf[-1] - 1), 4), "sharpe": round(sharpe(eqf), 4),
                           "maxdd": round(maxdd(eqf), 4)},
        "backtest_0fee": {"total_ret": round(float(eq0[-1] - 1), 4), "sharpe": round(sharpe(eq0), 4)},
    }
    if save:
        arts = repo_root() / "artifacts" / "notebooks_v2" / "09_crossasset"
        arts.mkdir(parents=True, exist_ok=True)
        np.save(arts / "oos_probs.npy", p_oos.astype(np.float32))
        np.save(arts / "oos_index.npy", oos_idx.astype("datetime64[ns]").astype(np.int64).values)
        np.save(arts / "wfo_probs.npy", probs.astype(np.float32))
        np.save(arts / "wfo_index.npy", df.index.astype("datetime64[ns]").astype(np.int64).values)
        json.dump(results, open(arts / "results.json", "w"), indent=2, default=float)
        grid.head(50).to_csv(arts / "grid_leaderboard.csv", index=False)
    if verbose:
        bt = results["backtest_wfees"]
        print(f"[crossasset] OOS ret={bt['total_ret']:+.1%} sharpe={bt['sharpe']:.2f} "
              f"maxdd={bt['maxdd']:.1%} trades={bt['n_trades']} (L{bt['n_long']}/S{bt['n_short']})")
    results["_probs"] = pd.Series(probs, index=df.index)
    results["_df"] = df
    return results


if __name__ == "__main__":
    run_pipeline()
