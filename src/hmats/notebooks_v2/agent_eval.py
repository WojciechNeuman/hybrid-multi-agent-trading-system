"""Honest agent evaluation utilities — the random-bracket null benchmark.

We discovered that an agent's headline OOS return is dominated by two things that are *not* signal
skill: (1) the asymmetric ATR bracket (cut losers fast / let winners run = positive convexity) and
(2) the trending OOS market (directional beta). A coin-flip entry through the *same* bracket on the
*same* window already makes a large return.

So the fair benchmark for a signal is not buy-&-hold — it is **"random entries through this agent's
own bracket."** The agent's *skill* is the percentile of its real return within that random
distribution (and the gap of its return over the random median). This module computes that.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mas07 import OOS_END, OOS_START, bracket_run, maxdd, sharpe


def _n_trades(pos: np.ndarray) -> int:
    return int((np.diff(np.sign(pos), prepend=0) != 0).sum())


def random_bracket_null(prob: pd.Series, df: pd.DataFrame, best_params: dict,
                        n_sims: int = 300, seed: int = 0,
                        prob_dn: pd.Series | None = None) -> dict:
    """Distribution of OOS returns from *random* long/short entries through the agent's bracket,
    trade-count-matched to the real signal. Returns the real result and its percentile vs random.
    """
    oos = (df.index >= OOS_START) & (df.index <= OOS_END)
    sub = df[oos]
    c, hi, lo = sub["close"].values, sub["high"].values, sub["low"].values
    atr = sub["atr_14_pct"].values
    n = len(sub)

    pdn = prob_dn[oos].values if prob_dn is not None else None
    eq, pos, _ = bracket_run(prob[oos].values, c, hi, lo, atr, with_fees=True,
                             prob_dn=pdn, **best_params)
    real_ret = float(eq[-1] - 1)
    n_tr = _n_trades(pos)

    lt, st = best_params["long_threshold"], best_params["short_threshold"]
    rng = np.random.default_rng(seed)
    rets = np.empty(n_sims)
    for k in range(n_sims):
        s = np.full(n, 0.5)
        # ~2x candidate entries; cooldown/min_hold thin them toward the real trade count
        locs = rng.choice(n, size=min(int(max(n_tr, 2) * 1.0), n), replace=False)
        s[locs] = rng.choice([lt + 0.05, st - 0.05], size=len(locs))
        e, _, _ = bracket_run(s, c, hi, lo, atr, with_fees=True, **best_params)
        rets[k] = e[-1] - 1
    pctile = float((rets < real_ret).mean())
    return {
        "real_ret": real_ret, "real_sharpe": float(sharpe(eq)), "real_maxdd": float(maxdd(eq)),
        "n_trades": n_tr,
        "null_mean": float(rets.mean()), "null_p50": float(np.percentile(rets, 50)),
        "null_p95": float(np.percentile(rets, 95)),
        "percentile": pctile,                 # fraction of random sims the agent beats
        "alpha_vs_null": real_ret - float(np.percentile(rets, 50)),  # return over random median
        "skill_significant": pctile >= 0.95,  # one-sided 5% test
    }


def null_table(agents: dict, df: pd.DataFrame, best_params: dict[str, dict],
               prob_dn: dict | None = None, n_sims: int = 300) -> pd.DataFrame:
    """Random-bracket-null skill table for a set of agents.

    ``agents`` maps name -> prob Series; ``best_params`` maps name -> bracket params.
    """
    rows = []
    for name, prob in agents.items():
        r = random_bracket_null(prob, df, best_params[name], n_sims=n_sims,
                                prob_dn=(prob_dn or {}).get(name))
        rows.append({"agent": name, "ret": f'{r["real_ret"]:+.1%}',
                     "null_p50": f'{r["null_p50"]:+.1%}', "null_p95": f'{r["null_p95"]:+.1%}',
                     "alpha_vs_null": f'{r["alpha_vs_null"]:+.1%}',
                     "pctile": f'{r["percentile"]:.0%}',
                     "skill_sig": "yes" if r["skill_significant"] else "no"})
    return pd.DataFrame(rows)
