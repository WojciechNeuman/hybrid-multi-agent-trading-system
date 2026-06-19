"""Builds src/hmats/notebooks_v2/05_rule_agents_v1.ipynb — a SELF-CONTAINED notebook (engine +
rule-agent logic inlined) that runs top-to-bottom in Jupyter with no local imports.
Run:  python _build_rule_nb.py
"""
import json
from pathlib import Path

from _nbinline import agent_module_source, cell, mas07_engine_source

cells = []


def md(s):
    cells.append(cell("markdown", s))


def code(s):
    cells.append(cell("code", s))


md(r"""# 08 — Rule-based strategy agents (self-contained)

Economically-motivated, **parameter-free, causal** strategy agents whose edge is *strategy logic*,
not a fit to the feature panel. The first three read the BTC price/feature panel; the last two read
a **different information source** (crowd sentiment, cross-asset capital rotation) and are therefore
structurally orthogonal to every price-feature model:

| Agent | Logic | Designed for |
|-------|-------|--------------|
| `trend`    | trend-following (MA / SuperTrend / MACD vote) | directional bull/bear |
| `meanrev`  | fade oscillator extremes (RSI / Stoch / Williams / MFI / Bollinger) | chop / range |
| `volbreak` | direction of a 24h range break (squeeze-confirmed) | expansion |
| `sentiment_regime`   | contrarian Fear & Greed (fade sentiment extremes) | capitulation / euphoria |
| `dominance_rotation` | cross-asset rotation (BTC dominance + ETH/BTC momentum) | BTC-leadership vs alt-season |

**Leak-free:** signals are vote-fractions over already-causal features with fixed economic
thresholds (nothing fitted → nothing leaks); ATR-bracket params are grid-searched on
2022-01→2024-05 only, then frozen for OOS; backtests use the same `bracket_run` engine as every
other agent. Artifacts → `artifacts/notebooks_v2/08_<agent>/` (drop-in for the coordinator,
notebook 06).

**Inclusion criteria.** An agent joins the final multi-agent system only if its OOS return is positive
*and* its max drawdown is no worse than BTC buy-and-hold. The random-bracket null determines only
how a passing agent is *described* — agents that fail to clear the 95th percentile are included as
diversification agents rather than as alpha sources:

| Agent | OOS ret | OOS maxdd | vs B&H maxdd (−50.1%) | random-bracket null percentile | status |
|---|---|---|---|---|---|
| `sentiment_regime`   | −29.9% | −42.8% | comparable | 7th (below random) | excluded — negative OOS return |
| `dominance_rotation` | +152.5% | −26.4% | better | 93–95th | accepted — diversification agent, not claimed as alpha |

This notebook is **self-contained**: the execution engine and the agent logic are inlined below.""")

md(r"""## 1 · Engine (ATR-bracket backtester, metrics, regime, coordinator) — inlined""")
code(mas07_engine_source())

md(r"""## 2 · Rule-agent signals, grid search & artifact export — inlined""")
code(agent_module_source("rule_agents.py"))

md(r"""## 3 · Build the agents (grid-search pre-OOS → freeze → evaluate OOS → save artifacts)""")
code("out = run_pipeline(save=True, verbose=True)\ndf = out['_df']")

md(r"""## 4 · OOS leaderboard""")
code(r"""import pandas as pd
rows = []
for a in RULE_AGENTS:
    bt = out[a]["backtest_wfees"]; bp = out[a]["best_params"]
    rows.append({"agent": a, "paradigm": RULE_PARADIGM[a], "ret": bt["total_ret"],
                 "sharpe": bt["sharpe"], "maxdd": bt["maxdd"], "n_trades": bt["n_trades"],
                 "n_long": bt["n_long"], "n_short": bt["n_short"]})
pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)""")

md(r"""## 4b · Random-bracket null — honest skill check

A signal's headline OOS return is dominated by two things that are **not** skill: the asymmetric ATR
bracket (cut losers / let winners run = positive convexity) and the trending OOS market. The fair
benchmark is therefore *not* buy-and-hold but **random entries through each agent's own bracket**,
trade-count-matched. An agent is only "alpha" if its real return clears the 95th percentile of that
random distribution. Agents that pass the acceptance gate but **not** this test are kept as
*diversification* agents only — their value is decorrelation, not predictive skill.""")
code(r"""def random_bracket_null(sig, df, bp, n_sims=400, seed=0):
    oos = (df.index >= OOS_START) & (df.index <= OOS_END); sub = df[oos]
    c, hi, lo = sub["close"].values, sub["high"].values, sub["low"].values
    atr = sub["atr_14_pct"].values; n = len(sub)
    eq, pos, _ = bracket_run(sig[oos].values, c, hi, lo, atr, with_fees=True, **bp)
    real = float(eq[-1] - 1)
    n_tr = int((np.diff(np.sign(pos), prepend=0) != 0).sum())
    lt, st = bp["long_threshold"], bp["short_threshold"]
    rng = np.random.default_rng(seed); rets = np.empty(n_sims)
    for k in range(n_sims):
        s = np.full(n, 0.5)
        locs = rng.choice(n, size=min(max(n_tr, 2), n), replace=False)
        s[locs] = rng.choice([lt + 0.05, st - 0.05], size=len(locs))
        e, _, _ = bracket_run(s, c, hi, lo, atr, with_fees=True, **bp)
        rets[k] = e[-1] - 1
    pct = float((rets < real).mean())
    return {"real_ret": real, "null_p50": float(np.percentile(rets, 50)),
            "null_p95": float(np.percentile(rets, 95)), "pctile": pct,
            "alpha_vs_null": real - float(np.percentile(rets, 50)),
            "skill_sig (>=95%)": "yes" if pct >= 0.95 else "no"}

rows = []
for a in RULE_AGENTS:
    r = random_bracket_null(SIGNALS[a](df).reindex(df.index), df, out[a]["best_params"])
    rows.append({"agent": a, **r})
nullt = pd.DataFrame(rows)
for c in ["real_ret", "null_p50", "null_p95", "alpha_vs_null"]:
    nullt[c] = (nullt[c] * 100).round(1)
nullt["pctile"] = (nullt["pctile"] * 100).round(0)
nullt""")

md(r"""## 5 · Leakage robustness — extra execution delay

A genuine signal survives an extra bar of delay; a look-ahead leak collapses immediately.""")
code(r"""oos = (df.index >= OOS_START) & (df.index <= OOS_END); sub = df[oos]
rows = []
for a in RULE_AGENTS:
    sig = SIGNALS[a](df); bp = out[a]["best_params"]
    for lag, tag in [(0, "base (i+1)"), (1, "+1 lag"), (2, "+2 lag")]:
        s = sig.shift(lag)[oos].values
        eq, _, _ = bracket_run(s, sub["close"].values, sub["high"].values, sub["low"].values,
                               sub["atr_14_pct"].values, with_fees=True, **bp)
        rows.append({"agent": a, "delay": tag, "ret": eq[-1] - 1, "sharpe": sharpe(eq)})
pd.DataFrame(rows).pivot(index="agent", columns="delay", values=["ret", "sharpe"]).round(3)""")

md(r"""## 6 · Equity curves & per-regime breakdown (OOS)""")
code(r"""import matplotlib.pyplot as plt, matplotlib.dates as mdates
colours = {"trend": "#43A047", "meanrev": "#FB8C00", "volbreak": "#5E35B1",
           "sentiment_regime": "#00897B", "dominance_rotation": "#D81B60"}
fig, ax = plt.subplots(figsize=(13, 5))
for a in RULE_AGENTS:
    eq = out[a]["_eq_oos"]
    ax.plot(eq.index, (eq.values - 1) * 100, lw=1.6, color=colours[a],
            label=f"{a} ({eq.values[-1]-1:+.0%}, Sharpe {sharpe(eq.values):.2f})")
bh = (1 + df["close"].pct_change().fillna(0))[oos].cumprod()
ax.plot(bh.index, (bh.values / bh.values[0] - 1) * 100, lw=1.1, ls=":", color="#9E9E9E",
        label=f"BTC B&H ({bh.values[-1]/bh.values[0]-1:+.0%})")
for r, c in [("chop", "#9E9E9E"), ("bull", "#26A69A"), ("bear", "#EF5350")]:
    s, e = REGIME_DATES[r]; ax.axvspan(s, min(e, bh.index[-1]), alpha=0.06, color=c)
ax.axhline(0, color="#9E9E9E", lw=0.6, ls=":"); ax.set_ylabel("Return (%)"); ax.legend(fontsize=8)
ax.set_title("Rule agents — OOS equity (shaded: chop / bull / bear)", fontweight="bold")
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
fig.tight_layout(); plt.show()

reg = []
for a in RULE_AGENTS:
    eq = out[a]["_eq_oos"]
    for r in REGIMES:
        s, e = REGIME_DATES[r]; mseg = (eq.index >= s) & (eq.index <= e)
        if mseg.sum() < 24: continue
        seg = eq[mseg].values / eq[mseg].values[0]
        reg.append({"agent": a, "regime": r, "ret": f"{seg[-1]-1:+.1%}"})
pd.DataFrame(reg).pivot(index="agent", columns="regime", values="ret")""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
dst = Path(__file__).resolve().parent / "05_rule_agents_v1.ipynb"
json.dump(nb, open(dst, "w"), indent=1)
print("wrote", dst, "| cells:", len(cells))
