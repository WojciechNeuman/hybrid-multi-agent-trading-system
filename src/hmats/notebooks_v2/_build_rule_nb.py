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

Three economically-motivated, **parameter-free, causal** strategy agents whose edge is *strategy
logic*, not a fit to the feature panel:

| Agent | Logic | Designed for |
|-------|-------|--------------|
| `trend`    | trend-following (MA / SuperTrend / MACD vote) | directional bull/bear |
| `meanrev`  | fade oscillator extremes (RSI / Stoch / Williams / MFI / Bollinger) | chop / range |
| `volbreak` | direction of a 24h range break (squeeze-confirmed) | expansion |

**Leak-free:** signals are vote-fractions over already-causal features with fixed economic
thresholds (nothing fitted → nothing leaks); ATR-bracket params are grid-searched on
2022-01→2024-05 only, then frozen for OOS; backtests use the same `bracket_run` engine as every
other agent. Artifacts → `artifacts/notebooks_v2/08_{trend,meanrev,volbreak}/` (drop-in for the
coordinator, notebook 07).

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
colours = {"trend": "#43A047", "meanrev": "#FB8C00", "volbreak": "#5E35B1"}
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
