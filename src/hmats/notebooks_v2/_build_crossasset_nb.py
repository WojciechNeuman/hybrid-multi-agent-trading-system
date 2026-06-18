"""Builds src/hmats/notebooks_v2/06_crossasset_v1.ipynb — a SELF-CONTAINED notebook (engine +
cross-asset LGBM agent inlined) that runs top-to-bottom in Jupyter. Run: python _build_crossasset_nb.py
"""
import json
from pathlib import Path

from _nbinline import agent_module_source, cell, mas07_engine_source

cells = []


def md(s):
    cells.append(cell("markdown", s))


def code(s):
    cells.append(cell("code", s))


md(r"""# 09 — Cross-asset / sentiment agent (self-contained)

The structurally-**orthogonal** learned agent: it trades a *different information source* than the
price-feature models — cross-asset relationships (ETH/BTC, BTC dominance), market sentiment
(Fear & Greed) and order-flow microstructure.

**Leak discipline.** Features exclude the known forward leak (`mkt_total_mcap_chg_24h`), the
momentum-redundant `cross_altcoin_breadth_24h` (0.56 corr with the realised 24h move), the sparse
`mkt_stablecoin_pct`, and raw price levels. Label = triple-barrier (2σ/24h). Model = LightGBM,
expanding-window walk-forward, retrained quarterly with a 48h embargo covering the label horizon.
Bracket params grid-searched on 2022-01→2024-05 only, frozen for OOS.

> **Honest expectation:** orthogonal info is a *diversification* hypothesis, not a guaranteed edge —
> the OOS AUC and the random-bracket null (notebook 07) tell you whether it carries real skill.

Self-contained: engine + agent inlined below. Artifacts → `artifacts/notebooks_v2/09_crossasset/`.""")

md(r"""## 1 · Engine — inlined""")
code(mas07_engine_source())

md(r"""## 2 · Cross-asset LGBM agent (TBM labels, WFO, grid search, export) — inlined""")
code(agent_module_source("crossasset_agent.py"))

md(r"""## 3 · Train walk-forward, evaluate OOS, save artifacts""")
code("out = run_pipeline(save=True, verbose=True)\ndf = out['_df']")

md(r"""## 4 · Result & diagnostics

`OOS AUC` near 0.5 means the orthogonal features do not predict BTC direction — in which case the
positive return is bracket-convexity + market beta, not signal skill (quantified by the
random-bracket null in notebook 07).""")
code(r"""import pandas as pd
r = out
bt = r["backtest_wfees"]
display(pd.Series({"OOS_return": f'{bt["total_ret"]:+.1%}', "sharpe": bt["sharpe"],
                   "maxdd": f'{bt["maxdd"]:.1%}', "n_trades": bt["n_trades"],
                   "n_long": bt["n_long"], "n_short": bt["n_short"],
                   "best_params": r["best_params"]}))
print("Features used (orthogonal only):"); print(", ".join(FEATURES))""")

md(r"""## 5 · OOS equity""")
code(r"""import matplotlib.pyplot as plt, matplotlib.dates as mdates
oos = (df.index >= OOS_START) & (df.index <= OOS_END); sub = df[oos]
p = out["_probs"][oos].values
eq, pos, _ = bracket_run(p, sub["close"].values, sub["high"].values, sub["low"].values,
                         sub["atr_14_pct"].values, with_fees=True, **out["best_params"])
bh = (sub["close"].values / sub["close"].values[0] - 1) * 100
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(sub.index, (eq - 1) * 100, lw=1.7, color="#00695C",
        label=f"crossasset ({eq[-1]-1:+.0%}, Sharpe {sharpe(eq):.2f})")
ax.plot(sub.index, bh, lw=1.1, ls=":", color="#9E9E9E", label=f"BTC B&H ({bh[-1]:+.0f}%)")
for rg, c in [("chop", "#9E9E9E"), ("bull", "#26A69A"), ("bear", "#EF5350")]:
    s, e = REGIME_DATES[rg]; ax.axvspan(s, min(e, sub.index[-1]), alpha=0.06, color=c)
ax.axhline(0, color="#9E9E9E", lw=0.6, ls=":"); ax.set_ylabel("Return (%)"); ax.legend(fontsize=9)
ax.set_title("Cross-asset / sentiment agent — OOS", fontweight="bold")
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
fig.tight_layout(); plt.show()""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
dst = Path(__file__).resolve().parent / "06_crossasset_v1.ipynb"
json.dump(nb, open(dst, "w"), indent=1)
print("wrote", dst, "| cells:", len(cells))
