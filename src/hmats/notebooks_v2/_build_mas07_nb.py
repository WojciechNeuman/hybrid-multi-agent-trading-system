"""Builds the final self-contained multi-agent notebook.

Output:
    src/hmats/notebooks_v2/06_multi_agent_v1.ipynb

The notebook is generated from the maintained source modules in ``src/hmats/mas`` so the thesis
artifact and runtime code cannot drift.
"""
import json
from pathlib import Path

from _nbinline import cell, helper_source, mas07_sections

HERE = Path(__file__).resolve().parent
cells = []


def md(s):
    cells.append(cell("markdown", s))


def code(s):
    cells.append(cell("code", s))


imports_chunk, sections = mas07_sections()

md(r"""# 06 — Hybrid Multi-Agent Trading System

This notebook is the final integration point for the thesis pipeline. It converts heterogeneous
model outputs into autonomous, risk-managed trading agents, compares several capital-allocation
methods, and saves publication-ready figures for the thesis.

The final agent set is intentionally conservative:

| Source notebook | Agent | Method family | Final status |
|---|---|---|---|
| `01_lgbm_v1` | `lgbm` | gradient boosting on tabular features | accepted |
| `02_mamba_v1` | `mamba` | selective state-space sequence model | accepted |
| `03_tcn_v1` | `tcn` | temporal convolutional network, TBM two-channel signal | accepted |
| `04_patchtst_v1` | `patch` | patch transformer, TBM two-channel signal | accepted |
| `05_rule_agents_v1` | `trend` | rule-based trend following | accepted |
| `05_rule_agents_v1` | `volbreak` | rule-based volatility breakout | accepted |
| `05_rule_agents_v1` | `dominance_rotation` | rule-based cross-asset dominance rotation | accepted as a **diversification** agent (OOS-profitable, drawdown < B&H, but straddles the 95% random-bracket null → not claimed as alpha) |
| `05_rule_agents_v1` | `meanrev` | rule-based mean reversion | excluded: negative OOS return |
| `05_rule_agents_v1` | `sentiment_regime` | rule-based contrarian Fear & Greed | excluded: negative OOS return (−29.9%), below random-bracket null |
| removed `06_crossasset_v1` | `crossasset` | cross-asset / sentiment learner | excluded: weak predictive skill and not significant vs random-bracket null |

The `dominance_rotation` rule agent reads BTC dominance and ETH/BTC cross-asset momentum rather than
the BTC price/feature panel. Its return stream is structurally decorrelated from every price-feature
model, and it is included for that diversification property — not as an alpha source, since its OOS
return sits at the 95% boundary of its own random-bracket null.

The key design decision is that the original
`softmax(trailing Sharpe) × regime competence` coordinator is treated as an **ablation**, not the
headline result. The final reported system is a leak-free capped inverse-volatility fund of
autonomous agents. This is still a multi-agent architecture in the defensible sense used in the thesis: agents
retain independent models, signals, positions, confidence paths, and realised return streams; the
system layer allocates capital across those agents after comparing multiple coordination rules.

**Leak discipline.** A weight decided with information up to bar *t* earns each agent's return over
*t → t+1*. Every trailing statistic is shifted by at least one bar plus the 48-hour embargo.
Per-regime competence priors are estimated on pre-OOS data only.

---
## Part A — Engine Inlined From `hmats.mas.mas07`""")

code("# Dependencies for the inlined engine\n" + imports_chunk)

for title, body in sections:
    md(f"### {title}")
    code(body)

md(r"""---
## Part B — Run The Final Multi-Agent System

The cell below verifies that all accepted agents have saved OOS probability artifacts, then runs the
full integration pipeline and writes results to `artifacts/notebooks_v2/06_mas/`.""")

code(r"""missing = [d for a, d in AGENT_DIR.items()
           if not (repo_root() / "artifacts" / "notebooks_v2" / d / "oos_probs.npy").exists()]
assert not missing, "Missing accepted-agent artifacts: " + ", ".join(missing)
print("Accepted agents:", AGENTS)
print("Excluded experiments:", EXCLUDED_AGENTS)""")

code("out = run_pipeline(save=True, verbose=True)")

md(r"""### Per-Regime Competence Priors

The table is estimated on the pre-OOS competence window. It is retained for interpretability and
for the coordinator ablation. The final reported allocator uses capped inverse-volatility weights
to avoid concentration in any single method.""")

code(r"""import pandas as pd, numpy as np

competence = pd.DataFrame(out["competence"])
display(competence.round(3))

panel, agents, idx = out["_panel"], out["_agents"], out["_idx"]
oosp = panel.loc[idx]
realised = {}
for a, ag in agents.items():
    g = ag.g.reindex(idx).values
    row = {}
    for reg in REGIMES:
        m = (oosp["regime"] == reg).values
        row[reg] = round(g[m].mean() / (g[m].std() + 1e-12) * ANN, 2) if m.sum() > 1 and g[m].std() > 0 else 0.0
    realised[a] = row

print("Realised OOS per-regime Sharpe:")
display(pd.DataFrame(realised).T.round(2))""")

md(r"""### Final Leaderboard

All rows use the same OOS window. `Final MAS fund (capped inverse-vol)` is the final system result.
`Coordinator ablation` is the regime-gated coordinator retained as a negative result: it shows
that this particular adaptive merging rule did not add value over capped inverse-volatility.""")

code(r"""lb = pd.DataFrame(out["leaderboard"]).copy()
for c in ["ret", "maxdd", "alpha"]:
    lb[c] = (lb[c] * 100).round(1)
lb["sharpe"] = lb["sharpe"].round(2)
lb["sortino"] = lb["sortino"].round(2)
display(lb[["name", "ret", "sharpe", "sortino", "maxdd", "alpha"]])""")

md(r"""### Final Fund Weights And Regime Breakdown""")

code(r"""print("Mean final capped inverse-volatility weights:")
display(pd.Series(out["mean_capped_inverse_vol_weights_oos"]).round(3))

print("Final MAS fund regime breakdown:")
display(pd.DataFrame(out["regime_breakdown"]))""")

md(r"""### Saved Figures

The plotting function saves four figures into `artifacts/notebooks_v2/06_mas/`:

1. `01_equity_comparison.png` — main equity chart, with the final MAS fund highlighted.
2. `02_capped_inverse_vol_weights.png` — capital allocation of the capped inverse-volatility fund.
3. `03_leaderboard_return_sharpe.png` — grouped return and Sharpe comparison.
4. `04_monthly_returns_comparison.png` — monthly MAS vs benchmark returns.
""")

code("fig = plot_results(out, save=True)\nfig")

md(r"""---
## Part C — Agent-Set And Merge-Method Bake-Off

This section evaluates several agent sets and allocation rules on the same OOS window. It is included
as an ablation table rather than as a separate optimisation stage.""")

code(r"""def _portfolio_row(name, agent_set, weights):
    sub = static_subset(agents, agent_set)
    eq = portfolio_equity(weights, sub, idx)
    return {
        "setup": name,
        "n_agents": len(sub),
        "ret": eq[-1] - 1,
        "sharpe": sharpe(eq),
        "sortino": sortino(eq),
        "maxdd": maxdd(eq),
    }

agent_sets = {
    "learned4": LEARNED_AGENTS,
    "rules_positive": RULE_AGENTS,
    "tcn_patch": ["tcn", "patch"],
    "tcn_patch_volbreak": ["tcn", "patch", "volbreak"],
    "final_agent_set": AGENTS,
}

rows = []
for label, agent_set in agent_sets.items():
    sub = static_subset(agents, agent_set)
    rows.append(_portfolio_row(f"{label} / EW", agent_set, equal_weight_weights(sub, panel)))
    rows.append(_portfolio_row(f"{label} / capped inverse-vol", agent_set, capped_inverse_vol_weights(sub, panel)))

rb = pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)
disp = rb.copy()
for c in ["ret", "maxdd"]:
    disp[c] = (disp[c] * 100).round(1)
disp["sharpe"] = disp["sharpe"].round(2)
disp["sortino"] = disp["sortino"].round(2)
display(disp)""")

md(r"""### Alternative Coordinator Rules

These allocators use the accepted agent set and identical gross exposure. They differ only in how
capital is routed across agents.""")

md("*Alternative allocators inlined from `hmats.mas.coordinators`:*")
code(helper_source("coordinators.py"))

code(r"""import matplotlib.pyplot as plt

mlb, mcurves = compare(agents=out["_agents"], panel=out["_panel"])
disp = mlb.copy()
for c in ["ret", "maxdd", "alpha_vs_BH"]:
    disp[c] = (disp[c] * 100).round(1)
disp["sharpe"] = disp["sharpe"].round(2)
disp["sortino"] = disp["sortino"].round(2)
disp["avg_turnover"] = disp["avg_turnover"].round(3)
display(disp)

fig, ax = plt.subplots(figsize=(13, 5))
for name, eq in mcurves.items():
    lw = 2.4 if "Capped inverse-vol" in name or name.startswith("Equal") else 1.2
    ax.plot(eq.index, (eq.values - 1) * 100, lw=lw, label=f"{name} ({eq.values[-1]-1:+.0%})")
ax.axhline(0, color="#9E9E9E", lw=0.6, ls=":")
ax.set_ylabel("Return (%)")
ax.legend(fontsize=8)
ax.set_title("Accepted-agent merge methods on the OOS window", fontweight="bold")
fig.tight_layout()
plt.show()""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

dst = HERE / "06_multi_agent_v1.ipynb"
json.dump(nb, open(dst, "w"), indent=1)
print("wrote", dst, "| cells:", len(cells))
