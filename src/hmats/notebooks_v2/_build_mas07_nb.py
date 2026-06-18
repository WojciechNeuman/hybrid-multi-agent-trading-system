"""Builds src/hmats/notebooks_v2/07_multi_agent_v1.ipynb — the SINGLE place where every agent is
merged into the coordinator.

Unlike a thin import-driver, this generates a *self-contained* notebook: the full coordinator
engine from `mas07.py` is inlined section-by-section (read straight from the source file, so it can
never drift), followed by the orchestration that loads all seven agents (4 learned + 3 rule),
estimates competence, allocates capital, and reports/plots the result.

Run:  python _build_mas07_nb.py
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

# ── intro ───────────────────────────────────────────────────────────────────────────────────
md(r"""# 07 — Hybrid Multi-Agent System · the merge point for all agents

This is the **single notebook where every agent is merged into one system.** The earlier notebooks
each *produce one agent* and save its signal to `artifacts/notebooks_v2/<agent>/`:

| # | Notebook | Agent | Kind |
|---|----------|-------|------|
| 01 | `01_lgbm`     | `lgbm`     | gradient boosting (tabular) |
| 02 | `02_mamba`    | `mamba`    | selective state-space |
| 03 | `03_tcn`      | `tcn`      | dilated causal conv. (TBM, 2-channel) |
| 04 | `05_patchtst` | `patch`    | patch transformer (TBM, 2-channel) |
| 05 | `08_trend`    | `trend`    | rule: trend-following |
| 05 | `08_meanrev`  | `meanrev`  | rule: mean-reversion |
| 05 | `08_volbreak` | `volbreak` | rule: volatility breakout |
| 06 | `09_crossasset` | `crossasset` | learned: cross-asset / sentiment / flow |

Here those eight independent signals are wrapped as **autonomous risk-managed `TradingAgent`s** and
fused by a **`Coordinator`** that allocates capital across them by `online skill × regime
competence`. This is the fund-of-agents layer — not an averaging ensemble (that was the now-archived averaging meta-learner).

**Self-contained:** this notebook inlines the full engine and all evaluation code — run it
top-to-bottom with no local imports. It reads each agent's saved artifacts, so run notebooks
01–04 (learned), 05 (rule) and 06 (cross-asset) first to produce them.

**Leak discipline (carried through every cell):** a weight decided with information up to bar *t*
earns each agent's return over *t → t+1*; every trailing statistic is shifted by ≥1 bar + embargo;
per-regime competence priors see pre-OOS data only.

---
## Part A — coordinator engine (inlined from `mas07.py`)""")

code("# Dependencies for the inlined engine\n" + imports_chunk)

for title, body in sections:
    md(f"### {title}")
    code(body)

# ── orchestration: merge all 7 agents ────────────────────────────────────────────────────────
md(r"""---
## Part B — merge the agents and run the coordinator

`AGENTS` (defined in *Configuration* above) is `LEARNED_AGENTS + RULE_AGENTS + CROSS_AGENTS` = the
full eight-agent roster. The cell below verifies every agent's artifacts exist on disk, then runs
the whole pipeline.""")

code(r"""# Verify all agents' artifacts exist (produced by notebooks 01–04, 05, 06).
missing = [d for a, d in AGENT_DIR.items()
           if not (repo_root() / "artifacts" / "notebooks_v2" / d / "oos_probs.npy").exists()]
assert not missing, ("missing artifacts — run the producing notebooks first: "
                     + ", ".join(missing) + " (05 → rule agents, 06 → cross-asset, 01–04 → learned)")
print("All", len(AGENT_DIR), "agents present:", list(AGENT_DIR))""")

code("out = run_pipeline(save=True, verbose=True)")

md(r"""### Per-regime competence priors (pre-OOS, leak-free)

Measured on 2023-01 → 2024-05-31 only — each agent's specialisation *before* it sees the test
window — and validated against realised OOS per-regime Sharpe below.""")
code(r"""import pandas as pd, numpy as np
competence = pd.DataFrame(out["competence"]); display(competence.round(3))

panel, agents, idx = out["_panel"], out["_agents"], out["_idx"]
oosp = panel.loc[idx]; val = {}
for a, ag in agents.items():
    g = ag.g.reindex(idx).values
    row = {}
    for reg in REGIMES:
        m = (oosp["regime"] == reg).values
        row[reg] = round(g[m].mean() / (g[m].std() + 1e-12) * ANN, 2) if m.sum() > 1 and g[m].std() > 0 else 0.0
    val[a] = row
print("Realised OOS per-regime Sharpe (validation of the priors):")
display(pd.DataFrame(val).T.round(2))""")

md(r"""### Leaderboard — coordinator vs every agent vs baselines

All rows on the identical OOS window. The honest benchmark is the **Naive EW fund** (equal weight
across all agents): if the coordinator cannot beat a blind equal split of the same roster, the
allocation logic — not the agents — is the bottleneck.""")
code(r"""lb = pd.DataFrame(out["leaderboard"]).copy()
for c in ["ret", "maxdd", "alpha"]:
    lb[c] = (lb[c] * 100).round(1)
lb["sharpe"] = lb["sharpe"].round(2); lb["sortino"] = lb["sortino"].round(2)
display(lb[["name", "ret", "sharpe", "sortino", "maxdd", "alpha"]])""")

md(r"""### Coordinator regime breakdown & mean capital weights""")
code(r"""display(pd.DataFrame(out["regime_breakdown"]))
print("Mean OOS capital weights:")
display(pd.Series(out["mean_weights_oos"]).round(3))""")

md(r"""### Figures — equity curves and capital allocation over time""")
code("fig = plot_results(out, save=True)\nfig")

md(r"""### Contribution of the rule agents (4 learned vs 7)

Re-run the coordinator on the learned-only roster and on the full roster to isolate what the
orthogonal rule agents add. Watch the **Naive EW fund** row — that is where the diversification
shows up.""")
code(r"""_full = list(AGENTS)
def _summary(ags, tag):
    global AGENTS
    AGENTS = ags
    r = run_pipeline(save=False, verbose=False)
    d = {x["name"]: x for x in r["leaderboard"]}
    c, ew = d["Coordinator (MAS)"], d["Naive EW fund"]
    return {"roster": tag, "EW_ret": f'{ew["ret"]:+.1%}', "EW_sharpe": round(ew["sharpe"], 2),
            "Coord_ret": f'{c["ret"]:+.1%}', "Coord_sharpe": round(c["sharpe"], 2)}
rows = [_summary(LEARNED_AGENTS, "4 learned"),
        _summary(LEARNED_AGENTS + RULE_AGENTS, "7 (learned+rule)"),
        _summary(LEARNED_AGENTS + RULE_AGENTS + CROSS_AGENTS, "8 (+ crossasset)")]
AGENTS = _full
display(pd.DataFrame(rows))""")

md(r"""### How much is *skill*? — the random-bracket null

A headline return is dominated by two things that are **not** signal skill: the asymmetric ATR
bracket (cut losers / let winners run = positive convexity) and the trending OOS market
(directional beta). The fair benchmark is therefore *random* long/short entries through the agent's
**own** bracket, trade-count-matched. An agent has demonstrable skill only if its real return sits
in the top tail of that random distribution (`pctile ≥ 95%`).""")
md("*Random-bracket null utilities (inlined):*")
code(helper_source("agent_eval.py"))
code(r"""panel = out["_panel"]; a2 = repo_root() / "artifacts" / "notebooks_v2"
probs = {a: panel[a] for a in AGENTS}
pdn = {a: panel[a + "_dn"] for a in AGENTS if a in MULTICLASS}
bps = {a: json.load(open(a2 / AGENT_DIR[a] / "results.json"))["best_params"] for a in AGENTS}
skill = null_table(probs, panel, bps, prob_dn=pdn, n_sims=250)
display(skill)""")

md(r"""**Reading it.** Only agents with `skill_sig = yes` carry predictive edge over the
bracket+beta baseline. Everything else "makes money" in this window the way a coin flip through the
same bracket would — useful only as diversification, not as alpha.""")

md(r"""### Alternative merge methods

The notebook-07 `Coordinator` is one allocation rule among many. `coordinators.py` scores several on
the identical OOS panel — all leak-free, all at gross exposure 1.0 (so differences are pure
allocation, not leverage). The question: does *any* allocator beat naive equal weight, and is the
trailing-Sharpe momentum gate actually helping or hurting?""")
md("*Alternative allocators (inlined):*")
code(helper_source("coordinators.py"))
code(r"""import matplotlib.pyplot as plt
mlb, mcurves = compare(agents=out["_agents"], panel=out["_panel"])
disp = mlb.copy()
for c in ["ret", "maxdd", "alpha_vs_BH"]:
    disp[c] = (disp[c] * 100).round(1)
disp["sharpe"] = disp["sharpe"].round(2); disp["sortino"] = disp["sortino"].round(2)
disp["avg_turnover"] = disp["avg_turnover"].round(3)
display(disp)

fig, ax = plt.subplots(figsize=(13, 5))
for name, eq in mcurves.items():
    lw = 2.4 if name.startswith("Equal") else 1.3
    ax.plot(eq.index, (eq.values - 1) * 100, lw=lw, label=f"{name} ({eq.values[-1]-1:+.0%})")
ax.axhline(0, color="#9E9E9E", lw=0.6, ls=":"); ax.set_ylabel("Return (%)"); ax.legend(fontsize=8)
ax.set_title("Agent-merge methods on the full roster (OOS)", fontweight="bold")
fig.tight_layout(); plt.show()""")

md(r"""---
## Part C — discussion

**What works.** Wrapping each model/strategy as a risk-managed agent and allocating across them by
`online skill × regime competence` is a coherent fund-of-agents. The **rule agents are structurally
orthogonal** to the learned models (their edge is strategy logic, not the shared feature panel), so
adding them sharply improves the diversified equal-weight fund and its drawdown — the
regime-robustness the hybrid thesis is about.

**The allocation verdict (from the merge-method table above).** The current
`softmax(trailing-Sharpe) × competence` coordinator is the **worst** allocator on the panel — it and
plain trailing-Sharpe softmax are the only methods that lose money. The trailing-Sharpe momentum gate
concentrates capital into an agent just as its edge mean-reverts (high turnover, negative alpha).

**The fix is a different merge method, not more agents.** Static / risk-based weightings win:
**inverse-volatility (risk parity)** delivers the best risk-adjusted result — higher Sharpe than
equal-weight at roughly half the drawdown, with near-zero turnover. The recommended path forward is
to replace the trailing-Sharpe gate with inverse-vol risk parity (optionally tilted by *significant*
competence only), and to remember the skill table: most agents are bracket+beta, so only **tcn** and
**patch** have earned a genuine skill premium in the weighting.""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}

dst = HERE / "07_multi_agent_v1.ipynb"
json.dump(nb, open(dst, "w"), indent=1)
print("wrote", dst, "| cells:", len(cells))
