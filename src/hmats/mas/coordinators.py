"""Alternative agent-merge methods for the multi-agent system.

The notebook-07 `Coordinator` allocates by `softmax(trailing-Sharpe) × competence`. Empirically it
*loses* to a naive equal-weight fund of the same agents — the trailing-Sharpe momentum gate
concentrates capital into an agent just as its edge mean-reverts. This module implements several
alternative allocators so they can be compared on the same OOS panel, all under one rule:

* **leak-free**: a weight decided with information up to bar ``t`` earns each agent's return over
  ``t -> t+1`` (``portfolio_equity`` enforces ``w[t-1] · g[t]``); every trailing statistic is
  shifted by ``1 + EMBARGO_H`` bars.
* **comparable**: every method produces weights that sum to 1 (gross exposure 1.0), so differences
  are pure *allocation*, not leverage.

Methods
-------
- ``equal_weight``         — 1/N (the benchmark to beat).
- ``inverse_vol``         — risk parity lite: w ∝ 1/trailing-vol. Down-weights the wild agents.
- ``trailing_sharpe``     — softmax over trailing Sharpe only (the gate, no competence tilt).
- ``mas_coordinator``     — the full notebook-07 allocator (softmax Sharpe × regime competence).
- ``shrink_to_ew``        — the MAS allocator blended λ toward equal weight (concentration cap).
- ``inverse_vol_competence`` — risk parity tilted by pre-OOS regime competence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import mas07 as m


def _trailing_vol(agents, win=m.PERF_WIN, embargo=m.EMBARGO_H) -> pd.DataFrame:
    out = {}
    for a, ag in agents.items():
        out[a] = ag.g.rolling(win, min_periods=200).std().shift(1 + embargo)
    return pd.DataFrame(out)[list(agents)]


def _norm(W: pd.DataFrame) -> pd.DataFrame:
    # divide each row by its row-sum → align on the index (axis=0), not the columns
    return W.div(W.sum(axis=1).replace(0, np.nan), axis=0).fillna(1.0 / W.shape[1])


def equal_weight(agents, panel, perf, competence) -> pd.DataFrame:
    return pd.DataFrame(1.0 / len(agents), index=panel.index, columns=list(agents))


def inverse_vol(agents, panel, perf, competence) -> pd.DataFrame:
    vol = _trailing_vol(agents)
    inv = 1.0 / (vol + 1e-9)
    return _norm(inv.reindex(panel.index)).fillna(1.0 / len(agents))


def softmax_sharpe(agents, panel, perf, competence) -> pd.DataFrame:
    z = (perf[list(agents)] / m.PERF_TEMP).clip(-10, 10)
    soft = np.exp(z)
    return _norm(soft)


def mas_coordinator(agents, panel, perf, competence) -> pd.DataFrame:
    return m.Coordinator(competence).allocate(agents, panel, perf)


def shrink_to_ew(agents, panel, perf, competence, lam: float = 0.5) -> pd.DataFrame:
    W = mas_coordinator(agents, panel, perf, competence)
    ew = 1.0 / len(agents)
    return (1 - lam) * W + lam * ew


def inverse_vol_competence(agents, panel, perf, competence) -> pd.DataFrame:
    vol = _trailing_vol(agents).reindex(panel.index)
    inv = (1.0 / (vol + 1e-9))
    tilt = m.smoothed_competence(competence, panel) + m.COMP_FLOOR
    return _norm((inv * tilt).fillna(0.0)).fillna(1.0 / len(agents))


METHODS = {
    "Equal weight (1/N)": equal_weight,
    "Inverse-vol (risk parity)": inverse_vol,
    "Trailing-Sharpe softmax": softmax_sharpe,
    "MAS coordinator (current)": mas_coordinator,
    "Shrink-to-EW (λ=0.5)": shrink_to_ew,
    "Inv-vol × competence": inverse_vol_competence,
}


def compare(agents=None, panel=None) -> pd.DataFrame:
    """Build the panel/agents once and score every merge method on the OOS window."""
    if panel is None:
        panel = m.load_panel()
    if agents is None:
        agents = m.build_agents(panel, m.repo_root() / "artifacts" / "notebooks_v2")
    competence = m.estimate_competence(agents, panel)
    perf = m.trailing_sharpe(agents)
    idx = m._oos(panel)
    bh = (1.0 + panel["ret"].reindex(idx)).cumprod()

    rows = []
    curves = {}
    for name, fn in METHODS.items():
        W = fn(agents, panel, perf, competence)
        eq = m.portfolio_equity(W, agents, idx)
        seg = eq / eq[0]
        turn = float(W.reindex(idx).diff().abs().sum(axis=1).mean())
        rows.append({"method": name, "ret": seg[-1] - 1, "sharpe": m.sharpe(seg),
                     "sortino": m.sortino(seg), "maxdd": m.maxdd(seg),
                     "alpha_vs_BH": seg[-1] - bh.values[-1] / bh.values[0],
                     "avg_turnover": turn})
        curves[name] = pd.Series(seg, index=idx)
    lb = pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)
    return lb, curves
