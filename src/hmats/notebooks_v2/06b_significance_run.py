#!/usr/bin/env python
"""06b — Significance / robustness test for the final MAS fund (audit Q4).

Refines the user's "run the fund on random sets, sample 30 times" idea into three
proper null/robustness tests on the EXACT agent return streams the real fund uses:

  A. Random-allocation null  — N random Dirichlet static weight vectors over the same 7
     agents. If the real capped-inverse-vol fund's Sharpe is not in the top tail of this
     distribution, the allocator adds nothing beyond "hold these 7 agents in some mix".
  B. Random-roster null      — random agent subsets (size 2..7), each run capped-IV.
     Tests how special the chosen 7-agent roster is.
  C. Block-bootstrap CI      — stationary block bootstrap (weekly blocks) of the real
     fund's own hourly returns -> 95% CI on Sharpe + P(Sharpe>0), P(Sharpe>1).
     This is the closest thing to a deflated/probabilistic Sharpe on a single path.

Read-only: writes a JSON report to artifacts/notebooks_v2/06_mas/significance.json.
"""
from __future__ import annotations
import json
import numpy as np, pandas as pd
from hmats.mas import mas07

RNG = np.random.default_rng(42)
N_ALLOC = 2000
N_ROSTER = 2000
N_BOOT = 5000
BLOCK = 168  # 1-week blocks for the stationary bootstrap

def sharpe_from_g(g):
    g = np.asarray(g, float)
    if g.std(ddof=1) < 1e-12: return 0.0
    return float(g.mean() / (g.std(ddof=1) + 1e-12) * mas07.ANN)

def main():
    panel = mas07.load_panel()
    a2 = mas07.repo_root() / 'artifacts' / 'notebooks_v2'
    agents = mas07.build_agents(panel, a2)
    idx = mas07._oos(panel)
    names = list(agents)
    # per-agent OOS log-return streams (what the coordinator allocates over)
    G = pd.DataFrame({a: agents[a].g for a in names}).reindex(idx).fillna(0.0)[names].values  # (T, N)
    T, N = G.shape
    print(f'OOS bars={T}  agents={names}', flush=True)

    # ---- Real fund (capped inverse-vol) ----
    iv_w = mas07.capped_inverse_vol_weights(agents, panel)
    real_eq = mas07.portfolio_equity(iv_w, agents, idx)
    real_g = np.diff(np.log(np.maximum(real_eq, 1e-12)), prepend=0.0)
    real_sharpe = sharpe_from_g(real_g[1:])
    print(f'REAL capped-IV fund: ret={real_eq[-1]-1:+.1%}  sharpe={real_sharpe:.3f}', flush=True)

    # ---- A. Random static-allocation null (Dirichlet weights, leak-free: w fixed, applied to g) ----
    # portfolio return each bar = sum_i w_i * g_i  (static weights -> no look-ahead)
    allocs = []
    for _ in range(N_ALLOC):
        w = RNG.dirichlet(np.ones(N))
        pg = G @ w
        allocs.append(sharpe_from_g(pg))
    allocs = np.array(allocs)
    a_pct = float((allocs < real_sharpe).mean())

    # 30-run sample exactly as the user described
    sample30 = allocs[:30]

    # ---- B. Random-roster null (subset size 2..7, capped inverse-vol on the subset) ----
    rosters = []
    for _ in range(N_ROSTER):
        k = RNG.integers(2, N + 1)
        sub = list(RNG.choice(names, size=k, replace=False))
        subA = {a: agents[a] for a in sub}
        wv = mas07.capped_inverse_vol_weights(subA, panel)
        eq = mas07.portfolio_equity(wv, subA, idx)
        rosters.append(sharpe_from_g(np.diff(np.log(np.maximum(eq, 1e-12)))))
    rosters = np.array(rosters)
    r_pct = float((rosters < real_sharpe).mean())

    # ---- C. Stationary block bootstrap CI on the real fund's Sharpe ----
    r = real_g[1:]
    boots = []
    nblocks = int(np.ceil(len(r) / BLOCK))
    for _ in range(N_BOOT):
        starts = RNG.integers(0, len(r), size=nblocks)
        seg = np.concatenate([np.take(r, range(s, s + BLOCK), mode='wrap') for s in starts])[:len(r)]
        boots.append(sharpe_from_g(seg))
    boots = np.array(boots)
    ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    p_gt0 = float((boots > 0).mean()); p_gt1 = float((boots > 1).mean())

    out = dict(
        oos_bars=T, agents=names, real_sharpe=round(real_sharpe, 4), real_ret=round(float(real_eq[-1]-1), 4),
        random_alloc_null=dict(n=N_ALLOC, mean=round(float(allocs.mean()),3), p50=round(float(np.percentile(allocs,50)),3),
            p95=round(float(np.percentile(allocs,95)),3), max=round(float(allocs.max()),3),
            real_percentile=round(a_pct,4), sample30_mean=round(float(sample30.mean()),3), sample30_max=round(float(sample30.max()),3)),
        random_roster_null=dict(n=N_ROSTER, mean=round(float(rosters.mean()),3), p50=round(float(np.percentile(rosters,50)),3),
            p95=round(float(np.percentile(rosters,95)),3), max=round(float(rosters.max()),3), real_percentile=round(r_pct,4)),
        block_bootstrap=dict(n=N_BOOT, block_h=BLOCK, sharpe_ci95=[round(ci[0],3),round(ci[1],3)],
            p_sharpe_gt_0=round(p_gt0,4), p_sharpe_gt_1=round(p_gt1,4)),
    )
    arts = a2 / '06_mas' / 'significance.json'
    json.dump(out, open(arts, 'w'), indent=2)
    print('\n=== SIGNIFICANCE REPORT ===')
    print(json.dumps(out, indent=2))
    print(f'\nsaved -> {arts}')
    print('\nINTERPRETATION:')
    print(f'  A) Real Sharpe {real_sharpe:.2f} sits at the {a_pct*100:.1f}th pct of {N_ALLOC} random weightings '
          f'(p95={np.percentile(allocs,95):.2f}). {"ALLOCATOR ADDS EDGE" if a_pct>0.95 else "allocator NOT distinguishable from random weights"}.')
    print(f'  B) Real Sharpe sits at the {r_pct*100:.1f}th pct of {N_ROSTER} random rosters.')
    print(f'  C) Bootstrap 95% CI on fund Sharpe = [{ci[0]:.2f}, {ci[1]:.2f}]; P(Sharpe>1)={p_gt1:.2f}. '
          f'{"robustly >1" if ci[0]>1 else ("robustly >0" if ci[0]>0 else "CI includes 0")}.')

if __name__ == '__main__':
    main()
