# notebooks_v2 — Final Thesis Pipeline

This folder is the canonical notebook sequence for the thesis **Hybrid Multi-Agent Trading System
Integrating Heterogeneous AI Methods**. The final integration notebook is now
`06_multi_agent_v1.ipynb`; the previous cross-asset notebook was removed from the final pipeline
because its signal did not show enough predictive skill versus the random-bracket null.

## Execution Order

```text
00_data_ingestion_v1
  -> 01_lgbm_v1
  -> 02_mamba_v1
  -> 03_tcn_v1
  -> 04_patchtst_v1
  -> 05_rule_agents_v1
  -> 06_multi_agent_v1
```

The final MAS notebook consumes the saved artifacts from notebooks `01` to `05`, wraps them as
autonomous trading agents, compares coordination/merging rules, and saves thesis-ready figures to:

```text
artifacts/notebooks_v2/06_mas/
```

## Runtime Modules

The maintained implementation lives in `src/hmats/mas/`; notebooks are generated from those source
files so the university-facing notebook and runtime code do not drift.

```text
src/hmats/mas/
├── mas07.py          # final engine, accepted-agent roster, allocator evaluation, thesis charts
├── rule_agents.py    # rule agents; final MAS keeps trend, volbreak, dominance_rotation; excludes meanrev, sentiment_regime
├── agent_eval.py     # random-bracket null utilities
├── coordinators.py   # alternative merge/allocation methods
└── crossasset_agent.py
    # experimental only; removed from final notebook roster
```

Regenerate the self-contained notebooks after editing source modules:

```bash
cd src/hmats/notebooks_v2
python _build_rule_nb.py
python _build_mas07_nb.py
```

`_build_mas07_nb.py` writes `06_multi_agent_v1.ipynb`.

## Final MAS Roster

Accepted agents:

| Agent | Source | Method family | Role in final system |
|---|---|---|---|
| `lgbm` | `01_lgbm_v1` | gradient boosting | tabular learned agent |
| `mamba` | `02_mamba_v1` | selective state-space model | sequence learned agent |
| `tcn` | `03_tcn_v1` | temporal convolutional network | TBM sequence learned agent |
| `patch` | `04_patchtst_v1` | patch transformer | TBM transformer learned agent |
| `trend` | `05_rule_agents_v1` | rule-based trend following | structurally different rule agent |
| `volbreak` | `05_rule_agents_v1` | rule-based volatility breakout | structurally different rule agent |
| `dominance_rotation` | `05_rule_agents_v1` | rule-based cross-asset dominance rotation | **diversification** rule agent (cross-asset flow, not price panel; not claimed as alpha) |

Excluded experiments:

| Agent | Reason |
|---|---|
| `meanrev` | negative OOS return |
| `sentiment_regime` | negative OOS return (−27.7%) and below its random-bracket null (7th percentile) |
| `crossasset` | weak directional evidence and not significant versus random-bracket null |

### Rule agent evaluation: `sentiment_regime` and `dominance_rotation`

Two heterogeneous rule agents were evaluated using only causal features in the unified parquet and
the same grid-search protocol as the existing rule agents (2022-01 → 2024-05 validation window,
frozen before OOS). Inclusion criteria: positive OOS return and max drawdown no worse than BTC
buy-and-hold (−50.1%). The random-bracket null determines whether a passing agent is described as
an alpha source or a diversification agent only.

| Agent | Inputs | OOS ret | OOS Sharpe | OOS maxdd | Random-bracket null percentile | Status |
|---|---|---:|---:|---:|---|---|
| `sentiment_regime` | Fear & Greed level / 7-day MA / 7-day change | −27.7% | −0.77 | −42.8% | 7th | Excluded — negative OOS return |
| `dominance_rotation` | BTC dominance Δ7d lagged 24h, ETH/BTC 24h & 72h momentum | +153.6% | 1.09 | −26.4% | 93–95th | Accepted — diversification agent, not claimed as alpha |

`dominance_rotation` reads cross-asset capital rotation (BTC dominance, ETH/BTC momentum) — an
information source no BTC price-feature model uses — and its return stream is therefore structurally
decorrelated from the four learned agents. Its OOS return sits at the 95% boundary of the
random-bracket null across multiple seeds, so it is not presented as alpha. The daily BTC-dominance
component is lagged by 24h before use on hourly bars to avoid same-day market-cap leakage.

## Final Conclusion To Carry Into The Thesis

The defensible claim is **hybrid multi-agent diversification**, not that a complex adaptive
coordinator adds alpha.

Current executed result in `artifacts/notebooks_v2/06_mas/results.json`:

| Strategy | OOS return | Sharpe | MaxDD | Interpretation |
|---|---:|---:|---:|---|
| Final MAS fund, capped inverse-vol | +56.8% | 2.02 | -5.9% | final system result |
| Naive EW fund | +53.8% | 1.76 | -5.2% | simple fund-of-agents baseline |
| Original coordinator ablation | +28.6% | 0.59 | -20.4% | tested and rejected as final allocator |
| BTC buy-and-hold | +15.6% | 0.16 | -50.1% | crypto benchmark |
| S&P 500 buy-and-hold | +43.6% | 1.11 | -18.8% | broad-market benchmark |

The original `softmax(trailing Sharpe) x regime competence` coordinator does not justify the thesis
claim as a headline mechanism. It remains useful as a negative ablation: it shows that not every
agentic-looking coordination rule improves the fund. The final thesis should present capped
inverse-volatility risk parity over heterogeneous agents as the robust merge method. The cap is
`2/N`, which limits any single agent to 2/7 ≈ 28.6% of capital in the seven-agent roster and
prevents long inactive periods dominated by one method.

The capped fund intentionally sacrifices part of the uncapped inverse-volatility result in exchange
for a more credible multi-agent allocation. In the OOS evaluation the seven-agent roster produces
mean weights of approximately LGBM 21.7%, Patch 22.0%, Mamba 14.6%, TCN 13.2%, VolBreak 10.9%,
Trend 8.9%, and DominanceRotation 8.7%.

## Thesis Figures To Use

Generated by executing `06_multi_agent_v1.ipynb`:

| Figure | File | Use in thesis |
|---|---|---|
| Main OOS equity comparison | `artifacts/notebooks_v2/06_mas/01_equity_comparison.png` | primary results figure; MAS fund highlighted against agents, BTC, and S&P 500 |
| Final capital allocation | `artifacts/notebooks_v2/06_mas/02_capped_inverse_vol_weights.png` | methodology/results figure showing capped dynamic risk-parity allocation |
| Leaderboard return and Sharpe | `artifacts/notebooks_v2/06_mas/03_leaderboard_return_sharpe.png` | compact comparison for experiments/results chapter |
| Monthly return comparison | `artifacts/notebooks_v2/06_mas/04_monthly_returns_comparison.png` | robustness and time-distribution figure |

Copy the selected PNGs into:

```text
latex_thesis_v4_26_06_07/Master_Thesis___Hybrid_Multi_Agent_Trading_System/Images/
```

Suggested thesis wording:

> The final system is best understood as a fund of heterogeneous autonomous trading agents. The
> attempted adaptive coordinator did not outperform simple baselines, so the final architecture uses
> leak-free capped inverse-volatility risk parity as the capital-allocation layer. This produces a stronger
> risk-adjusted OOS profile than BTC buy-and-hold, S&P 500 buy-and-hold, and the original coordinator
> ablation, while preserving the multi-agent framing through independent models, rule agents,
> positions, confidence paths, and realised return streams.

## Artifact Contract

Every producing notebook writes a consistent schema:

```text
artifacts/notebooks_v2/{agent_dir}/
├── oos_probs.npy
├── oos_index.npy
├── wfo_probs.npy
├── wfo_index.npy
├── results.json
└── model or chart files
```

The final notebook writes:

```text
artifacts/notebooks_v2/06_mas/
├── results.json
├── leaderboard.csv
├── competence.csv
├── capped_inverse_vol_weights_oos.csv
├── coordinator_weights_oos.csv
├── final_equity.npy
├── coord_ablation_equity.npy
├── 01_equity_comparison.png
├── 02_capped_inverse_vol_weights.png
├── 03_leaderboard_return_sharpe.png
└── 04_monthly_returns_comparison.png
```

## Notes On Notebook Generation

`_nbinline.py` strips package imports from `hmats.mas` modules and inlines their source into the
self-contained notebooks. If module imports move, update `_nbinline.py` before regenerating.

Do not regenerate the removed cross-asset notebook for the final thesis sequence. The
`crossasset_agent.py` module may remain as an archived experiment, but it is not part of the final
accepted MAS result.
