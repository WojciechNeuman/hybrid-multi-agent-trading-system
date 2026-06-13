# Diagrams — notes (2026-06-11)

Mermaid charts for the `notebooks_v2` ensemble. Render in any Markdown viewer with
Mermaid support (VS Code "Markdown Preview Mermaid Support", GitHub, Obsidian).

| File | What it shows | Use it for |
|---|---|---|
| `agent_communication.md` | Data/message flow: 4 base agents → artifacts → meta-learner → backtest, plus the communication contract table | Thesis "system architecture" section; shows the **current** v2 design (the root `system_architecture.md` is stale — it still has DRL/GP and OOS 2024-01-01) |
| `feature_pipeline.md` | Raw sources → 4 feature libraries (V1/V3/V4/Structural, 281 ML cols) → 22 V1 groups → per-agent selection | Thesis "feature engineering" section; visualises the funnel and flags the leaked column |
| `meta_dataflow.md` | (1) meta-labeling flow, (2) WFO coverage timeline, (3) the regime-mismatch failure mode | Explaining *why* the meta currently underperforms |

## Notes / caveats

- **Counts** come from `docs/features.md`. Unified parquet has 292 columns; ML-eligible
  ≈ 281 after excluding OHLCV/targets and the leaked `mkt_total_mcap_chg_24h`.
- **Coverage** of each base model's `wfo_probs.npy` (verified 2026-06-11):
  LGBM 2017-12→2026-05, Mamba 2022-01→2026-05, TCN & PatchTST 2023-01→2026-05.
  This staggered start is why the meta auto-sets `META_TRAIN_START ≈ 2022-02`.
- The **regime-mismatch** chart is the core thesis-worthy finding: the failure is not a
  coding artefact (after the sizing fix) but a *methodological* property of tuning base
  grids on a bearish grid-val window then testing across a bull.
- These diagrams are descriptive of the **as-built** system. The target/fixed design is
  tracked in `docs/TASKS_meta_overhaul.md`; update these charts when those tasks land.
