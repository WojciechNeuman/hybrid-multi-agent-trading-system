# Task List — Ensemble & Meta-Learner Overhaul (2026-06-11)

Prioritised, self-contained tasks for a coding agent. Each has: **why**, **where**,
**done-when**. Full diagnosis in `docs/meta_analysis_2026-06-11.md`.

Ground rules (from the project owner):
- **Do NOT change base-model *training* logic** (`01_lgbm`, `02_mamba`, `03_tcn`,
  `05_patchtst`). They produce the signals we want. Touch only seeds / artifact plumbing.
- All randomness must be **seeded & reproducible** across runs.
- OOS test window is fixed: **2024-05-31 → 2026-05-16**. Meta trains only on bars
  strictly before it (minus embargo).
- After any base-notebook edit, the owner re-runs it (Mamba is Colab-only).

---

## P0 — Correctness (blockers)

### T1 — Fix meta sizing backtest capital accounting  ✅ DONE 2026-06-11
- **Why:** `pos_eq = cur*pos_size` then `cur = pos_eq*(1+net)` discarded the uninvested
  `(1-pos_size)` capital every trade → equity decays to 0 regardless of P&L (the −100%
  wipeout). 
- **Where:** `04_meta_learning_v1.ipynb`, sized-backtest cell.
- **Fix applied:** `pos_eq=cur; psize=pos_size`; mark-to-market and exit now apply
  `psize*net` to full equity. Re-ran: equity no longer wipes out (AUC unchanged — this
  was a P&L bug, not a ranking bug).
- **Done when:** meta `total_ret` is finite and matches a hand-check on 2–3 trades. ✔

### T2 — Mamba reproducibility (torch seeding)
- **Why:** `02_mamba_v1.ipynb` sets only `random_state` (LightGBM); the Mamba torch model
  is **not** seeded → its `wfo_probs` change every run, so the meta is irreproducible.
- **Where:** `02_mamba_v1.ipynb` cell 1 (imports). Add the same block `03_tcn`/`05_patchtst`
  use: `SEED=42; random.seed; np.random.seed; torch.manual_seed; torch.cuda.manual_seed_all;
  torch.backends.cudnn.deterministic=True; torch.use_deterministic_algorithms(True, warn_only=True)`,
  and pass a seeded `generator=` + `worker_init_fn` to every `DataLoader(shuffle=True)`.
- **Constraint:** do not touch the model architecture or training hyperparameters.
- **Done when:** two consecutive Colab runs produce identical `oos_probs.npy` (bitwise or
  `np.allclose` < 1e-6). Owner re-runs on Colab.

### T3 — Tighten determinism in TCN / PatchTST DataLoaders
- **Why:** both set global seeds but `DataLoader(..., shuffle=True)` has no `generator`,
  so batch order (hence early-stopping epoch) can drift.
- **Where:** `03_tcn_v1.ipynb`, `05_patchtst_v1.ipynb` training cells.
- **Done when:** re-running each notebook twice yields identical `oos_probs.npy`. Owner re-runs.

---

## P1 — Make the meta actually use the agents' edge

### T4 — Stop averaging heterogeneous-label probabilities
- **Why:** `01_lgbm` predicts a **next-bar direction** label; `02/03/05` predict a
  **TBM ±2σ/24h** label. `ensemble_p_up = mean(...)` averages probabilities that mean
  different things → muddies the primary signal.
- **Options (pick one, document choice):**
  1. **Standardise the label:** retrain LGBM on the same TBM label (cheap, LGBM is fast)
     so all four probs are comparable. *(Allowed — LGBM training change is low-risk and the
     owner explicitly wants comparable signals; confirm before doing.)*
  2. **Per-model calibration:** isotonic/Platt-scale each model's prob against `meta_y`
     on the train window before they enter the ensemble/meta features. No base-training change.
- **Done when:** the four `*_p_up` features are on a common, calibrated scale; document
  reliability curves.

### T5 — Feed the meta the agents' *tuned* signals, not just raw probs
- **Why:** each base model's standalone edge lives in its `results.json["best_params"]`
  (thresholds, SL/TP, holds) — TCN +73.8% / PatchTST +36.1% standalone. The meta ignores
  these and re-derives signals with one fixed untuned rule (ENS thresholds + SL1.5/TP2.5),
  throwing the alpha away.
- **Where:** `04_meta_learning_v1.ipynb`. Add, per base model, an engineered feature =
  that model's tuned directional position at each bar (run its `_run_backtest`/signal rule
  with its own `best_params`). Then meta-label/size on the **union of per-model trade
  proposals**, or stack the per-model positions.
- **Done when:** meta OOS Sharpe ≥ best single base model's, or a clear written reason why not.

### T6 — Address the short-bias / regime overfit of base grids
- **Why:** grids are tuned on grid-val **2022→2024-05** (bear/chop) → all models short-biased
  (Mamba 244S/70L, TCN 232S/40L, PatchTST 129S/0L); shorts lose through the 2024-11→2025-10 bull.
- **Where:** the grid-search cells of each base notebook (parameter selection, **not** model
  training) and/or the meta.
- **Options:** (a) add a long/short trade-count balance guard to the grid filter;
  (b) purged K-fold over grid-val so a bull slice is always represented;
  (c) de-bias probabilities per regime before the ensemble.
- **Done when:** OOS long/short trade ratio is not pathologically one-sided and bull-regime
  drawdown shrinks.

---

## P2 — Meta robustness & hygiene

### T7 — Remove/neutralise monotonic time features from the meta
- **Why:** meta feature importance is dominated by `halving_cycle_pos` (357, ~2× the next),
  a near-monotonic time index → the meta memorises *when* it trained, not *what works* →
  OOS AUC 0.464 (< 0.5). 
- **Where:** `REGIME_FEATURES` in `04_meta_learning_v1.ipynb` cell 2.
- **Fix:** drop `halving_cycle_pos` (and audit `dom_*`, `quarter_*`) from meta features;
  keep stationary regime descriptors (`atr_14_pct`, `hurst_*`, `bb_width_pct`, `vol_ratio_*`).
- **Done when:** meta OOS AUC > 0.50 and importance is spread across model probs + stationary
  regime features.

### T8 — Decide Mamba's inclusion
- **Why:** Mamba OOS AUC 0.5319 is fine, but its standalone backtest is −27% (Sharpe −0.52,
  MaxDD −47%) — worst of the four; it drags the unweighted ensemble.
- **Action:** after T2 (reproducible) + T6, re-evaluate; if still a net drag, exclude from the
  ensemble mean but optionally keep `mamba_p_up` as a meta feature (the meta can down-weight it).
- **Done when:** documented include/exclude decision with before/after meta metrics.

### T9 — Refresh stale architecture doc  ✅ DONE 2026-06-11
- **Why:** `docs/system_architecture.md` still described the old `notebooks/` design (DRL+GP,
  `08_meta`, OOS 2024-01-01). The live system is `notebooks_v2` (LGBM+Mamba+TCN+PatchTST,
  `04_meta`, OOS 2024-05-31).
- **Where:** `docs/system_architecture.md`; reuse the charts in `docs/diagrams/`.
- **Fix applied:** full rewrite to the as-built v2 pipeline — 4 base agents, `04_meta`,
  unified OOS 2024-05-31→2026-05-16, corrected sizing, current numbers, and the known-issues
  section (T4/T5/T6/T7). Mermaid chart, signal-type table, and WFO table all updated.
- **Done when:** doc matches the as-built v2 pipeline. ✔

---

## Re-run checklist for the owner (after the above)
| Notebook | Re-run? | Why |
|---|---|---|
| `01_lgbm_v1` | only if T4-opt-1 chosen | label change |
| `02_mamba_v1` (Colab) | **yes** | T2 seeds → reproducible artifacts |
| `03_tcn_v1` | optional | T3 determinism |
| `05_patchtst_v1` | optional | T3 determinism |
| `04_meta_learning_v1` | **yes, last** | consumes refreshed artifacts; T1/T5/T7 changes |
