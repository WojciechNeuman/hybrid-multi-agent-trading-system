# Meta-Learning & Ensemble Analysis — 2026-06-11

Thorough analysis of why the 4-model ensemble + LightGBM meta-learner underperforms,
despite two strong base agents (TCN, PatchTST). Companion charts in `docs/diagrams/`,
action items in `docs/TASKS_meta_overhaul.md`.

## TL;DR

- The meta-agent's catastrophic **−100% wipeout was a capital-accounting bug** in the
  sizing backtest, now fixed (→ −4.1%). **Fixing it does not make the meta profitable.**
- The meta's **OOS AUC is 0.464 (< 0.5)** — it ranks trades *anti-*correctly out of sample.
  This is methodological, not a bug, and has three compounding causes:
  1. The meta **memorises the training regime** (top feature `halving_cycle_pos`, a
     monotonic time index, importance 357 ≈ 2× the next).
  2. Base agents are **short-biased** (grids tuned on a 2022→2024-05 bear/chop window),
     so the primary signal shorts into the 2024-11→2025-10 bull.
  3. The meta **throws away each agent's tuned trading rule** and re-derives signals from
     raw, label-inconsistent probabilities.
- LGBM is **not broken** — its drop vs "v12" is the harder OOS window, not a regression.
- The train/test *structure* the owner intended is essentially correct; the *content*
  (features, label consistency, regime coverage) is what fails.

## 1 · Current state (unified OOS 2024-05-31 → 2026-05-16)

| Model | Label | OOS AUC | Trades | L/S | Ret w/fees | Sharpe | MaxDD |
|---|---|---|---|---|---|---|---|
| 01 LGBM | direction | **0.536** | 214 | 132/82 | +12.0% | 0.24 | −17.0% |
| 02 Mamba | TBM | 0.532 | 314 | 70/244 | **−27.0%** | −0.52 | −47.4% |
| 03 TCN | TBM | 0.513 | 272 | 40/232 | **+73.8%** | **1.14** | −13.6% |
| 05 PatchTST | TBM | 0.509 | 129 | 0/129 | +36.1% | 0.87 | −10.7% |
| **04 Meta** | meta-label | **0.464** | 145 | — | −4.1% (was −100%) | −0.43 | −9.5% |

Standalone, TCN and PatchTST are genuinely good. The meta is *worse than every base model.*
A correct ensemble should not be able to do that — so the integration layer is the problem.

## 2 · Is the train/test design correct? (mostly yes)

The owner's intent — *base models emit p(up) over both a training period and the 2-year
test period; the meta trains on the training-period signals and is evaluated only on the
2-year window* — **is implemented correctly in structure**:

- `_load_model_probs` prefers each model's `wfo_probs.npy` (full walk-forward range) over
  `oos_probs.npy`. Verified coverage 2026-06-11:
  LGBM 2017-12→2026-05, **Mamba 2022-01→2026-05**, TCN & PatchTST 2023-01→2026-05.
- `META_TRAIN_START` auto-set to ~2022-02 (2nd-earliest model start +1mo); meta WFO is
  expanding, 3-month step, 48h embargo; evaluation filtered to `index ≥ 2024-05-31`.
- The base-period probs are genuine OOS (LGBM rolling WFO; TCN/Patch train ≤2022 then
  predict 2023+; Mamba sliding WFO) → **no obvious look-ahead leak into the meta**.

Two structural caveats, both fixable:
- The earlier meta run loaded Mamba from `oos_probs` (OOS-only) because its `wfo_probs`
  hadn't been refreshed — i.e. **ensemble composition silently differed between train
  (3 models) and test (4 models)**. With Mamba's `wfo_probs` now present this is resolved,
  but it must stay in sync (T2/T8).
- TCN/PatchTST contribute nothing before 2023, so 2022 meta-train bars see only LGBM(+Mamba).

## 3 · Root causes of meta AUC < 0.5

### 3a · Regime memorisation via monotonic time features
Meta feature importance (last fold): `halving_cycle_pos` **357**, `ret_24h` 180,
`bb_width_pct` 170, `hurst_168h` 169, … `patch_p_up` 128. The single most important
feature is a near-monotonic cycle position — the model learns *“in this slice of the
halving cycle, shorts paid”* and applies it to a different slice OOS. → **T7: drop
monotonic calendar features from the meta.**

### 3b · Short-biased primary signal (grid regime overfit)
All base grids are tuned on grid-val **2022-01→2024-05**, which is dominated by bear/chop.
Result: Mamba 244S/70L, TCN 232S/40L, PatchTST 129S/0L. The ensemble mean p(up) sits below
0.5, so the **primary signal shorts** — 2,312 shorts vs 962 longs over OOS. Those shorts get
run over by the 2024-11→2025-10 bull, so `meta_y` positive-rate is only 0.43 and the
prob→outcome relationship *flips* between train and test. → **T6.**

### 3c · The meta discards each agent's tuned edge + mixes label semantics
- TCN/PatchTST's profit comes from their **tuned `best_params`** (thresholds, ATR-scaled
  SL/TP, holds). The meta ignores `best_params` and rebuilds signals with one fixed rule
  (`ENS_LONG=0.56/SHORT=0.44`, `SL=1.5/TP=2.5`). The alpha never reaches the meta. → **T5.**
- `ensemble_p_up = mean(p)` averages a **next-bar-direction** prob (LGBM) with three
  **TBM ±2σ/24h** probs — different events, different base rates. Averaging them is
  semantically incoherent. → **T4.**

### 3d · (Fixed) sizing backtest capital bug
The sizing loop set `pos_eq = cur * pos_size` and then `cur = pos_eq*(1+net)`, so the
uninvested `(1-pos_size)` capital was deleted on every trade → equity → 0 regardless of
P&L. Fixed to apply `pos_size*net` to full equity (commit 2026-06-11). Wipeout −100% → −4.1%;
AUC unaffected (it was never a ranking issue).

## 4 · Why LGBM looks "worse than v12" (it isn't broken)

`01_lgbm_v1_as_in_v12` reported very high returns on OOS **2024-01-01 →** (a shorter window
ending before / early into the bear, and excluding the 2025-11+ bear leg). The current
`01_lgbm_v1` runs the **unified 2-year window 2024-05-31 → 2026-05-16** (full Chop+Bull+Bear)
and already uses v12's 11 Boruta-locked features + asymmetric grid (ported earlier this
session). Its OOS AUC **0.536** is in fact *higher* than the old table value (0.524); the
lower headline return is the **harder, longer test window**, not a regression. (Also note:
neither version uses the leaked `mkt_total_mcap_chg_24h`, so the old high numbers are not
leak-inflated — they're window-inflated.) **No code change required for parity; it's a
window definition difference, which is intentional and non-negotiable for the ensemble.**

## 5 · Why Mamba underperforms

- OOS AUC 0.532 is fine, but the grid (tuned on the bearish grid-val) chose a heavy-short
  configuration (244S/70L) that loses −27% through the bull (MaxDD −47%). Same regime-overfit
  mechanism as 3b, worst-affected.
- Mamba is **not torch-seeded** (only `random_state` for an unrelated LightGBM call), so its
  `wfo_probs` are non-reproducible run-to-run → the meta is non-reproducible. → **T2.**
- Recommendation: after seeding + de-biasing, re-evaluate; if still a net drag, keep
  `mamba_p_up` as a meta feature but drop it from the unweighted ensemble mean. → **T8.**

## 6 · Reproducibility audit

| Notebook | Seeds present | Gap |
|---|---|---|
| 01 LGBM | `random_state=42` | deterministic enough (single seed) |
| 02 Mamba | only LightGBM `random_state` | **torch model unseeded** → T2 |
| 03 TCN | SEED, random/np/torch, cudnn | DataLoader has no `generator` → T3 |
| 05 PatchTST | SEED, random/np/torch, cudnn | DataLoader has no `generator` → T3 |
| 04 Meta | LightGBM `random_state=42` | deterministic |

## 7 · What was changed this session

- **`04_meta_learning_v1.ipynb`** — wired in the 4th model (`PATCH_DIR`, loaded into
  `available`, added to overlay); **fixed the sizing capital bug** (T1). Re-ran locally.
- **`02_mamba_v1.ipynb` / `03_mamba_colab_a100.ipynb`** — `ARTS_DIR` now resolves to
  `artifacts/notebooks_v2/02_mamba` locally (Colab `/content` fallback) for folder unification.
- **`05_patchtst_v1.ipynb`** — new 4th base model (built + validated earlier this session).
- Docs: this file, `docs/diagrams/*`, `docs/TASKS_meta_overhaul.md`.

## 8 · Recommended order of work
T7 (drop monotonic meta features) and T5 (use tuned per-model signals) are the two changes
most likely to push meta AUC above 0.5 and turn the ensemble positive. Do T2/T3 first so
results are reproducible while iterating. See `docs/TASKS_meta_overhaul.md`.
