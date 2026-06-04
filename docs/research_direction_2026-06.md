# Research Direction — Hybrid Multi-Agent Trading System (2026-06-04)

Grounded review of the current agent stack, the `meta_labeling_foundation.txt`
blueprint, and the third-party critique of it. Numbers below are from the
committed `artifacts/*/results.json` (OOS 2024-01-01 → 2026-05-16, ~29 months,
with the real fee model).

## 1. Where we actually are

| Agent | OOS return (fees) | Sharpe | Zero-fee return | Trades | Verdict |
|---|---|---|---|---|---|
| **LGBM v12** | **+59.1%** | **+0.62** | +105% | 416 | **Only edge in the system** |
| DRL PPO v3 | −93.0% | −3.12 | **−17.2%** | 2484 | Negative edge *before* fees |
| GP v2 | −96.5% | −3.82 | **−71.1%** | 2151 | Negative edge *before* fees |
| TCN v1 | (no results.json) | — | — | — | Unmeasured |

The single most important fact: **DRL and GP lose money even at zero fees**
(−17% and −71%). Fees turn a bad strategy catastrophic, but they are not the root
cause. The root cause is that both make a *fresh directional bet roughly every bar
on an hourly series whose Hurst ≈ 0.52* (near-random). The foundation doc is right
about the diagnosis; it is wrong that its prescribed fixes are sufficient — see §3.

LGBM works for a structural reason the other two lack: it does **not** trade per
bar. It fires an entry, then holds inside an ATR bracket (`min_hold=8`,
`max_hold=48`, TP=2.5·ATR / SL=1.5·ATR), and it is 408-long / 8-short — it mostly
rides a bull market with disciplined exits. The holding mechanism *is* the edge.

## 2. The meta-labeling architecture is currently miswired

`build_signal_df()` defines the primary signal as **`lgbm | tcn | drl | gp`
(logical OR)**. Three of the four sources fire almost every bar and are
money-losing. So:

- The primary-signal set is dominated by DRL/GP noise. The meta-model's job
  collapses to "veto ~95% of incoming signals," which is a very hard ask and
  throws away the one clean signal (LGBM) by drowning it in the union.
- `primary_side` is the OR-consensus direction — mostly the losing agents' votes.
  We are meta-labeling a near-random primary, which violates the premise of
  meta-labeling (it presumes a primary with real recall that you sharpen for
  precision).

**Meta-labeling only makes sense on top of a primary that already has positive
expectancy. Today that is LGBM, and only LGBM.**

## 3. Why the foundation doc's fixes were tried and still failed

The DRL v3 and GP v2 runs already applied the headline prescriptions, yet results
got no better:

- **DRL: `min_hold=6` was applied.** Still 2484 trades, still −17% at zero fees.
  `min_hold` only blocks a flip *while in a position*; it does nothing about the
  policy having no directional edge at 1h, and Short↔Long reversals through the
  gate remain cheap. The agent is optimizing a per-step PnL reward over an H≈0.52
  process — there is no stable gradient to climb.
- **GP: `logic_only=True` and `flat_threshold=0.5` were applied.** Still 2151
  trades, avg hold 5.6h, −71% at zero fees. In-sample fold fitness was **38–42**
  (annualized Sharpe ~40) against an OOS Sharpe of −3.8 — a textbook overfit
  signature. The two fixes that would actually bite were **not** applied:
  fee-in-fitness and a structural exit. The GP objective
  (`strategy_rets = signals * log_rets`, recomputed every bar, no cost term)
  literally rewards catching every intrabar wiggle, so evolution finds trees that
  flip constantly in-sample and die OOS.

Correcting my own first guess on the record: I suspected GP's `flat_threshold`
was defeated by unnormalized features (rsi 0–100). It is not — the GP feature diet
is already scaled to ~[0,1], and tree outputs sit in {−1,0,1}, so the threshold
behaves. The defect is the *objective*, not the scaling.

## 4. The third-party critique of the foundation doc — verdict

The critique's three "landmines" are technically correct but two are moot here:

1. **ApEn is O(N²) and self-biased → use Sample Entropy.** Correct and worth
   doing. Implemented in `src/hmats/features/microstructure.py` (`sample_entropy`,
   `rolling_sample_entropy`), numpy-vectorized since numba is not in this env.
2. **VPIN can't be computed on time bars → use taker imbalance.** Correct — and we
   already have it. `data/raw/BTCUSDT_1h_taker.parquet` carries the real
   taker-buy/-sell split, and the V4 feature set already ships `tfi_*`
   (taker-flow-imbalance). The critique's author didn't know these exist.
   `compute_volume_imbalance()` is provided for OHLCV-only notebooks.
3. **Fractional differentiation can leak future info → causal weights only.**
   Correct in principle; we already ship `fracdiff_close_d0.2` in V4. Action item:
   *verify* that column was built with a strictly one-sided window (audit the
   generator), don't re-derive it.

Net: the genuinely new, useful artifacts are Sample Entropy and Roll/Amihud, all
now implemented causally in `microstructure.py`. VPIN and fracdiff are already
covered by existing features.

## 5. Recommended direction (in priority order)

1. **Rewire meta-labeling onto LGBM as the sole primary (side) model.** Drop the
   OR-union. Meta-model predicts P(LGBM entry hits TP before SL) from regime +
   microstructure context, and outputs veto/size. The honest bar to clear is the
   standalone LGBM (+59% / Sharpe 0.62) — the meta-layer must beat that on
   risk-adjusted terms or it is dead weight. Use DRL/GP/TCN outputs as *context
   features* of the meta-model, never as primary triggers.
2. **Add proper purging + embargo to the meta WFO.** TBM labels are
   path-dependent over `max_hold`; training events within `max_hold` of an OOS
   boundary leak. `MetaSupervisoryAgent.run_wfo` currently has no purge between
   `X_tr` and `step_start`. This is a real leakage bug, not a nicety.
3. **Either fix GP/DRL structurally or retire them as signal sources.** If kept,
   both must adopt LGBM's regime: signal = *entry trigger only*, exit governed by
   ATR brackets + min-hold; and GP's fitness must include the fee term. Without
   that they cannot stop churning. Given the zero-fee negative edge, retiring them
   to context-only is the higher-EV default.
4. **Consider a 4h timeframe variant for the trend-following sleeve.** The doc's
   Hurst argument (H≈0.64 daily vs ≈0.52 hourly) is sound; 4h quarters the
   decision count and the fee surface. Worth one experiment, but secondary to
   fixing the architecture above.
5. **Add the four `microstructure.py` features to the meta-model candidate pool**
   and let the LGBM meta-classifier's importance filter decide. They are cheap,
   causal, and orthogonal to the existing tabular diet (entropy is a
   forward-looking volatility proxy; Amihud/imbalance are liquidity-regime gates).

## 6. Concrete next experiment

`08_meta_learning_v2`: LGBM-primary meta-labeling.
- Primary side = LGBM v12 long/short triggers only.
- Meta-features = `lgbm_p_up`, `tcn_p_up/down`, `drl_action`, `gp_action` (as
  context), plus `hurst_24h/72h`, `tfi_*`, `fracdiff_close_d0.2`, and the four new
  `roll_measure_50 / amihud_50 / vol_imbalance_50 / sampen_48`.
- Purged + embargoed WFO (embargo = `max_hold`).
- Success criteria: OOS Sharpe > 0.62 **and** approved-trade rate < primary-signal
  rate (i.e. the meta-model actually vetoes), **and** meta-AUC > 0.52 on held-out
  signals.

If the LGBM-primary meta-layer cannot beat standalone LGBM, the conclusion is that
the ensemble adds nothing at 1h and the thesis contribution is the
single-model + disciplined-exit result, not the multi-agent supervisor.
