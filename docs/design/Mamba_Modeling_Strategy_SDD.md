# Mamba Modeling Strategy — Software Design Document

**Task ID:** `MODEL-MAMBA-ZERO-FEE-005`  
**Notebook:** `src/hmats/notebooks/03_mamba_omni_0fee_v1.ipynb`  
**Version:** 0.1 — Draft  
**Date:** 2026-05-28  

---

## 1. Executive Summary

This SDD describes the design rationale for the first clean-slate Mamba (Selective State Space Model) experiment on 5-minute BTCUSDT structural data.

**Core research question:**  
Does reading the full chronological sequence of 288 bars (24 hours) provide a predictive edge that LGBM's single-row feature lookups cannot capture?

Mamba reads a 24-hour window before every prediction. Theoretically, it can recognize temporal patterns across the full session:

- Volatility compression → explosion sequences (squeeze buildup)
- Progressive volume absorption into a level (institutional accumulation)
- Failed breakout → reversal fingerprints
- Session-boundary momentum shifts (London/NY overlap)

The LGBM baseline (V1 AUC = 0.52 on clean features, V2 AUC ≈ 0.52 across 3 targets) provides the null hypothesis: if Mamba cannot beat this with sequence context, the 5-minute structural feature set likely lacks causal edge regardless of model class.

---

## 2. Architecture

### 2.1 Fixed Hyperparameters (Frozen)

No architectural grid search. Every compute cycle goes toward evaluating feature subsets and target labels.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Sequence length | 288 bars (24h) | Matches Macro TBM vertical barrier; full session |
| d\_model | 64 | Minimal embedding; ~250k total params per model |
| d\_state | 8 | SSM hidden state channels per feature dimension |
| d\_inner | 128 (= d\_model × 2) | Standard Mamba expand ratio |
| n\_layers | 2 | Enough depth for hierarchical temporal abstraction |
| d\_conv | 4 | Depthwise causal conv for local context |
| dt\_rank | 4 (= d\_model // 16) | Rank of discretization projection |

Total trainable parameters per model: **≈ 250 000** — deliberately small.

### 2.2 Block Architecture (Pure PyTorch, CPU/MPS)

```
MambaClassifier
├── input_proj : Linear(n_features → 64)
├── MambaBlock × 2
│   ├── LayerNorm(64)               pre-norm
│   ├── in_proj : Linear(64 → 256)  x branch + z gate branch
│   ├── Conv1d(128, kernel=4)       causal depthwise conv
│   ├── SelectiveSSM(128, 8)        S6 scan
│   │   ├── A_log : Param(128, 8)   log-scale SSM matrix
│   │   ├── D     : Param(128)      skip connection
│   │   ├── x_proj: Linear(128→20)  dt_rank + 2×d_state projections
│   │   └── dt_proj: Linear(4→128)  discretization step
│   ├── SiLU gating
│   └── out_proj: Linear(128 → 64)
├── LayerNorm(64)
└── head : Linear(64 → 2)          binary logits [not-TP, TP]
```

**Why pure PyTorch (not `mamba-ssm`):** The `mamba-ssm` package requires CUDA-specific kernels (Triton) and is not compatible with Apple MPS or CPU-only environments. Our implementation replicates the S6 selective scan in standard PyTorch operations, running efficiently on MPS (Apple Silicon) with ~250 ms per training batch.

### 2.3 Inference Mode

At prediction time, the model processes a **sliding window**: for bar `t`, the input is bars `[t − 287, ..., t]`. The output probability at `t` is used as the trading signal. The window moves forward one bar at a time (stride = 1) during final backtest inference.

---

## 3. Tensor Reshaping Methodology

### 3.1 Sequence Windowing

Raw data shape: `(N_bars, N_features)` — 2D, one row per 5-minute bar.

**Training:** Sliding window with configurable stride to control dataset density.

```
stride = 144 bars (12h)    →   ~4,600 sequences for training split
```

Choosing stride = SEQ\_LEN // 2 produces sequences with 50% overlap between consecutive windows. This gives sufficient data volume without saturating the dataset with near-duplicate windows.

**Validation AUC monitoring (during training):** stride = 12 bars (1h) for fast epoch-level feedback.

**Final backtest inference:** stride = 1 bar — every possible bar in the validation set gets a prediction. Batched at 256 sequences per forward pass (no gradient computation).

### 3.2 Label Assignment

The label for each sequence is always the label of the **last bar in the window** (`window[−1]`). This is the bar for which the prediction is made. All bars `window[0..−2]` are context only.

### 3.3 Normalization Pipeline

```
1. Extract flat training feature matrix: X_train  shape (n_train_bars, n_features)
2. Fit QuantileTransformer(n_quantiles=1000, output_distribution='normal') on X_train
3. Transform X_train and X_val with the fitted scaler
4. Store scaler per (feature_subset, target) run for reproducibility
```

**Why QuantileTransformer:** Neural networks are sensitive to heavy-tailed distributions common in financial data (e.g., volume spikes). The QT maps each feature to an approximately Gaussian distribution using rank statistics, which is more robust to outliers than standard normalization and doesn't require clipping.

**Lookahead check:** The scaler is fit strictly on `X_train` (bars up to 2023-12-31). The validation split (2024-01-01 to 2024-12-31) is transformed using the train-fitted scaler — it sees no validation statistics.

---

## 4. Anti-Bias Measures

### 4.1 The Legacy Problem

The previous Mamba experiment produced a **72% short-side directional bias** — the model predicted "market will go down" for nearly three-quarters of its trades. Root causes:

1. **Class imbalance ignored:** Macro TBM produces ~37% TP events (+1), ~26% SL events (−1), and ~37% timeouts (0). A model that labels everything "not TP" achieves ~63% accuracy and cross-entropy loss near the prior — but has zero edge.
2. **Symmetric loss:** Standard binary cross-entropy gives equal weight to easy (correctly classified, high-confidence) and hard examples, allowing the model to achieve low loss by memorizing the prior distribution.

### 4.2 Solutions Implemented

**Focal Loss (γ = 2.0):**

```
FL(p_t) = −(1 − p_t)^γ · log(p_t)
```

When the model is already confident and correct (`p_t → 1`), the factor `(1 − p_t)^2 → 0` suppresses that example's contribution. The model is forced to focus learning cycles on hard, uncertain examples near the decision boundary — exactly the bars where real predictive edge lives.

**Class-weighted Focal Loss:**

```
weight_positive  = n_negative / (n_positive + n_negative)
weight_negative  = n_positive / (n_positive + n_negative)
```

Applied as `alpha` in the Focal Loss. Compensates for the imbalanced positive rate (~37% for Macro TBM).

**Per-epoch bias monitoring:**

Each epoch reports:
- Mean predicted probability for TP bars (`y = 1`) vs non-TP bars (`y = 0`)
- Separation score: `mean(p|y=1) − mean(p|y=0)`  — healthy range: > 0.05
- Fraction of val predictions above threshold: should be << 100% (model shouldn't be firing everywhere)

**Gradient clipping:** `max_norm = 1.0` prevents exploding gradients from destabilizing early training.

---

## 5. Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (weight\_decay = 1e-4) |
| Initial LR | 3 × 10⁻⁴ |
| Schedule | Cosine Annealing (T\_max = MAX\_EPOCHS) with linear warmup (3 epochs) |
| Batch size | 256 |
| Max epochs | 50 |
| Early stopping patience | 10 epochs (val AUC metric) |
| Gradient clip | 1.0 (L2 norm) |
| Loss | Focal Loss (γ = 2.0) + class weights |

---

## 6. Experimental Plan

**8 training runs** = 4 feature subsets × 2 target labels.

### Feature Subsets

| ID | Name | Features | Count |
|----|------|----------|-------|
| M1 | Structure only | `struct_*` | 11 |
| M2 | Liquidity only | `liq_*` | 10 |
| M3 | Volatility only | `volat_*` | 8 |
| M4 | Omni | all structural + momentum | 43 |

### Target Labels

| ID | Name | Definition | Positive rate |
|----|------|------------|---------------|
| T1 | Macro TBM | TP = 2.5×ATR, SL = 1.5×ATR, 24h horizon | ≈ 37% |
| T2 | Fixed Horizon | close[t+72] / close[t] − 1 > 0.3% | ≈ 40–45% |

---

## 7. Evaluation Protocol

**Primary metric:** ROC-AUC on validation set (2024-01-01 to 2024-12-31, chronological).

**Bias diagnostics:**
- Separation score: `E[p | y=1] − E[p | y=0]`
- Calibration check: fraction of predictions above each threshold

**Zero-fee backtest:**
- Signal threshold: `p > 0.70` (primary), fallback to `p > 0.60` if < 5 signals
- No-overlap constraint: next trade starts only after current exits
- EV survival gate: `EV/trade > 0.40%` (4× Binance maker fee)

---

## 8. Results (populated after notebook execution)

| Run | AUC | n\_trades | Win % | EV/trade | Separation | Converge epoch |
|-----|-----|-----------|-------|----------|------------|----------------|
| M1\_macro\_tbm | — | — | — | — | — | — |
| M1\_fixed\_hor | — | — | — | — | — | — |
| M2\_macro\_tbm | — | — | — | — | — | — |
| M2\_fixed\_hor | — | — | — | — | — | — |
| M3\_macro\_tbm | — | — | — | — | — | — |
| M3\_fixed\_hor | — | — | — | — | — | — |
| M4\_macro\_tbm | — | — | — | — | — | — |
| M4\_fixed\_hor | — | — | — | — | — | — |

---

## 9. LGBM Comparison Baseline

Results from `02_lgbm_omni_0fee_v1.ipynb` and `02_lgbm_omni_0fee_v2.ipynb`:

| Metric | LGBM V1 (Macro TBM) | LGBM V2 (Micro TBM) | LGBM V2 (Fixed Hor.) | Mamba Target |
|--------|---------------------|---------------------|----------------------|--------------|
| Val AUC | 0.520 | — | — | > 0.54 |
| n\_trades (val) | 0 at p>0.75 | — | — | > 5/day |
| EV/trade | — | — | — | > 0.40% |

The LGBM models at AUC ≈ 0.52 produced 0 trades above `p > 0.65` threshold, indicating that the model probability mass was too compressed around the prior (≈ 0.37). If Mamba's sequence context produces better-calibrated, higher-confidence predictions on a subset of bars, it should generate tradeable signals at `p > 0.70` while LGBM cannot.

---

## 10. Next Steps

| Scenario | Implication |
|----------|-------------|
| Mamba AUC >> 0.52, EV > 0.40% | Proceed to `04_mamba_fees_wfo.ipynb` — real fees + walk-forward |
| Mamba AUC >> 0.52, EV < 0.40% | Use Mamba probabilities as meta-features in LGBM stacking model |
| Mamba AUC ≈ 0.52 | 5-minute structural features lack sequence-level edge; switch to 1h base timeframe |
| Mamba develops bias again | Lower γ (→ 1.5) or increase class weight ratio in FocalLoss |
