"""09 — Recurrent (GRU / LSTM) agent: walk-forward training + ATR-bracket grid.

Standard recurrent deep-learning baseline requested by the thesis advisor, built as a *real
candidate agent* in the notebooks_v2 -> mas07 pipeline (binary P(up) on Triple-Barrier labels).

This is a faithful clone of ``02_mamba_v1`` (same TBM labels, expanding-window WFO with 3-month
retrain, SEQ_LEN=24, focal/weighted-CE loss, fold-local percentile signal transform, identical
ATR-bracket trading grid and fee model). The *only* substantive change is the encoder:
``MambaClassifier`` -> ``RecurrentClassifier`` (``nn.GRU`` or ``nn.LSTM`` selected by ``CELL_TYPE``),
so the comparison is "a standard recurrent net on the same inputs and the same execution".

Run locally (CPU / CUDA / Apple MPS):

    uv run python -m hmats.notebooks_v2._rnn_train --cell GRU
    uv run python -m hmats.notebooks_v2._rnn_train --cell LSTM

Artifacts -> ``artifacts/notebooks_v2/09_gru`` and ``09_lstm`` (the standard contract consumed by
``hmats.mas.mas07``: ``wfo_probs.npy`` / ``wfo_index.npy`` / ``oos_probs.npy`` / ``oos_index.npy`` /
``results.json`` with ``best_params``).
"""
from __future__ import annotations

import argparse
import calendar
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import QuantileTransformer
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# ── Repo / data ───────────────────────────────────────────────────────────────
def _repo_root() -> Path:
    p = Path.cwd()
    while p != p.parent:
        if (p / "pyproject.toml").exists():
            return p
        p = p.parent
    raise RuntimeError("pyproject.toml not found")


REPO = _repo_root()
UNIFIED = REPO / "data" / "features" / "BTCUSDT_1h_unified.parquet"

# ── OOS / WFO config (unified across notebooks_v2 — identical to 02_mamba) ─────
GRID_VAL_START = pd.Timestamp("2023-06-01")
GRID_VAL_END = pd.Timestamp("2024-05-31")
OOS_START = pd.Timestamp("2024-06-01")

RETRAIN_MONTHS = 3
TRAIN_START = pd.Timestamp("2019-01-01")
WFO_SCHEME = "expanding"
VAL_FRAC = 0.15
EMBARGO_H = 24

SEQ_LEN = 24
STRIDE = 1
BATCH = 1024
EPOCHS = 40
PATIENCE = 8
LR = 3e-4
FOCAL_GAMMA = 0.0  # weighted CE (TBM is balanced/noisy) — matches mamba

# Recurrent architecture (kept comparable to mamba: d_model=64, 2 layers)
D_MODEL = 64
N_LAYERS = 2
DROPOUT = 0.2

# Fee model — identical to the base agents' backtests
MAKER_FEE = 0.0
SPOT_TAKER_FEE = 0.0005
FUTURES_TAKER_FEE = 0.0005
BUFFER = 0.0005
SHORT_FUNDING_H = 0.0000077

LABEL_COL = "y_dir"
TBM_VOL_WINDOW = 24
TBM_MULT = 2.0
TBM_VERT_H = 24
EXCLUDE_COLS = {"open", "high", "low", "close", "volume", "label", "y_dir",
                "mkt_total_mcap_chg_24h"}

TRADING_GRID = {
    "signal_quantile": [0.020, 0.030, 0.050, 0.075, 0.100, 0.150, 0.200],
    "trade_direction": ["both", "long_only", "short_only"],
    "entry_atr_mult": [0.3, 0.6, 1.0],
    "sl_atr_mult": [1.5, 2.0, 2.5],
    "tp_atr_mult": [2.0, 2.5, 3.0],
    "min_sl": [0.010, 0.015],
    "min_hold": [4, 8],
    "max_hold": [24, 48],
    "cooldown": [2, 3],
}

ANN = np.sqrt(24 * 365)


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = _device()


# ── Model ─────────────────────────────────────────────────────────────────────
class RecurrentClassifier(nn.Module):
    """Standard recurrent direction classifier: feature projection -> GRU/LSTM stack ->
    last hidden state -> 2-class head. ``cell_type`` selects the recurrent cell."""

    def __init__(self, n_features: int, cell_type: str = "GRU",
                 d_model: int = D_MODEL, n_layers: int = N_LAYERS, dropout: float = DROPOUT):
        super().__init__()
        self.cell_type = cell_type.upper()
        self.input_proj = nn.Linear(n_features, d_model)
        rnn_cls = nn.GRU if self.cell_type == "GRU" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=d_model, hidden_size=d_model, num_layers=n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, 2)

    def forward(self, x):
        # Reference (documented) baseline uses a tanh on the projected inputs. The interactive
        # notebook 09_rnn_train_v1 exposes this as a toggle (tanh / layernorm / none) for
        # experimentation; LayerNorm and 'none' were tried and did not beat this on OOS.
        x = torch.tanh(self.input_proj(x))
        out, _ = self.rnn(x)
        return self.head(self.dropout(out[:, -1, :]))


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha)

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none", weight=self.alpha)
        return ((1 - torch.exp(-ce)) ** self.gamma * ce).mean()


# ── Data prep ─────────────────────────────────────────────────────────────────
def _tbm_labels(df: pd.DataFrame) -> np.ndarray:
    logc = np.log(df["close"])
    rvol = logc.diff().rolling(TBM_VOL_WINDOW).std().values
    close = df["close"].values; high = df["high"].values; low = df["low"].values
    n = len(df)
    y = np.full(n, np.nan, dtype=np.float32)
    for i in range(n):
        s = rvol[i]
        if np.isnan(s) or s == 0:
            continue
        sig = s * close[i]
        up = close[i] + TBM_MULT * sig; dn = close[i] - TBM_MULT * sig
        end_j = min(i + TBM_VERT_H, n - 1)
        for j in range(i + 1, end_j + 1):
            hit_up = high[j] >= up; hit_dn = low[j] <= dn
            if hit_up and hit_dn:
                y[i] = 1 if close[j] >= close[i] else 0
                break
            if hit_up:
                y[i] = 1; break
            if hit_dn:
                y[i] = 0; break
        else:
            y[i] = -1
    return y


def build_sequences(X, y, seq_len, stride=1):
    ends = np.arange(seq_len - 1, len(X), stride)
    valid = np.isin(y[ends], (0, 1))
    ends = ends[valid]
    return (np.stack([X[e - seq_len + 1:e + 1] for e in ends], 0).astype(np.float32),
            y[ends].astype(np.int64))


def make_focal(y_arr):
    counts = np.bincount(y_arr.astype(int), minlength=2)
    alpha = torch.tensor(1.0 / (counts + 1), dtype=torch.float32, device=DEVICE)
    return FocalLoss(FOCAL_GAMMA, alpha / alpha.sum())


def train_fold(X_tr_s, y_tr_s, X_vl_s, y_vl_s, n_feat, cell_type):
    ld_tr = DataLoader(TensorDataset(torch.from_numpy(X_tr_s), torch.from_numpy(y_tr_s)),
                       BATCH, shuffle=True, drop_last=True)
    ld_vl = DataLoader(TensorDataset(torch.from_numpy(X_vl_s), torch.from_numpy(y_vl_s)),
                       BATCH, shuffle=False)
    model = RecurrentClassifier(n_feat, cell_type=cell_type).to(DEVICE)
    crit = make_focal(y_tr_s)
    opt = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = SequentialLR(opt, [LinearLR(opt, 0.1, 1.0, total_iters=5),
                               CosineAnnealingLR(opt, T_max=EPOCHS - 5, eta_min=1e-6)],
                         milestones=[5])
    best_auc = 0.0; best_state = None; best_ep = 0
    for ep in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in ld_tr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval(); ps = []; ys = []
        with torch.no_grad():
            for xb, yb in ld_vl:
                ps.append(torch.softmax(model(xb.to(DEVICE)), -1)[:, 1].cpu().numpy())
                ys.append(yb.numpy())
        p = np.concatenate(ps); yv = np.concatenate(ys)
        auc = roc_auc_score(yv, p) if 0 < yv.sum() < len(yv) else 0.5
        if auc > best_auc:
            best_auc = auc; best_ep = ep
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ep - best_ep >= PATIENCE:
            break
    model.load_state_dict(best_state)
    return model, best_auc, best_ep


def _predict_proba(model, X_np, batch=2048):
    model.eval(); preds = []
    with torch.no_grad():
        for ch in torch.split(torch.from_numpy(X_np), batch):
            preds.append(torch.softmax(model(ch.to(DEVICE)), -1)[:, 1].cpu().numpy())
    return np.concatenate(preds) if preds else np.array([], dtype=np.float32)


def _fold_percentile_score(preds, ref_preds):
    ref = np.sort(ref_preds[np.isfinite(ref_preds)])
    if len(ref) < 50:
        return np.full_like(preds, np.nan, dtype=np.float32)
    return (np.searchsorted(ref, preds, side="right") / len(ref)).astype(np.float32)


# ── ATR-bracket backtester (identical logic + fees to mamba/base agents) ───────
def _run_backtest(probs_arr, close_arr, high_arr, low_arr, atr_arr, long_threshold,
                  short_threshold, entry_atr_mult, sl_atr_mult, tp_atr_mult, min_sl,
                  min_hold, max_hold, cooldown, trade_direction="both", with_fees=True):
    n = len(close_arr); eq = np.ones(n); cur = 1.0; trades = []
    in_pos = False; direction = None
    entry_px = sl_px = tp_px = pos_eq = entry_fee = 0.0
    hold_cnt = cd_cnt = 0; funding = 0.0; pending = None
    for i in range(n):
        lo = low_arr[i]; hi = high_arr[i]; px = close_arr[i]
        if in_pos:
            hold_cnt += 1
            if direction == "short":
                funding += SHORT_FUNDING_H
            eq[i] = pos_eq * (px / entry_px if direction == "long" else 1 + (entry_px - px) / entry_px)
            exited = False; exit_px = 0.0; reason = ""; exit_fee = 0.0
            if hold_cnt >= min_hold:
                if direction == "long":
                    if lo <= sl_px:
                        exit_px = sl_px; exited = True; reason = "sl"; exit_fee = SPOT_TAKER_FEE if with_fees else 0.0
                    elif hi >= tp_px:
                        exit_px = tp_px; exited = True; reason = "tp"; exit_fee = MAKER_FEE
                    elif hold_cnt >= max_hold:
                        exit_px = px; exited = True; reason = "timeout"; exit_fee = SPOT_TAKER_FEE if with_fees else 0.0
                else:
                    if hi >= sl_px:
                        exit_px = sl_px; exited = True; reason = "sl"; exit_fee = FUTURES_TAKER_FEE if with_fees else 0.0
                    elif lo <= tp_px:
                        exit_px = tp_px; exited = True; reason = "tp"; exit_fee = MAKER_FEE
                    elif hold_cnt >= max_hold:
                        exit_px = px; exited = True; reason = "timeout"; exit_fee = FUTURES_TAKER_FEE if with_fees else 0.0
            if exited:
                gross = ((exit_px - entry_px) / entry_px if direction == "long"
                         else (entry_px - exit_px) / entry_px)
                net = gross - (entry_fee + exit_fee if with_fees else 0.0) - funding
                cur = pos_eq * (1.0 + net); eq[i] = cur
                trades.append({"direction": direction, "reason": reason, "gross": gross,
                               "net": net, "hold": hold_cnt})
                in_pos = False; cd_cnt = cooldown; funding = 0.0
        elif pending is not None:
            d, lim, p_sl, p_tp = pending
            if d == "long":
                filled = lo <= lim + BUFFER
                ef = MAKER_FEE if (filled and with_fees) else (SPOT_TAKER_FEE if with_fees else 0.0)
            else:
                filled = hi >= lim - BUFFER
                ef = MAKER_FEE if (filled and with_fees) else (FUTURES_TAKER_FEE if with_fees else 0.0)
            entry_px = lim if filled else px; sl_px = p_sl; tp_px = p_tp; entry_fee = ef
            direction = d; in_pos = True; pos_eq = cur; hold_cnt = 0; funding = 0.0
            pending = None; eq[i] = cur
        elif cd_cnt > 0:
            cd_cnt -= 1; eq[i] = cur
        elif not np.isnan(probs_arr[i]) and i + 1 < n:
            atr = max(atr_arr[i], min_sl)
            go_long = trade_direction in ("both", "long_only") and probs_arr[i] > long_threshold
            go_short = trade_direction in ("both", "short_only") and probs_arr[i] < short_threshold
            if go_long:
                pending = ("long", px * (1 - entry_atr_mult * atr), px * (1 - sl_atr_mult * atr),
                           px * (1 + tp_atr_mult * atr))
            elif go_short:
                pending = ("short", px * (1 + entry_atr_mult * atr), px * (1 + sl_atr_mult * atr),
                           px * (1 - tp_atr_mult * atr))
            eq[i] = cur
        else:
            eq[i] = cur
    if in_pos:
        gross = ((px - entry_px) / entry_px if direction == "long" else (entry_px - px) / entry_px)
        taker = SPOT_TAKER_FEE if direction == "long" else FUTURES_TAKER_FEE
        net = gross - (entry_fee + (taker if with_fees else 0.0)) - funding
        cur = pos_eq * (1.0 + net); eq[-1] = cur
    return eq, trades


def _sharpe(eq):
    r = np.diff(np.log(np.maximum(eq, 1e-12)))
    return float(r.mean() / (r.std(ddof=1) + 1e-12) * ANN)


def _maxdd(eq):
    pk = np.maximum.accumulate(eq)
    return float(((eq - pk) / (pk + 1e-12)).min())


def _adaptive_thresholds(q):
    q = float(q)
    return 1.0 - q, q


MAX_TRADES_GRID = 260
MIN_TRADES_GRID = 40
TARGET_TRADES_GRID = 120


def run(cell_type: str, save: bool = True, verbose: bool = True) -> dict:
    import itertools

    cell_type = cell_type.upper()
    arts = REPO / "artifacts" / "notebooks_v2" / ("09_gru" if cell_type == "GRU" else "09_lstm")
    arts.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(UNIFIED)
    df.index = df.index.tz_localize(None) if df.index.tz else df.index
    feats = [c for c in df.columns
             if c not in EXCLUDE_COLS and pd.api.types.is_numeric_dtype(df[c])]
    if verbose:
        print(f"[{cell_type}] device={DEVICE} | {df.shape} | {len(feats)} features")

    t0 = time.time()
    df["y_dir"] = _tbm_labels(df)
    y_all = df[LABEL_COL].values.astype(float)
    X_raw = np.nan_to_num(df[feats].values.astype(np.float32))
    if verbose:
        up = int((y_all == 1).sum()); dn = int((y_all == 0).sum())
        print(f"[{cell_type}] TBM labels in {time.time()-t0:.0f}s | Up={up:,} Down={dn:,}")

    # ── Walk-forward ──────────────────────────────────────────────────────────
    probs_raw = np.full(len(df), np.nan)
    probs = np.full(len(df), np.nan)
    anchors = []
    d = pd.Timestamp("2023-06-01")
    while d <= df.index[-1]:
        anchors.append(d); d += pd.DateOffset(months=RETRAIN_MONTHS)

    best_model = None; fold_log = []; t_all = time.time()
    for fi, a_start in enumerate(anchors):
        a_end = a_start + pd.DateOffset(months=RETRAIN_MONTHS)
        tr_cut = a_start - pd.Timedelta(hours=EMBARGO_H)
        tr_begin = TRAIN_START
        tr_mask = (df.index >= tr_begin) & (df.index < tr_cut)
        if tr_mask.sum() < 2000:
            continue
        tr_idx = np.where(tr_mask)[0]
        n_val = int(len(tr_idx) * VAL_FRAC)
        if n_val < 200 or len(tr_idx) < n_val + EMBARGO_H + 500:
            continue
        tr_rows = tr_idx[:-(n_val + EMBARGO_H)]
        vl_rows = tr_idx[-n_val:]

        qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal", random_state=42)
        qt.fit(X_raw[tr_rows])
        X_scaled = qt.transform(X_raw)

        X_tr_s, y_tr_s = build_sequences(X_scaled[tr_rows[0]:tr_rows[-1] + 1],
                                         y_all[tr_rows[0]:tr_rows[-1] + 1], SEQ_LEN, STRIDE)
        X_vl_s, y_vl_s = build_sequences(X_scaled[vl_rows[0] - SEQ_LEN + 1:vl_rows[-1] + 1],
                                         y_all[vl_rows[0] - SEQ_LEN + 1:vl_rows[-1] + 1], SEQ_LEN, 1)
        if len(y_tr_s) < 500 or len(np.unique(y_tr_s)) < 2:
            continue

        model, auc, ep = train_fold(X_tr_s, y_tr_s, X_vl_s, y_vl_s, len(feats), cell_type)

        oos_mask = (df.index >= a_start) & (df.index < a_end)
        oos_rows = np.where(oos_mask)[0]; oos_rows = oos_rows[oos_rows >= SEQ_LEN - 1]
        if len(oos_rows):
            val_ref = _predict_proba(model, X_vl_s)
            Xo = np.stack([X_scaled[e - SEQ_LEN + 1:e + 1] for e in oos_rows]).astype(np.float32)
            raw_pred = _predict_proba(model, Xo)
            probs_raw[oos_rows] = raw_pred
            probs[oos_rows] = _fold_percentile_score(raw_pred, val_ref)
        best_model = model
        fold_log.append({"fold": fi, "val_auc": round(auc, 4), "ep": ep,
                         "train": f"{tr_begin.date()}->{tr_cut.date()}",
                         "oos": f"{a_start.date()}->{a_end.date()}"})
        if verbose:
            print(f"  fold {fi}: train[{tr_begin.date()}->{tr_cut.date()}] "
                  f"OOS {a_start.date()}->{a_end.date()} auc={auc:.4f}@{ep} ({len(oos_rows)} bars)")
    if verbose:
        print(f"[{cell_type}] WFO done in {(time.time()-t_all)/60:.1f} min | {len(fold_log)} folds")

    oos_m = (df.index >= OOS_START) & (~np.isnan(probs_raw))
    auc_m = oos_m & np.isin(y_all, (0, 1))
    auc_oos_raw = float(roc_auc_score(y_all[auc_m], probs_raw[auc_m]))
    auc_oos_signal = float(roc_auc_score(y_all[auc_m], probs[auc_m]))
    oos_probs_series = pd.Series(probs, index=df.index)[df.index >= OOS_START]
    oos_df = df[df.index >= OOS_START].copy()
    if verbose:
        print(f"[{cell_type}] OOS AUC raw={auc_oos_raw:.4f} signal-rank={auc_oos_signal:.4f}")

    # ── Trading grid on grid-val window ───────────────────────────────────────
    grid_keys = list(TRADING_GRID)
    grid_combos = list(itertools.product(*TRADING_GRID.values()))
    gv_m = (df.index >= GRID_VAL_START) & (df.index <= GRID_VAL_END)
    gv_df = df[gv_m]; gv_p = probs[gv_m]
    valid = gv_p[~np.isnan(gv_p)]
    if len(valid) == 0:
        raise RuntimeError("Grid-val probs contain no finite values after WFO.")

    rows = []; skipped_low = skipped_high = 0; tg = time.time()
    for vals in grid_combos:
        p = dict(zip(grid_keys, vals))
        if p["max_hold"] < p["min_hold"]:
            continue
        long_th, short_th = _adaptive_thresholds(p["signal_quantile"])
        p_bt = {**p, "long_threshold": long_th, "short_threshold": short_th}
        del p_bt["signal_quantile"]
        eq, tr = _run_backtest(gv_p, gv_df["close"].values, gv_df["high"].values,
                               gv_df["low"].values, gv_df["atr_14_pct"].values,
                               with_fees=True, **p_bt)
        if len(tr) < MIN_TRADES_GRID:
            skipped_low += 1; continue
        if len(tr) > MAX_TRADES_GRID:
            skipped_high += 1; continue
        nl = sum(1 for t in tr if t["direction"] == "long"); ns = len(tr) - nl
        sharpe = _sharpe(eq); total_ret = float(eq[-1] - 1); maxdd = _maxdd(eq)
        sparse_penalty = 0.20 * max(0.0, (TARGET_TRADES_GRID - len(tr)) / TARGET_TRADES_GRID)
        turnover_penalty = 0.10 * max(0.0, (len(tr) - TARGET_TRADES_GRID) / (MAX_TRADES_GRID - TARGET_TRADES_GRID))
        dd_penalty = 0.50 * max(0.0, abs(maxdd) - 0.20)
        score = sharpe + 0.10 * np.tanh(4.0 * total_ret) - sparse_penalty - turnover_penalty - dd_penalty
        rows.append({**p_bt, "signal_quantile": p["signal_quantile"], "score": score,
                     "sharpe": sharpe, "total_ret": total_ret, "maxdd": maxdd,
                     "win_rate": float(np.mean([t["net"] > 0 for t in tr])),
                     "n_trades": len(tr), "n_long": nl, "n_short": ns})
    if not rows:
        raise RuntimeError(f"Trading grid: 0 valid combos (low={skipped_low}, high={skipped_high}).")
    grid_df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    INT = {"min_hold", "max_hold", "cooldown"}
    best = {k: (int(grid_df.iloc[0][k]) if k in INT else grid_df.iloc[0][k])
            for k in ["long_threshold", "short_threshold", "entry_atr_mult", "sl_atr_mult",
                      "tp_atr_mult", "min_sl", "min_hold", "max_hold", "cooldown", "trade_direction"]}
    for k, v in list(best.items()):
        if k not in INT and k != "trade_direction":
            best[k] = float(v)
    best_grid = {**best, "signal_quantile": float(grid_df.iloc[0]["signal_quantile"])}
    if verbose:
        print(f"[{cell_type}] grid done in {time.time()-tg:.0f}s | {len(grid_df)} combos | best={best_grid}")

    # ── OOS backtest ──────────────────────────────────────────────────────────
    o_p = oos_probs_series.values
    eq_fees, tdf_fees = _run_backtest(o_p, oos_df["close"].values, oos_df["high"].values,
                                      oos_df["low"].values, oos_df["atr_14_pct"].values,
                                      with_fees=True, **best)
    eq_0fee, tdf_0fee = _run_backtest(o_p, oos_df["close"].values, oos_df["high"].values,
                                      oos_df["low"].values, oos_df["atr_14_pct"].values,
                                      with_fees=False, **best)
    n_years = (oos_df.index[-1] - oos_df.index[0]).days / 365.25
    ann_ret = float((eq_fees[-1]) ** (1.0 / n_years) - 1.0)
    TF = pd.DataFrame(tdf_fees) if tdf_fees else pd.DataFrame(columns=["direction", "net"])

    def _bt_metrics(eq, t):
        wr = float((t["net"] > 0).mean()) if len(t) else 0.0
        nl = int((t["direction"] == "long").sum()) if len(t) else 0
        ns = int((t["direction"] == "short").sum()) if len(t) else 0
        return {"n_trades": len(t), "n_long": nl, "n_short": ns, "win_rate": round(wr, 4),
                "total_ret": round(float(eq[-1] - 1), 4), "ann_ret": round(ann_ret, 4),
                "sharpe": round(_sharpe(eq), 4), "maxdd": round(_maxdd(eq), 4)}

    m_fees = _bt_metrics(eq_fees, TF)
    if verbose:
        print(f"[{cell_type}] OOS w/fees: ret={m_fees['total_ret']:+.1%} "
              f"ann={ann_ret:+.1%} sharpe={m_fees['sharpe']:.2f} maxdd={m_fees['maxdd']:.1%} "
              f"trades={m_fees['n_trades']}")

    results = {
        "notebook": "09_rnn_v1", "created": pd.Timestamp.now().isoformat(),
        "model": f"Recurrent classifier ({cell_type}, {N_LAYERS}x{D_MODEL})",
        "cell_type": cell_type,
        "wfo": {"type": WFO_SCHEME, "train_start_floor": str(TRAIN_START.date()),
                "retrain_months": RETRAIN_MONTHS, "folds": fold_log},
        "signal_transform": "fold_validation_percentile_score",
        "grid_val": f"{GRID_VAL_START.date()}->{GRID_VAL_END.date()}",
        "oos_period": f"{OOS_START.date()}->{oos_df.index[-1].date()}",
        "oos_auc": round(auc_oos_raw, 4), "oos_auc_raw": round(auc_oos_raw, 4),
        "oos_auc_signal_rank": round(auc_oos_signal, 4),
        "architecture": {"cell_type": cell_type, "d_model": D_MODEL, "n_layers": N_LAYERS,
                         "seq_len": SEQ_LEN, "batch": BATCH, "dropout": DROPOUT},
        "selected_features": feats,
        "best_params": best_grid,
        "backtest_wfees": m_fees,
        "backtest_0fee": _bt_metrics(eq_0fee, pd.DataFrame(tdf_0fee) if tdf_0fee else pd.DataFrame(columns=["direction", "net"])),
        "annualised_return_wfees": round(ann_ret, 4),
        "decision_gate_10pct": bool(ann_ret >= 0.10),
    }

    if save:
        np.save(arts / "oos_probs.npy", oos_probs_series.values.astype(np.float32))
        np.save(arts / "oos_probs_raw.npy",
                pd.Series(probs_raw, index=df.index)[df.index >= OOS_START].values.astype(np.float32))
        np.save(arts / "oos_index.npy", oos_df.index.astype("datetime64[ns]").astype(np.int64).values)
        np.save(arts / "wfo_probs.npy", probs.astype(np.float32))
        np.save(arts / "wfo_probs_raw.npy", probs_raw.astype(np.float32))
        np.save(arts / "wfo_index.npy", df.index.astype("datetime64[ns]").astype(np.int64).values)
        if best_model is not None:
            torch.save(best_model.state_dict(), arts / "model_lastfold.pt")
        grid_df.to_csv(arts / "trading_grid_leaderboard.csv", index=False)
        with open(arts / "results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        # equity arrays for downstream plotting / the notebook
        np.save(arts / "oos_equity_wfees.npy", eq_fees.astype(np.float64))
        if verbose:
            print(f"[{cell_type}] artifacts -> {arts}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", choices=["GRU", "LSTM", "gru", "lstm"], default="GRU")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(42); np.random.seed(42)
    run(args.cell, save=not args.no_save)
