"""Builds src/hmats/notebooks_v2/09_rnn_train_v1.ipynb — a SELF-CONTAINED, hands-on training
notebook for the recurrent (GRU / LSTM) agent, meant to be run and tweaked interactively.

Everything you would want to experiment with is inline and editable (config, model architecture,
input normalisation, training loop, walk-forward scheme). The *shared, frozen* execution engine
(the ATR-bracket backtester, TBM labels, sequence builder, metrics, trading grid) is imported from
``hmats.notebooks_v2._rnn_train`` so it stays byte-identical to what every other agent uses — do not
re-implement those if you want the comparison to remain fair.

Run once to (re)generate the notebook:  ``uv run python -m hmats.notebooks_v2._build_rnn_train_nb``
"""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []
def md(s):   cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))

md(r"""# 09 — Train the recurrent (GRU / LSTM) agent yourself

This notebook trains a **standard recurrent network** as a candidate agent in the
`notebooks_v2 -> mas07` pipeline, on the **current** 285-feature unified panel — the same data,
Triple-Barrier labels, walk-forward scheme, and ATR-bracket execution as the Mamba agent. Only the
encoder changes.

**What is yours to tweak** (cells below are inline and editable):
- the **config** — cell type, architecture, input normalisation, sequence length, training budget;
- the **model** (`RecurrentClassifier`);
- the **training loop** and the **walk-forward** scheme.

**What stays frozen** (imported from `_rnn_train`, identical for every agent — changing it would make
the comparison unfair): the ATR-bracket backtester, the TBM labels, the sequence builder, the metric
definitions, and the trading grid.

**Decision gate (predeclared):** the agent is only promoted into the fund if its standalone OOS
return is **≥ 10%/year**. As of the last run both GRU and LSTM are *below* that (negative after
fees, OOS AUC ≈ 0.50–0.52), so they are reported as honest negatives. The signal is weak enough that
architecture tweaks mostly shuffle a near-random OOS outcome — keep an eye on **validation vs OOS**:
a high validation Sharpe that does not survive OOS is overfitting, not skill.
""")

md("## 1 · Config — edit me")
code(r"""# ── Model / training (tweak freely) ──────────────────────────────────────────
CELL_TYPE   = "GRU"      # "GRU" or "LSTM"
D_MODEL     = 64         # hidden width
N_LAYERS    = 2
DROPOUT     = 0.2
INPUT_NORM  = "tanh"     # projected-input handling: "tanh" (documented baseline) | "layernorm" | "none"
SEQ_LEN     = 24         # lookback window (bars)
EPOCHS      = 40
PATIENCE    = 8
LR          = 3e-4
BATCH       = 1024
SEED        = 42

# ── Walk-forward (tweak freely) ──────────────────────────────────────────────
WFO_SCHEME      = "expanding"   # "expanding" or "sliding"
RETRAIN_MONTHS  = 3
TRAIN_WINDOW_MONTHS = 36        # only used when WFO_SCHEME == "sliding"
VAL_FRAC        = 0.15
EMBARGO_H       = 24

import numpy as np, torch
torch.manual_seed(SEED); np.random.seed(SEED)
""")

md("## 2 · Imports + shared frozen engine")
code(r"""import json, time, warnings
from pathlib import Path
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import QuantileTransformer
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt, matplotlib as mpl

# Shared, FROZEN engine — identical for every agent (do not re-implement here):
from hmats.notebooks_v2 import _rnn_train as R
from hmats.notebooks_v2._rnn_train import (
    build_sequences, _tbm_labels, _run_backtest, _sharpe, _maxdd,
    _adaptive_thresholds, TRADING_GRID, EXCLUDE_COLS,
    GRID_VAL_START, GRID_VAL_END, OOS_START, TRAIN_START,
)
warnings.filterwarnings("ignore")
mpl.rcParams.update({"font.family":"serif","axes.spines.top":False,"axes.spines.right":False,
                     "figure.dpi":120})
ACCENT="#F7931A"; BLUE="#2962FF"; GREY="#9E9E9E"

DEVICE = R._device(); print("device:", DEVICE)
REPO = R.REPO
ARTS = REPO/"artifacts"/"notebooks_v2"/("09_gru" if CELL_TYPE=="GRU" else "09_lstm")
ARTS.mkdir(parents=True, exist_ok=True)
""")

md("## 3 · Data + Triple-Barrier labels")
code(r"""df = pd.read_parquet(R.UNIFIED)
df.index = df.index.tz_localize(None) if df.index.tz else df.index
FEATS = [c for c in df.columns if c not in EXCLUDE_COLS and pd.api.types.is_numeric_dtype(df[c])]
df["y_dir"] = _tbm_labels(df)
y_all = df["y_dir"].values.astype(float)
X_raw = np.nan_to_num(df[FEATS].values.astype(np.float32))
print(f"{df.shape} | {len(FEATS)} features | Up={int((y_all==1).sum()):,} Down={int((y_all==0).sum()):,}")
""")

md("## 4 · Model — edit me")
code(r"""class RecurrentClassifier(nn.Module):
    def __init__(self, n_features, cell_type=CELL_TYPE, d_model=D_MODEL,
                 n_layers=N_LAYERS, dropout=DROPOUT, input_norm=INPUT_NORM):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.input_norm = input_norm
        self.ln = nn.LayerNorm(d_model) if input_norm == "layernorm" else None
        rnn_cls = nn.GRU if cell_type.upper() == "GRU" else nn.LSTM
        self.rnn = rnn_cls(d_model, d_model, num_layers=n_layers, batch_first=True,
                           dropout=dropout if n_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, 2)

    def forward(self, x):
        x = self.input_proj(x)
        if self.input_norm == "layernorm": x = self.ln(x)
        elif self.input_norm == "tanh":    x = torch.tanh(x)
        out, _ = self.rnn(x)
        return self.head(self.dropout(out[:, -1, :]))

def make_focal(y_arr):  # weighted CE (FOCAL_GAMMA=0) — class-balanced
    counts = np.bincount(y_arr.astype(int), minlength=2)
    alpha = torch.tensor(1.0/(counts+1), dtype=torch.float32, device=DEVICE)
    return R.FocalLoss(0.0, alpha/alpha.sum())

n_params = sum(p.numel() for p in RecurrentClassifier(len(FEATS)).parameters())
print(f"{CELL_TYPE} params: {n_params:,} | input_norm={INPUT_NORM}")
""")

md("## 5 · Training loop — edit me")
code(r"""def train_fold(X_tr, y_tr, X_vl, y_vl, n_feat):
    ld_tr = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
                       BATCH, shuffle=True, drop_last=True)
    ld_vl = DataLoader(TensorDataset(torch.from_numpy(X_vl), torch.from_numpy(y_vl)),
                       BATCH, shuffle=False)
    model = RecurrentClassifier(n_feat).to(DEVICE)
    crit = make_focal(y_tr)
    opt = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = SequentialLR(opt, [LinearLR(opt, 0.1, 1.0, total_iters=5),
                               CosineAnnealingLR(opt, T_max=EPOCHS-5, eta_min=1e-6)], milestones=[5])
    best_auc=0.0; best_state=None; best_ep=0
    for ep in range(1, EPOCHS+1):
        model.train()
        for xb, yb in ld_tr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE); opt.zero_grad()
            loss = crit(model(xb), yb); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step()
        model.eval(); ps=[]; ys=[]
        with torch.no_grad():
            for xb, yb in ld_vl:
                ps.append(torch.softmax(model(xb.to(DEVICE)), -1)[:,1].cpu().numpy()); ys.append(yb.numpy())
        p=np.concatenate(ps); yv=np.concatenate(ys)
        auc = roc_auc_score(yv, p) if 0 < yv.sum() < len(yv) else 0.5
        if auc > best_auc: best_auc=auc; best_ep=ep; best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}
        if ep-best_ep >= PATIENCE: break
    model.load_state_dict(best_state)
    return model, best_auc, best_ep

def predict_proba(model, X_np, batch=2048):
    model.eval(); preds=[]
    with torch.no_grad():
        for ch in torch.split(torch.from_numpy(X_np), batch):
            preds.append(torch.softmax(model(ch.to(DEVICE)), -1)[:,1].cpu().numpy())
    return np.concatenate(preds) if preds else np.array([], dtype=np.float32)

def fold_percentile(preds, ref):
    ref = np.sort(ref[np.isfinite(ref)])
    if len(ref) < 50: return np.full_like(preds, np.nan, dtype=np.float32)
    return (np.searchsorted(ref, preds, side="right")/len(ref)).astype(np.float32)
""")

md("## 6 · Walk-forward training (this is the slow cell, ~3–4 min/cell type on MPS)")
code(r"""probs = np.full(len(df), np.nan); probs_raw = np.full(len(df), np.nan)
anchors=[]; d=pd.Timestamp("2023-06-01")
while d <= df.index[-1]: anchors.append(d); d += pd.DateOffset(months=RETRAIN_MONTHS)

best_model=None; fold_log=[]; t0=time.time()
for fi, a_start in enumerate(anchors):
    a_end = a_start + pd.DateOffset(months=RETRAIN_MONTHS)
    tr_cut = a_start - pd.Timedelta(hours=EMBARGO_H)
    tr_begin = TRAIN_START if WFO_SCHEME=="expanding" else max(TRAIN_START, a_start - pd.DateOffset(months=TRAIN_WINDOW_MONTHS))
    tr_mask = (df.index >= tr_begin) & (df.index < tr_cut)
    if tr_mask.sum() < 2000: continue
    tr_idx = np.where(tr_mask)[0]; n_val = int(len(tr_idx)*VAL_FRAC)
    if n_val < 200 or len(tr_idx) < n_val + EMBARGO_H + 500: continue
    tr_rows = tr_idx[:-(n_val+EMBARGO_H)]; vl_rows = tr_idx[-n_val:]
    qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal", random_state=42).fit(X_raw[tr_rows])
    Xs = qt.transform(X_raw)
    Xtr, ytr = build_sequences(Xs[tr_rows[0]:tr_rows[-1]+1], y_all[tr_rows[0]:tr_rows[-1]+1], SEQ_LEN, 1)
    Xvl, yvl = build_sequences(Xs[vl_rows[0]-SEQ_LEN+1:vl_rows[-1]+1], y_all[vl_rows[0]-SEQ_LEN+1:vl_rows[-1]+1], SEQ_LEN, 1)
    if len(ytr) < 500 or len(np.unique(ytr)) < 2: continue
    model, auc, ep = train_fold(Xtr, ytr, Xvl, yvl, len(FEATS))
    oos_rows = np.where((df.index >= a_start) & (df.index < a_end))[0]; oos_rows = oos_rows[oos_rows >= SEQ_LEN-1]
    if len(oos_rows):
        ref = predict_proba(model, Xvl)
        Xo = np.stack([Xs[e-SEQ_LEN+1:e+1] for e in oos_rows]).astype(np.float32)
        raw = predict_proba(model, Xo); probs_raw[oos_rows]=raw; probs[oos_rows]=fold_percentile(raw, ref)
    best_model=model; fold_log.append({"fold":fi,"val_auc":round(auc,4),"ep":ep})
    print(f"  fold {fi}: OOS {a_start.date()}->{a_end.date()} auc={auc:.4f}@{ep} ({len(oos_rows)} bars)")
print(f"WFO done in {(time.time()-t0)/60:.1f} min | {len(fold_log)} folds")

oos_m = (df.index >= OOS_START) & np.isfinite(probs_raw); auc_m = oos_m & np.isin(y_all,(0,1))
print(f"OOS AUC raw = {roc_auc_score(y_all[auc_m], probs_raw[auc_m]):.4f}")
""")

md("## 7 · Trading grid (frozen) → best ATR-bracket params, tuned on the validation window only")
code(r"""import itertools
gk = list(TRADING_GRID); gv_m=(df.index>=GRID_VAL_START)&(df.index<=GRID_VAL_END)
gv_df=df[gv_m]; gv_p=probs[gv_m]
rows=[]
for vals in itertools.product(*TRADING_GRID.values()):
    p=dict(zip(gk,vals))
    if p["max_hold"]<p["min_hold"]: continue
    lt,st=_adaptive_thresholds(p["signal_quantile"]); pbt={**p,"long_threshold":lt,"short_threshold":st}; del pbt["signal_quantile"]
    eq,tr=_run_backtest(gv_p,gv_df["close"].values,gv_df["high"].values,gv_df["low"].values,gv_df["atr_14_pct"].values,with_fees=True,**pbt)
    if not (40 <= len(tr) <= 260): continue
    sh=_sharpe(eq); ret=float(eq[-1]-1); dd=_maxdd(eq)
    score=sh+0.10*np.tanh(4*ret)-0.20*max(0,(120-len(tr))/120)-0.10*max(0,(len(tr)-120)/140)-0.50*max(0,abs(dd)-0.20)
    rows.append({**pbt,"signal_quantile":p["signal_quantile"],"score":score,"sharpe":sh,"total_ret":ret,"maxdd":dd,"n_trades":len(tr)})
grid=pd.DataFrame(rows).sort_values("score",ascending=False).reset_index(drop=True)
INTk={"min_hold","max_hold","cooldown"}
BEST={k:(int(grid.iloc[0][k]) if k in INTk else grid.iloc[0][k]) for k in ["long_threshold","short_threshold","entry_atr_mult","sl_atr_mult","tp_atr_mult","min_sl","min_hold","max_hold","cooldown","trade_direction"]}
for k,v in list(BEST.items()):
    if k not in INTk and k!="trade_direction": BEST[k]=float(v)
print("best params:", BEST)
""")

md("## 8 · OOS backtest + the 10%/yr decision gate")
code(r"""oos = df[df.index >= OOS_START]
o_p = pd.Series(probs, index=df.index)[df.index >= OOS_START].values
eq,tr = _run_backtest(o_p,oos["close"].values,oos["high"].values,oos["low"].values,oos["atr_14_pct"].values,with_fees=True,**BEST)
TF = pd.DataFrame(tr) if tr else pd.DataFrame(columns=["net"])
n_years=(oos.index[-1]-oos.index[0]).days/365.25; ann=float(eq[-1]**(1/n_years)-1)
print(f"OOS w/fees: return={eq[-1]-1:+.1%}  annualised={ann:+.1%}  Sharpe={_sharpe(eq):.2f}  "
      f"MaxDD={_maxdd(eq):.1%}  trades={len(TF)}")
print(f"DECISION GATE (>= 10%/yr): {'PASS -> promote into fund' if ann>=0.10 else 'FAIL -> honest negative'}")

# validation-window Sharpe (watch for validation-vs-OOS collapse = overfitting)
vm=(df.index>=GRID_VAL_START)&(df.index<=GRID_VAL_END); vdf=df[vm]
veq,_=_run_backtest(pd.Series(probs,index=df.index)[vm].values,vdf["close"].values,vdf["high"].values,vdf["low"].values,vdf["atr_14_pct"].values,with_fees=True,**BEST)
print(f"validation-window Sharpe={_sharpe(veq):.2f} (return {veq[-1]-1:+.1%}) — compare to OOS above")

bh=(oos["close"].values/oos["close"].values[0]-1)*100
fig,ax=plt.subplots(figsize=(13,5))
ax.plot(oos.index[:len(eq)],(eq-1)*100,color=ACCENT,lw=1.5,label=f"{CELL_TYPE} (w/ fees) {eq[-1]-1:+.1%}")
ax.plot(oos.index,bh,color=GREY,lw=1.0,ls=":",label="BTC B&H")
ax.axhline(0,color=GREY,lw=0.6,ls=":"); ax.set_ylabel("Return (%)"); ax.legend()
ax.set_title(f"{CELL_TYPE} — OOS, net of fees | Sharpe {_sharpe(eq):.2f}",fontweight="bold")
plt.show()
""")

md("## 9 · (Optional) Save artifacts in the standard contract consumed by `mas07`")
code(r"""SAVE = False   # set True to overwrite artifacts/notebooks_v2/09_{gru,lstm}/
if SAVE:
    np.save(ARTS/"oos_probs.npy", pd.Series(probs,index=df.index)[df.index>=OOS_START].values.astype(np.float32))
    np.save(ARTS/"oos_probs_raw.npy", pd.Series(probs_raw,index=df.index)[df.index>=OOS_START].values.astype(np.float32))
    np.save(ARTS/"oos_index.npy", oos.index.astype("datetime64[ns]").astype(np.int64).values)
    np.save(ARTS/"wfo_probs.npy", probs.astype(np.float32))
    np.save(ARTS/"wfo_probs_raw.npy", probs_raw.astype(np.float32))
    np.save(ARTS/"wfo_index.npy", df.index.astype("datetime64[ns]").astype(np.int64).values)
    np.save(ARTS/"oos_equity_wfees.npy", eq.astype(np.float64))
    if best_model is not None: torch.save(best_model.state_dict(), ARTS/"model_lastfold.pt")
    res={"notebook":"09_rnn_train_v1","cell_type":CELL_TYPE,"input_norm":INPUT_NORM,
         "architecture":{"d_model":D_MODEL,"n_layers":N_LAYERS,"seq_len":SEQ_LEN,"dropout":DROPOUT},
         "best_params":{**BEST,"signal_quantile":float(grid.iloc[0]["signal_quantile"])},
         "backtest_wfees":{"total_ret":round(float(eq[-1]-1),4),"ann_ret":round(ann,4),
                            "sharpe":round(_sharpe(eq),4),"maxdd":round(_maxdd(eq),4),"n_trades":len(TF)},
         "annualised_return_wfees":round(ann,4),"decision_gate_10pct":bool(ann>=0.10),
         "selected_features":FEATS}
    json.dump(res, open(ARTS/"results.json","w"), indent=2, default=str)
    print("saved ->", ARTS)
else:
    print("SAVE=False (set True to write artifacts)")
""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
out = Path(__file__).parent / "09_rnn_train_v1.ipynb"
nbf.write(nb, out)
print(f"wrote {out}")
