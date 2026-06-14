#!/usr/bin/env python
"""PatchTST walk-forward retraining runner.

Replaces the static train<=2022 split (which left the model regime-stuck with no
OOS signal) with a sliding 24-month / 3-month-step walk-forward, mirroring the
Mamba protocol. Produces wfo_probs across 2022-06+ (genuine per-fold OOS) so the
meta-learner gets a real signal, plus the 2-year unified OOS slice.

Run: python _patch_wfo.py   (writes artifacts/notebooks_v2/05_patchtst/)
"""
import itertools, json, math, random, time, warnings
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import QuantileTransformer
from torch.utils.data import DataLoader, TensorDataset
warnings.filterwarnings('ignore')
SEED=42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE=torch.device('mps') if torch.backends.mps.is_available() else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

def _repo():
    p=Path.cwd()
    while p!=p.parent:
        if (p/'pyproject.toml').exists(): return p
        p=p.parent
    raise RuntimeError('repo root')
REPO=_repo(); ART=REPO/'artifacts'/'notebooks_v2'/'05_patchtst'; ART.mkdir(parents=True,exist_ok=True)

# ── config (identical model/label spec to the notebook) ─────────────────────
OOS_START=pd.Timestamp('2024-05-31'); WFO_START=pd.Timestamp('2022-06-01')
TRAIN_MONTHS=24; STEP_MONTHS=3; EMBARGO_H=12
TBM_VOL_WINDOW=24; TBM_MULT=2.0; TBM_VERT_H=24; FRAC_D=0.4; FFD_THRES=1e-4
SEQ_LEN=48; PATCH_LEN=8; PATCH_STRIDE=4; D_MODEL=64; N_HEADS=4; N_LAYERS=2; D_FF=128; TR_DROPOUT=0.20
EPOCHS=60; BATCH_SIZE=256; LR=3e-4; WEIGHT_DECAY=1e-4; PATIENCE=8; LAMBDA_VOL=0.50; AUX_FWD_H=6; VAL_FRAC=0.15
MAKER=0.0; STAKER=0.0005; FTAKER=0.0005; BUF=0.0005; FUND=0.0000077
LGBM_CORE=['stoch_k_14','ret_2h','rsi_divergence','close_vs_sma_7','bear_streak','close_vs_s1','macd_hist_5_13','ad_z_48h','ret_3h']
V1_EXTRA=['ret_1h','rsi_14','vol_ratio_24h','bb_position_20','hour_sin','hour_cos','atr_14_pct','hurst_168h','trend_score','close_vs_sma_50','ma_bull_score']
V4_FEATURES=['close_vs_true_vwap','hurst_24h','hurst_72h','tfi_pct','tfi_z_24h','bb_width_pct','sideways_flag']
STRUCT=['liq_vwap_dev_24h','volat_atr_20_pct','mtf_alignment','mtf_h4_rsi']
BASE=LGBM_CORE+V1_EXTRA+V4_FEATURES+STRUCT; FFD='ffd_log_price'; ALL=BASE+[FFD]

class PatchTST(nn.Module):
    def __init__(s,n_feat,seq_len,patch_len,stride,d_model,n_heads,n_layers,d_ff,dropout,n_classes=2):
        super().__init__(); s.patch_len=patch_len; s.stride=stride; s.n_patches=(seq_len-patch_len)//stride+1
        s.proj=nn.Linear(patch_len*n_feat,d_model); s.cls=nn.Parameter(torch.randn(1,1,d_model)*0.02)
        s.pos=nn.Parameter(torch.randn(1,s.n_patches+1,d_model)*0.02); s.in_drop=nn.Dropout(dropout)
        enc=nn.TransformerEncoderLayer(d_model,n_heads,d_ff,dropout,activation='gelu',batch_first=True,norm_first=True)
        s.encoder=nn.TransformerEncoder(enc,n_layers); s.norm=nn.LayerNorm(d_model); s.hd=nn.Dropout(dropout)
        s.cls_head=nn.Linear(d_model,n_classes); s.vol_head=nn.Linear(d_model,1)
    def forward(s,x):
        B,T,Fd=x.shape
        p=x.unfold(1,s.patch_len,s.stride).permute(0,1,3,2).reshape(B,s.n_patches,s.patch_len*Fd)
        h=s.in_drop(s.proj(p)); h=torch.cat([s.cls.expand(B,-1,-1),h],dim=1)+s.pos
        h=s.encoder(h); pooled=s.hd(s.norm(h.mean(dim=1)))
        return s.cls_head(pooled), s.vol_head(pooled).squeeze(-1)

def make_seqs(X,yd,yv,seq_len):
    Xs=[];Yd=[];Yv=[]
    for i in range(seq_len-1,len(X)):
        if yd[i] not in (0,1): continue
        Xs.append(X[i-seq_len+1:i+1]); Yd.append(int(yd[i])); Yv.append(yv[i] if not np.isnan(yv[i]) else 0.0)
    if not Xs: return None
    return np.stack(Xs).astype(np.float32),np.array(Yd,np.int64),np.array(Yv,np.float32)

def train_fold(Xtr,ytr_d,ytr_v,Xvl,yvl_d,yvl_v):
    g=torch.Generator(); g.manual_seed(SEED)
    m=PatchTST(Xtr.shape[2],SEQ_LEN,PATCH_LEN,PATCH_STRIDE,D_MODEL,N_HEADS,N_LAYERS,D_FF,TR_DROPOUT).to(DEVICE)
    cnt=np.bincount(ytr_d,minlength=2); w=torch.tensor(1.0/np.maximum(cnt,1),dtype=torch.float32).to(DEVICE); w/=w.sum()
    cc=nn.CrossEntropyLoss(weight=w); vc=nn.MSELoss()
    tl=DataLoader(TensorDataset(torch.from_numpy(Xtr),torch.from_numpy(ytr_d),torch.from_numpy(ytr_v)),BATCH_SIZE,shuffle=True,drop_last=True,generator=g)
    vl=DataLoader(TensorDataset(torch.from_numpy(Xvl),torch.from_numpy(yvl_d),torch.from_numpy(yvl_v)),BATCH_SIZE,shuffle=False)
    opt=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=LR,epochs=EPOCHS,steps_per_epoch=max(1,len(tl)))
    best=1e9;best_state=None;best_ep=0
    for ep in range(1,EPOCHS+1):
        m.train()
        for xb,yd,yv in tl:
            xb,yd,yv=xb.to(DEVICE),yd.to(DEVICE),yv.to(DEVICE)
            opt.zero_grad(); lg,vp=m(xb); loss=cc(lg,yd)+LAMBDA_VOL*vc(vp,yv)
            loss.backward(); nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step(); sched.step()
        m.eval(); v=0.
        with torch.no_grad():
            for xb,yd,yv in vl:
                xb,yd,yv=xb.to(DEVICE),yd.to(DEVICE),yv.to(DEVICE)
                lg,vp=m(xb); v+=(cc(lg,yd)+LAMBDA_VOL*vc(vp,yv)).item()
        v/=max(1,len(vl))
        if v<best: best=v;best_ep=ep;best_state={k:val.cpu().clone() for k,val in m.state_dict().items()}
        if ep-best_ep>=PATIENCE: break
    m.load_state_dict(best_state)
    # temperature scaling on val
    m.eval(); lgs=[];ys=[]
    with torch.no_grad():
        for xb,yd,yv in vl:
            lg,_=m(xb.to(DEVICE)); lgs.append(lg.cpu()); ys.append(yd)
    lgs=torch.cat(lgs); ys=torch.cat(ys); T=torch.ones(1,requires_grad=True); ot=torch.optim.LBFGS([T],lr=0.05,max_iter=200); nll=nn.CrossEntropyLoss()
    def cl(): ot.zero_grad(); l=nll(lgs/T.clamp(min=1e-2),ys); l.backward(); return l
    ot.step(cl); return m,float(T.detach().clamp(min=1e-2))

def predict(m,T,X):
    m.eval(); out=[]
    with torch.no_grad():
        for i in range(SEQ_LEN-1,len(X)):
            xb=torch.from_numpy(X[i-SEQ_LEN+1:i+1][None]).to(DEVICE)
            lg,_=m(xb); out.append(torch.softmax(lg/T,-1).cpu().numpy()[0][1])
    return np.array(out,np.float32)

def main():
    df=pd.read_parquet(REPO/'data'/'features'/'BTCUSDT_1h_unified.parquet'); df.index=df.index.tz_localize(None) if df.index.tz else df.index
    # FFD
    lc=np.log(df['close']); w=[1.0];k=1
    while True:
        wk=-w[-1]*(FRAC_D-k+1)/k; w.append(wk); k+=1
        if abs(wk)<FFD_THRES: break
    w=np.array(w[::-1]); nw=len(w); ffd=np.full(len(df),np.nan)
    lcv=lc.values
    for i in range(nw-1,len(df)): ffd[i]=float(np.dot(w,lcv[i-nw+1:i+1]))
    df[FFD]=ffd
    # TBM labels
    lr=lc.diff(); rv=lr.rolling(TBM_VOL_WINDOW).std().values; cv=df['close'].values; n=len(df)
    yd=np.full(n,np.nan,np.float32); yv=np.full(n,np.nan,np.float32)
    for i in range(n):
        if np.isnan(rv[i]) or rv[i]==0: continue
        s=rv[i]*cv[i]; up=cv[i]+TBM_MULT*s; dn=cv[i]-TBM_MULT*s; ej=min(i+TBM_VERT_H,n)
        for j in range(i+1,ej):
            if cv[j]>=up: yd[i]=1;break
            if cv[j]<=dn: yd[i]=0;break
        else: yd[i]=-1
        fwd=lr.values[i+1:i+1+AUX_FWD_H];
        if len(fwd): yv[i]=float(np.std(fwd))
    df['y_dir']=yd; df['y_fwd_vol']=yv
    feat_ok=df[ALL].notna().all(axis=1).values
    full_probs=pd.Series(np.nan,index=df.index)
    # WFO loop
    blocks=[]; d=WFO_START
    while d<=df.index[-1]:
        blocks.append(d); d+=pd.DateOffset(months=STEP_MONTHS)
    t0=time.time(); last_m=None; last_T=1.0
    for bi,bstart in enumerate(blocks):
        bend=bstart+pd.DateOffset(months=STEP_MONTHS)
        tr_start=bstart-pd.DateOffset(months=TRAIN_MONTHS)
        tr_mask=(df.index>=tr_start)&(df.index<bstart-pd.Timedelta(hours=EMBARGO_H))&feat_ok
        if tr_mask.sum()<2000: continue
        tr_df=df[tr_mask]
        # split train/val tail
        cut=int(len(tr_df)*(1-VAL_FRAC)); tri=tr_df.index[:cut]; vli=tr_df.index[cut:]
        qt=QuantileTransformer(output_distribution='normal',n_quantiles=1000,random_state=SEED)
        Xtr_raw=np.nan_to_num(df.loc[tri,ALL].values.astype(np.float32)); qt.fit(Xtr_raw)
        def seqs_for(idx):
            sub=df.loc[idx]; X=qt.transform(np.nan_to_num(sub[ALL].values.astype(np.float32))).astype(np.float32)
            return make_seqs(X,sub['y_dir'].values,sub['y_fwd_vol'].values,SEQ_LEN)
        s_tr=seqs_for(tri); s_vl=seqs_for(vli)
        if s_tr is None or s_vl is None or len(np.unique(s_tr[1]))<2: continue
        m,T=train_fold(*s_tr,*s_vl)
        # predict OOS block (with SEQ_LEN-1 preceding context)
        pre=bstart-pd.Timedelta(hours=SEQ_LEN)
        blk=df[(df.index>=pre)&(df.index<bend)&feat_ok]
        if len(blk)<=SEQ_LEN: continue
        Xb=qt.transform(np.nan_to_num(blk[ALL].values.astype(np.float32))).astype(np.float32)
        p=predict(m,T,Xb); idx=blk.index[SEQ_LEN-1:]
        keep=idx>=bstart
        full_probs.loc[idx[keep]]=p[keep]
        last_m,last_T=m,T
        print(f'[{bi+1}/{len(blocks)}] OOS {bstart.date()}→{bend.date()} train={tr_mask.sum()} T={T:.2f} ({(time.time()-t0)/60:.1f}m)',flush=True)
    # AUC on OOS
    oos=full_probs[(full_probs.index>=OOS_START)].dropna()
    yb=(df.loc[oos.index,'y_dir'].values==1).astype(int); mask=df.loc[oos.index,'y_dir'].isin([0,1]).values
    auc=roc_auc_score(yb[mask],oos.values[mask]) if mask.sum() else float('nan')
    print(f'\nWFO done {(time.time()-t0)/60:.1f}m  OOS bars={len(oos)}  OOS AUC={auc:.4f}')
    bull=oos[(oos.index>=pd.Timestamp('2024-11-06'))&(oos.index<=pd.Timestamp('2025-10-31'))]
    print(f'OOS prob mean={oos.mean():.3f}  Bull mean={bull.mean():.3f}')
    # save artifacts (probs only; backtest/grid done in notebook or follow-up)
    np.save(ART/'wfo_probs.npy', full_probs.values.astype(np.float32))
    np.save(ART/'wfo_index.npy', full_probs.index.values.astype('int64'))
    oosdf=df[df.index>=OOS_START]
    np.save(ART/'oos_probs.npy', full_probs.reindex(oosdf.index).values.astype(np.float32))
    np.save(ART/'oos_index.npy', oosdf.index.values.astype('int64'))
    if last_m is not None: torch.save(last_m.state_dict(), ART/'model.pt')
    json.dump({'notebook':'05_patchtst','scheme':'walk-forward sliding 24mo/3mo','oos_auc_tbm':float(auc),
               'wfo_start':str(WFO_START.date()),'train_months':TRAIN_MONTHS,'temperature_last':last_T,
               'note':'Converted from static <=2022 split to walk-forward retraining.'},
              open(ART/'results_wfo.json','w'),indent=2,default=float)
    print('saved wfo_probs + results_wfo.json →',ART)

if __name__=='__main__': main()
