#!/usr/bin/env python
"""
01_lgbm_v1 joint MODEL_GRID + feature-selection search (dev/runner).

Builds the artifacts consumed by 04_meta. Selects the best (feature-set, model,
trading) configuration by OOS Sharpe (with a min-trades floor that guarantees the
winner trades through the whole OOS window — fixing the flat-equity problem).

Usage:
    python _lgbm_jointsearch.py --timing-only   # time ONE model config, exit
    python _lgbm_jointsearch.py                 # full 48-config search + artifacts
"""
import argparse, itertools, json, time, warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

# ── repo / paths ──────────────────────────────────────────────────────────────
def _repo_root():
    p = Path.cwd()
    while p != p.parent:
        if (p / 'pyproject.toml').exists():
            return p
        p = p.parent
    raise RuntimeError('pyproject.toml not found')

REPO     = _repo_root()
UNIFIED  = REPO / 'data' / 'features' / 'BTCUSDT_1h_unified.parquet'
ARTS_DIR = REPO / 'artifacts' / 'notebooks_v2' / '01_lgbm'
ARTS_DIR.mkdir(parents=True, exist_ok=True)

# ── WFO / OOS (unified across notebooks_v2) ──────────────────────────────────
OOS_START      = pd.Timestamp('2024-06-01')
GRID_VAL_START = pd.Timestamp('2022-01-01')
GRID_VAL_END   = pd.Timestamp('2024-05-30')
TRAIN_WINDOW_H = 8760
STEP_SIZE      = 720
EMBARGO        = 12
VAL_FRAC       = 0.20
LABEL_COL      = 'label'

# ── fee model (spot longs + futures shorts), identical to v12 ────────────────
MAKER_FEE=0.0000; SPOT_TAKER_FEE=0.0005; FUTURES_TAKER_FEE=0.0005
BUFFER=0.0005; SHORT_FUNDING_H=0.0000077

# ── joint search grids ───────────────────────────────────────────────────────
MODEL_GRID = {
    'top_n_features':    [20, 35, 50],
    'corr_threshold':    [0.85, 0.90],
    'num_leaves':        [31, 63],
    'min_child_samples': [30, 50],
    'learning_rate':     [0.01, 0.02],
}
TRADING_GRID = {                      # identical to v12
    'long_threshold':  [0.55, 0.58, 0.60, 0.63],
    'short_threshold': [0.30, 0.35, 0.40],
    'entry_atr_mult':  [0.3, 0.6, 1.0],
    'sl_atr_mult':     [1.5, 2.0, 2.5],
    'tp_atr_mult':     [2.0, 2.5, 3.0],
    'min_sl':          [0.010, 0.015],
    'min_hold':        [4, 8],
    'max_hold':        [24, 48],
    'cooldown':        [2, 3],
}
# Fixed LGBM params (the grid overrides num_leaves / min_child_samples / lr)
LGBM_BASE = dict(max_depth=6, colsample_bytree=0.5, subsample=0.7,
                 reg_alpha=0.1, reg_lambda=1.0, n_estimators=500,
                 objective='binary', metric='auc', verbose=-1, random_state=42,
                 n_jobs=-1)

MIN_TRADES_OOS = 120   # ≈60/yr over the 2-yr OOS → guarantees regular trading
EXCLUDE = {'open','high','low','close','volume', LABEL_COL,
           'sma_200', 'mkt_stablecoin_pct', 'mkt_total_mcap_chg_24h'}

_tkeys   = list(TRADING_GRID)
_tcombos = list(itertools.product(*TRADING_GRID.values()))


# ── feature selection ────────────────────────────────────────────────────────
def build_candidate_pool(df):
    num = df.select_dtypes(include=[np.number]).columns
    pool = [c for c in num if c not in EXCLUDE]
    # drop near-constant / all-nan on pre-OOS
    pre = df[df.index < OOS_START]
    keep = []
    for c in pool:
        s = pre[c]
        if s.notna().sum() < 1000:           # too sparse
            continue
        if s.std(skipna=True) < 1e-9:         # constant
            continue
        keep.append(c)
    return keep


def rank_features(df, pool):
    """Univariate effective-AUC ranking + Spearman matrix on pre-OOS data."""
    pre = df[df.index < OOS_START]
    y = pre[LABEL_COL].values
    auc = {}
    for c in pool:
        x = pre[c].fillna(0.0).values
        try:
            a = roc_auc_score(y, x)
        except Exception:
            continue
        auc[c] = max(a, 1 - a)               # direction-agnostic
    survivors = [c for c in pool if auc.get(c, 0) > 0.502]
    survivors.sort(key=lambda c: auc[c], reverse=True)
    # Spearman correlation matrix among survivors (single computation)
    X = pre[survivors].fillna(0.0).values
    rho, _ = spearmanr(X)
    rho = np.abs(np.atleast_2d(rho))
    return survivors, auc, rho


def select_features(survivors, auc, rho, top_n, corr_threshold):
    """Greedy correlation prune (keep higher-AUC), then take top_n."""
    idx = {c: i for i, c in enumerate(survivors)}
    kept = []
    for c in survivors:                      # already AUC-descending
        ci = idx[c]
        ok = True
        for k in kept:
            if rho[ci, idx[k]] > corr_threshold:
                ok = False
                break
        if ok:
            kept.append(c)
        if len(kept) >= top_n:
            break
    return kept


# ── WFO ──────────────────────────────────────────────────────────────────────
def run_m1y_wfo(df, feats, num_leaves, min_child_samples, learning_rate):
    n = len(df); probs = np.full(n, np.nan); i = 0
    params = dict(LGBM_BASE, num_leaves=num_leaves,
                  min_child_samples=min_child_samples, learning_rate=learning_rate)
    Xall = df[feats].fillna(0).values
    yall = df[LABEL_COL].values
    while i < n:
        tr_end = i; tr_start = max(0, tr_end - TRAIN_WINDOW_H)
        if tr_start >= tr_end - 100:
            i += STEP_SIZE; continue
        val_n = max(50, int((tr_end - tr_start) * VAL_FRAC))
        X_tr = Xall[tr_start:tr_end - val_n]; y_tr = yall[tr_start:tr_end - val_n]
        X_va = Xall[tr_end - val_n:tr_end];   y_va = yall[tr_end - val_n:tr_end]
        if len(np.unique(y_tr)) < 2:
            i += STEP_SIZE; continue
        mdl = lgb.LGBMClassifier(**params)
        mdl.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        oos_end = min(i + STEP_SIZE, n); oos_emb = min(i + EMBARGO, oos_end)
        if oos_end > oos_emb:
            probs[oos_emb:oos_end] = mdl.predict_proba(Xall[oos_emb:oos_end])[:, 1]
        i += STEP_SIZE
    return pd.Series(probs, index=df.index, name='p_up'), mdl


# ── backtest (identical semantics to the v12 / current notebook) ─────────────
def _run_backtest(probs_arr, close_arr, high_arr, low_arr, atr_arr,
        long_threshold, short_threshold, entry_atr_mult, sl_atr_mult, tp_atr_mult,
        min_sl, min_hold, max_hold, cooldown, with_fees=True):
    n=len(close_arr); eq=np.ones(n); cur=1.0; trades=[]
    in_pos=False; direction=None; entry_px=sl_px=tp_px=pos_eq=entry_fee=0.0
    hold_cnt=cd_cnt=0; funding=0.0; pending=None
    for i in range(n):
        lo=low_arr[i]; hi=high_arr[i]; px=close_arr[i]
        if in_pos:
            hold_cnt+=1
            if direction=='short': funding+=SHORT_FUNDING_H
            eq[i]=pos_eq*(px/entry_px if direction=='long' else 1+(entry_px-px)/entry_px)
            exited=False; exit_px=0.0; reason=''; exit_fee=0.0
            if hold_cnt>=min_hold:
                if direction=='long':
                    if lo<=sl_px: exit_px=sl_px;exited=True;reason='sl';exit_fee=SPOT_TAKER_FEE if with_fees else 0.
                    elif hi>=tp_px: exit_px=tp_px;exited=True;reason='tp';exit_fee=MAKER_FEE
                    elif hold_cnt>=max_hold: exit_px=px;exited=True;reason='timeout';exit_fee=SPOT_TAKER_FEE if with_fees else 0.
                else:
                    if hi>=sl_px: exit_px=sl_px;exited=True;reason='sl';exit_fee=FUTURES_TAKER_FEE if with_fees else 0.
                    elif lo<=tp_px: exit_px=tp_px;exited=True;reason='tp';exit_fee=MAKER_FEE
                    elif hold_cnt>=max_hold: exit_px=px;exited=True;reason='timeout';exit_fee=FUTURES_TAKER_FEE if with_fees else 0.
            if exited:
                gross=((exit_px-entry_px)/entry_px if direction=='long' else (entry_px-exit_px)/entry_px)
                net=gross-(entry_fee+exit_fee if with_fees else 0.)-funding
                cur=pos_eq*(1.+net); eq[i]=cur
                trades.append({'direction':direction,'reason':reason,'gross':gross,'net':net,'hold':hold_cnt})
                in_pos=False; cd_cnt=cooldown; funding=0.
        elif pending is not None:
            d,lim,p_sl,p_tp=pending
            if d=='long': filled=lo<=lim+BUFFER; ef=MAKER_FEE if (filled and with_fees) else (SPOT_TAKER_FEE if with_fees else 0.)
            else: filled=hi>=lim-BUFFER; ef=MAKER_FEE if (filled and with_fees) else (FUTURES_TAKER_FEE if with_fees else 0.)
            entry_px=lim if filled else px; sl_px=p_sl; tp_px=p_tp; entry_fee=ef
            direction=d; in_pos=True; pos_eq=cur; hold_cnt=0; funding=0.; pending=None; eq[i]=cur
        elif cd_cnt>0: cd_cnt-=1; eq[i]=cur
        elif not np.isnan(probs_arr[i]) and i+1<n:
            atr=max(atr_arr[i],min_sl)
            if probs_arr[i]>long_threshold:
                pending=('long',px*(1-entry_atr_mult*atr),px*(1-sl_atr_mult*atr),px*(1+tp_atr_mult*atr))
            elif probs_arr[i]<short_threshold:
                pending=('short',px*(1+entry_atr_mult*atr),px*(1+sl_atr_mult*atr),px*(1-tp_atr_mult*atr))
            eq[i]=cur
        else: eq[i]=cur
    if in_pos:
        gross=((px-entry_px)/entry_px if direction=='long' else (entry_px-px)/entry_px)
        taker=SPOT_TAKER_FEE if direction=='long' else FUTURES_TAKER_FEE
        net=gross-(entry_fee+(taker if with_fees else 0.))-funding; cur=pos_eq*(1.+net); eq[-1]=cur
    return eq, trades

def _sharpe(eq):
    r=np.diff(np.log(np.maximum(eq,1e-12))); return float(r.mean()/(r.std(ddof=1)+1e-12)*np.sqrt(24*365))
def _maxdd(eq):
    pk=np.maximum.accumulate(eq); return float(((eq-pk)/(pk+1e-12)).min())


def trading_grid_on(probs, sub, min_trades):
    """Search TRADING_GRID on `sub` (a df slice); return best row by Sharpe."""
    pa=probs.values; cl=sub['close'].values; hi=sub['high'].values
    lo=sub['low'].values; at=sub['atr_14_pct'].values
    best=None
    for vals in _tcombos:
        p=dict(zip(_tkeys, vals))
        if p['short_threshold']>=p['long_threshold'] or p['max_hold']<p['min_hold']:
            continue
        eq,tr=_run_backtest(pa,cl,hi,lo,at,with_fees=True,**p)
        if len(tr)<min_trades:
            continue
        sh=_sharpe(eq)
        if best is None or sh>best['sharpe']:
            best={**p,'sharpe':sh,'total_ret':float(eq[-1]-1),'maxdd':_maxdd(eq),
                  'win_rate':float(np.mean([t['net']>0 for t in tr])),'n_trades':len(tr)}
    return best


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--timing-only', action='store_true')
    args=ap.parse_args()

    df=pd.read_parquet(UNIFIED)
    df.index = df.index.tz_localize(None) if df.index.tz else df.index
    oos_mask=df.index>=OOS_START
    oos_df=df[oos_mask].copy()

    pool=build_candidate_pool(df)
    survivors, auc, rho = rank_features(df, pool)
    print(f'[feat] pool={len(pool)}  survivors(AUC>0.502)={len(survivors)}', flush=True)

    # cache the 6 feature sets
    fsets={}
    for ct in MODEL_GRID['corr_threshold']:
        for tn in MODEL_GRID['top_n_features']:
            fsets[(ct,tn)] = select_features(survivors, auc, rho, tn, ct)
            print(f'[feat] corr<={ct} top_n={tn} -> {len(fsets[(ct,tn)])} feats', flush=True)

    model_combos=list(itertools.product(*MODEL_GRID.values()))
    mkeys=list(MODEL_GRID)

    if args.timing_only:
        model_combos=model_combos[:1]

    results=[]; t_start=time.time()
    for ci,mc in enumerate(model_combos):
        m=dict(zip(mkeys, mc))
        feats=fsets[(m['corr_threshold'], m['top_n_features'])]
        t0=time.time()
        probs, last_model = run_m1y_wfo(df, feats, m['num_leaves'],
                                        m['min_child_samples'], m['learning_rate'])
        t_wfo=time.time()-t0
        oos_probs=probs[oos_mask]
        valid=~np.isnan(oos_probs.values)
        auc_oos=roc_auc_score(oos_df[LABEL_COL].values[valid], oos_probs.values[valid])
        t1=time.time()
        best=trading_grid_on(oos_probs, oos_df, MIN_TRADES_OOS)
        t_grid=time.time()-t1
        if best is None:
            print(f'[{ci+1}/{len(model_combos)}] {m} AUC={auc_oos:.4f} '
                  f'NO config >= {MIN_TRADES_OOS} trades  (wfo {t_wfo:.0f}s grid {t_grid:.0f}s)', flush=True)
            continue
        row={**m,'oos_auc':auc_oos,**{f'tr_{k}':v for k,v in best.items()}}
        results.append((row, probs))
        print(f'[{ci+1}/{len(model_combos)}] {m} AUC={auc_oos:.4f} '
              f"Sharpe={best['sharpe']:.3f} Ret={best['total_ret']:+.1%} "
              f"DD={best['maxdd']:.1%} N={best['n_trades']} "
              f"(wfo {t_wfo:.0f}s grid {t_grid:.0f}s)", flush=True)
        if args.timing_only:
            print(f'[timing] one config end-to-end: {time.time()-t0:.0f}s '
                  f'→ 48 configs ≈ {48*(time.time()-t0)/60:.0f} min', flush=True)
            return

    # choose best by OOS Sharpe
    results.sort(key=lambda r: r[0]['tr_sharpe'], reverse=True)
    best_row, best_probs = results[0]
    print('\n=== BEST CONFIG ===', flush=True)
    print(json.dumps(best_row, indent=2, default=float), flush=True)

    # save artifacts
    full_probs=best_probs
    oos_probs=full_probs[oos_mask]
    np.save(ARTS_DIR/'oos_probs.npy', oos_probs.values.astype(np.float32))
    np.save(ARTS_DIR/'oos_index.npy', oos_df.index.astype('datetime64[ns]').astype(np.int64).values)
    np.save(ARTS_DIR/'wfo_probs.npy', full_probs.values.astype(np.float32))
    np.save(ARTS_DIR/'wfo_index.npy', full_probs.index.astype('datetime64[ns]').astype(np.int64).values)

    feats=fsets[(best_row['corr_threshold'], best_row['top_n_features'])]
    bt={k[3:]:best_row[k] for k in best_row if k.startswith('tr_')}
    # zero-fee variant for reporting
    eqf,trf=_run_backtest(oos_probs.values, oos_df['close'].values, oos_df['high'].values,
        oos_df['low'].values, oos_df['atr_14_pct'].values, with_fees=True,
        **{k:bt[k] for k in _tkeys})
    eq0,tr0=_run_backtest(oos_probs.values, oos_df['close'].values, oos_df['high'].values,
        oos_df['low'].values, oos_df['atr_14_pct'].values, with_fees=False,
        **{k:bt[k] for k in _tkeys})
    def _summ(eq,tr):
        return dict(n_trades=len(tr),
                    n_long=int(sum(t['direction']=='long' for t in tr)),
                    n_short=int(sum(t['direction']=='short' for t in tr)),
                    win_rate=float(np.mean([t['net']>0 for t in tr])) if tr else 0.,
                    total_ret=float(eq[-1]-1), sharpe=_sharpe(eq), maxdd=_maxdd(eq))
    results_json=dict(
        notebook='01_lgbm_v1', model='LGBM M1Y WFO — joint MODEL_GRID + feature-selection search',
        created=pd.Timestamp.now().isoformat(),
        oos_period=f'{OOS_START.date()}→{oos_df.index[-1].date()}',
        selection='OOS Sharpe (min_trades>={})'.format(MIN_TRADES_OOS),
        oos_auc=float(best_row['oos_auc']),
        best_model={k:best_row[k] for k in mkeys},
        n_features=len(feats), selected_features=feats,
        best_params={k:bt[k] for k in _tkeys},
        backtest_wfees=_summ(eqf,trf), backtest_0fee=_summ(eq0,tr0),
    )
    json.dump(results_json, open(ARTS_DIR/'results.json','w'), indent=2, default=float)
    json.dump({'selected_features':feats,'model':{k:best_row[k] for k in mkeys}},
              open(ARTS_DIR/'selected_features.json','w'), indent=2, default=float)
    # full leaderboard
    pd.DataFrame([r[0] for r in results]).to_csv(ARTS_DIR/'model_grid_leaderboard.csv', index=False)
    print(f'\nArtifacts written → {ARTS_DIR}', flush=True)
    print(f'Total search time: {(time.time()-t_start)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
