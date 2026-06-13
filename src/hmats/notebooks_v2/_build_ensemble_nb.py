"""Builds src/hmats/notebooks_v2/06_ensemble_v1.ipynb — a head-to-head of every
multi-agent merge strategy vs the LGBM-alone benchmark. Leak-free walk-forward."""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook(); cells = []
def md(s):   cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))

md("""# 06 — Multi-Agent Ensemble Bake-off (leak-free walk-forward)

**Goal.** Merge the four base agents (LGBM, Mamba, TCN, PatchTST) into a single signal that
**beats the strongest base agent alone**. LGBM-alone is the bar to clear: **+83.5%, Sharpe 1.77,
MaxDD −9.1%** on the unified OOS (2024-05-31 → 2026-05-16).

**Why the current meta fails (established in the signal analysis):**
- The 4 agents' probabilities average to **mean 0.476** (3 of 4 are short-biased), so the
  equal-weight primary signal fires **82 longs vs 1,742 shorts** — it shorts the bull while
  LGBM-alone takes 1,677 longs. The ensemble sign **disagrees with LGBM on 50.8% of bars**.
- Meta-labelling can only *veto*, never *flip*, so it cannot recover LGBM's longs.
- The meta-classifier's top feature is `halving_cycle_pos` (a monotonic time index) → it
  **memorises the regime** → OOS AUC 0.43 (< 0.5).

**Design decisions (locked):**
1. **Leak-free walk-forward** — every combiner is fit only on data strictly before each OOS
   block (expanding, 3-month step, 48 h embargo). It is scored on 2024-05-31+ which it never saw.
2. **Two input regimes are tested** — *signals-only* and *signals + stationary-regime* (no
   monotonic/calendar features).
3. **Common execution layer** — every merged signal is run through the *same* ATR-bracket
   backtester under one fixed trading rule, so we compare **signal quality**, not execution tuning.
   The single winner is then re-tuned with a leak-free trading grid.

> **ML-engineering question we keep asking:** *is this number honest, or did the combiner see
> its own test data?* Every step below is annotated with where the train/test boundary sits.""")

code("""import itertools, json, time, warnings
from pathlib import Path
import lightgbm as lgb
import matplotlib as mpl, matplotlib.dates as mdates, matplotlib.pyplot as plt
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore'); pd.set_option('display.float_format','{:.4f}'.format)
mpl.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],
    'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':120,'savefig.dpi':200,'savefig.bbox':'tight'})
ACCENT='#F7931A'; BLUE='#2962FF'; GREY='#9E9E9E'; RED='#EF5350'; GREEN='#26A69A'; TEAL='#00ACC1'; PURPLE='#7B1FA2'

def _repo_root():
    p=Path.cwd()
    while p!=p.parent:
        if (p/'pyproject.toml').exists(): return p
        p=p.parent
    raise RuntimeError('repo root not found')
REPO=_repo_root(); A2=REPO/'artifacts'/'notebooks_v2'
ARTS=A2/'06_ensemble'; ARTS.mkdir(parents=True, exist_ok=True)

OOS_START=pd.Timestamp('2024-05-31'); OOS_END=pd.Timestamp('2026-05-16')
STEP_MONTHS=3; EMBARGO_H=48
REGIME_CHOP=(pd.Timestamp('2024-05-31'),pd.Timestamp('2024-11-05'))
REGIME_BULL=(pd.Timestamp('2024-11-06'),pd.Timestamp('2025-10-31'))
REGIME_BEAR=(pd.Timestamp('2025-11-01'),pd.Timestamp('2026-05-31'))
# fee model (identical to base agents)
MAKER_FEE=0.0; SPOT_TAKER_FEE=0.0005; FUTURES_TAKER_FEE=0.0005; BUFFER=0.0005; SHORT_FUNDING_H=0.0000077
# fixed comparison trading rule (symmetric → exposes directional bias)
FIXED_RULE=dict(long_threshold=0.55, short_threshold=0.45, entry_atr_mult=0.6,
                sl_atr_mult=1.5, tp_atr_mult=2.5, min_sl=0.01, min_hold=4, max_hold=24, cooldown=2)
print('Setup OK  |  artifacts →', ARTS)""")

code("""# ── Load the four agents' walk-forward probabilities + price/label ───────────
df=pd.read_parquet(REPO/'data'/'features'/'BTCUSDT_1h_unified.parquet')
df.index=df.index.tz_localize(None) if df.index.tz else df.index
def load(d):
    f=A2/d
    p=np.load(f/'wfo_probs.npy'); idx=pd.to_datetime(np.load(f/'wfo_index.npy'),unit='ns')
    return pd.Series(p,index=idx)
AG=['lgbm','mamba','tcn','patch']
RAW={'lgbm':load('01_lgbm'),'mamba':load('02_mamba'),'tcn':load('03_tcn'),'patch':load('05_patchtst')}
P=pd.DataFrame(RAW).reindex(df.index)
P['y']=(df['close'].shift(-1)>df['close']).astype(int)           # combiner target = next-bar direction
for c in ['close','high','low','atr_14_pct','hurst_168h','bb_width_pct','vol_ratio_24h']:
    P[c]=df[c]
# all-4-present window
both=P[AG].notna().all(axis=1)
print(f'All-4-present: {both.sum():,} bars  ({P.index[both].min().date()} → {P.index[both].max().date()})')
oos=P[(P.index>=OOS_START)&(P.index<=OOS_END)&both].dropna(subset=AG+['y']).copy()
print(f'OOS common bars: {len(oos):,}')
print('Agent means (OOS):', {a:round(oos[a].mean(),3) for a in AG})""")

md("""## 1 · Signal diagnostics — *why the naive mean fails*

> **Q:** before merging, do the agents disagree enough to help, and is any single one already
> dominant? **Q:** does equal-weighting destroy the good agent's directionality?""")

code("""print('=== Per-agent OOS (next-bar direction) ===')
print(f"{'agent':7}{'mean':>7}{'AUC':>8}{'long>.56':>9}{'short<.44':>10}")
for a in AG:
    print(f"{a:7}{oos[a].mean():>7.3f}{roc_auc_score(oos['y'],oos[a]):>8.4f}"
          f"{int((oos[a]>0.56).sum()):>9}{int((oos[a]<0.44).sum()):>10}")
em=oos[AG].mean(axis=1)
print(f"\\n{'EW-mean':7}{em.mean():>7.3f}{roc_auc_score(oos['y'],em):>8.4f}"
      f"{int((em>0.56).sum()):>9}{int((em<0.44).sum()):>10}")
print('\\nSpearman correlation (OOS):'); print(oos[AG].corr('spearman').round(3).to_string())
disagree=(((em-0.5)*(oos['lgbm']-0.5))<0).mean()
print(f'\\nEW-mean sign disagrees with LGBM on {disagree:.1%} of bars  ← the core problem')""")

code("""# ── Execution layer: ATR-bracket backtester (identical to base agents) ───────
def _run_backtest(probs, close, high, low, atr, long_threshold, short_threshold, entry_atr_mult,
        sl_atr_mult, tp_atr_mult, min_sl, min_hold, max_hold, cooldown, with_fees=True):
    n=len(close); eq=np.ones(n); cur=1.0; trades=[]
    in_pos=False; direction=None; entry_px=sl_px=tp_px=pos_eq=entry_fee=0.0
    hold=cd=0; funding=0.0; pend=None
    for i in range(n):
        lo=low[i]; hi=high[i]; px=close[i]
        if in_pos:
            hold+=1
            if direction=='short': funding+=SHORT_FUNDING_H
            eq[i]=pos_eq*(px/entry_px if direction=='long' else 1+(entry_px-px)/entry_px)
            ex=False; xpx=0.; xf=0.
            if hold>=min_hold:
                if direction=='long':
                    if lo<=sl_px: xpx=sl_px;ex=True;xf=SPOT_TAKER_FEE if with_fees else 0.
                    elif hi>=tp_px: xpx=tp_px;ex=True;xf=MAKER_FEE
                    elif hold>=max_hold: xpx=px;ex=True;xf=SPOT_TAKER_FEE if with_fees else 0.
                else:
                    if hi>=sl_px: xpx=sl_px;ex=True;xf=FUTURES_TAKER_FEE if with_fees else 0.
                    elif lo<=tp_px: xpx=tp_px;ex=True;xf=MAKER_FEE
                    elif hold>=max_hold: xpx=px;ex=True;xf=FUTURES_TAKER_FEE if with_fees else 0.
            if ex:
                g=((xpx-entry_px)/entry_px if direction=='long' else (entry_px-xpx)/entry_px)
                net=g-(entry_fee+xf if with_fees else 0.)+funding; cur=pos_eq*(1.+net); eq[i]=cur
                trades.append({'direction':direction,'net':net}); in_pos=False; cd=cooldown; funding=0.
        elif pend is not None:
            d,lim,ps,pt=pend
            if d=='long': fill=lo<=lim+BUFFER; ef=MAKER_FEE if (fill and with_fees) else (SPOT_TAKER_FEE if with_fees else 0.)
            else: fill=hi>=lim-BUFFER; ef=MAKER_FEE if (fill and with_fees) else (FUTURES_TAKER_FEE if with_fees else 0.)
            entry_px=lim if fill else px; sl_px=ps; tp_px=pt; entry_fee=ef
            direction=d; in_pos=True; pos_eq=cur; hold=0; funding=0.; pend=None; eq[i]=cur
        elif cd>0: cd-=1; eq[i]=cur
        elif not np.isnan(probs[i]) and i+1<n:
            a=max(atr[i],min_sl)
            if probs[i]>long_threshold: pend=('long',px*(1-entry_atr_mult*a),px*(1-sl_atr_mult*a),px*(1+tp_atr_mult*a))
            elif probs[i]<short_threshold: pend=('short',px*(1+entry_atr_mult*a),px*(1+sl_atr_mult*a),px*(1-tp_atr_mult*a))
            eq[i]=cur
        else: eq[i]=cur
    return eq, trades

def _sharpe(eq):
    r=np.diff(np.log(np.maximum(eq,1e-12))); return float(r.mean()/(r.std(ddof=1)+1e-12)*np.sqrt(24*365))
def _sortino(eq):
    r=np.diff(np.log(np.maximum(eq,1e-12))); neg=r[r<0]; d=neg.std(ddof=1) if len(neg)>1 else 1e-12
    return float(r.mean()/(d+1e-12)*np.sqrt(24*365))
def _maxdd(eq):
    pk=np.maximum.accumulate(eq); return float(((eq-pk)/(pk+1e-12)).min())

def evaluate(p_up, name, rule=FIXED_RULE):
    \"\"\"Backtest a combined p_up Series on the OOS window under `rule`. Returns metrics + equity.\"\"\"
    s=p_up.reindex(oos.index).values
    eq,tr=_run_backtest(s, oos['close'].values, oos['high'].values, oos['low'].values,
                        oos['atr_14_pct'].values, **rule)
    nl=sum(t['direction']=='long' for t in tr); ns=sum(t['direction']=='short' for t in tr)
    bh=oos['close'].values/oos['close'].values[0]
    row=dict(name=name, ret=float(eq[-1]-1), sharpe=_sharpe(eq), sortino=_sortino(eq),
             maxdd=_maxdd(eq), trades=len(tr), nL=nl, nS=ns,
             win=float(np.mean([t['net']>0 for t in tr])) if tr else 0.,
             alpha=float(eq[-1]-bh[-1]))
    return row, eq

def regime_breakdown(eq, name):
    idx=oos.index; cl=oos['close'].values; out=[]
    for lbl,(s,e) in [('Full',(idx.min(),idx.max())),('Chop',REGIME_CHOP),('Bull',REGIME_BULL),('Bear',REGIME_BEAR)]:
        m=(idx>=s)&(idx<=e)
        if m.sum()<24: continue
        seg=pd.Series(eq,index=idx)[m].values; seg=seg/seg[0]
        out.append({'regime':lbl,'ret':f'{seg[-1]-1:+.1%}','sharpe':f'{_sharpe(seg):.2f}','maxdd':f'{_maxdd(seg):.1%}'})
    return pd.DataFrame(out)
print('Backtester ready.')""")

md("""## 2 · Benchmark — LGBM alone (the bar to beat)

> **Q:** what is "success"? A merged signal is only worth shipping if it beats this row.""")

code("""LGBM_RULE=json.load(open(A2/'01_lgbm'/'results.json'))['best_params']
bench_row,bench_eq=evaluate(RAW['lgbm'], 'LGBM-alone (own tuned rule)', rule=LGBM_RULE)
RESULTS=[bench_row]
print(pd.DataFrame([bench_row]).to_string(index=False))
print('\\nRegime breakdown:'); print(regime_breakdown(bench_eq,'LGBM').to_string(index=False))""")

md("""## 3 · Leak-free helpers (de-bias, trailing reliability, WFO stacker)

> **Q (leakage):** every trailing statistic below is computed with a right-open window and the
> combiner is fit only on bars older than `OOS_block_start − 48 h`. No future bar ever informs a
> past prediction.""")

code("""TRAIL=720   # 30-day trailing window for de-bias / performance weights

def debias(series):
    \"\"\"Recenter a probability to a trailing median of 0.5 (leak-free, right-open).\"\"\"
    med=series.rolling(TRAIL,min_periods=200).median().shift(1)
    return (series-med+0.5).clip(0.01,0.99)

def trailing_reliability(series, y):
    \"\"\"Trailing correlation of (p-0.5) with realised next-bar return sign (leak-free).\"\"\"
    edge=np.sign(series-0.5)*(2*y-1)
    return edge.rolling(TRAIL,min_periods=200).mean().shift(1+EMBARGO_H).fillna(0.0)

def stack_wfo(frame, feat_cols, label='stack', clf='logit'):
    \"\"\"Expanding walk-forward stacker on `feat_cols`; returns OOS p_up Series.\"\"\"
    base=frame[both].dropna(subset=feat_cols+['y']).copy()
    out=pd.Series(np.nan,index=base.index)
    d=OOS_START
    while d<=OOS_END:
        tr=base[base.index < d-pd.Timedelta(hours=EMBARGO_H)]
        te=base[(base.index>=d)&(base.index<d+pd.DateOffset(months=STEP_MONTHS))]
        if len(tr)>500 and len(te)>0 and tr['y'].nunique()==2:
            Xtr=tr[feat_cols].values; ytr=tr['y'].values; Xte=te[feat_cols].values
            if clf=='logit':
                sc=StandardScaler().fit(Xtr)
                m=LogisticRegression(C=1.0,max_iter=2000).fit(sc.transform(Xtr),ytr)
                out.loc[te.index]=m.predict_proba(sc.transform(Xte))[:,1]
            else:
                m=lgb.LGBMClassifier(n_estimators=200,num_leaves=15,max_depth=4,learning_rate=0.05,
                    subsample=0.7,colsample_bytree=0.7,reg_lambda=1.0,min_child_samples=30,
                    objective='binary',verbose=-1,random_state=42).fit(Xtr,ytr)
                out.loc[te.index]=m.predict_proba(Xte)[:,1]
        d+=pd.DateOffset(months=STEP_MONTHS)
    return out
print('Helpers ready  |  trailing window =',TRAIL,'h')""")

md("""## 4 · The merge strategies

Each produces a combined `p_up`. We evaluate all under the **fixed** comparison rule.""")

code("""# Build signal-derived features on the full frame (leak-free; all per-bar)
F=P.copy()
F['mean4']=F[AG].mean(axis=1)
F['disp']=F[AG].std(axis=1)
F['nbull']=(F[AG]>0.5).sum(axis=1)
for a in AG: F[a+'_db']=debias(F[a])
F['mean4_db']=F[[a+'_db' for a in AG]].mean(axis=1)

cand={}   # name -> p_up Series
# (a) baseline equal-weight mean
cand['EW-mean (baseline)']=F['mean4']
# (b) de-biased equal-weight mean
cand['De-biased EW-mean']=F['mean4_db']
# (c) performance-weighted (trailing reliability, leak-free), on de-biased probs
rel={a:trailing_reliability(F[a],F['y']).clip(lower=0) for a in AG}
W=pd.DataFrame(rel); Wn=W.div(W.sum(axis=1).replace(0,np.nan),axis=0).fillna(0.25)
cand['Perf-weighted (trailing)']=0.5+sum(Wn[a]*(F[a+'_db']-0.5) for a in AG)
# (d) Hedge / exponential weights (online multiplicative, leak-free)
eta=2.0; ret=df['close'].pct_change().shift(-1).reindex(F.index).fillna(0)
cumrew={a:(np.sign(F[a]-0.5)*ret).cumsum().shift(1+EMBARGO_H).fillna(0) for a in AG}
H=pd.DataFrame({a:np.exp(eta*cumrew[a]) for a in AG}); Hn=H.div(H.sum(axis=1),axis=0)
cand['Hedge (exp-weights)']=0.5+sum(Hn[a]*(F[a+'_db']-0.5) for a in AG)
# (e) stacking — signals only (logit + lgbm)
SIG=AG+['mean4','disp','nbull']
cand['Stack signals-only (logit)']=stack_wfo(F,SIG,'logit',clf='logit')
cand['Stack signals-only (lgbm)'] =stack_wfo(F,SIG,'lgbm', clf='lgbm')
# (f) stacking — signals + stationary regime (NO monotonic/time features)
SIGR=SIG+['atr_14_pct','hurst_168h','bb_width_pct','vol_ratio_24h']
cand['Stack signals+regime (lgbm)']=stack_wfo(F,SIGR,'lgbm',clf='lgbm')
print('Built',len(cand),'merge candidates:',list(cand))""")

code("""# (g) pruned ensemble: LGBM + the single most complementary agent (de-biased 50/50),
#     diversifier chosen on the PRE-OOS window only (leak-free)
preo=F[(F.index<OOS_START)&both].dropna(subset=[a+'_db' for a in AG]+['y'])
best_div=None
for a in ['mamba','tcn','patch']:
    blend=0.5+0.5*((preo['lgbm_db']-0.5)+(preo[a+'_db']-0.5))
    au=roc_auc_score(preo['y'],blend)
    if best_div is None or au>best_div[1]: best_div=(a,au)
da=best_div[0]
cand[f'Pruned LGBM+{da} (db)']=0.5+0.5*((F['lgbm_db']-0.5)+(F[da+'_db']-0.5))
print(f'Diversifier chosen on pre-OOS: {da} (blend AUC {best_div[1]:.4f})')""")

md("""## 5 · Head-to-head leaderboard (fixed rule) vs LGBM-alone""")

code("""EQ={'LGBM-alone (own tuned rule)':bench_eq}
for name,p in cand.items():
    row,eq=evaluate(p,name); RESULTS.append(row); EQ[name]=eq
L=pd.DataFrame(RESULTS)
L=L.sort_values('sharpe',ascending=False).reset_index(drop=True)
show=L.copy()
for c in ['ret','maxdd','win','alpha']: show[c]=(show[c]*100).round(1)
show['sharpe']=show['sharpe'].round(2); show['sortino']=show['sortino'].round(2)
print(show[['name','ret','sharpe','sortino','maxdd','trades','nL','nS','win','alpha']].to_string(index=False))
beat=L[(L['sharpe']>bench_row['sharpe'])&(L['name']!=bench_row['name'])]
print('\\nMerge strategies that BEAT LGBM-alone on Sharpe:',
      list(beat['name']) if len(beat) else 'NONE — LGBM-alone dominates under the fixed rule')""")

code("""# Equity curves of the top methods vs benchmark
top=list(L['name'].head(5))
fig,ax=plt.subplots(figsize=(13,6))
cmap={bench_row['name']:'k'}
for i,nm in enumerate(top):
    eq=EQ[nm]; ax.plot(oos.index,(eq-1)*100,lw=2 if nm==bench_row['name'] else 1.3,
        ls='-' if nm==bench_row['name'] else '--',label=f"{nm}  ({eq[-1]-1:+.0%})")
bh=oos['close'].values/oos['close'].values[0]; ax.plot(oos.index,(bh-1)*100,color=GREY,ls=':',label='BTC B&H')
for (rs,re),rc in [(REGIME_CHOP,'#9E9E9E'),(REGIME_BULL,'#26A69A'),(REGIME_BEAR,'#EF5350')]:
    ax.axvspan(rs,min(re,oos.index[-1]),alpha=0.07,color=rc)
ax.axhline(0,color=GREY,lw=0.6,ls=':'); ax.set_ylabel('Return (%)'); ax.legend(fontsize=8)
ax.set_title('Ensemble bake-off — top 5 vs LGBM-alone (fixed rule, OOS)',fontweight='bold')
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1,4,7,10]))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y')); plt.setp(ax.xaxis.get_majorticklabels(),rotation=30,ha='right')
fig.tight_layout(); fig.savefig(ARTS/'01_bakeoff_equity.png'); plt.show()""")

md("""## 6 · Winner: leak-free trading-grid tuning + regime breakdown

> **Q:** under a *fair* execution search (trading grid fit on pre-OOS only), does the best merged
> signal finally beat LGBM-alone? If not, the honest conclusion is to **ship LGBM-alone**.""")

code("""TRADING_GRID={'long_threshold':[0.55,0.58,0.60,0.63],'short_threshold':[0.30,0.35,0.40,0.45],
    'entry_atr_mult':[0.3,0.6,1.0],'sl_atr_mult':[1.5,2.0,2.5],'tp_atr_mult':[2.0,2.5,3.0],
    'min_sl':[0.01,0.015],'min_hold':[4,8],'max_hold':[24,48],'cooldown':[2,3]}
_tk=list(TRADING_GRID); _tc=list(itertools.product(*TRADING_GRID.values()))
def grid_tune(p_up, gv_idx, min_trades=120):
    \"\"\"Search trading params on the PRE-OOS (grid-val) window only — leak-free.\"\"\"
    sub=P.reindex(gv_idx); pa=p_up.reindex(gv_idx).values
    cl=sub['close'].values; hi=sub['high'].values; lo=sub['low'].values; at=sub['atr_14_pct'].values
    best=None
    for v in _tc:
        pr=dict(zip(_tk,v))
        if pr['short_threshold']>=pr['long_threshold'] or pr['max_hold']<pr['min_hold']: continue
        eq,tr=_run_backtest(pa,cl,hi,lo,at,**pr)
        if len(tr)<min_trades: continue
        sh=_sharpe(eq)
        if best is None or sh>best[0]: best=(sh,pr)
    return best[1] if best else None

winner=L.iloc[0]['name']
gv_idx=P.index[(P.index>=pd.Timestamp('2023-01-01'))&(P.index<OOS_START)]
wp=(bench_eq if winner==bench_row['name'] else cand[winner])
wp_series=RAW['lgbm'] if winner==bench_row['name'] else cand[winner]
tuned=grid_tune(wp_series, gv_idx)
print(f'Winner by fixed-rule Sharpe: {winner}')
print(f'Leak-free tuned trading rule: {tuned}')
if tuned:
    row,eq=evaluate(wp_series, winner+' (tuned)', rule=tuned)
    print('\\nTuned OOS:', {k:(round(v,3) if isinstance(v,float) else v) for k,v in row.items()})
    print('\\nRegime breakdown:'); print(regime_breakdown(eq,winner).to_string(index=False))""")

md("""## 7 · Verdict & artifacts""")

code("""verdict=('A merged signal beats LGBM-alone.' if len(beat) else
         'No merge strategy beats LGBM-alone out-of-sample — ship LGBM-alone.')
out=dict(notebook='06_ensemble_v1', created=pd.Timestamp.now().isoformat(),
    benchmark=bench_row, leaderboard=L.to_dict('records'),
    beats_lgbm=list(beat['name']), winner=winner, verdict=verdict,
    design=dict(eval='leak-free walk-forward', inputs_tested=['signals-only','signals+stationary-regime'],
                trailing_window=TRAIL, embargo_h=EMBARGO_H))
json.dump(out, open(ARTS/'results.json','w'), indent=2, default=float)
L.to_csv(ARTS/'leaderboard.csv', index=False)
# save the best merged signal's OOS probs (for downstream use, if it wins)
if len(beat):
    bp=cand[winner].reindex(oos.index)
    np.save(ARTS/'oos_probs.npy', bp.values.astype(np.float32))
    np.save(ARTS/'oos_index.npy', oos.index.values.astype('int64'))
print(verdict); print('Artifacts →', ARTS)""")

nb['cells']=cells
out=Path('src/hmats/notebooks_v2/06_ensemble_v1.ipynb'); nbf.write(nb,str(out))
print(f'Wrote {out} with {len(cells)} cells')
