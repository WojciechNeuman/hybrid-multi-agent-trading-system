#!/usr/bin/env python
# coding: utf-8

# # 07 -- LGBM Walk-Forward Optimization (vtrain7)
# 
# ## What changed vs vtrain6
# 
# | # | Problem (vtrain6) | Fix applied here |
# |---|-------------------|-----------------|
# | 1 | Naive binary label (next-close > current-close) | **Triple Barrier Method (TBM)**: Class 1=Long (hit +2xATR), Class 0=Short (hit -2xATR), Class 2=Neutral (24h timeout) |
# | 2 | Binary LGBM (P_up only) | **Multiclass LGBM** (`objective: multiclass`, `num_class: 3`); signals from `P_up=probs[:,1]` and `P_down=probs[:,0]` |
# | 3 | Single model config (fixed hyperparams) | **Combined Model+Trading Grid Search**: 48 model configs x trading grid, decoupled via OOS probability cache |
# | 4 | All trades routed to Futures (funding drag on longs) | **Spot/Futures routing**: Longs -> Spot (0% funding), Shorts -> Futures (+0.00077%/h funding *received*) |
# | 5 | Fixed feature selection (pre-computed CSV) | Dynamic RF importance + correlation filter per model config |
# 
# ## Architecture
# 
# ```
# Phase 0: TBM Labels      -> Replace `label` with 3-class TBM targets
# Phase 1: Combined Grid Search
#   For each MODEL_GRID config (48):
#     Feature selection: RF importance + corr filter -> top N
#     Purged K-Fold (K=5, embargo=168h) -> OOS probs (N, 3)  [cached]
#     For each TRADING_GRID combo:
#       run_backtest_v7(p_up, p_down, ...) -> score
# Phase 2: WFO on test set (best model config + best trading params)
# Phase 3: Final evaluation + comparison
# ```
# 
# ## Spot vs Futures execution
# - **Longs (Spot):** limit entry 0% fee / TP 0% fee / SL+time 0.05% taker / Funding 0.00%/h
# - **Shorts (Futures):** limit entry 0% fee / TP 0% fee / SL+time 0.05% taker / Funding +0.00077%/h RECEIVED
# 

# In[1]:


SYMBOL    = 'BTCUSDT'
INTERVAL  = '1h'
TRAIN_END = '2024-06-01'
VAL_END   = '2024-11-10'

# Purged K-Fold
K                    = 5
EMBARGO              = 168
KFOLD_INTERNAL_VAL_H = 2500
EARLY_STOPPING_ROUNDS = 50

# Walk-Forward Optimization
WFO_STEP_H         = 720
WFO_INTERNAL_VAL_H = 2500
WFO_PATIENCE       = 50

# TBM labeling
# Class 1 = Long  (upper barrier close*(1+TBM_ATR_MULT*atr) hit first)
# Class 0 = Short (lower barrier close*(1-TBM_ATR_MULT*atr) hit first)
# Class 2 = Neutral (TBM_HORIZON vertical barrier hit first)
TBM_ATR_MULT = 2.0
TBM_HORIZON  = 24

# Execution: Spot/Futures routing
MAKER_FEE         = 0.0000     # 0%    - limit orders (entries + TP exits, both markets)
SPOT_TAKER_FEE    = 0.0005     # 0.05% - Spot market orders (SL/time exits, longs)
FUTURES_TAKER_FEE = 0.0005     # 0.05% - Futures market orders (SL/time exits, shorts)
BUFFER            = 0.0005     # 5bp penetration buffer for limit fill confirmation
SPOT_FUNDING_H    = 0.0        # Spot longs: zero funding drag
SHORT_FUNDING_H   = 0.0000077  # Futures shorts: +0.00077%/h RECEIVED per hour held

# Optimisation
OPTIMISE_METRIC = 'sharpe'
MIN_TRADES      = 30
TOP_N           = 20

# Model grid (3x2x2x2x2 = 48 configs)
RF_N_ESTIMATORS = 200

MODEL_GRID = {
    'top_n_features':    [20, 35, 50],
    'corr_threshold':    [0.85, 0.90],
    'num_leaves':        [31, 63],
    'min_child_samples': [30, 50],
    'learning_rate':     [0.01, 0.02],
}

BASE_LGB_PARAMS = {
    'objective':        'multiclass',
    'num_class':        3,
    'metric':           'multi_logloss',
    'n_estimators':     1000,
    'max_depth':        -1,
    'subsample':        0.8,
    'colsample_bytree': 0.8,
    'reg_alpha':        0.1,
    'reg_lambda':       1.0,
    'random_state':     42,
    'n_jobs':           -1,
    'verbose':          -1,
}

# Trading grid
# long_threshold  = min P(class 1) to enter long
# short_threshold = min P(class 0) to enter short
TRADING_GRID = {
    'long_threshold':   [0.40, 0.45, 0.50, 0.55],
    'short_threshold':  [0.40, 0.45, 0.50, 0.55],
    'entry_atr_mult':   [0.3, 0.6, 1.0],
    'sl_atr_multiplier':[1.5, 2.0, 2.5],
    'tp_atr_multiplier':[2.0, 2.5, 3.0],
    'min_sl':           [0.010, 0.015],
    'min_hold':         [4, 8],
    'max_hold':         [24, 48],
    'cooldown':         [2, 3],
}


# In[2]:


import itertools
import json
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from hmats.data.splits import calendar_split

warnings.filterwarnings('ignore')

mpl.rcParams.update({
    'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelsize': 10, 'axes.titlesize': 11,
    'xtick.labelsize': 9,  'ytick.labelsize': 9,
    'legend.fontsize': 9,  'legend.framealpha': 0.85,
    'figure.dpi': 120, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})
ACCENT = '#F7931A'; BLUE = '#2962FF'; GREY = '#9E9E9E'
RED    = '#EF5350'; GREEN = '#26A69A'; PURPLE = '#7B1FA2'

REPO_ROOT    = Path.cwd().parents[2]
if not (REPO_ROOT / 'pyproject.toml').exists():
    REPO_ROOT = Path.cwd()
FEATURES_DIR = REPO_ROOT / 'data' / 'features'
RAW_DIR      = REPO_ROOT / 'data' / 'raw'
MODELS_DIR   = REPO_ROOT / 'local_models'
if not MODELS_DIR.exists():
    MODELS_DIR = REPO_ROOT / 'models'
FIGURES_DIR  = REPO_ROOT / 'figures' / 'grid_lgbm'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print(f'REPO_ROOT  : {REPO_ROOT}')
print(f'MODELS_DIR : {MODELS_DIR}')
print(f'vtrain7 -- TBM labels | multiclass LGBM | Spot/Futures routing')
print(f'Spot funding={SPOT_FUNDING_H*100:.4f}%/h  Short funding received={SHORT_FUNDING_H*100:.5f}%/h')


# In[3]:


feat_df = pd.read_parquet(FEATURES_DIR / f'{SYMBOL}_{INTERVAL}_features.parquet')
feat_df.index = feat_df.index.tz_localize(None) if feat_df.index.tz else feat_df.index

raw_df = pd.read_parquet(RAW_DIR / f'{SYMBOL}_{INTERVAL}.parquet')
raw_df.index = raw_df.index.tz_localize(None) if raw_df.index.tz else raw_df.index
feat_df = feat_df.join(raw_df[['high', 'low']], how='left')
feat_df = feat_df.dropna(subset=['high', 'low'])


def make_tbm_labels(close_arr, high_arr, low_arr, atr_arr, atr_mult=2.0, horizon=24):
    """Triple Barrier Method. Scans forward `horizon` bars from each row.
    Class 1: upper barrier close*(1+atr_mult*atr) hit first (Long).
    Class 0: lower barrier close*(1-atr_mult*atr) hit first (Short).
    Class 2: time barrier hit first (Neutral).
    Last `horizon` rows labelled -1 (dropped -- insufficient lookahead).
    """
    N = len(close_arr)
    labels = np.full(N, 2, dtype=np.int8)
    for i in range(N - horizon):
        c     = close_arr[i]
        atr   = atr_arr[i]
        upper = c * (1.0 + atr_mult * atr)
        lower = c * (1.0 - atr_mult * atr)
        for j in range(i + 1, i + horizon + 1):
            if high_arr[j] >= upper:
                labels[i] = 1; break
            elif low_arr[j] <= lower:
                labels[i] = 0; break
    labels[N - horizon:] = -1
    return labels


tbm = make_tbm_labels(
    feat_df['close'].values, feat_df['high'].values, feat_df['low'].values,
    feat_df['atr_14_pct'].values, TBM_ATR_MULT, TBM_HORIZON,
)
feat_df['tbm_label'] = tbm
feat_df = feat_df[feat_df['tbm_label'] >= 0].copy()

# Split
train_df, val_df, test_df = calendar_split(feat_df, train_end=TRAIN_END, val_end=VAL_END)
trainval_df = pd.concat([train_df, val_df]).sort_index()

# Feature columns
_EXCLUDE = {'open', 'high', 'low', 'close', 'volume', 'label', 'tbm_label',
            'return', 'log_return', 'target', 'future_return'}
feature_cols = [c for c in feat_df.columns
                if c not in _EXCLUDE
                and not c.startswith('future_')
                and pd.api.types.is_numeric_dtype(feat_df[c])]

def _arr(df, col): return df[col].values.astype(np.float64)

X_tv   = trainval_df[feature_cols].values.astype(np.float32)
y_tv   = trainval_df['tbm_label'].values.astype(int)
X_test = test_df[feature_cols].values.astype(np.float32)
y_test = test_df['tbm_label'].values.astype(int)

tv_close = _arr(trainval_df, 'close');  tv_high  = _arr(trainval_df, 'high')
tv_low   = _arr(trainval_df, 'low');    tv_atr   = _arr(trainval_df, 'atr_14_pct')
tv_index = trainval_df.index

test_close = _arr(test_df, 'close');    test_high  = _arr(test_df, 'high')
test_low   = _arr(test_df, 'low');      test_atr   = _arr(test_df, 'atr_14_pct')
test_index = test_df.index

print(f'Features : {len(feature_cols)}')
for split, df in [('TrainVal', trainval_df), ('Test', test_df)]:
    vc = df['tbm_label'].value_counts().sort_index()
    n  = len(df)
    print(f'{split:8s}: {n:>7,} bars | '
          f'Short(0)={vc.get(0,0):,} ({vc.get(0,0)/n:.1%}) | '
          f'Long(1)={vc.get(1,0):,} ({vc.get(1,0)/n:.1%}) | '
          f'Neutral(2)={vc.get(2,0):,} ({vc.get(2,0)/n:.1%})')

fig, axes = plt.subplots(1, 2, figsize=(12, 3))
for ax, (lbl, df) in zip(axes, [('TrainVal', trainval_df), ('Test', test_df)]):
    vc = df['tbm_label'].value_counts().sort_index()
    ax.bar(['Short(0)', 'Long(1)', 'Neutral(2)'],
           [vc.get(0, 0), vc.get(1, 0), vc.get(2, 0)],
           color=[RED, GREEN, GREY], alpha=0.8)
    ax.set_title(f'TBM label distribution -- {lbl}', fontweight='bold')
    ax.set_ylabel('Count'); ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURES_DIR / 'tbm_label_distribution_v7.png'); plt.show()


# In[4]:


def select_features(X_train, y_train, cols, top_n, corr_thresh, n_estimators=200):
    """RF importance + greedy correlation filter -> top_n features."""
    rf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=8,
        min_samples_leaf=50, n_jobs=-1, random_state=42,
    )
    rf.fit(X_train, y_train)
    imp_df = (pd.DataFrame({'feature': cols, 'importance': rf.feature_importances_})
              .sort_values('importance', ascending=False).reset_index(drop=True))
    corr_mat = pd.DataFrame(X_train, columns=cols).corr().abs()
    kept = []
    for feat in imp_df['feature']:
        if not any(corr_mat.loc[feat, k] > corr_thresh for k in kept):
            kept.append(feat)
        if len(kept) >= top_n:
            break
    return kept[:top_n]


print('select_features() defined')


# In[5]:


def run_backtest_v7(
        p_up, p_down, close_arr, high_arr, low_arr, atr_arr, params,
        spot_taker=SPOT_TAKER_FEE, fut_taker=FUTURES_TAKER_FEE,
        maker_fee=MAKER_FEE, buf=BUFFER,
        spot_fund=SPOT_FUNDING_H, short_fund=SHORT_FUNDING_H):
    """Multiclass backtester with Spot/Futures routing.

    Longs  (Spot)    : 0% entry/TP fee | SPOT_TAKER_FEE on SL/time | zero funding
    Shorts (Futures) : 0% entry/TP fee | FUTURES_TAKER_FEE on SL/time | funding RECEIVED
    Pessimistic both-hit: SL wins when wick penetrates both SL and TP in same bar.
    Exit-confidence for longs : exit if p_up   < (1 - long_threshold)
    Exit-confidence for shorts: exit if p_down < (1 - short_threshold)
    """
    lt         = params['long_threshold']
    st         = params['short_threshold']
    exit_long  = 1.0 - lt
    exit_short = 1.0 - st
    ent_atr    = params['entry_atr_mult']
    sl_m       = params['sl_atr_multiplier']
    tp_m       = params['tp_atr_multiplier']
    min_sl     = params['min_sl']
    min_hold   = int(params['min_hold'])
    max_hold   = int(params['max_hold'])
    cd_n       = int(params['cooldown'])

    cash = 1.0; units = 0.0; entry_cash = 0.0
    in_pos = False; direction = None
    entry_px = 0.0; dynamic_sl = 0.0; dynamic_tp = 0.0
    hold_count = 0; cooldown = 0; entry_bar = -1
    pending = None

    equity_curve = [1.0]
    trade_log    = []
    n_signals = 0; n_fills = 0; n_expires = 0

    N = len(close_arr)
    for i in range(N):
        px  = close_arr[i]; hi = high_arr[i]; lo = low_arr[i]
        pup = p_up[i];      pdn = p_down[i];  atr = atr_arr[i]
        if cooldown > 0:
            cooldown -= 1

        # 1. Fill / expire pending limit order (TIF = 1 bar)
        if pending is not None:
            lp = pending['limit_px']
            if pending['direction'] == 'long':
                if lo < lp * (1.0 - buf):
                    units = cash * (1.0 - maker_fee) / lp
                    cash = 0.0; in_pos = True; direction = 'long'
                    entry_px = lp; entry_bar = i; hold_count = 0
                    dynamic_sl = pending['sl']; dynamic_tp = pending['tp']
                    n_fills += 1
                else:
                    n_expires += 1
            else:
                if hi > lp * (1.0 + buf):
                    entry_cash = cash * (1.0 - maker_fee)
                    cash = 0.0; units = entry_cash / lp
                    in_pos = True; direction = 'short'
                    entry_px = lp; entry_bar = i; hold_count = 0
                    dynamic_sl = pending['sl']; dynamic_tp = pending['tp']
                    n_fills += 1
                else:
                    n_expires += 1
            pending = None

        # 2. Manage open position
        if in_pos and i > entry_bar:
            hold_count += 1
            if direction == 'long':
                units      *= (1.0 - spot_fund)   # Spot: zero drag (spot_fund=0)
            else:
                entry_cash *= (1.0 + short_fund)   # Futures short: receive funding

            reason = None; exit_px = px; pnl = 0.0

            if direction == 'long':
                sl_price = entry_px * (1.0 - dynamic_sl)
                tp_price = entry_px * (1.0 + dynamic_tp)
                tp_check = tp_price * (1.0 + buf)
                sl_hit = lo <= sl_price
                tp_hit = hi > tp_check
                if sl_hit and tp_hit:
                    reason = 'sl'; exit_px = sl_price
                    pnl  = (exit_px - entry_px) / entry_px
                    cash = units * exit_px * (1.0 - spot_taker); units = 0.0
                elif sl_hit:
                    reason = 'sl'; exit_px = sl_price
                    pnl  = (exit_px - entry_px) / entry_px
                    cash = units * exit_px * (1.0 - spot_taker); units = 0.0
                elif tp_hit:
                    reason = 'tp'; exit_px = tp_price
                    pnl  = (exit_px - entry_px) / entry_px
                    cash = units * exit_px * (1.0 - maker_fee); units = 0.0
                elif hold_count >= max_hold:
                    reason = 'max_hold'; exit_px = px
                    pnl  = (px - entry_px) / entry_px
                    cash = units * px * (1.0 - spot_taker); units = 0.0
                elif hold_count >= min_hold and pup < exit_long:
                    reason = 'conf'; exit_px = px
                    pnl  = (px - entry_px) / entry_px
                    cash = units * px * (1.0 - spot_taker); units = 0.0
            else:
                sl_price = entry_px * (1.0 + dynamic_sl)
                tp_price = entry_px * (1.0 - dynamic_tp)
                tp_check = tp_price * (1.0 - buf)
                sl_hit = hi >= sl_price
                tp_hit = lo < tp_check
                if sl_hit and tp_hit:
                    reason = 'sl'; exit_px = sl_price
                    gross = (entry_px - exit_px) / entry_px; pnl = gross
                    cash  = entry_cash * (1.0 + gross) * (1.0 - fut_taker)
                elif sl_hit:
                    reason = 'sl'; exit_px = sl_price
                    gross = (entry_px - exit_px) / entry_px; pnl = gross
                    cash  = entry_cash * (1.0 + gross) * (1.0 - fut_taker)
                elif tp_hit:
                    reason = 'tp'; exit_px = tp_price
                    gross = (entry_px - exit_px) / entry_px; pnl = gross
                    cash  = entry_cash * (1.0 + gross) * (1.0 - maker_fee)
                elif hold_count >= max_hold:
                    reason = 'max_hold'; exit_px = px
                    gross = (entry_px - px) / entry_px; pnl = gross
                    cash  = entry_cash * (1.0 + gross) * (1.0 - fut_taker)
                elif hold_count >= min_hold and pdn < exit_short:
                    reason = 'conf'; exit_px = px
                    gross = (entry_px - px) / entry_px; pnl = gross
                    cash  = entry_cash * (1.0 + gross) * (1.0 - fut_taker)

            if reason:
                trade_log.append({'direction': direction, 'pnl_pct': pnl,
                                   'hold_h': hold_count, 'reason': reason})
                in_pos = False; direction = None; hold_count = 0; cooldown = cd_n

        # 3. Place new pending limit order if flat
        if not in_pos and pending is None and cooldown == 0:
            sl_v = max(sl_m * atr, min_sl); tp_v = tp_m * atr
            if pup >= lt:
                limit_px = px * (1.0 - ent_atr * atr)
                pending = {'direction': 'long',  'limit_px': limit_px, 'sl': sl_v, 'tp': tp_v}
                n_signals += 1
            elif pdn >= st:
                limit_px = px * (1.0 + ent_atr * atr)
                pending = {'direction': 'short', 'limit_px': limit_px, 'sl': sl_v, 'tp': tp_v}
                n_signals += 1

        # Equity mark-to-market
        if   in_pos and direction == 'long':
            equity_curve.append(units * px)
        elif in_pos and direction == 'short':
            equity_curve.append(entry_cash * (1.0 + (entry_px - px) / entry_px))
        else:
            equity_curve.append(cash)

    # Force-close at end
    if in_pos:
        px = close_arr[-1]
        if direction == 'long':
            gross = (px - entry_px) / entry_px
            cash  = units * px * (1.0 - spot_taker)
        else:
            gross = (entry_px - px) / entry_px
            cash  = entry_cash * (1.0 + gross) * (1.0 - fut_taker)
        trade_log.append({'direction': direction, 'pnl_pct': gross,
                           'hold_h': hold_count, 'reason': 'eod'})
        equity_curve[-1] = cash

    tdf = pd.DataFrame(trade_log)
    tdf.attrs['n_signals'] = n_signals
    tdf.attrs['n_fills']   = n_fills
    tdf.attrs['n_expires'] = n_expires
    return np.array(equity_curve[1:]), tdf


def score_equity(equity_arr, trades_df, metric):
    if trades_df.empty: return -np.inf
    eq  = equity_arr
    ret = np.log(np.maximum(eq[1:], 1e-12) / np.maximum(eq[:-1], 1e-12))
    ann = 24 * 365
    if metric == 'sharpe':       return float(ret.mean() / (ret.std(ddof=1) + 1e-12) * np.sqrt(ann))
    if metric == 'total_return': return float(eq[-1] - 1)
    if metric == 'calmar':
        ar  = float((eq[-1] ** (ann / len(eq))) - 1)
        pk  = np.maximum.accumulate(eq)
        mdd = float(((eq - pk) / (pk + 1e-12)).min())
        return ar / (abs(mdd) + 1e-6)
    if metric == 'win_rate':     return float((trades_df['pnl_pct'] > 0).mean())
    if metric == 'profit_factor':
        g = trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].sum()
        l = trades_df[trades_df['pnl_pct'] < 0]['pnl_pct'].abs().sum()
        return float(g / (l + 1e-6))
    return -np.inf


print('run_backtest_v7() defined')
print(f'  Spot  longs  : maker=0%  TP=0%  SL/time={SPOT_TAKER_FEE*100:.3f}%  funding=0%/h')
print(f'  Futures shorts: maker=0%  TP=0%  SL/time={FUTURES_TAKER_FEE*100:.3f}%  '
      f'funding received=+{SHORT_FUNDING_H*100:.5f}%/h')


# In[6]:


def purged_kfold_oos_probs(X, y, lgb_params,
                            k=K, embargo=EMBARGO,
                            internal_val_h=KFOLD_INTERNAL_VAL_H,
                            es_rounds=EARLY_STOPPING_ROUNDS):
    """Purged K-Fold cross-validation for multiclass LGBM.
    Returns OOS probability array shape (N, num_class); NaN where fold was embargoed.
    """
    N         = len(X)
    fold_size = N // k
    num_class = lgb_params.get('num_class', 2)
    oos_probs = np.full((N, num_class), np.nan)

    for fold_k in range(k):
        fold_start = fold_k * fold_size
        fold_end   = (fold_k + 1) * fold_size if fold_k < k - 1 else N
        emb_start  = max(0, fold_start - embargo)
        emb_end    = min(N, fold_end   + embargo)

        if emb_start >= 200:
            n_int     = min(internal_val_h, emb_start)
            int_end   = emb_start
            int_start = emb_start - n_int
        else:
            n_int     = min(internal_val_h, N - emb_end)
            int_start = emb_end
            int_end   = min(N, emb_end + n_int)
            n_int     = int_end - int_start

        mask = np.ones(N, dtype=bool)
        mask[emb_start:emb_end] = False
        mask[int_start:int_end] = False

        ds_tr = lgb.Dataset(X[mask],              label=y[mask])
        ds_vl = lgb.Dataset(X[int_start:int_end], label=y[int_start:int_end],
                             reference=ds_tr)
        model = lgb.train(
            lgb_params, ds_tr,
            valid_sets=[ds_tr, ds_vl], valid_names=['train', 'val'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=es_rounds, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        oos_probs[fold_start:fold_end] = model.predict(X[fold_start:fold_end])
        del model

    return oos_probs   # shape (N, 3)


print('purged_kfold_oos_probs() defined')


# In[7]:


model_keys   = list(MODEL_GRID.keys())
model_combos = list(itertools.product(*[MODEL_GRID[k] for k in model_keys]))

trading_keys   = list(TRADING_GRID.keys())
trading_combos = list(itertools.product(*[TRADING_GRID[k] for k in trading_keys]))
valid_trading  = [
    dict(zip(trading_keys, v)) for v in trading_combos
    if v[0] >= 0.40 and v[1] >= 0.40
]

total = len(model_combos) * len(valid_trading)
print(f'Model configs    : {len(model_combos):,}')
print(f'Trading combos   : {len(valid_trading):,}')
print(f'Total evaluations: {total:,}  '
      f'(train {len(model_combos)} models, run {len(valid_trading)} backtests each)')

all_results    = []
oos_prob_cache = {}   # model_id -> dict with arrays + metadata
t0 = time.perf_counter()

for mid, m_vals in enumerate(tqdm(model_combos, desc='Model configs')):
    cfg = dict(zip(model_keys, m_vals))

    # Feature selection on full trainval (once per model config)
    sel_feats = select_features(
        X_tv, y_tv, feature_cols,
        top_n=cfg['top_n_features'],
        corr_thresh=cfg['corr_threshold'],
        n_estimators=RF_N_ESTIMATORS,
    )
    fi       = [feature_cols.index(f) for f in sel_feats]
    X_tv_sel = X_tv[:, fi]

    lgb_params = {
        **BASE_LGB_PARAMS,
        'num_leaves':        cfg['num_leaves'],
        'min_child_samples': cfg['min_child_samples'],
        'learning_rate':     cfg['learning_rate'],
    }

    # Purged K-Fold -> OOS probs shape (N, 3)
    oos_probs = purged_kfold_oos_probs(X_tv_sel, y_tv, lgb_params)

    valid_idx = np.isfinite(oos_probs[:, 0])
    p_up  = oos_probs[valid_idx, 1].astype(np.float64)
    p_dn  = oos_probs[valid_idx, 0].astype(np.float64)
    c_arr = tv_close[valid_idx]; h_arr = tv_high[valid_idx]
    l_arr = tv_low[valid_idx];   a_arr = tv_atr[valid_idx]
    oos_y = y_tv[valid_idx]

    auc_long  = roc_auc_score((oos_y == 1).astype(int), p_up)
    auc_short = roc_auc_score((oos_y == 0).astype(int), p_dn)
    tqdm.write(
        f'  [{mid:03d}] top_n={cfg["top_n_features"]} lr={cfg["learning_rate"]:.2f} '
        f'leaves={cfg["num_leaves"]} mc={cfg["min_child_samples"]} '
        f'corr={cfg["corr_threshold"]} | '
        f'AUC_long={auc_long:.4f}  AUC_short={auc_short:.4f}'
    )

    oos_prob_cache[mid] = {
        'p_up': p_up, 'p_dn': p_dn,
        'c': c_arr, 'h': h_arr, 'l': l_arr, 'a': a_arr,
        'oos_idx': tv_index[valid_idx],
        'sel_feats': sel_feats, 'fi': fi,
        'config': cfg, 'auc_long': auc_long, 'auc_short': auc_short,
    }

    # Inner loop: trading grid
    for tp in valid_trading:
        eq, tdf = run_backtest_v7(p_up, p_dn, c_arr, h_arr, l_arr, a_arr, tp)
        if len(tdf) < MIN_TRADES:
            continue
        s   = score_equity(eq, tdf, OPTIMISE_METRIC)
        ret = np.log(np.maximum(eq[1:], 1e-12) / np.maximum(eq[:-1], 1e-12))
        pk  = np.maximum.accumulate(eq)
        fr  = tdf.attrs.get('n_fills', 0) / max(tdf.attrs.get('n_signals', 1), 1)
        all_results.append({
            'model_id':     mid,
            **tp,
            'score':        s,
            'total_return': float(eq[-1] - 1),
            'sharpe':       float(ret.mean() / (ret.std(ddof=1) + 1e-12) * np.sqrt(24 * 365)),
            'max_dd':       float(((eq - pk) / (pk + 1e-12)).min()),
            'n_trades':     len(tdf),
            'win_rate':     float((tdf['pnl_pct'] > 0).mean()),
            'n_long':       int((tdf['direction'] == 'long').sum()),
            'n_short':      int((tdf['direction'] == 'short').sum()),
            'n_sl':         int((tdf['reason'] == 'sl').sum()),
            'n_tp':         int((tdf['reason'] == 'tp').sum()),
            'fill_rate':    fr,
        })

results_df = (pd.DataFrame(all_results)
              .sort_values('score', ascending=False)
              .reset_index(drop=True))
elapsed = time.perf_counter() - t0
print(f'\nDone in {elapsed:.1f}s ({elapsed/60:.1f} min) -- {len(results_df):,} valid results')
if not results_df.empty:
    print(f'Best {OPTIMISE_METRIC} : {results_df["score"].iloc[0]:.4f}')
    print(f'Best return      : {results_df["total_return"].iloc[0]:+.2%}')
    print(f'Best model_id    : {int(results_df["model_id"].iloc[0])}')


# In[8]:


from IPython.display import display

display_cols = [
    'score', 'total_return', 'sharpe', 'max_dd', 'win_rate', 'fill_rate',
    'n_trades', 'n_long', 'n_short', 'n_sl', 'n_tp', 'model_id',
    'long_threshold', 'short_threshold', 'entry_atr_mult',
    'sl_atr_multiplier', 'tp_atr_multiplier', 'min_sl',
    'min_hold', 'max_hold', 'cooldown',
]
top = results_df[display_cols].head(TOP_N).copy()
for col, fmt in [('total_return', '{:+.2%}'), ('max_dd', '{:.2%}'),
                 ('win_rate', '{:.1%}'),    ('fill_rate', '{:.1%}'),
                 ('score', '{:.4f}'),       ('sharpe', '{:.3f}')]:
    top[col] = top[col].map(fmt.format)

best        = results_df.iloc[0]
best_mid    = int(best['model_id'])
best_params = {k: best[k] for k in trading_keys}
best_cfg    = oos_prob_cache[best_mid]['config']

print(f'Top {TOP_N} by {OPTIMISE_METRIC}:\n')
print(top.to_string(index=True))
print(f'\n-- Best config ------------------------------------------')
print(f'  Model config (id={best_mid}):')
for k, v in best_cfg.items():
    print(f'    {k:<24}: {v}')
print(f'  Trading params:')
for k, v in best_params.items():
    print(f'    {k:<24}: {v}')
print(f'  AUC Long  : {oos_prob_cache[best_mid]["auc_long"]:.4f}')
print(f'  AUC Short : {oos_prob_cache[best_mid]["auc_short"]:.4f}')
print(f'  Sharpe    : {best["sharpe"]:.4f}')
print(f'  Return    : {best["total_return"]:+.2%}')
print(f'  MaxDD     : {best["max_dd"]:.2%}')
print(f'  Trades    : {int(best["n_trades"])}')
print(f'  Fill rate : {best["fill_rate"]:.1%}')


# In[9]:


if not results_df.empty:
    plot_cols = [
        ('score',        f'Opt. metric ({OPTIMISE_METRIC})',  BLUE),
        ('total_return', 'Total return (OOS)',                ACCENT),
        ('sharpe',       'Sharpe (ann., OOS)',                BLUE),
        ('max_dd',       'Max drawdown (OOS)',                RED),
        ('win_rate',     'Win rate (OOS)',                    GREEN),
        ('fill_rate',    'Order fill rate',                   PURPLE),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, (col, title, color) in zip(axes.flat, plot_cols):
        data = results_df[col]
        ax.hist(data, bins=50, color=color, alpha=0.75, edgecolor='none')
        ax.axvline(data.median(), color='black', lw=1.2, ls='--',
                   label=f'Median {data.median():.3f}')
        ax.axvline(results_df[col].iloc[0], color=ACCENT, lw=1.5,
                   label=f'Best {results_df[col].iloc[0]:.3f}')
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel(col); ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
    fig.suptitle(
        f'Grid distributions -- TBM multiclass | Spot/Futures routing | vtrain7 '
        f'(n={len(results_df):,})',
        fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'grid_distributions_v7.png'); plt.show()


# In[10]:


if not results_df.empty:
    param_keys = trading_keys
    ncols = 3; nrows = int(np.ceil(len(param_keys) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3.5))
    for ax, param in zip(axes.flat, param_keys):
        grp = results_df.groupby(param)['score'].agg(['median', 'std']).reset_index()
        x   = grp[param].astype(str)
        ax.bar(x, grp['median'], color=BLUE, alpha=0.7)
        ax.errorbar(x, grp['median'], yerr=grp['std'],
                    fmt='none', color='black', capsize=4, lw=1.2)
        ax.set_title(param, fontweight='bold')
        ax.set_xlabel('Value'); ax.set_ylabel(f'Median {OPTIMISE_METRIC}')
        ax.grid(axis='y', alpha=0.3)
        best_val = str(best[param])
        for tick in ax.get_xticklabels():
            if tick.get_text() == best_val:
                tick.set_color(ACCENT); tick.set_fontweight('bold')
    for ax in axes.flat[len(param_keys):]:
        ax.set_visible(False)
    fig.suptitle('Trading parameter sensitivity (vtrain7) -- orange bar = best value',
                 fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'grid_sensitivity_v7.png'); plt.show()

    # Per-model leaderboard
    pm = (results_df.groupby('model_id')
          .agg(best_score=('score', 'max'), best_return=('total_return', 'max'),
               median_score=('score', 'median'), n_valid=('score', 'count'))
          .sort_values('best_score', ascending=False).reset_index())
    pm['auc_long']  = pm['model_id'].map({k: v['auc_long']  for k, v in oos_prob_cache.items()})
    pm['auc_short'] = pm['model_id'].map({k: v['auc_short'] for k, v in oos_prob_cache.items()})
    print('Per-model ranking (by best score):\n')
    print(pm.head(10).to_string(index=False))


# In[11]:


cache   = oos_prob_cache[best_mid]
eq_oos, tdf_oos = run_backtest_v7(
    cache['p_up'], cache['p_dn'],
    cache['c'], cache['h'], cache['l'], cache['a'],
    best_params,
)
oos_idx = cache['oos_idx']
bh_oos  = cache['c'] / cache['c'][0]

min_len = min(len(oos_idx), len(eq_oos), len(bh_oos))
oos_idx = oos_idx[:min_len]; eq_oos = eq_oos[:min_len]; bh_oos = bh_oos[:min_len]

fig, axes = plt.subplots(2, 1, figsize=(13, 8),
                          gridspec_kw={'height_ratios': [3, 1.2], 'hspace': 0.08})
ax = axes[0]
ax.plot(oos_idx, eq_oos, color=ACCENT, lw=1.4, label='Best config OOS (Purged K-Fold)')
ax.plot(oos_idx, bh_oos, color=BLUE,   lw=1.2, ls='--', label='Buy & Hold')
ax.axhline(1.0, color=GREY, lw=0.7, ls=':')
ax.set_ylabel('Portfolio value'); ax.legend(); ax.grid(axis='y', alpha=0.3)
ax.set_title(
    f'vtrain7 OOS equity -- model_id={best_mid} | '
    f'TBM(+-{TBM_ATR_MULT}xATR, {TBM_HORIZON}h) | Spot longs / Futures shorts',
    fontweight='bold')

ax = axes[1]
pk_s = np.maximum.accumulate(eq_oos); pk_b = np.maximum.accumulate(bh_oos)
ax.fill_between(oos_idx, (eq_oos - pk_s) / (pk_s + 1e-12) * 100, 0,
                color=ACCENT, alpha=0.45, label='Best config')
ax.fill_between(oos_idx, (bh_oos - pk_b) / (pk_b + 1e-12) * 100, 0,
                color=BLUE, alpha=0.25, label='Buy & Hold')
ax.set_ylabel('Drawdown (%)'); ax.legend(); ax.grid(axis='y', alpha=0.3)

for ax in axes:
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
fig.tight_layout()
fig.savefig(FIGURES_DIR / 'oos_best_equity_v7.png'); plt.show()

ret_oos = np.log(np.maximum(eq_oos[1:], 1e-12) / np.maximum(eq_oos[:-1], 1e-12))
pk_oos  = np.maximum.accumulate(eq_oos)
print(f'OOS K-Fold | Sharpe={ret_oos.mean()/(ret_oos.std(ddof=1)+1e-12)*np.sqrt(24*365):.3f}  '
      f'Return={eq_oos[-1]-1:+.2%}  MaxDD={((eq_oos-pk_oos)/(pk_oos+1e-12)).min():.2%}  '
      f'Trades={len(tdf_oos)}  '
      f'Fill={tdf_oos.attrs["n_fills"]/max(tdf_oos.attrs["n_signals"],1):.1%}  '
      f'WR={(tdf_oos["pnl_pct"]>0).mean():.1%}')
print(f'Long={int((tdf_oos["direction"]=="long").sum())}  '
      f'Short={int((tdf_oos["direction"]=="short").sum())}  '
      f'TP={int((tdf_oos["reason"]=="tp").sum())}  '
      f'SL={int((tdf_oos["reason"]=="sl").sum())}  '
      f'Conf={int((tdf_oos["reason"]=="conf").sum())}')


# In[12]:


all_df = pd.concat([trainval_df, test_df]).sort_index()
n_tv   = len(trainval_df); n_test = len(test_df)

fi_best     = oos_prob_cache[best_mid]['fi']
X_all       = np.concatenate([X_tv, X_test], axis=0)[:, fi_best]
y_all       = np.concatenate([y_tv, y_test])

lgb_params_best = {
    **BASE_LGB_PARAMS,
    'num_leaves':        best_cfg['num_leaves'],
    'min_child_samples': best_cfg['min_child_samples'],
    'learning_rate':     best_cfg['learning_rate'],
}

wfo_probs   = np.full((n_test, 3), np.nan)
wfo_n_trees = []; step = 0
t0_wfo = time.perf_counter(); t = 0

while t < n_test:
    step_end    = min(t + WFO_STEP_H, n_test)
    n_wfo_train = n_tv + t
    X_wfo = X_all[:n_wfo_train]; y_wfo = y_all[:n_wfo_train]
    n_int   = min(WFO_INTERNAL_VAL_H, int(0.10 * n_wfo_train))
    X_vl_w  = X_wfo[-n_int:]; y_vl_w = y_wfo[-n_int:]
    X_tr_w  = X_wfo[:-n_int]; y_tr_w = y_wfo[:-n_int]

    ds_tr_w = lgb.Dataset(X_tr_w, label=y_tr_w)
    ds_vl_w = lgb.Dataset(X_vl_w, label=y_vl_w, reference=ds_tr_w)
    wfo_model = lgb.train(
        lgb_params_best, ds_tr_w,
        valid_sets=[ds_tr_w, ds_vl_w], valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=WFO_PATIENCE, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    wfo_probs[t:step_end] = wfo_model.predict(X_all[n_tv + t : n_tv + step_end])
    wfo_n_trees.append(wfo_model.best_iteration)
    step += 1
    print(f'Step {step:>2}  train={n_wfo_train:,}  int_val={n_int:,}  '
          f'best_iter={wfo_model.best_iteration:>4}  {step_end/n_test*100:.0f}%')
    del wfo_model; t = step_end

wfo_p_up = wfo_probs[:, 1]; wfo_p_dn = wfo_probs[:, 0]
print(f'WFO done -- {step} steps  {time.perf_counter()-t0_wfo:.1f}s  '
      f'mean_trees={np.mean(wfo_n_trees):.0f}')
print(f'P_up  range: [{wfo_p_up.min():.3f}, {wfo_p_up.max():.3f}]  mean={wfo_p_up.mean():.3f}')
print(f'P_down range: [{wfo_p_dn.min():.3f}, {wfo_p_dn.max():.3f}]  mean={wfo_p_dn.mean():.3f}')


# In[13]:


bh_test = test_close / test_close[0]

# vtrain7 WFO
eq_wfo7, tdf_wfo7 = run_backtest_v7(
    wfo_p_up, wfo_p_dn, test_close, test_high, test_low, test_atr, best_params)

# vtrain7 static (single full-trainval retrain)
X_tv_best = X_tv[:, fi_best]; X_te_best = X_test[:, fi_best]
n_int_st  = min(WFO_INTERNAL_VAL_H, int(0.10 * len(X_tv_best)))
ds_tr_st  = lgb.Dataset(X_tv_best[:-n_int_st], label=y_tv[:-n_int_st])
ds_vl_st  = lgb.Dataset(X_tv_best[-n_int_st:],  label=y_tv[-n_int_st:], reference=ds_tr_st)
static_model = lgb.train(
    lgb_params_best, ds_tr_st,
    valid_sets=[ds_tr_st, ds_vl_st], valid_names=['train', 'val'],
    callbacks=[lgb.early_stopping(stopping_rounds=WFO_PATIENCE, verbose=False),
               lgb.log_evaluation(period=0)],
)
static_probs = static_model.predict(X_te_best)
eq_static7, tdf_static7 = run_backtest_v7(
    static_probs[:, 1], static_probs[:, 0],
    test_close, test_high, test_low, test_atr, best_params)
print(f'Static model trees: {static_model.best_iteration}')

# vtrain6 reference (if available)
v6_path = MODELS_DIR / 'lgbm_best_trading_params_v6.json'
if v6_path.exists():
    with open(v6_path) as f: v6_data = json.load(f)
    print(f'vtrain6 test Sharpe (reference): {v6_data["test_wfo"]["sharpe"]:.4f}')
    print(f'vtrain6 test Return (reference): {v6_data["test_wfo"]["total_return"]:+.2%}')


def quick_metrics(eq, tdf, label):
    ret = np.log(np.maximum(eq[1:], 1e-12) / np.maximum(eq[:-1], 1e-12))
    pk  = np.maximum.accumulate(eq)
    sh  = float(ret.mean() / (ret.std(ddof=1) + 1e-12) * np.sqrt(24 * 365))
    mdd = float(((eq - pk) / (pk + 1e-12)).min())
    an  = float((eq[-1] ** (24 * 365 / max(len(eq), 1))) - 1)
    wr  = float((tdf['pnl_pct'] > 0).mean()) if not tdf.empty else float('nan')
    fr  = (tdf.attrs.get('n_fills', 0) / max(tdf.attrs.get('n_signals', 1), 1)
           if hasattr(tdf, 'attrs') else float('nan'))
    pf  = 0.0
    if not tdf.empty:
        g = tdf[tdf['pnl_pct'] > 0]['pnl_pct'].sum()
        l = tdf[tdf['pnl_pct'] < 0]['pnl_pct'].abs().sum()
        pf = g / (l + 1e-6)
    return {
        'Strategy':      label,
        'Total Return':  f'{eq[-1]-1:+.2%}',
        'Ann. Return':   f'{an:+.2%}',
        'Sharpe (ann.)': f'{sh:.3f}',
        'Max DD':        f'{mdd:.2%}',
        'Calmar':        f'{an/(abs(mdd)+1e-6):.3f}',
        'Win Rate':      f'{wr:.1%}' if not np.isnan(wr) else 'N/A',
        'Profit Factor': f'{pf:.3f}',
        'Trades':        str(len(tdf)),
        'Fill Rate':     f'{fr:.1%}' if not np.isnan(fr) else 'N/A',
    }


rows = [
    quick_metrics(eq_wfo7,    tdf_wfo7,    'vtrain7 WFO    (TBM, multiclass, Spot/Futures)'),
    quick_metrics(eq_static7, tdf_static7, 'vtrain7 Static (TBM, multiclass, Spot/Futures)'),
    quick_metrics(bh_test,    pd.DataFrame(), 'Buy & Hold'),
]
summary = pd.DataFrame(rows).set_index('Strategy')
print('\n' + '='*100)
print('  FINAL TEST-SET SUMMARY -- vtrain7')
print('='*100)
print(summary.to_string())
print('='*100)


# In[14]:


fig, axes = plt.subplots(3, 1, figsize=(13, 12),
                          gridspec_kw={'height_ratios': [3, 1.2, 1.2], 'hspace': 0.10})

ax = axes[0]
ax.plot(test_index, eq_wfo7,    color=ACCENT, lw=1.6,
        label='vtrain7 WFO (TBM, multiclass, Spot/Futures)')
ax.plot(test_index, eq_static7, color=GREEN,  lw=1.2, ls=':',  label='vtrain7 Static')
ax.plot(test_index, bh_test,    color=BLUE,   lw=1.2, ls='--', label='Buy & Hold')
ax.axhline(1.0, color=GREY, lw=0.7, ls=':')
ax.set_ylabel('Portfolio value')
ax.set_title('vtrain7 -- TBM labels | Multiclass LGBM | Spot longs / Futures shorts',
             fontweight='bold')
ax.legend(ncol=2); ax.grid(axis='y', alpha=0.3); ax.grid(axis='x', alpha=0.15)

ax = axes[1]
ax.plot(test_index, wfo_p_up, color=GREEN, lw=0.6, alpha=0.7, label='P(Long) WFO')
ax.plot(test_index, wfo_p_dn, color=RED,   lw=0.6, alpha=0.7, label='P(Short) WFO')
ax.axhspan(best_params['long_threshold'],  1.0, alpha=0.07, color=GREEN,
           label=f'Long >={best_params["long_threshold"]}')
ax.axhspan(0.0, best_params['short_threshold'], alpha=0.07, color=RED,
           label=f'Short >={best_params["short_threshold"]}')
ax.axhline(1/3, color=GREY, ls=':', lw=0.7, label='1/3 baseline')
ax.set_ylim(0, 1); ax.set_ylabel('Class probability')
ax.set_title('WFO model class probabilities')
ax.legend(ncol=4, fontsize=8); ax.grid(axis='y', alpha=0.3)

ax = axes[2]
for eq, idx, color, lbl in [
    (eq_wfo7,    test_index, ACCENT, 'vtrain7 WFO'),
    (eq_static7, test_index, GREEN,  'vtrain7 Static'),
    (bh_test,    test_index, BLUE,   'B&H'),
]:
    pk = np.maximum.accumulate(eq)
    ax.fill_between(idx, (eq - pk) / (pk + 1e-12) * 100, 0, alpha=0.35, label=lbl)
ax.set_ylabel('Drawdown (%)'); ax.legend(ncol=3)
ax.grid(axis='y', alpha=0.3); ax.grid(axis='x', alpha=0.15)

for ax in axes:
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')

fig.text(0.5, 0.005,
    f'model_id={best_mid} top_n={best_cfg["top_n_features"]} lr={best_cfg["learning_rate"]} '
    f'leaves={best_cfg["num_leaves"]} | '
    f'entry_atr={best_params["entry_atr_mult"]}x  SL={best_params["sl_atr_multiplier"]}x  '
    f'TP={best_params["tp_atr_multiplier"]}x  '
    f'long_thr={best_params["long_threshold"]}  short_thr={best_params["short_threshold"]}  '
    f'hold={best_params["min_hold"]}-{best_params["max_hold"]}h  cd={best_params["cooldown"]}h',
    ha='center', fontsize=8, color=GREY, style='italic')
fig.savefig(FIGURES_DIR / 'wfo_equity_v7.png'); plt.show()


# In[16]:


results_path = MODELS_DIR / 'lgbm_grid_results_v7.csv'
results_df.to_csv(results_path, index=False)
print(f'Saved {len(results_df):,} grid results -> {results_path}')

ret_wfo  = np.log(np.maximum(eq_wfo7[1:],  1e-12) / np.maximum(eq_wfo7[:-1],  1e-12))
ret_oos2 = np.log(np.maximum(eq_oos[1:],   1e-12) / np.maximum(eq_oos[:-1],   1e-12))
pk_wfo   = np.maximum.accumulate(eq_wfo7)
pk_oos2  = np.maximum.accumulate(eq_oos)

output = {
    'version': 'vtrain7',
    'changes': [
        f'TBM labels: +-{TBM_ATR_MULT}xATR barriers, {TBM_HORIZON}h vertical barrier',
        'Multiclass LGBM: objective=multiclass, num_class=3',
        'Combined model+trading grid search (48 model configs x trading grid)',
        'Spot/Futures routing: longs->Spot (0% funding), shorts->Futures (+funding received)',
        f'Spot taker fee: {SPOT_TAKER_FEE*100:.3f}%  Futures taker fee: {FUTURES_TAKER_FEE*100:.3f}%',
        f'Short funding received: +{SHORT_FUNDING_H*100:.5f}%/h',
    ],
    'execution': {
        'maker_fee':         MAKER_FEE,
        'spot_taker_fee':    SPOT_TAKER_FEE,
        'futures_taker_fee': FUTURES_TAKER_FEE,
        'buffer':            BUFFER,
        'spot_funding_h':    SPOT_FUNDING_H,
        'short_funding_h':   SHORT_FUNDING_H,
    },
    'tbm': {'atr_mult': TBM_ATR_MULT, 'horizon': TBM_HORIZON},
    'best_model_id':       best_mid,
    'best_model_config':   best_cfg,
    'best_trading_params': best_params,
    'oos_auc': {
        'long':  round(oos_prob_cache[best_mid]['auc_long'],  4),
        'short': round(oos_prob_cache[best_mid]['auc_short'], 4),
    },
    'oos_kfold': {
        'sharpe':       round(float(ret_oos2.mean() / (ret_oos2.std(ddof=1) + 1e-12) * np.sqrt(24*365)), 4),
        'total_return': round(float(eq_oos[-1] - 1), 4),
        'max_dd':       round(float(((eq_oos - pk_oos2) / (pk_oos2 + 1e-12)).min()), 4),
        'n_trades':     int(len(tdf_oos)),
        'fill_rate':    round(tdf_oos.attrs.get('n_fills', 0) / max(tdf_oos.attrs.get('n_signals', 1), 1), 4),
    },
    'test_wfo': {
        'sharpe':       round(float(ret_wfo.mean() / (ret_wfo.std(ddof=1) + 1e-12) * np.sqrt(24*365)), 4),
        'total_return': round(float(eq_wfo7[-1] - 1), 4),
        'max_dd':       round(float(((eq_wfo7 - pk_wfo) / (pk_wfo + 1e-12)).min()), 4),
        'n_trades':     int(len(tdf_wfo7)),
        'fill_rate':    round(tdf_wfo7.attrs.get('n_fills', 0) / max(tdf_wfo7.attrs.get('n_signals', 1), 1), 4),
    },
}

params_path = MODELS_DIR / 'lgbm_best_trading_params_v7.json'
with open(params_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f'Saved best params -> {params_path}')
print()
print(json.dumps(output, indent=2))


# In[18]:


# Trade Timeline: Weekly Panels
# Re-run WFO backtest with bar-index tracing to recover entry/exit timestamps

def run_backtest_v7_traced(p_up, p_down, close_arr, high_arr, low_arr, atr_arr, params,
                            spot_taker=SPOT_TAKER_FEE, fut_taker=FUTURES_TAKER_FEE,
                            maker_fee=MAKER_FEE, buf=BUFFER,
                            spot_fund=SPOT_FUNDING_H, short_fund=SHORT_FUNDING_H):
    # Same logic as run_backtest_v7 but records entry_bar/exit_bar/entry_px/exit_px
    lt         = params['long_threshold'];    st       = params['short_threshold']
    exit_long  = 1.0 - lt;                   exit_short = 1.0 - st
    ent_atr    = params['entry_atr_mult'];    sl_m     = params['sl_atr_multiplier']
    tp_m       = params['tp_atr_multiplier']; min_sl   = params['min_sl']
    min_hold   = int(params['min_hold']);      max_hold = int(params['max_hold'])
    cd_n       = int(params['cooldown'])

    cash = 1.0; units = 0.0; entry_cash = 0.0
    in_pos = False; direction = None
    entry_px = 0.0; dynamic_sl = 0.0; dynamic_tp = 0.0
    hold_count = 0; cooldown = 0; entry_bar = -1
    pending = None
    equity_curve = [1.0]; trade_log = []

    N = len(close_arr)
    for i in range(N):
        px  = close_arr[i]; hi = high_arr[i]; lo = low_arr[i]
        pup = p_up[i];      pdn = p_down[i];  atr = atr_arr[i]
        if cooldown > 0: cooldown -= 1

        if pending is not None:
            lp = pending['limit_px']
            if pending['direction'] == 'long':
                if lo < lp * (1.0 - buf):
                    units = cash * (1.0 - maker_fee) / lp
                    cash = 0.0; in_pos = True; direction = 'long'
                    entry_px = lp; entry_bar = i; hold_count = 0
                    dynamic_sl = pending['sl']; dynamic_tp = pending['tp']
            else:
                if hi > lp * (1.0 + buf):
                    entry_cash = cash * (1.0 - maker_fee)
                    cash = 0.0; units = entry_cash / lp
                    in_pos = True; direction = 'short'
                    entry_px = lp; entry_bar = i; hold_count = 0
                    dynamic_sl = pending['sl']; dynamic_tp = pending['tp']
            pending = None

        if in_pos and i > entry_bar:
            hold_count += 1
            if direction == 'long': units      *= (1.0 - spot_fund)
            else:                   entry_cash *= (1.0 + short_fund)
            reason = None; exit_px_t = px; pnl = 0.0

            if direction == 'long':
                sl_p = entry_px * (1.0 - dynamic_sl)
                tp_p = entry_px * (1.0 + dynamic_tp)
                sl_hit = lo <= sl_p; tp_hit = hi > tp_p * (1.0 + buf)
                if (sl_hit and tp_hit) or sl_hit:
                    reason = 'sl'; exit_px_t = sl_p
                    pnl  = (exit_px_t - entry_px) / entry_px
                    cash = units * exit_px_t * (1.0 - spot_taker); units = 0.0
                elif tp_hit:
                    reason = 'tp'; exit_px_t = tp_p
                    pnl  = (exit_px_t - entry_px) / entry_px
                    cash = units * exit_px_t * (1.0 - maker_fee); units = 0.0
                elif hold_count >= max_hold:
                    reason = 'max_hold'; exit_px_t = px
                    pnl  = (px - entry_px) / entry_px
                    cash = units * px * (1.0 - spot_taker); units = 0.0
                elif hold_count >= min_hold and pup < exit_long:
                    reason = 'conf'; exit_px_t = px
                    pnl  = (px - entry_px) / entry_px
                    cash = units * px * (1.0 - spot_taker); units = 0.0
            else:
                sl_p = entry_px * (1.0 + dynamic_sl)
                tp_p = entry_px * (1.0 - dynamic_tp)
                sl_hit = hi >= sl_p; tp_hit = lo < tp_p * (1.0 - buf)
                if (sl_hit and tp_hit) or sl_hit:
                    reason = 'sl'; exit_px_t = sl_p
                    gross = (entry_px - exit_px_t) / entry_px; pnl = gross
                    cash  = entry_cash * (1.0 + gross) * (1.0 - fut_taker)
                elif tp_hit:
                    reason = 'tp'; exit_px_t = tp_p
                    gross = (entry_px - exit_px_t) / entry_px; pnl = gross
                    cash  = entry_cash * (1.0 + gross) * (1.0 - maker_fee)
                elif hold_count >= max_hold:
                    reason = 'max_hold'; exit_px_t = px
                    gross = (entry_px - px) / entry_px; pnl = gross
                    cash  = entry_cash * (1.0 + gross) * (1.0 - fut_taker)
                elif hold_count >= min_hold and pdn < exit_short:
                    reason = 'conf'; exit_px_t = px
                    gross = (entry_px - px) / entry_px; pnl = gross
                    cash  = entry_cash * (1.0 + gross) * (1.0 - fut_taker)

            if reason:
                trade_log.append({
                    'direction': direction, 'pnl_pct': pnl,
                    'hold_h': hold_count,   'reason': reason,
                    'entry_bar': entry_bar,  'exit_bar': i,
                    'entry_px': entry_px,    'exit_px': exit_px_t,
                })
                in_pos = False; direction = None; hold_count = 0; cooldown = cd_n

        if not in_pos and pending is None and cooldown == 0:
            sl_v = max(sl_m * atr, min_sl); tp_v = tp_m * atr
            if pup >= lt:
                pending = {'direction': 'long',
                           'limit_px': px * (1.0 - ent_atr * atr), 'sl': sl_v, 'tp': tp_v}
            elif pdn >= st:
                pending = {'direction': 'short',
                           'limit_px': px * (1.0 + ent_atr * atr), 'sl': sl_v, 'tp': tp_v}

        if   in_pos and direction == 'long':
            equity_curve.append(units * px)
        elif in_pos and direction == 'short':
            equity_curve.append(entry_cash * (1.0 + (entry_px - px) / entry_px))
        else:
            equity_curve.append(cash)

    if in_pos:
        px = close_arr[-1]
        if direction == 'long':
            gross = (px - entry_px) / entry_px
            cash  = units * px * (1.0 - spot_taker)
        else:
            gross = (entry_px - px) / entry_px
            cash  = entry_cash * (1.0 + gross) * (1.0 - fut_taker)
        trade_log.append({
            'direction': direction, 'pnl_pct': gross,
            'hold_h': hold_count,   'reason': 'eod',
            'entry_bar': entry_bar,  'exit_bar': N - 1,
            'entry_px': entry_px,    'exit_px': px,
        })
        equity_curve[-1] = cash

    trades = pd.DataFrame(trade_log)
    if not trades.empty:
        tidx = pd.DatetimeIndex(test_index)
        trades['entry_ts'] = tidx[trades['entry_bar'].astype(int).clip(0, len(tidx)-1)]
        trades['exit_ts']  = tidx[trades['exit_bar'].astype(int).clip(0, len(tidx)-1)]
    return np.array(equity_curve[1:]), trades


# Run traced backtest
_, trades_traced = run_backtest_v7_traced(
    wfo_p_up, wfo_p_dn, test_close, test_high, test_low, test_atr, best_params)

print(f'Traced trades: {len(trades_traced)}')
if not trades_traced.empty:
    show = ['direction','reason','pnl_pct','hold_h','entry_ts','exit_ts','entry_px','exit_px']
    print(trades_traced[show].to_string())


# Weekly panel plotter
from matplotlib.lines import Line2D

def plot_trade_weeks(close_arr, idx, eq_arr, trades_df, p_up_arr, p_dn_arr,
                     long_thr, short_thr, atr_arr_ref,
                     weeks_per_fig=2, fig_w=30, save_dir=None):
    if trades_df.empty:
        print('No trades.'); return

    dates   = pd.DatetimeIndex(idx)
    mon0    = dates[0] - pd.Timedelta(days=dates[0].dayofweek)
    w_starts = pd.date_range(mon0.normalize(), dates[-1], freq='7D')
    buckets = []
    for ws in w_starts:
        we   = ws + pd.Timedelta(days=7)
        mask = (dates >= ws) & (dates < we)
        if mask.sum() > 0:
            buckets.append((ws, we, np.where(mask)[0]))

    n_figs = int(np.ceil(len(buckets) / weeks_per_fig))
    print(f'{len(buckets)} weeks -> {n_figs} figures')

    for fig_i in range(n_figs):
        wks  = buckets[fig_i * weeks_per_fig : (fig_i+1) * weeks_per_fig]
        bars = np.concatenate([b[2] for b in wks])
        if len(bars) == 0: continue
        bar_set = set(bars.tolist())

        w_idx   = dates[bars]
        w_close = close_arr[bars]
        w_eq    = eq_arr[bars]
        w_pup   = p_up_arr[bars]
        w_pdn   = p_dn_arr[bars]
        eq_base = eq_arr[bars[0]]

        fig = plt.figure(figsize=(fig_w, 9))
        gs  = fig.add_gridspec(3, 1, height_ratios=[4, 1.4, 1.4], hspace=0.05)
        ax_px = fig.add_subplot(gs[0])
        ax_eq = fig.add_subplot(gs[1], sharex=ax_px)
        ax_pb = fig.add_subplot(gs[2], sharex=ax_px)

        # Price line
        ax_px.plot(w_idx, w_close, color='#90A4AE', lw=0.9, zorder=1)

        # Weekend shading
        for ws, we, _ in wks:
            for d in range(7):
                day = ws + pd.Timedelta(days=d)
                if day.weekday() >= 5:
                    ax_px.axvspan(day, day + pd.Timedelta(days=1),
                                  color=GREY, alpha=0.07, zorder=0)

        # Draw trades
        for _, tr in trades_df.iterrows():
            eb = int(tr['entry_bar']); xb = int(tr['exit_bar'])
            if eb not in bar_set and xb not in bar_set: continue

            e_ts = tr['entry_ts']; x_ts = tr['exit_ts']
            e_px = tr['entry_px']; x_px = tr['exit_px']
            won  = tr['pnl_pct'] > 0
            is_long = tr['direction'] == 'long'

            e_col = GREEN if is_long else RED
            x_col = GREEN if won     else RED
            e_mrk = '^'   if is_long else 'v'
            x_mrk = 'v'   if is_long else '^'

            # SL/TP dashed levels for visible bars
            in_bars = [b for b in range(eb, xb+1) if b in bar_set]
            if in_bars:
                atr_e  = atr_arr_ref[eb]
                sl_lvl = e_px*(1 - best_params['sl_atr_multiplier']*atr_e) if is_long \
                         else e_px*(1 + best_params['sl_atr_multiplier']*atr_e)
                tp_lvl = e_px*(1 + best_params['tp_atr_multiplier']*atr_e) if is_long \
                         else e_px*(1 - best_params['tp_atr_multiplier']*atr_e)
                t0 = dates[in_bars[0]]; t1 = dates[in_bars[-1]]
                ax_px.hlines(sl_lvl, t0, t1, colors=RED,   lw=1.1, ls='--', alpha=0.55, zorder=2)
                ax_px.hlines(tp_lvl, t0, t1, colors=GREEN, lw=1.1, ls='--', alpha=0.55, zorder=2)
                if eb in bar_set and xb in bar_set:
                    ax_px.plot([e_ts, x_ts], [e_px, x_px],
                               color=x_col, lw=0.9, alpha=0.5, zorder=3)

            # Entry marker + PnL label
            if eb in bar_set:
                ax_px.scatter(e_ts, e_px, marker=e_mrk, s=170,
                              color=e_col, zorder=6, linewidths=0)
                offset = 16 if is_long else -20
                ax_px.annotate(f"{tr['pnl_pct']:+.2%}",
                               xy=(e_ts, e_px), xytext=(0, offset),
                               textcoords='offset points',
                               fontsize=7.5, color=x_col, fontweight='bold',
                               ha='center')

            # Exit marker
            if xb in bar_set:
                ax_px.scatter(x_ts, x_px, marker=x_mrk, s=140,
                              color=x_col, edgecolors='white', lw=0.8, zorder=6)

        ax_px.set_ylabel('BTC Price (USDT)', fontsize=9)
        ax_px.grid(axis='y', alpha=0.2); ax_px.grid(axis='x', alpha=0.08)

        legend_h = [
            Line2D([0],[0], marker='^', color='w', markerfacecolor=GREEN,  ms=10, label='Long entry'),
            Line2D([0],[0], marker='v', color='w', markerfacecolor=RED,    ms=10, label='Short entry'),
            Line2D([0],[0], marker='v', color='w', markerfacecolor=GREEN,  ms=10, label='Exit (profit)'),
            Line2D([0],[0], marker='v', color='w', markerfacecolor=RED,    ms=10, label='Exit (loss)'),
            Line2D([0],[0], color=GREEN, lw=1.1, ls='--', label='TP level'),
            Line2D([0],[0], color=RED,   lw=1.1, ls='--', label='SL level'),
        ]
        ax_px.legend(handles=legend_h, loc='upper left', ncol=6, fontsize=8, framealpha=0.9)

        d0 = wks[0][0].strftime('%d %b %Y')
        d1 = wks[-1][1].strftime('%d %b %Y')
        ax_px.set_title(
            f'vtrain7 WFO  |  {d0} -- {d1}  |  '
            f'long P>={long_thr}  short P>={short_thr}  '
            f'SL={best_params["sl_atr_multiplier"]}xATR  TP={best_params["tp_atr_multiplier"]}xATR',
            fontweight='bold', fontsize=10)

        # Equity panel
        ax_eq.plot(w_idx, w_eq, color=ACCENT, lw=1.4)
        ax_eq.axhline(eq_base, color=GREY, lw=0.8, ls=':')
        ax_eq.fill_between(w_idx, w_eq, eq_base,
                           where=(w_eq >= eq_base), color=GREEN, alpha=0.18)
        ax_eq.fill_between(w_idx, w_eq, eq_base,
                           where=(w_eq <  eq_base), color=RED,   alpha=0.18)
        ax_eq.set_ylabel('Equity', fontsize=9)
        ax_eq.yaxis.set_major_formatter(
            mpl.ticker.FuncFormatter(lambda v, _: f'{v:.3f}'))
        ax_eq.grid(axis='y', alpha=0.2)

        # Prob panel
        ax_pb.plot(w_idx, w_pup, color=GREEN, lw=0.9, alpha=0.85, label='P(Long)')
        ax_pb.plot(w_idx, w_pdn, color=RED,   lw=0.9, alpha=0.85, label='P(Short)')
        ax_pb.axhline(long_thr,  color=GREEN, lw=0.8, ls=':', alpha=0.7)
        ax_pb.axhline(short_thr, color=RED,   lw=0.8, ls=':', alpha=0.7)
        ax_pb.axhline(1/3, color=GREY, lw=0.5, ls=':', alpha=0.5, label='1/3')
        ax_pb.set_ylim(0, 1); ax_pb.set_ylabel('Class prob', fontsize=9)
        ax_pb.legend(loc='upper right', ncol=3, fontsize=8, framealpha=0.9)
        ax_pb.grid(axis='y', alpha=0.2)

        ax_pb.xaxis.set_major_locator(mdates.DayLocator())
        ax_pb.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
        plt.setp(ax_pb.xaxis.get_majorticklabels(), rotation=35, ha='right', fontsize=8)
        plt.setp(ax_px.xaxis.get_ticklabels(), visible=False)
        plt.setp(ax_eq.xaxis.get_ticklabels(), visible=False)

        fig.tight_layout()
        if save_dir is not None:
            fname = save_dir / f'trade_timeline_v7_{fig_i+1:03d}.png'
            fig.savefig(fname, dpi=150, bbox_inches='tight')
            print(f'  Saved {fname.name}')
        plt.show()
        plt.close(fig)


plot_trade_weeks(
    close_arr    = test_close,
    idx          = test_index,
    eq_arr       = eq_wfo7,
    trades_df    = trades_traced,
    p_up_arr     = wfo_p_up,
    p_dn_arr     = wfo_p_dn,
    long_thr     = best_params['long_threshold'],
    short_thr    = best_params['short_threshold'],
    atr_arr_ref  = test_atr,
    weeks_per_fig = 2,
    fig_w        = 30,
    save_dir     = FIGURES_DIR,
)


# In[19]:


# ── Trade Statistics Dashboard ────────────────────────────────────────────────
import scipy.stats as stats_scipy

# Enrich trades_traced with entry-bar context
if not trades_traced.empty:
    eb = trades_traced['entry_bar'].astype(int).clip(0, len(test_index)-1).values
    trades_traced['p_up_entry']  = wfo_p_up[eb]
    trades_traced['p_dn_entry']  = wfo_p_dn[eb]
    trades_traced['atr_entry']   = test_atr[eb]
    trades_traced['hour_entry']  = trades_traced['entry_ts'].dt.hour
    trades_traced['dow_entry']   = trades_traced['entry_ts'].dt.dayofweek   # 0=Mon
    trades_traced['dow_name']    = trades_traced['entry_ts'].dt.day_name().str[:3]
    trades_traced['win']         = (trades_traced['pnl_pct'] > 0).astype(int)
    # Confidence used at entry: p_up for longs, p_dn for shorts
    trades_traced['conf_entry']  = np.where(
        trades_traced['direction'] == 'long',
        trades_traced['p_up_entry'],
        trades_traced['p_dn_entry'],
    )
    # ATR quintile label
    trades_traced['atr_q'] = pd.qcut(
        trades_traced['atr_entry'], q=5,
        labels=['Q1 low vol', 'Q2', 'Q3', 'Q4', 'Q5 high vol'])

T  = trades_traced
n  = len(T)
longs  = T[T['direction'] == 'long']
shorts = T[T['direction'] == 'short']

# ── 1. PRINTED SUMMARY TABLE ─────────────────────────────────────────────────
def _pf(df):
    g = df[df['pnl_pct'] > 0]['pnl_pct'].sum()
    l = df[df['pnl_pct'] < 0]['pnl_pct'].abs().sum()
    return g / (l + 1e-9)

def _fmt(df, label):
    wr  = df['win'].mean()
    avg = df['pnl_pct'].mean()
    med = df['pnl_pct'].median()
    std = df['pnl_pct'].std()
    aw  = df.loc[df['pnl_pct'] > 0, 'pnl_pct'].mean() if (df['pnl_pct'] > 0).any() else 0
    al  = df.loc[df['pnl_pct'] < 0, 'pnl_pct'].mean() if (df['pnl_pct'] < 0).any() else 0
    pf  = _pf(df)
    exp = avg  # expectancy per trade
    return {
        'Subset': label, 'N': len(df),
        'Win Rate':       f'{wr:.1%}',
        'Avg PnL':        f'{avg:+.3%}',
        'Median PnL':     f'{med:+.3%}',
        'Std PnL':        f'{std:.3%}',
        'Avg Win':        f'{aw:+.3%}',
        'Avg Loss':       f'{al:+.3%}',
        'Win/Loss Ratio': f'{abs(aw/(al-1e-9)):.2f}',
        'Profit Factor':  f'{pf:.3f}',
        'Expectancy':     f'{exp:+.4%}',
        'Max Win':        f'{df["pnl_pct"].max():+.3%}',
        'Max Loss':       f'{df["pnl_pct"].min():+.3%}',
        'Avg Hold (h)':   f'{df["hold_h"].mean():.1f}',
    }

rows = [_fmt(T, 'All'), _fmt(longs, 'Longs (Spot)'), _fmt(shorts, 'Shorts (Futures)')]
summary_stats = pd.DataFrame(rows).set_index('Subset')
print('=' * 90)
print('  TRADE STATISTICS SUMMARY — vtrain7 WFO')
print('=' * 90)
print(summary_stats.T.to_string())
print('=' * 90)

# Funding benefit on shorts
if not shorts.empty:
    avg_hold_short = shorts['hold_h'].mean()
    total_funding  = SHORT_FUNDING_H * avg_hold_short * len(shorts)
    print(f'\n  Shorts: avg hold={avg_hold_short:.1f}h  '
          f'=> avg funding received per trade={SHORT_FUNDING_H*avg_hold_short*100:.4f}%  '
          f'=> total across {len(shorts)} shorts={total_funding*100:.3f}%')


# ── 2. PnL DISTRIBUTION ───────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(18, 9))

# 2a. Histogram + KDE all trades
ax = axes[0, 0]
vals = T['pnl_pct'].values * 100
ax.hist(vals, bins=40, color=BLUE, alpha=0.6, density=True, label='All trades')
kde_x = np.linspace(vals.min(), vals.max(), 300)
kde   = stats_scipy.gaussian_kde(vals)
ax.plot(kde_x, kde(kde_x), color=ACCENT, lw=2.0, label='KDE')
ax.axvline(0,              color='black', lw=0.9, ls='--', alpha=0.6)
ax.axvline(np.mean(vals),  color=ACCENT,  lw=1.3, ls=':',  label=f'Mean {np.mean(vals):+.2f}%')
ax.axvline(np.median(vals),color=GREEN,   lw=1.3, ls=':',  label=f'Median {np.median(vals):+.2f}%')
ax.set_xlabel('Trade PnL (%)'); ax.set_title('PnL distribution — all trades', fontweight='bold')
ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)

# 2b. Long vs Short overlay
ax = axes[0, 1]
bins = np.linspace(vals.min(), vals.max(), 35)
if not longs.empty:
    ax.hist(longs['pnl_pct'].values*100,  bins=bins, color=GREEN, alpha=0.55, density=True, label=f'Long (n={len(longs)})')
if not shorts.empty:
    ax.hist(shorts['pnl_pct'].values*100, bins=bins, color=RED,   alpha=0.55, density=True, label=f'Short (n={len(shorts)})')
ax.axvline(0, color='black', lw=0.9, ls='--', alpha=0.6)
ax.set_xlabel('Trade PnL (%)'); ax.set_title('PnL distribution — Long vs Short', fontweight='bold')
ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)

# 2c. ECDF
ax = axes[1, 0]
for df, color, lbl in [(T, BLUE, 'All'), (longs, GREEN, 'Longs'), (shorts, RED, 'Shorts')]:
    if df.empty: continue
    sorted_p = np.sort(df['pnl_pct'].values * 100)
    ecdf     = np.arange(1, len(sorted_p)+1) / len(sorted_p)
    ax.plot(sorted_p, ecdf, color=color, lw=1.5, label=lbl)
ax.axvline(0, color='black', lw=0.9, ls='--', alpha=0.6)
ax.set_xlabel('Trade PnL (%)'); ax.set_ylabel('Cumulative probability')
ax.set_title('ECDF — trade PnL', fontweight='bold')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# 2d. Q-Q plot (normality check)
ax = axes[1, 1]
(osm, osr), (slope, intercept, r) = stats_scipy.probplot(T['pnl_pct'].values*100, dist='norm')
ax.scatter(osm, osr, s=12, color=BLUE, alpha=0.6)
ax.plot(osm, slope*np.array(osm)+intercept, color=RED, lw=1.5, label=f'Normal fit (R²={r**2:.3f})')
ax.set_xlabel('Theoretical quantiles'); ax.set_ylabel('Sample quantiles')
ax.set_title('Q-Q plot (vs Normal) — fat tails?', fontweight='bold')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

fig.suptitle('PnL Distributions — vtrain7', fontweight='bold', fontsize=12)
fig.tight_layout()
fig.savefig(FIGURES_DIR / 'stats_pnl_distribution_v7.png', dpi=150)
plt.show()


# ── 3. EXIT REASON BREAKDOWN ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

reasons = T['reason'].value_counts().index.tolist()
reason_stats = (T.groupby('reason')['pnl_pct']
                 .agg(['mean', 'median', 'count',
                       lambda x: (x > 0).mean()])
                 .rename(columns={'<lambda_0>': 'win_rate'})
                 .sort_values('count', ascending=False))

# Count
ax = axes[0]
colors_r = [GREEN if reason_stats.loc[r, 'win_rate'] >= 0.5 else RED for r in reason_stats.index]
ax.bar(reason_stats.index, reason_stats['count'], color=colors_r, alpha=0.8)
ax.set_title('Trades per exit reason', fontweight='bold')
ax.set_ylabel('Count'); ax.grid(axis='y', alpha=0.3)
for i, (r, row) in enumerate(reason_stats.iterrows()):
    ax.text(i, row['count'] + 0.3, str(int(row['count'])), ha='center', fontsize=9)

# Win rate
ax = axes[1]
ax.bar(reason_stats.index, reason_stats['win_rate']*100, color=colors_r, alpha=0.8)
ax.axhline(50, color='black', lw=0.9, ls='--', alpha=0.5)
ax.set_title('Win rate per exit reason', fontweight='bold')
ax.set_ylabel('Win rate (%)'); ax.set_ylim(0, 100); ax.grid(axis='y', alpha=0.3)
for i, (r, row) in enumerate(reason_stats.iterrows()):
    ax.text(i, row['win_rate']*100 + 1, f'{row["win_rate"]:.1%}', ha='center', fontsize=9)

# Avg PnL
ax = axes[2]
bar_cols = [GREEN if v > 0 else RED for v in reason_stats['mean']]
ax.bar(reason_stats.index, reason_stats['mean']*100, color=bar_cols, alpha=0.8)
ax.axhline(0, color='black', lw=0.9, ls='--', alpha=0.5)
ax.set_title('Avg PnL per exit reason', fontweight='bold')
ax.set_ylabel('Avg PnL (%)'); ax.grid(axis='y', alpha=0.3)
for i, (r, row) in enumerate(reason_stats.iterrows()):
    ax.text(i, row['mean']*100 + (0.05 if row['mean'] > 0 else -0.12),
            f'{row["mean"]:+.2%}', ha='center', fontsize=8)

fig.suptitle('Exit Reason Analysis — vtrain7', fontweight='bold', fontsize=12)
fig.tight_layout()
fig.savefig(FIGURES_DIR / 'stats_exit_reasons_v7.png', dpi=150)
plt.show()
print(reason_stats.to_string())


# ── 4. MODEL CONFIDENCE vs OUTCOME ────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(18, 9))

conf_bins = np.linspace(T['conf_entry'].min(), T['conf_entry'].max(), 9)
T['conf_bin'] = pd.cut(T['conf_entry'], bins=conf_bins)
conf_grp = T.groupby('conf_bin', observed=True)['pnl_pct'].agg(
    ['mean', 'count', lambda x: (x > 0).mean()]).rename(columns={'<lambda_0>': 'wr'})
conf_grp['mid'] = conf_grp.index.map(lambda iv: iv.mid)

# Avg PnL vs confidence
ax = axes[0, 0]
bar_c = [GREEN if v > 0 else RED for v in conf_grp['mean']]
ax.bar(range(len(conf_grp)), conf_grp['mean']*100, color=bar_c, alpha=0.8,
       tick_label=[f'{m:.2f}' for m in conf_grp['mid']])
ax.axhline(0, color='black', lw=0.8, ls='--', alpha=0.5)
ax.set_xlabel('Model confidence at entry (binned)')
ax.set_ylabel('Avg PnL (%)')
ax.set_title('Avg PnL vs model confidence at entry', fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=35, ha='right', fontsize=8)

# Win rate vs confidence
ax = axes[0, 1]
ax.bar(range(len(conf_grp)), conf_grp['wr']*100, color=BLUE, alpha=0.75,
       tick_label=[f'{m:.2f}' for m in conf_grp['mid']])
ax.axhline(50, color='black', lw=0.8, ls='--', alpha=0.5)
ax.set_xlabel('Model confidence at entry (binned)')
ax.set_ylabel('Win rate (%)')
ax.set_title('Win rate vs model confidence at entry', fontweight='bold')
ax.set_ylim(0, 100); ax.grid(axis='y', alpha=0.3)
for i, (_, row) in enumerate(conf_grp.iterrows()):
    ax.text(i, row['wr']*100 + 1, f'n={int(row["count"])}', ha='center', fontsize=7)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=35, ha='right', fontsize=8)

# Scatter: confidence vs PnL
ax = axes[1, 0]
cols_s = [GREEN if w else RED for w in T['win']]
ax.scatter(T['conf_entry'], T['pnl_pct']*100, c=cols_s, s=20, alpha=0.5)
# regression line
m_conf, b_conf, r_conf, p_conf, _ = stats_scipy.linregress(T['conf_entry'], T['pnl_pct']*100)
xr = np.linspace(T['conf_entry'].min(), T['conf_entry'].max(), 100)
ax.plot(xr, m_conf*xr + b_conf, color=ACCENT, lw=1.8,
        label=f'slope={m_conf:.3f}  R²={r_conf**2:.4f}  p={p_conf:.3f}')
ax.axhline(0, color='black', lw=0.8, ls='--', alpha=0.5)
ax.set_xlabel('Model confidence at entry'); ax.set_ylabel('Trade PnL (%)')
ax.set_title('Confidence vs PnL scatter (green=win, red=loss)', fontweight='bold')
ax.legend(fontsize=8); ax.grid(alpha=0.2)

# Confidence distribution: wins vs losses
ax = axes[1, 1]
ax.hist(T.loc[T['win']==1, 'conf_entry'], bins=25, color=GREEN, alpha=0.6,
        density=True, label='Winners')
ax.hist(T.loc[T['win']==0, 'conf_entry'], bins=25, color=RED,   alpha=0.6,
        density=True, label='Losers')
w_mean = T.loc[T['win']==1, 'conf_entry'].mean()
l_mean = T.loc[T['win']==0, 'conf_entry'].mean()
ax.axvline(w_mean, color=GREEN, lw=1.5, ls=':', label=f'Win mean={w_mean:.3f}')
ax.axvline(l_mean, color=RED,   lw=1.5, ls=':', label=f'Loss mean={l_mean:.3f}')
ax.set_xlabel('Model confidence at entry')
ax.set_title('Confidence distribution: winners vs losers', fontweight='bold')
ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)

fig.suptitle('Model Confidence vs Trade Outcome — vtrain7', fontweight='bold', fontsize=12)
fig.tight_layout()
fig.savefig(FIGURES_DIR / 'stats_confidence_vs_outcome_v7.png', dpi=150)
plt.show()


# ── 5. HOLD TIME vs OUTCOME ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Scatter
ax = axes[0]
ax.scatter(T.loc[T['win']==1,'hold_h'], T.loc[T['win']==1,'pnl_pct']*100,
           color=GREEN, s=18, alpha=0.55, label='Win')
ax.scatter(T.loc[T['win']==0,'hold_h'], T.loc[T['win']==0,'pnl_pct']*100,
           color=RED,   s=18, alpha=0.55, label='Loss')
ax.axhline(0, color='black', lw=0.8, ls='--', alpha=0.5)
ax.set_xlabel('Hold time (hours)'); ax.set_ylabel('PnL (%)')
ax.set_title('Hold time vs PnL', fontweight='bold')
ax.legend(fontsize=8); ax.grid(alpha=0.2)

# Binned avg PnL
T['hold_bin'] = pd.cut(T['hold_h'],
                        bins=[0, 4, 8, 12, 18, 24, 36, 48, T['hold_h'].max()+1],
                        labels=['0-4h','4-8h','8-12h','12-18h','18-24h','24-36h','36-48h','48h+'])
hold_grp = T.groupby('hold_bin', observed=True)['pnl_pct'].agg(
    ['mean','count', lambda x: (x>0).mean()]).rename(columns={'<lambda_0>':'wr'})

ax = axes[1]
bc = [GREEN if v > 0 else RED for v in hold_grp['mean']]
ax.bar(range(len(hold_grp)), hold_grp['mean']*100, color=bc, alpha=0.8,
       tick_label=hold_grp.index.tolist())
ax.axhline(0, color='black', lw=0.8, ls='--', alpha=0.5)
ax.set_xlabel('Hold time bucket'); ax.set_ylabel('Avg PnL (%)')
ax.set_title('Avg PnL by hold time', fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=35, ha='right', fontsize=8)

ax = axes[2]
ax.bar(range(len(hold_grp)), hold_grp['wr']*100, color=BLUE, alpha=0.75,
       tick_label=hold_grp.index.tolist())
ax.axhline(50, color='black', lw=0.8, ls='--', alpha=0.5)
ax.set_xlabel('Hold time bucket'); ax.set_ylabel('Win rate (%)')
ax.set_title('Win rate by hold time', fontweight='bold')
ax.set_ylim(0, 100); ax.grid(axis='y', alpha=0.3)
for i, (_, row) in enumerate(hold_grp.iterrows()):
    ax.text(i, row['wr']*100 + 1, f'n={int(row["count"])}', ha='center', fontsize=7)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=35, ha='right', fontsize=8)

fig.suptitle('Hold Time Analysis — vtrain7', fontweight='bold', fontsize=12)
fig.tight_layout()
fig.savefig(FIGURES_DIR / 'stats_hold_time_v7.png', dpi=150)
plt.show()


# ── 6. TIME-OF-DAY + DAY-OF-WEEK ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(18, 9))

# Hour of day: win rate
hr_grp = T.groupby('hour_entry').agg(
    wr=('win','mean'), avg_pnl=('pnl_pct','mean'), n=('win','count'))

ax = axes[0, 0]
bar_c = [GREEN if v >= 0.5 else RED for v in hr_grp['wr']]
ax.bar(hr_grp.index, hr_grp['wr']*100, color=bar_c, alpha=0.8)
ax.axhline(50, color='black', lw=0.8, ls='--', alpha=0.5)
ax.set_xlabel('Hour of entry (UTC)'); ax.set_ylabel('Win rate (%)')
ax.set_title('Win rate by hour of day', fontweight='bold')
ax.set_ylim(0, 100); ax.set_xticks(range(0, 24, 2)); ax.grid(axis='y', alpha=0.3)

ax = axes[0, 1]
bar_c2 = [GREEN if v > 0 else RED for v in hr_grp['avg_pnl']]
ax.bar(hr_grp.index, hr_grp['avg_pnl']*100, color=bar_c2, alpha=0.8)
ax.axhline(0, color='black', lw=0.8, ls='--', alpha=0.5)
ax.set_xlabel('Hour of entry (UTC)'); ax.set_ylabel('Avg PnL (%)')
ax.set_title('Avg PnL by hour of day', fontweight='bold')
ax.set_xticks(range(0, 24, 2)); ax.grid(axis='y', alpha=0.3)

# Day of week: win rate
dow_order = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
dow_grp = (T.groupby('dow_name')
            .agg(wr=('win','mean'), avg_pnl=('pnl_pct','mean'), n=('win','count'))
            .reindex([d for d in dow_order if d in T['dow_name'].unique()]))

ax = axes[1, 0]
bar_c = [GREEN if v >= 0.5 else RED for v in dow_grp['wr']]
bars = ax.bar(range(len(dow_grp)), dow_grp['wr']*100, color=bar_c, alpha=0.8,
              tick_label=dow_grp.index.tolist())
ax.axhline(50, color='black', lw=0.8, ls='--', alpha=0.5)
ax.set_ylabel('Win rate (%)'); ax.set_title('Win rate by day of week', fontweight='bold')
ax.set_ylim(0, 100); ax.grid(axis='y', alpha=0.3)
for i, (_, row) in enumerate(dow_grp.iterrows()):
    ax.text(i, row['wr']*100 + 1, f'n={int(row["n"])}', ha='center', fontsize=8)

ax = axes[1, 1]
bar_c2 = [GREEN if v > 0 else RED for v in dow_grp['avg_pnl']]
ax.bar(range(len(dow_grp)), dow_grp['avg_pnl']*100, color=bar_c2, alpha=0.8,
       tick_label=dow_grp.index.tolist())
ax.axhline(0, color='black', lw=0.8, ls='--', alpha=0.5)
ax.set_ylabel('Avg PnL (%)'); ax.set_title('Avg PnL by day of week', fontweight='bold')
ax.grid(axis='y', alpha=0.3)

fig.suptitle('Time-of-Day & Day-of-Week Patterns — vtrain7', fontweight='bold', fontsize=12)
fig.tight_layout()
fig.savefig(FIGURES_DIR / 'stats_time_patterns_v7.png', dpi=150)
plt.show()


# ── 7. VOLATILITY REGIME AT ENTRY ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

atr_grp = T.groupby('atr_q', observed=True).agg(
    wr=('win','mean'), avg_pnl=('pnl_pct','mean'), n=('win','count'),
    avg_atr=('atr_entry','mean'))
atr_labels = [f'{lbl}\n(ATR={row.avg_atr:.3%})' for lbl, row in atr_grp.iterrows()]

ax = axes[0]
bar_c = [GREEN if v >= 0.5 else RED for v in atr_grp['wr']]
ax.bar(range(len(atr_grp)), atr_grp['wr']*100, color=bar_c, alpha=0.8, tick_label=atr_labels)
ax.axhline(50, color='black', lw=0.8, ls='--', alpha=0.5)
ax.set_ylabel('Win rate (%)'); ax.set_title('Win rate by ATR quintile', fontweight='bold')
ax.set_ylim(0, 100); ax.grid(axis='y', alpha=0.3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=20, ha='right', fontsize=8)

ax = axes[1]
bar_c2 = [GREEN if v > 0 else RED for v in atr_grp['avg_pnl']]
ax.bar(range(len(atr_grp)), atr_grp['avg_pnl']*100, color=bar_c2, alpha=0.8, tick_label=atr_labels)
ax.axhline(0, color='black', lw=0.8, ls='--', alpha=0.5)
ax.set_ylabel('Avg PnL (%)'); ax.set_title('Avg PnL by ATR quintile', fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=20, ha='right', fontsize=8)

ax = axes[2]
ax.scatter(T['atr_entry']*100, T['pnl_pct']*100,
           c=[GREEN if w else RED for w in T['win']], s=15, alpha=0.45)
m_a, b_a, r_a, p_a, _ = stats_scipy.linregress(T['atr_entry']*100, T['pnl_pct']*100)
xr = np.linspace(T['atr_entry'].min()*100, T['atr_entry'].max()*100, 100)
ax.plot(xr, m_a*xr+b_a, color=ACCENT, lw=1.8,
        label=f'slope={m_a:.3f}  p={p_a:.3f}')
ax.axhline(0, color='black', lw=0.8, ls='--', alpha=0.5)
ax.set_xlabel('ATR at entry (%)'); ax.set_ylabel('Trade PnL (%)')
ax.set_title('ATR vs PnL scatter', fontweight='bold')
ax.legend(fontsize=8); ax.grid(alpha=0.2)

fig.suptitle('Volatility Regime at Entry — vtrain7', fontweight='bold', fontsize=12)
fig.tight_layout()
fig.savefig(FIGURES_DIR / 'stats_volatility_regime_v7.png', dpi=150)
plt.show()


# ── 8. ROLLING PERFORMANCE ────────────────────────────────────────────────────
ROLL_W = min(20, max(5, n // 8))   # adaptive window: ~1/8 of total trades, min 5

fig, axes = plt.subplots(2, 1, figsize=(18, 7), sharex=True)

roll_wr  = T['win'].rolling(ROLL_W).mean()
roll_pnl = T['pnl_pct'].rolling(ROLL_W).mean() * 100
x        = T['entry_ts'].values

ax = axes[0]
ax.plot(x, roll_wr*100, color=BLUE, lw=1.4)
ax.axhline(50,                    color='black', lw=0.8, ls='--', alpha=0.5, label='50%')
ax.axhline(T['win'].mean()*100,   color=ACCENT,  lw=1.2, ls=':',  label=f'Overall {T["win"].mean():.1%}')
ax.fill_between(x, roll_wr*100, 50,
                where=(roll_wr*100 >= 50), color=GREEN, alpha=0.18)
ax.fill_between(x, roll_wr*100, 50,
                where=(roll_wr*100  < 50), color=RED,   alpha=0.18)
ax.set_ylabel(f'Rolling {ROLL_W}-trade win rate (%)')
ax.set_title(f'Rolling performance (window={ROLL_W} trades)', fontweight='bold')
ax.set_ylim(0, 100); ax.legend(fontsize=8); ax.grid(alpha=0.2)

ax = axes[1]
ax.plot(x, roll_pnl, color=ACCENT, lw=1.4)
ax.axhline(0,                     color='black', lw=0.8, ls='--', alpha=0.5)
ax.axhline(T['pnl_pct'].mean()*100, color=BLUE, lw=1.2, ls=':',
           label=f'Overall mean {T["pnl_pct"].mean():+.3%}')
ax.fill_between(x, roll_pnl, 0,
                where=(roll_pnl >= 0), color=GREEN, alpha=0.18)
ax.fill_between(x, roll_pnl, 0,
                where=(roll_pnl  < 0), color=RED,   alpha=0.18)
ax.set_ylabel(f'Rolling {ROLL_W}-trade avg PnL (%)')
ax.legend(fontsize=8); ax.grid(alpha=0.2)

ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')

fig.tight_layout()
fig.savefig(FIGURES_DIR / 'stats_rolling_performance_v7.png', dpi=150)
plt.show()


# ── 9. STREAK ANALYSIS ────────────────────────────────────────────────────────
streaks = []
cur_streak = 0; cur_type = None
for w in T['win'].values:
    t = 'W' if w else 'L'
    if t == cur_type:
        cur_streak += 1
    else:
        if cur_type is not None:
            streaks.append((cur_type, cur_streak))
        cur_type = t; cur_streak = 1
if cur_type:
    streaks.append((cur_type, cur_streak))

w_streaks = [s for tp, s in streaks if tp == 'W']
l_streaks = [s for tp, s in streaks if tp == 'L']

print('\n-- Streak Analysis --')
print(f'  Longest win streak : {max(w_streaks) if w_streaks else 0}')
print(f'  Longest loss streak: {max(l_streaks) if l_streaks else 0}')
print(f'  Avg win streak     : {np.mean(w_streaks):.1f}' if w_streaks else '  No win streaks')
print(f'  Avg loss streak    : {np.mean(l_streaks):.1f}' if l_streaks else '  No loss streaks')

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for ax, (streak_list, color, label) in zip(axes, [
    (w_streaks, GREEN, 'Win streaks'),
    (l_streaks, RED,   'Loss streaks'),
]):
    if streak_list:
        ax.hist(streak_list, bins=range(1, max(streak_list)+2), color=color, alpha=0.8,
                edgecolor='white', align='left')
        ax.axvline(np.mean(streak_list), color='black', lw=1.2, ls='--',
                   label=f'Mean={np.mean(streak_list):.1f}  Max={max(streak_list)}')
    ax.set_xlabel('Streak length'); ax.set_ylabel('Frequency')
    ax.set_title(f'{label} distribution', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)

fig.suptitle('Win/Loss Streak Analysis — vtrain7', fontweight='bold')
fig.tight_layout()
fig.savefig(FIGURES_DIR / 'stats_streaks_v7.png', dpi=150)
plt.show()

print('\nAll statistics saved to FIGURES_DIR.')

