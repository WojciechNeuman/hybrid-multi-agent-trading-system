"""Meta-Labeling Framework (López de Prado, AFML Ch. 3).

Two-model pipeline to filter low-conviction trades before fees eat the edge:

  Primary model  → generates raw Long signals (high recall, any rule-based method)
  Secondary model → LightGBM trained *only* on primary signal timestamps,
                    predicts P(trade hits TP | signal fired)

If secondary probability < threshold → discard the trade (no fee bleed).

Pipeline
--------
1. ``generate_primary_signals()`` — EMA crossover (or plug in structural signals)
2. ``build_meta_labels()``        — for each primary signal, label 1 if TP hit first
3. ``train_meta_model()``         — fit LightGBM on (features at signal time, meta_label)
4. ``MetaLabelingBacktester``     — full OOS backtest with gatekeeper threshold

Quick-start
-----------
    from hmats.data.meta_labeling import (
        generate_primary_signals, build_meta_labels,
        train_meta_model, MetaLabelingBacktester,
    )
    import pandas as pd

    v1    = pd.read_parquet("data/features/BTCUSDT_1h_features.parquet")
    v3    = pd.read_parquet("data/features/BTCUSDT_1h_v3_features.parquet")
    raw   = pd.read_parquet("data/raw/BTCUSDT_1h.parquet")

    # 1. Primary signals on pre-OOS data
    signals = generate_primary_signals(raw, fast=12, slow=48)

    # 2. Meta-labels
    meta_df = build_meta_labels(raw, signals, sl_mult=1.5, tp_mult=2.0)

    # 3. Merge features at signal timestamps
    features = v1.join(v3, how="left")
    FEAT_COLS = [c for c in features.columns
                 if c not in ["label", "close", "sma_200", "atr_14_pct"]]
    X = features.loc[meta_df.index, FEAT_COLS].fillna(0)
    y = meta_df["meta_label"]

    # 4. Train meta-model (walk-forward ready)
    model = train_meta_model(X, y)

    # 5. Backtest
    bt = MetaLabelingBacktester(primary_signals=signals, meta_model=model,
                                 feature_cols=FEAT_COLS, threshold=0.60)
    equity, trade_log = bt.run(raw.loc[raw.index >= "2024-01-01"], features)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# 1. Primary Signal Generator — EMA Crossover (high-recall baseline)
# ──────────────────────────────────────────────────────────────────────────────

def generate_primary_signals(
    ohlcv: pd.DataFrame,
    fast: int = 12,
    slow: int = 48,
    direction: str = "long",
) -> pd.Series:
    """Generate primary Long signals via EMA crossover.

    Signal fires on the bar where fast EMA crosses above slow EMA.

    Parameters
    ----------
    ohlcv:
        DataFrame with a ``close`` column (lowercase).
    fast / slow:
        EMA spans in bars. Default: 12h / 48h.
    direction:
        ``"long"`` (fast > slow) or ``"both"`` (any cross).

    Returns
    -------
    Boolean Series of signal timestamps (True = signal on that bar).
    """
    close = ohlcv["close"]
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()

    above  = ema_f > ema_s
    cross_up = above & ~above.shift(1).fillna(False)

    if direction == "long":
        return cross_up.rename("primary_signal")
    elif direction == "both":
        cross_down = ~above & above.shift(1).fillna(True)
        return (cross_up | cross_down).rename("primary_signal")
    else:
        raise ValueError(f"direction must be 'long' or 'both', got {direction!r}")


def generate_primary_signals_from_structural(
    signal_series: pd.Series,
) -> pd.Series:
    """Wrap an existing external signal series as a primary signal.

    Use this to plug in V2 structural breakout signals or any other rule.
    ``signal_series`` should be boolean or 0/1 integer.
    """
    return signal_series.astype(bool).rename("primary_signal")


# ──────────────────────────────────────────────────────────────────────────────
# 2. Meta-Label Construction
# ──────────────────────────────────────────────────────────────────────────────

def build_meta_labels(
    ohlcv: pd.DataFrame,
    primary_signals: pd.Series,
    atr_col: Optional[pd.Series] = None,
    sl_mult: float = 1.5,
    tp_mult: float = 2.0,
    max_hold: int = 48,
    min_atr_pct: float = 0.005,
) -> pd.DataFrame:
    """Construct the meta-label target matrix.

    For each primary signal bar t:
      - Define SL = close[t] * (1 - sl_mult * atr_pct[t])
      - Define TP = close[t] * (1 + tp_mult * atr_pct[t])
      - Walk forward bar by bar:
          * If low[t+k] <= SL first → meta_label = 0
          * If high[t+k] >= TP first → meta_label = 1
          * If max_hold reached first → meta_label = 0 (trade timed out)

    Parameters
    ----------
    ohlcv:
        Raw OHLCV DataFrame with columns ``open``, ``high``, ``low``, ``close``.
    primary_signals:
        Boolean Series from ``generate_primary_signals()``.
    atr_col:
        Precomputed ATR-pct series. If None, computed internally as
        ``rolling(14).mean()`` of ``|log_ret_1h|``.
    sl_mult / tp_mult:
        ATR multiples for stop-loss / take-profit.
    max_hold:
        Maximum bars to hold before labelling as timeout (0).
    min_atr_pct:
        Floor for ATR (prevents degenerate SL/TP on very quiet bars).

    Returns
    -------
    DataFrame indexed to signal timestamps with columns:
      ``meta_label``  — 1 (TP hit) or 0 (SL / timeout)
      ``sl_px``       — stop-loss price
      ``tp_px``       — take-profit price
      ``exit_bar``    — index position of exit
      ``hold_bars``   — bars held
      ``exit_reason`` — 'tp', 'sl', 'timeout'
    """
    close_arr = ohlcv["close"].values
    high_arr  = ohlcv["high"].values
    low_arr   = ohlcv["low"].values
    idx       = ohlcv.index
    n         = len(ohlcv)

    # ATR-pct proxy: 14-bar rolling mean of |log returns|
    if atr_col is not None:
        atr_arr = atr_col.reindex(ohlcv.index).values
    else:
        log_ret = np.abs(np.log(close_arr[1:] / (close_arr[:-1] + 1e-12)))
        log_ret = np.concatenate([[np.nan], log_ret])
        atr_series = pd.Series(log_ret).rolling(14, min_periods=7).mean().values
        atr_arr = atr_series

    sig_arr = primary_signals.reindex(ohlcv.index).fillna(False).values
    signal_idx = np.where(sig_arr)[0]

    rows = []
    for t in signal_idx:
        px   = close_arr[t]
        atr  = max(float(atr_arr[t]) if not np.isnan(atr_arr[t]) else min_atr_pct,
                   min_atr_pct)
        sl   = px * (1.0 - sl_mult * atr)
        tp   = px * (1.0 + tp_mult * atr)

        label  = 0
        reason = "timeout"
        exit_t = min(t + max_hold, n - 1)
        hold   = 0

        for k in range(1, max_hold + 1):
            j = t + k
            if j >= n:
                exit_t = n - 1
                break
            hold = k
            if low_arr[j] <= sl:
                label  = 0
                reason = "sl"
                exit_t = j
                break
            if high_arr[j] >= tp:
                label  = 1
                reason = "tp"
                exit_t = j
                break
        else:
            exit_t = min(t + max_hold, n - 1)

        rows.append({
            "signal_bar":  t,
            "meta_label":  label,
            "sl_px":       sl,
            "tp_px":       tp,
            "entry_px":    px,
            "exit_bar":    exit_t,
            "hold_bars":   hold,
            "exit_reason": reason,
        })

    result = pd.DataFrame(rows, index=idx[signal_idx])
    result.index.name = ohlcv.index.name
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 3. Secondary Model Training (LightGBM)
# ──────────────────────────────────────────────────────────────────────────────

def train_meta_model(
    X: pd.DataFrame,
    y: pd.Series,
    val_frac: float = 0.15,
    lgbm_params: Optional[dict] = None,
    verbose: bool = True,
):
    """Train a LightGBM meta-model on (features at signal time, meta_label).

    Only signal-timestamp rows are used — the secondary model never sees
    non-signal bars.

    Parameters
    ----------
    X:
        Feature matrix indexed to signal timestamps (from meta_df.index).
    y:
        Binary meta-label Series (1 = TP hit, 0 = SL / timeout).
    val_frac:
        Fraction of X/y to use for early-stopping validation (time-ordered).
    lgbm_params:
        LightGBM hyperparameters. Sensible defaults are used if None.

    Returns
    -------
    Fitted ``lgb.LGBMClassifier`` instance.
    """
    import lightgbm as lgb  # lazy import

    if lgbm_params is None:
        lgbm_params = dict(
            num_leaves=31, max_depth=6, learning_rate=0.05,
            n_estimators=500, colsample_bytree=0.7,
            subsample=0.7, min_child_samples=20,
            reg_alpha=0.1, reg_lambda=1.0,
        )

    n_val  = max(30, int(len(X) * val_frac))
    X_tr, X_va = X.iloc[:-n_val], X.iloc[-n_val:]
    y_tr, y_va = y.iloc[:-n_val], y.iloc[-n_val:]

    if y_tr.nunique() < 2 or y_va.nunique() < 2:
        raise ValueError(
            "Meta-label has only one class in train or val split. "
            "Check primary signal frequency and label balance."
        )

    model = lgb.LGBMClassifier(**lgbm_params, verbose=-1, random_state=42, n_jobs=-1)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(30, verbose=verbose)],
    )

    if verbose:
        from sklearn.metrics import roc_auc_score
        probs_va = model.predict_proba(X_va)[:, 1]
        auc = roc_auc_score(y_va, probs_va)
        pos_rate = y.mean()
        print(f"Meta-model trained on {len(X_tr)} signals | val AUC={auc:.4f} | "
              f"label positive rate={pos_rate:.3f}")

    return model


# ──────────────────────────────────────────────────────────────────────────────
# 4. Meta-Labeling Backtester
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MetaLabelingBacktester:
    """Gatekeeper backtester: primary signal + meta-model probability filter.

    Attributes
    ----------
    primary_signals:
        Boolean Series of raw trade signals (from ``generate_primary_signals``).
    meta_model:
        Fitted LGBMClassifier from ``train_meta_model``.
    feature_cols:
        Column names the meta-model was trained on.
    threshold:
        Meta-probability threshold. Trades with P < threshold are discarded.
    sl_mult / tp_mult:
        ATR multiples for SL / TP (must match those used in ``build_meta_labels``).
    max_hold / cooldown_bars:
        Hold limit and cooldown between trades.
    maker_fee / taker_fee:
        Per-side futures fees (applied on notional).
    leverage:
        Fixed leverage for the backtest (1.0 = no leverage).
    """

    primary_signals:  pd.Series
    meta_model:       object
    feature_cols:     list[str]
    threshold:        float = 0.60
    sl_mult:          float = 1.5
    tp_mult:          float = 2.0
    max_hold:         int   = 48
    cooldown_bars:    int   = 4
    maker_fee:        float = 0.0002
    taker_fee:        float = 0.0005
    leverage:         float = 1.0
    trade_log_:       list  = field(default_factory=list, init=False)

    def run(
        self,
        ohlcv: pd.DataFrame,
        features: pd.DataFrame,
        atr_col: Optional[pd.Series] = None,
    ) -> tuple[np.ndarray, pd.DataFrame]:
        """Run the meta-labeling backtest over an OOS window.

        Parameters
        ----------
        ohlcv:
            Raw OHLCV DataFrame for the OOS period (must contain
            ``open``, ``high``, ``low``, ``close``).
        features:
            Feature DataFrame (same index as ohlcv or wider).
        atr_col:
            ATR-pct series. If None, computed internally.

        Returns
        -------
        (equity_array, trade_log_df)
        """
        n         = len(ohlcv)
        close_arr = ohlcv["close"].values
        high_arr  = ohlcv["high"].values
        low_arr   = ohlcv["low"].values
        idx       = ohlcv.index

        if atr_col is not None:
            atr_arr = atr_col.reindex(idx).values
        else:
            log_ret = np.abs(np.log(close_arr[1:] / (close_arr[:-1] + 1e-12)))
            log_ret = np.concatenate([[np.nan], log_ret])
            atr_arr = pd.Series(log_ret).rolling(14, min_periods=7).mean().values

        sig_arr  = self.primary_signals.reindex(idx).fillna(False).values
        feat_mat = features.reindex(idx)[self.feature_cols].fillna(0)

        equity_arr = np.ones(n)
        cur_eq     = 1.0
        self.trade_log_ = []

        in_pos       = False
        pos_eq       = 1.0
        entry_px     = sl_px = tp_px = 0.0
        entry_bar    = hold_count = cooldown_count = 0
        entry_fee_pd = 0.0
        lev          = self.leverage
        pending      = None  # (limit_px, sl_px, tp_px)

        for i in range(n):
            if in_pos:
                hold_count += 1
                price_ret = close_arr[i] / entry_px - 1
                equity_arr[i] = pos_eq * max(1 + price_ret * lev, 0.0)

                exited = False; exit_px = 0.0; reason = ""; exit_fee = 0.0
                if   low_arr[i]  <= sl_px:           exit_px = sl_px;        exited = True; reason = "sl";      exit_fee = self.taker_fee
                elif high_arr[i] >= tp_px:           exit_px = tp_px;        exited = True; reason = "tp";      exit_fee = self.maker_fee
                elif hold_count  >= self.max_hold:   exit_px = close_arr[i]; exited = True; reason = "timeout"; exit_fee = self.taker_fee

                if exited:
                    pr       = exit_px / entry_px - 1
                    fee_cost = (entry_fee_pd + exit_fee) * lev
                    net_ret  = max(pr * lev - fee_cost, -1.0)
                    cur_eq   = pos_eq * (1.0 + net_ret)
                    equity_arr[i] = cur_eq
                    self.trade_log_.append({
                        "entry_bar":  entry_bar, "exit_bar": i,
                        "entry_time": idx[entry_bar], "exit_time": idx[i],
                        "entry_px":   float(entry_px), "exit_px": float(exit_px),
                        "leverage":   float(lev),
                        "gross_pct":  float(pr * lev * 100),
                        "fee_pct":    float(fee_cost * 100),
                        "pnl_pct":    float(net_ret * 100),
                        "reason":     reason, "hold_bars": hold_count,
                    })
                    in_pos = False; cooldown_count = self.cooldown_bars

            elif pending is not None:
                lim_px, p_sl, p_tp = pending
                entry_px     = lim_px if low_arr[i] <= lim_px else close_arr[i]
                entry_fee_pd = self.maker_fee if low_arr[i] <= lim_px else self.taker_fee
                sl_px = p_sl; tp_px = p_tp
                in_pos = True; pos_eq = cur_eq; entry_bar = i; hold_count = 0; pending = None
                equity_arr[i] = cur_eq

            elif cooldown_count > 0:
                cooldown_count -= 1; equity_arr[i] = cur_eq

            elif sig_arr[i] and i + 1 < n:
                # ── Gatekeeper: query meta-model ──────────────────────────────
                x_row = feat_mat.iloc[i].values.reshape(1, -1)
                meta_prob = float(self.meta_model.predict_proba(x_row)[0, 1])

                if meta_prob >= self.threshold:
                    atr = max(float(atr_arr[i]) if not np.isnan(atr_arr[i]) else 0.02, 0.002)
                    px  = close_arr[i]
                    pending = (
                        px * (1.0 - 0.30 * atr),
                        px * (1.0 - self.sl_mult * atr),
                        px * (1.0 + self.tp_mult * atr),
                    )

                equity_arr[i] = cur_eq

            else:
                equity_arr[i] = cur_eq

        if in_pos:
            pr       = close_arr[-1] / entry_px - 1
            fee_cost = (entry_fee_pd + self.taker_fee) * lev
            net_ret  = max(pr * lev - fee_cost, -1.0)
            cur_eq   = pos_eq * (1.0 + net_ret)
            equity_arr[-1] = cur_eq
            self.trade_log_.append({
                "entry_bar": entry_bar, "exit_bar": n - 1,
                "entry_time": idx[entry_bar], "exit_time": idx[-1],
                "entry_px": float(entry_px), "exit_px": float(close_arr[-1]),
                "leverage": float(lev),
                "gross_pct": float(pr * lev * 100),
                "fee_pct": float(fee_cost * 100),
                "pnl_pct": float(net_ret * 100),
                "reason": "eod", "hold_bars": hold_count,
            })

        return equity_arr, pd.DataFrame(self.trade_log_)

    @property
    def n_trades(self) -> int:
        return len(self.trade_log_)

    @property
    def win_rate(self) -> float:
        if not self.trade_log_:
            return 0.0
        return float(np.mean([r["pnl_pct"] > 0 for r in self.trade_log_]))


# ──────────────────────────────────────────────────────────────────────────────
# Walk-Forward Meta-Labeling (for OOS evaluation)
# ──────────────────────────────────────────────────────────────────────────────

def run_meta_wfo(
    ohlcv: pd.DataFrame,
    features: pd.DataFrame,
    feature_cols: list[str],
    oos_start: str = "2024-01-01",
    step_size: int = 720,
    embargo: int = 12,
    val_frac: float = 0.15,
    threshold: float = 0.60,
    sl_mult: float = 1.5,
    tp_mult: float = 2.0,
    max_hold: int = 48,
    fast_ema: int = 12,
    slow_ema: int = 48,
    lgbm_params: Optional[dict] = None,
    verbose: bool = True,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Walk-forward meta-labeling pipeline over the OOS period.

    At each WFO step:
      1. Generate primary signals on the training slice.
      2. Build meta-labels on the training slice.
      3. Train LightGBM meta-model on training-slice signal rows.
      4. Apply the gatekeeper to the test slice (step_size bars).

    Returns concatenated equity array and trade log for the full OOS period.
    """
    oos_mask = ohlcv.index >= pd.Timestamp(oos_start)
    oos_ohlcv = ohlcv[oos_mask].copy()
    n_oos     = len(oos_ohlcv)

    equity_full    = np.ones(n_oos)
    trade_log_all  = []
    cur_eq         = 1.0

    step = 0
    while step * step_size < n_oos:
        t0 = step * step_size
        t1 = min((step + 1) * step_size, n_oos)

        first_test_ts = oos_ohlcv.index[t0]
        abs_test_start = ohlcv.index.searchsorted(first_test_ts)
        train_end_abs  = abs_test_start - embargo

        if train_end_abs < 500:
            step += 1
            continue

        train_ohlcv    = ohlcv.iloc[:train_end_abs]
        train_features = features.iloc[:train_end_abs]

        # Primary signals on training data
        primary_sig = generate_primary_signals(train_ohlcv, fast=fast_ema, slow=slow_ema)
        sig_times   = primary_sig[primary_sig].index

        if len(sig_times) < 50:
            step += 1
            continue

        # Meta-labels on training data
        meta_df = build_meta_labels(
            train_ohlcv, primary_sig,
            sl_mult=sl_mult, tp_mult=tp_mult, max_hold=max_hold,
        )

        X_meta = train_features.loc[meta_df.index, feature_cols].fillna(0)
        y_meta = meta_df["meta_label"]

        if y_meta.nunique() < 2:
            step += 1
            continue

        try:
            model = train_meta_model(X_meta, y_meta, val_frac=val_frac,
                                     lgbm_params=lgbm_params, verbose=False)
        except Exception as e:
            if verbose:
                print(f"  Step {step}: model training failed ({e}), skipping")
            step += 1
            continue

        # Run gatekeeper on test slice
        test_ohlcv = oos_ohlcv.iloc[t0:t1]
        full_primary = generate_primary_signals(
            pd.concat([train_ohlcv, oos_ohlcv.iloc[:t1]]),
            fast=fast_ema, slow=slow_ema,
        )
        test_primary = full_primary.reindex(test_ohlcv.index)

        bt = MetaLabelingBacktester(
            primary_signals=test_primary,
            meta_model=model,
            feature_cols=feature_cols,
            threshold=threshold,
            sl_mult=sl_mult,
            tp_mult=tp_mult,
            max_hold=max_hold,
        )
        # Carry equity from previous step
        test_feat = features.reindex(test_ohlcv.index)
        eq_step, tdf_step = bt.run(test_ohlcv, test_feat)

        # Chain equity
        eq_step = eq_step * cur_eq
        cur_eq  = eq_step[-1]
        equity_full[t0:t1] = eq_step

        if len(tdf_step) > 0:
            trade_log_all.append(tdf_step)

        if verbose:
            n_tr = len(tdf_step)
            wr   = (tdf_step["pnl_pct"] > 0).mean() if n_tr > 0 else 0.0
            print(f"  Step {step:>3}: {n_tr:>3} trades  WR={wr:.0%}  "
                  f"step_eq={eq_step[-1] / cur_eq:.3f}  cum_eq={cur_eq:.3f}")

        step += 1

    all_trades = pd.concat(trade_log_all) if trade_log_all else pd.DataFrame()
    return equity_full, all_trades
