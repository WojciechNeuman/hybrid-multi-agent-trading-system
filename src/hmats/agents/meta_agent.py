"""Meta-Labeling Supervisory Agent (López de Prado, AFML Ch. 3 — multi-agent variant).

Combines signals from heterogeneous base agents (LGBM, TCN, DRL) and learns
when their collective opinion is trustworthy enough to bet on.

Pipeline
--------
1. Primary signal: any base agent fires (long or short) on bar t.
2. Triple Barrier Method labels bar t: 1 if trade hits TP, 0 if SL/timeout.
3. Meta-features at bar t: base-agent probabilities/actions + market context.
4. Meta-model (LightGBM): trained to predict P(success | signal fired).
5. Sizing: position = sign(primary_side) × meta_prob  (continuous in [−1, 1]).

Walk-Forward OOS Discipline
---------------------------
The meta-model is trained on base-agent signals from a *prior* period
(meta-training window) and evaluated on a subsequent hold-out period.
This prevents the meta-model from seeing its own OOS period during fitting.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# ── Market-context features appended to base-agent signals ────────────────────
CONTEXT_FEATURES: list[str] = [
    "atr_14_pct", "vol_ratio_24h", "bb_width_pct", "trend_score",
    "close_vs_sma_50", "rsi_14", "hour_sin", "hour_cos",
    "hurst_24h", "ret_1h",
]

_META_LGBM_DEFAULTS: dict = dict(
    num_leaves=31, max_depth=5, learning_rate=0.05,
    n_estimators=500, colsample_bytree=0.7, subsample=0.8,
    min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
    verbose=-1, random_state=42,
    n_jobs=1,   # avoids OMP segfault on macOS when called in a loop
)


# ── Triple Barrier Method (vectorised, signal-bar scoped) ─────────────────────

def _tbm_label_signals(
    signal_bars: np.ndarray,
    close_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    atr_arr: np.ndarray,
    side_arr: np.ndarray,
    sl_mult: float = 1.5,
    tp_mult: float = 2.5,
    max_hold: int = 48,
    min_atr: float = 0.005,
) -> pd.DataFrame:
    """Label each signal bar using Triple Barrier Method.

    Parameters
    ----------
    signal_bars:
        Integer indices (into the full array) where a signal fired.
    side_arr:
        +1 (long) or −1 (short) for each bar in the full array.
    Returns a DataFrame indexed to signal_bars with columns:
        ``meta_label``, ``exit_bar``, ``hold_bars``, ``exit_reason``.
    """
    rows: list[dict] = []
    n = len(close_arr)
    for t in signal_bars:
        px = close_arr[t]
        atr = max(float(atr_arr[t]) if not np.isnan(atr_arr[t]) else min_atr, min_atr)
        side = int(side_arr[t])

        if side == 1:   # long
            tp = px * (1.0 + tp_mult * atr)
            sl = px * (1.0 - sl_mult * atr)
        else:           # short
            tp = px * (1.0 - tp_mult * atr)
            sl = px * (1.0 + sl_mult * atr)

        label = 0
        reason = "timeout"
        exit_t = min(t + max_hold, n - 1)
        hold = 0

        for k in range(1, max_hold + 1):
            j = t + k
            if j >= n:
                exit_t = n - 1
                break
            hold = k
            if side == 1:
                if low_arr[j] <= sl:
                    label = 0; reason = "sl"; exit_t = j; break
                if high_arr[j] >= tp:
                    label = 1; reason = "tp"; exit_t = j; break
            else:
                if high_arr[j] >= sl:
                    label = 0; reason = "sl"; exit_t = j; break
                if low_arr[j] <= tp:
                    label = 1; reason = "tp"; exit_t = j; break
        else:
            exit_t = min(t + max_hold, n - 1)

        rows.append({
            "signal_bar": t,
            "meta_label": label,
            "exit_bar": exit_t,
            "hold_bars": hold,
            "exit_reason": reason,
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["signal_bar", "meta_label", "exit_bar", "hold_bars", "exit_reason"]
    )


# ── Signal combination helpers ────────────────────────────────────────────────

def build_signal_df(
    ohlcv_df: pd.DataFrame,
    lgbm_p_up: Optional[pd.Series] = None,
    tcn_signals: Optional[pd.DataFrame] = None,
    drl_action: Optional[pd.Series] = None,
    gp_action: Optional[pd.Series] = None,
    long_threshold: float = 0.58,
    short_threshold: float = 0.35,
    drl_long_val: int = 1,
    drl_short_val: int = -1,
    context_features: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Merge base-agent signals and market context into a single DataFrame.

    All base-agent series are reindexed to ``ohlcv_df.index`` (forward-filled
    within the same bar; bars with no signal get NaN for the probability columns
    and 0 for the discrete-action columns).

    Returns a DataFrame with columns:
        ``lgbm_p_up``, ``tcn_p_up``, ``tcn_p_down``, ``drl_action``, ``gp_action``,
        ``primary_signal`` (bool), ``primary_side`` (+1/−1),
        + all available ``context_features`` from ohlcv_df.
    """
    ctx_cols = context_features or CONTEXT_FEATURES
    df = ohlcv_df[["close", "high", "low", "atr_14_pct"]].copy()

    # Base-agent signals
    if lgbm_p_up is not None:
        df["lgbm_p_up"] = lgbm_p_up.reindex(df.index)
    else:
        df["lgbm_p_up"] = np.nan

    if tcn_signals is not None:
        for col in ["tcn_p_up", "tcn_p_down"]:
            if col in tcn_signals.columns:
                df[col] = tcn_signals[col].reindex(df.index)
            else:
                df[col] = np.nan
    else:
        df["tcn_p_up"] = np.nan
        df["tcn_p_down"] = np.nan

    if drl_action is not None:
        df["drl_action"] = drl_action.reindex(df.index).fillna(0).astype(int)
    else:
        df["drl_action"] = 0

    if gp_action is not None:
        df["gp_action"] = gp_action.reindex(df.index).fillna(0).astype(int)
    else:
        df["gp_action"] = 0

    # Market context
    for col in ctx_cols:
        if col in ohlcv_df.columns:
            df[col] = ohlcv_df[col]

    # Primary signal: any base agent fires long or short
    lgbm_long  = df["lgbm_p_up"].fillna(0.5) > long_threshold
    lgbm_short = df["lgbm_p_up"].fillna(0.5) < short_threshold
    tcn_long   = df["tcn_p_up"].fillna(0.5) > long_threshold
    tcn_short  = df["tcn_p_down"].fillna(0.5) > (1 - short_threshold)
    drl_long   = df["drl_action"] == drl_long_val
    drl_short  = df["drl_action"] == drl_short_val
    gp_long    = df["gp_action"] == 1
    gp_short   = df["gp_action"] == -1

    primary_long  = lgbm_long  | tcn_long  | drl_long  | gp_long
    primary_short = lgbm_short | tcn_short | drl_short | gp_short

    df["primary_signal"] = primary_long | primary_short
    # When both fire (rare), long takes precedence
    df["primary_side"] = np.where(primary_long, 1, np.where(primary_short, -1, 0))

    return df


# ── Meta-Labeling Agent ───────────────────────────────────────────────────────

@dataclass
class MetaSupervisoryAgent:
    """Walk-forward meta-labeling supervisor.

    Accepts a ``signals_df`` produced by ``build_signal_df`` and runs a
    walk-forward meta-model that learns to predict P(trade success).

    Parameters
    ----------
    sl_mult / tp_mult:
        Triple-barrier ATR multiples for meta-label generation.
    max_hold:
        Maximum hold period for TBM (bars).
    train_window:
        Number of *signal* events (rows) used per meta-model training fold.
        The meta-model trains on the most recent ``train_window`` signal events.
    step_months:
        OOS step size in calendar months (default 3).
    threshold:
        Meta-probability cutoff: trades with P < threshold are vetoed (size → 0).
    lgbm_params:
        Override default meta-model hyperparameters.
    """

    sl_mult: float = 1.5
    tp_mult: float = 2.5
    max_hold: int = 48
    train_window: int = 500
    step_months: int = 3
    threshold: float = 0.55
    lgbm_params: dict = field(default_factory=dict)

    # learned attributes (populated by fit / run_wfo)
    _feature_cols: list[str] = field(default_factory=list, init=False, repr=False)
    _model: object | None = field(default=None, init=False, repr=False)

    def _get_feature_cols(self, signals_df: pd.DataFrame) -> list[str]:
        candidates = [
            "lgbm_p_up", "tcn_p_up", "tcn_p_down", "drl_action", "gp_action",
            "primary_side",
        ] + CONTEXT_FEATURES
        return [c for c in candidates if c in signals_df.columns]

    def _build_meta_dataset(
        self, signals_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Run TBM over all signal bars and return (X, y, sides)."""
        signal_mask = signals_df["primary_signal"].fillna(False).astype(bool)
        sig_idx = np.where(signal_mask.values)[0]

        close_arr = signals_df["close"].values
        high_arr  = signals_df["high"].values
        low_arr   = signals_df["low"].values
        atr_arr   = signals_df["atr_14_pct"].values
        side_arr  = signals_df["primary_side"].values

        meta_df = _tbm_label_signals(
            sig_idx, close_arr, high_arr, low_arr, atr_arr, side_arr,
            sl_mult=self.sl_mult, tp_mult=self.tp_mult, max_hold=self.max_hold,
        )
        if meta_df.empty:
            return pd.DataFrame(), pd.Series(dtype=int), pd.Series(dtype=int)

        meta_df.index = signals_df.index[meta_df["signal_bar"].values]
        feat_cols = self._get_feature_cols(signals_df)
        X = signals_df.iloc[meta_df["signal_bar"].values][feat_cols].fillna(0).copy()
        X.index = meta_df.index
        y = meta_df["meta_label"]
        sides = pd.Series(side_arr[meta_df["signal_bar"].values], index=meta_df.index)
        return X, y, sides

    def _train_meta_model(self, X: pd.DataFrame, y: pd.Series) -> lgb.LGBMClassifier:
        params = {**_META_LGBM_DEFAULTS, **self.lgbm_params}
        val_n = max(20, int(len(X) * 0.15))
        X_tr, X_va = X.iloc[:-val_n], X.iloc[-val_n:]
        y_tr, y_va = y.iloc[:-val_n], y.iloc[-val_n:]

        if y_tr.nunique() < 2 or y_va.nunique() < 2:
            # Fallback: no validation split when only one class present
            model = lgb.LGBMClassifier(**params)
            model.fit(X, y)
            return model

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        return model

    # ── Public API ────────────────────────────────────────────────────────────

    def run_wfo(
        self,
        signals_df: pd.DataFrame,
        oos_start: pd.Timestamp,
        verbose: bool = True,
    ) -> tuple[pd.Series, pd.DataFrame]:
        """Walk-forward meta-labeling over the OOS period.

        The meta-model trains on all signal events *before* each OOS step,
        using the most recent ``self.train_window`` events.

        Parameters
        ----------
        signals_df:
            Output of ``build_signal_df`` — must span the full period
            (pre-OOS for meta-training, OOS for evaluation).
        oos_start:
            First bar of the meta-model's hold-out OOS window.

        Returns
        -------
        (sizing_series, trade_log_df)
            sizing_series: continuous position in [−1, 1] for each OOS bar.
            trade_log_df: one row per meta-approved trade.
        """
        self._feature_cols = self._get_feature_cols(signals_df)
        X_all, y_all, sides_all = self._build_meta_dataset(signals_df)

        if X_all.empty:
            if verbose:
                print("[MetaAgent] No primary signals found in signals_df.")
            empty = pd.Series(0.0, index=signals_df.index, name="meta_position")
            return empty[signals_df.index >= oos_start], pd.DataFrame()

        # OOS step boundaries (calendar months)
        oos_index = signals_df.index[signals_df.index >= oos_start]
        if len(oos_index) == 0:
            if verbose:
                print(f"[MetaAgent] No data found at or after oos_start={oos_start}.")
            empty = pd.Series(0.0, index=signals_df.index, name="meta_position")
            return empty, pd.DataFrame()

        step_starts = pd.date_range(
            start=oos_start,
            end=oos_index[-1],
            freq=f"{self.step_months}ME",
        )
        if len(step_starts) == 0:
            step_starts = pd.DatetimeIndex([oos_start])

        sizing = pd.Series(0.0, index=oos_index, name="meta_position")
        trades: list[dict] = []
        fold = 0
        meta_model: lgb.LGBMClassifier | None = None

        for k, step_start in enumerate(step_starts):
            step_end = step_starts[k + 1] if k + 1 < len(step_starts) else oos_index[-1] + pd.Timedelta(hours=1)

            # Meta-training: all signal events strictly before this step
            train_mask = X_all.index < step_start
            X_tr = X_all[train_mask].tail(self.train_window)
            y_tr = y_all[train_mask].tail(self.train_window)

            if len(X_tr) < 30 or y_tr.nunique() < 2:
                if verbose:
                    print(f"  [MetaAgent] fold {fold}: skipping (only {len(X_tr)} training signals)")
                fold += 1
                continue

            meta_model = self._train_meta_model(X_tr, y_tr)
            fold += 1

            tr_auc = roc_auc_score(y_tr, meta_model.predict_proba(X_tr)[:, 1])

            # Inference on signal bars in this step's OOS window
            step_signal_mask = (
                (X_all.index >= step_start) & (X_all.index < step_end)
            )
            X_step = X_all[step_signal_mask]
            y_step = y_all[step_signal_mask]
            sides_step = sides_all[step_signal_mask]

            if X_step.empty:
                if verbose:
                    print(f"  [MetaAgent] fold {fold}: no signals in OOS step, skipping")
                continue

            meta_probs = meta_model.predict_proba(X_step)[:, 1]
            val_auc = roc_auc_score(y_step, meta_probs) if y_step.nunique() > 1 else float("nan")

            approved = 0
            for ts, prob, side in zip(X_step.index, meta_probs, sides_step.values):
                if prob >= self.threshold and ts in sizing.index:
                    # Continuous sizing: prob above threshold, scaled to [0, 1]
                    size = float(side) * (prob - 0.5) * 2.0  # maps [0.5, 1.0] → [0, 1]
                    sizing.loc[ts] = size
                    trades.append({
                        "ts": ts, "meta_prob": prob, "side": side,
                        "raw_size": size, "step": fold,
                    })
                    approved += 1

            if verbose:
                print(
                    f"  [MetaAgent] fold {fold:>2}  "
                    f"[{step_start.date()} → {step_end.date()}]  "
                    f"train_signals={len(X_tr):>4}  train_AUC={tr_auc:.4f}  "
                    f"val_AUC={val_auc:.4f}  signals={len(X_step):>3}  approved={approved}"
                )

        self._model = meta_model
        return sizing, pd.DataFrame(trades)

    def get_feature_importance(self) -> pd.DataFrame | None:
        """Feature importance from the last trained meta-model fold."""
        if self._model is None:
            return None
        imp = self._model.feature_importances_
        return pd.DataFrame(
            {"feature": self._feature_cols, "importance": imp}
        ).sort_values("importance", ascending=False).reset_index(drop=True)


# ── Backtest: sizing-based equity simulation ──────────────────────────────────

def run_sized_backtest(
    sizing: pd.Series,
    ohlcv_df: pd.DataFrame,
    taker_fee: float = 0.0005,
    funding_h: float = 0.0000077,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Simulate a portfolio driven by the meta-agent's continuous sizing signal.

    The sizing signal is treated as a fractional position in [−1, 1]:
      - Positive → fractional long
      - Negative → fractional short
      - Zero → flat

    Position changes incur a taker fee proportional to the change in exposure.
    Short positions receive the hourly funding rate.

    Returns (equity_array, trade_log_df) aligned to the sizing index.
    """
    n = len(sizing)
    index = sizing.index
    close_arr = ohlcv_df["close"].reindex(index).values
    log_rets = np.log(np.maximum(close_arr[1:] / close_arr[:-1], 1e-12))
    log_rets = np.concatenate([[0.0], log_rets])

    eq = np.ones(n)
    cur = 1.0
    prev_pos = 0.0
    trades: list[dict] = []

    for i in range(n):
        pos = float(sizing.iloc[i])
        pos_change = abs(pos - prev_pos)
        fee = taker_fee * pos_change
        funding = funding_h * (-pos) if pos < 0 else 0.0   # receive funding when short
        step_ret = pos * log_rets[i] + funding - fee
        cur *= np.exp(step_ret)
        eq[i] = cur

        if pos_change > 0.05:   # record position changes as "trade events"
            trades.append({
                "ts": index[i],
                "old_pos": prev_pos,
                "new_pos": pos,
                "fee_pct": fee * 100,
            })
        prev_pos = pos

    return eq, pd.DataFrame(trades)
