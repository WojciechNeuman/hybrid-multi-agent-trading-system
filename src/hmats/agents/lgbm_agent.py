"""LightGBM WFO Agent — extracted from 02_lgbm_omni_0fee_v12.

Generates OOS P(up) binary-class probabilities via M1Y sliding walk-forward.
Output is a pd.Series named ``lgbm_p_up`` aligned to the OOS DatetimeIndex.
"""

from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# ── Feature set locked from v8 4-stage selection ──────────────────────────────
SELECTED_FEATURES: list[str] = [
    "close_vs_true_vwap", "stoch_k_14", "ret_2h", "rsi_divergence",
    "close_vs_sma_7", "bear_streak", "close_vs_s1", "macd_hist_5_13",
    "hurst_24h", "ad_z_48h", "ret_3h",
]

_DEFAULT_LGBM_PARAMS: dict = dict(
    num_leaves=31, max_depth=6, learning_rate=0.05,
    colsample_bytree=0.5, min_child_samples=50,
    subsample=0.7, reg_alpha=0.1, reg_lambda=1.0,
    n_estimators=500, objective="binary",
    metric="auc", verbose=-1, random_state=42,
)


class LGBMAgent:
    """Walk-forward LightGBM agent producing OOS P(up) probability signals.

    Mirrors the M1Y WFO scheme from ``02_lgbm_omni_0fee_v12.ipynb``.

    Parameters
    ----------
    features:
        Input feature columns. Defaults to the 11 features locked from v8.
    label_col:
        Binary label column (1 = Up, 0 = Down/Flat).
    train_window_h:
        Training window in bars (default 8760 = 1 year of hourly data).
    step_size:
        Step size between WFO folds in bars (default 720 = 1 month).
    embargo:
        Gap between train end and OOS start in bars (default 12 = 12h).
    val_frac:
        Fraction of training window reserved for early-stopping validation.
    lgbm_params:
        Override default LightGBM hyperparameters.
    """

    AGENT_ID = "lgbm_v12"
    SIGNAL_COL = "lgbm_p_up"

    def __init__(
        self,
        features: list[str] | None = None,
        label_col: str = "label",
        train_window_h: int = 8760,
        step_size: int = 720,
        embargo: int = 12,
        val_frac: float = 0.20,
        lgbm_params: dict | None = None,
    ) -> None:
        self.features = features or SELECTED_FEATURES
        self.label_col = label_col
        self.train_window_h = train_window_h
        self.step_size = step_size
        self.embargo = embargo
        self.val_frac = val_frac
        self.lgbm_params = {**_DEFAULT_LGBM_PARAMS, **(lgbm_params or {})}

    # ── Internal WFO loop ─────────────────────────────────────────────────────

    def _run_wfo(self, df: pd.DataFrame, verbose: bool) -> np.ndarray:
        """Walk-forward loop over df; returns probability array aligned to df."""
        n = len(df)
        probs = np.full(n, np.nan)
        steps = 0
        i = 0

        while i < n:
            tr_end = i
            tr_start = max(0, tr_end - self.train_window_h)
            if tr_start >= tr_end - 100:
                i += self.step_size
                continue

            tr_slice = df.iloc[tr_start:tr_end]
            val_n = max(50, int(len(tr_slice) * self.val_frac))
            X_tr = tr_slice.iloc[:-val_n][self.features].fillna(0).values
            y_tr = tr_slice.iloc[:-val_n][self.label_col].values
            X_va = tr_slice.iloc[-val_n:][self.features].fillna(0).values
            y_va = tr_slice.iloc[-val_n:][self.label_col].values

            if len(np.unique(y_tr)) < 2:
                i += self.step_size
                continue

            mdl = lgb.LGBMClassifier(**self.lgbm_params)
            mdl.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
            )

            oos_end = min(i + self.step_size, n)
            oos_emb = min(i + self.embargo, oos_end)
            X_oos = df.iloc[oos_emb:oos_end][self.features].fillna(0).values
            if len(X_oos):
                probs[oos_emb:oos_end] = mdl.predict_proba(X_oos)[:, 1]

            steps += 1
            if verbose and steps % 6 == 1:
                pct = min(oos_end / n * 100, 100)
                print(
                    f"  [{self.AGENT_ID}] step {steps:>3}  "
                    f"train [{df.index[tr_start].date()} → {df.index[tr_end - 1].date()}]  "
                    f"OOS [{df.index[oos_emb].date()} → {df.index[oos_end - 1].date()}]  "
                    f"{pct:.0f}%"
                )
            i += self.step_size

        if verbose:
            valid = ~np.isnan(probs)
            print(
                f"[{self.AGENT_ID}] WFO done: {steps} steps  valid={valid.sum():,}  "
                f"mean P(up)={probs[valid].mean():.3f}"
            )
        return probs

    # ── Public API ────────────────────────────────────────────────────────────

    def generate_signals(
        self,
        df: pd.DataFrame,
        oos_start: pd.Timestamp,
        verbose: bool = True,
        full_history: bool = False,
    ) -> pd.Series:
        """Run M1Y WFO over ``df`` and return the OOS P(up) signal series.

        Parameters
        ----------
        df:
            Full feature DataFrame including historical training bars AND the
            OOS period. Must contain all columns in ``self.features`` and
            ``self.label_col``.
        oos_start:
            First timestamp of the hold-out OOS window.
        full_history:
            If True, return valid WFO signals for the *entire* period (used by
            the meta-labeling orchestrator to build its training set). If False
            (default), restrict to bars >= oos_start.
        """
        probs = self._run_wfo(df, verbose)
        full_series = pd.Series(probs, index=df.index, name=self.SIGNAL_COL)
        if full_history:
            return full_series
        return full_series[df.index >= oos_start].copy()

    def get_oos_auc(
        self,
        df: pd.DataFrame,
        oos_start: pd.Timestamp,
        verbose: bool = False,
    ) -> float:
        """Compute OOS AUC without storing WFO probs (convenience for quick checks)."""
        probs = self._run_wfo(df, verbose=verbose)
        mask = df.index >= oos_start
        y = df.loc[mask, self.label_col].values
        p = probs[mask]
        valid = ~np.isnan(p)
        return float(roc_auc_score(y[valid], p[valid]))
