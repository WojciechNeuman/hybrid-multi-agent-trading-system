"""Classical ARIMA / random-walk baseline for the OOS window.

The thesis advisor asked for an ARIMA (and naive random-walk) baseline next to BTC Buy & Hold and
the S&P 500 — the sharp version of the question being *"if an unpromising classical model returns
nearly the same as the fancy AI agents, the fancy ones aren't justified."*

This module provides two things over the canonical OOS window (2024-06-01 -> 2026-05-31):

1. **A forecasting-accuracy comparison** (`forecast_accuracy`): walk-forward 1-step-ahead forecast
   of the hourly log-return, ARIMA(p,d,q) vs the naive random-walk null (forecast = 0 return).
   Reports RMSE and directional accuracy. The expected, thesis-supporting outcome is that ARIMA
   collapses toward the random walk (~50% directional) — i.e. no exploitable linear structure.

2. **A trading-baseline equity curve** (`arima_equity`): the standard ARIMA trading rule — go long
   when the 1-step forecast is positive, short when negative — net of the same taker fee the agents
   pay, so it sits in the same results table as BTC B&H and the S&P 500.

For tractability over ~17.5k hourly bars the ARIMA parameters are re-estimated periodically
(``refit_every`` bars, weekly by default); between refits the fixed-parameter state-space model is
*filtered forward* one observation at a time (`SARIMAXResults.append`, no re-estimation), which is
the correct leak-free 1-step walk-forward. Results are cached to ``data/external/arima_oos.parquet``.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Fees / constants mirror hmats.mas.mas07 so the baseline is comparable to the agents.
TAKER_FEE = 0.0005
ANN = np.sqrt(24 * 365)
OOS_START = pd.Timestamp("2024-06-01")
OOS_END = pd.Timestamp("2026-05-31 23:00:00")

DEFAULT_ORDER = (1, 1, 1)
TRAIN_WINDOW = 2000   # trailing bars used to (re)estimate ARIMA params
REFIT_EVERY = 168     # weekly re-estimation
CONVICTION_K = 0.5    # only trade when |forecast| > K * trailing forecast-vol (cuts fee churn)
MIN_HOLD = 24         # minimum holding period in bars (~1 day, matches the agents' horizon)


def _repo_root() -> Path:
    p = Path.cwd()
    while p != p.parent:
        if (p / "pyproject.toml").exists():
            return p
        p = p.parent
    raise RuntimeError("pyproject.toml not found")


def _sharpe(eq: np.ndarray) -> float:
    r = np.diff(np.log(np.maximum(eq, 1e-12)))
    return float(r.mean() / (r.std(ddof=1) + 1e-12) * ANN)


def _maxdd(eq: np.ndarray) -> float:
    pk = np.maximum.accumulate(eq)
    return float(((eq - pk) / (pk + 1e-12)).min())


def _walk_forward_forecasts(logp: pd.Series, order, train_window, refit_every):
    """Walk-forward 1-step-ahead forecast of log-price over the OOS window.

    Returns a DataFrame indexed by the *forecast target* bar with columns
    ``y_true`` (realised log-price), ``yhat_arima`` (ARIMA 1-step forecast),
    ``yhat_rw`` (random-walk: previous log-price).
    """
    from statsmodels.tsa.arima.model import ARIMA

    idx = logp.index
    oos_positions = np.where((idx >= OOS_START) & (idx <= OOS_END))[0]
    oos_positions = oos_positions[oos_positions > 0]

    out_idx, y_true, yhat_a, yhat_rw = [], [], [], []
    res = None; last_fit_pos = -(10**9); fitted_through = -1
    vals = logp.values

    for t in oos_positions:  # forecast bar t using data up to t-1
        need_refit = (res is None) or (t - 1 - last_fit_pos >= refit_every)
        if need_refit:
            start = max(0, t - 1 - train_window + 1)
            hist = vals[start:t]  # up to and including t-1
            res = ARIMA(hist, order=order).fit()
            last_fit_pos = t - 1
            fitted_through = t - 1
        else:
            # extend the fixed-parameter filter with the observations seen since the last fit
            new = vals[fitted_through + 1:t]
            if len(new):
                res = res.append(new, refit=False)
                fitted_through = t - 1
        fc = float(res.forecast(steps=1)[0])
        out_idx.append(idx[t]); y_true.append(vals[t])
        yhat_a.append(fc); yhat_rw.append(vals[t - 1])

    return pd.DataFrame({"y_true": y_true, "yhat_arima": yhat_a, "yhat_rw": yhat_rw},
                        index=pd.DatetimeIndex(out_idx))


def compute(close: pd.Series, order=DEFAULT_ORDER, train_window=TRAIN_WINDOW,
            refit_every=REFIT_EVERY, cache=True) -> pd.DataFrame:
    """Walk-forward ARIMA forecasts + the ARIMA / random-walk trading equity over OOS.

    Returns a DataFrame (indexed by OOS bar) with: ``close``, ``ret`` (realised next-context return),
    ``yhat_ret_arima``, ``pos_arima`` (+1/-1), ``eq_arima`` (net of fees), plus the random-walk
    forecast columns for the accuracy table.
    """
    close = close.sort_index()
    close.index = close.index.tz_localize(None) if close.index.tz else close.index
    logp = np.log(close.astype(float))

    fc = _walk_forward_forecasts(logp, order, train_window, refit_every)
    # 1-step forecast of the *return* = forecast log-price - last observed log-price
    prev_logp = logp.reindex(fc.index)  # log-price at target bar t (y_true); we need t-1
    # y_true is logp at t; forecasts were made from t-1, so forecast return = yhat - logp[t-1]
    logp_tm1 = logp.shift(1).reindex(fc.index)
    fc["yhat_ret_arima"] = fc["yhat_arima"] - logp_tm1
    fc["true_ret"] = fc["y_true"] - logp_tm1
    fc["close"] = close.reindex(fc.index)

    # Trading rule: a *conviction-filtered, min-hold* ARIMA strategy so the equity reflects the
    # forecast's edge rather than a fee-churn artifact (a naive every-bar flip on a ~coin-flip
    # signal is destroyed by paying 5 bps each hour). Enter long/short only when the forecast
    # exceeds K * its trailing volatility; hold at least MIN_HOLD bars before re-deciding.
    yhat = fc["yhat_ret_arima"].values
    sig = pd.Series(yhat).rolling(MIN_HOLD, min_periods=MIN_HOLD).std().bfill().values
    thr = CONVICTION_K * sig
    desired = np.where(yhat > thr, 1.0, np.where(yhat < -thr, -1.0, 0.0))

    pos = np.zeros(len(yhat)); cur = 0.0; held = 0
    for i in range(len(yhat)):
        if cur != 0.0 and held < MIN_HOLD:
            held += 1
        else:
            if desired[i] != cur:
                cur = desired[i]; held = 0
            else:
                held += 1
        pos[i] = cur

    realised = fc["true_ret"].values
    gross = pos * realised
    flips = np.abs(np.diff(np.concatenate([[0.0], pos])))  # turnover when position changes
    fee = flips * TAKER_FEE
    eq = np.exp(np.cumsum(gross - fee))
    eq_0fee = np.exp(np.cumsum(gross))

    fc["pos_arima"] = pos
    fc["eq_arima"] = eq
    fc["eq_arima_0fee"] = eq_0fee

    if cache:
        out = _repo_root() / "data" / "external" / "arima_oos.parquet"
        fc.to_parquet(out)
    return fc


def forecast_accuracy(fc: pd.DataFrame) -> dict:
    """RMSE and directional accuracy of ARIMA vs the random-walk null on 1-step return forecasts."""
    true_ret = fc["true_ret"].values
    rmse_arima = float(np.sqrt(np.nanmean((fc["yhat_ret_arima"].values - true_ret) ** 2)))
    rmse_rw = float(np.sqrt(np.nanmean((0.0 - true_ret) ** 2)))  # RW return forecast = 0
    da_arima = float(np.nanmean(np.sign(fc["yhat_ret_arima"].values) == np.sign(true_ret)))
    return {"rmse_arima": rmse_arima, "rmse_random_walk": rmse_rw,
            "rmse_improvement_pct": round(100 * (rmse_rw - rmse_arima) / rmse_rw, 4),
            "directional_accuracy_arima": round(da_arima, 4)}


def arima_equity(idx: pd.DatetimeIndex, close: pd.Series | None = None,
                 order=DEFAULT_ORDER, use_cache=True) -> pd.Series:
    """ARIMA trading-baseline equity aligned to ``idx`` (full-history index, OOS-active).

    Mirrors ``load_sp500_benchmark``: returns an equity Series starting at 1.0, reindexed to ``idx``
    and forward-filled (flat = 1.0 before the OOS window begins)."""
    cache_path = _repo_root() / "data" / "external" / "arima_oos.parquet"
    if use_cache and cache_path.exists():
        fc = pd.read_parquet(cache_path)
    else:
        if close is None:
            raise ValueError("close series required when cache is unavailable")
        fc = compute(close, order=order)
    eq = fc["eq_arima"].copy()
    eq = eq.reindex(idx).ffill().fillna(1.0)
    return eq.rename(f"ARIMA{order}")


if __name__ == "__main__":
    repo = _repo_root()
    df = pd.read_parquet(repo / "data" / "features" / "BTCUSDT_1h_unified.parquet")
    df.index = df.index.tz_localize(None) if df.index.tz else df.index
    fc = compute(df["close"])
    acc = forecast_accuracy(fc)
    eq = fc["eq_arima"].values
    print("=== ARIMA walk-forward (OOS) ===")
    print(f"bars={len(fc)}  return={eq[-1]-1:+.1%}  sharpe={_sharpe(eq):.3f}  maxdd={_maxdd(eq):.1%}")
    print("=== Forecast accuracy: ARIMA vs random-walk null ===")
    for k, v in acc.items():
        print(f"  {k:32s} {v}")
