"""Hourly feature engineering for NEAT/PPO agents (Binance-style data)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


FEATURE_COLS: list[str] = [
    "log_ret_1",
    "vol_24",
    "vol_72",
    "sma_ratio_24_72",
    "macd",
    "macd_signal",
    "macd_hist",
    "mom_24",
    "mom_72",
    "rsi_14",
    "volu_z_72",
    "z_close_72",
]


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 12 technical features from an OHLCV DataFrame.

    Expects columns ``close`` and ``volume`` (lowercase, Binance-style).
    Returns a DataFrame with the feature columns plus ``close``, rows with
    NaN features dropped.  Values are clipped to [-10, 10].
    """
    out = df.copy()

    out["log_close"] = np.log(out["close"])
    out["log_ret_1"] = out["log_close"].diff()

    out["vol_24"] = out["log_ret_1"].rolling(24).std()
    out["vol_72"] = out["log_ret_1"].rolling(72).std()

    out["sma_24"] = out["close"].rolling(24).mean()
    out["sma_72"] = out["close"].rolling(72).mean()
    out["sma_ratio_24_72"] = out["sma_24"] / (out["sma_72"] + 1e-12) - 1.0

    out["ema_12"] = out["close"].ewm(span=12, adjust=False).mean()
    out["ema_26"] = out["close"].ewm(span=26, adjust=False).mean()
    out["macd"] = out["ema_12"] - out["ema_26"]
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    out["mom_24"] = out["close"] / (out["close"].shift(24) + 1e-12) - 1.0
    out["mom_72"] = out["close"] / (out["close"].shift(72) + 1e-12) - 1.0

    out["rsi_14"] = _rsi(out["close"], 14)

    out["volu_z_72"] = (out["volume"] - out["volume"].rolling(72).mean()) / (
        out["volume"].rolling(72).std() + 1e-12
    )
    out["z_close_72"] = (out["close"] - out["close"].rolling(72).mean()) / (
        out["close"].rolling(72).std() + 1e-12
    )

    feats = out[FEATURE_COLS].dropna().copy()
    feats = feats.clip(lower=-10.0, upper=10.0)
    feats["close"] = out.loc[feats.index, "close"]

    return feats


def standardise(
    train: pd.DataFrame,
    *others: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray, list[np.ndarray]]:
    """Standardise feature arrays using train-set statistics only.

    Returns
    -------
    mu, sd, [X_train, X_other1, ...], P_train, [P_other1, ...]
        Where ``X_*`` are standardised feature matrices and ``P_*`` are price
        arrays.
    """
    cols = feature_cols or FEATURE_COLS

    x_train = train[cols].values.astype(np.float32)
    mu = x_train.mean(axis=0, keepdims=True)
    sd = x_train.std(axis=0, keepdims=True) + 1e-8

    x_train_std = (x_train - mu) / sd
    p_train = train["close"].values.astype(np.float32)

    x_others = []
    p_others = []
    for df in others:
        x = (df[cols].values.astype(np.float32) - mu) / sd
        x_others.append(x)
        p_others.append(df["close"].values.astype(np.float32))

    return mu, sd, [x_train_std, *x_others], p_train, p_others
