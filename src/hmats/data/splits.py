"""Calendar-based train/val/test splitting and rolling test windows."""

from __future__ import annotations

import pandas as pd


def calendar_split(
    df: pd.DataFrame,
    train_end: str,
    val_end: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a datetime-indexed DataFrame into train / val / test by calendar dates.

    Parameters
    ----------
    df:
        DataFrame with a :class:`DatetimeIndex`.
    train_end:
        Last date (inclusive) for the training set, e.g. ``"2023-12-31"``.
    val_end:
        Last date (inclusive) for the validation set, e.g. ``"2024-06-30"``.

    Returns
    -------
    tuple of (train, val, test) DataFrames.
    """
    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)

    train = df[df.index <= train_end_ts].copy()
    val = df[(df.index > train_end_ts) & (df.index <= val_end_ts)].copy()
    test = df[df.index > val_end_ts].copy()

    return train, val, test


def rolling_test_windows(
    df: pd.DataFrame,
    start: str,
    window_days: int = 140,
) -> list[pd.DataFrame]:
    """Generate non-overlapping rolling windows from *start* onwards.

    Parameters
    ----------
    df:
        DataFrame with a :class:`DatetimeIndex`.
    start:
        First date of the test region, e.g. ``"2024-07-01"``.
    window_days:
        Length of each window in calendar days.

    Returns
    -------
    List of DataFrame slices, one per window.  The last window may be shorter
    than *window_days* if there isn't enough remaining data.
    """
    start_ts = pd.Timestamp(start)
    subset = df[df.index >= start_ts].copy()

    if subset.empty:
        return []

    windows: list[pd.DataFrame] = []
    window_delta = pd.Timedelta(days=window_days)
    cursor = start_ts

    while True:
        window_end = cursor + window_delta
        chunk = subset[(subset.index >= cursor) & (subset.index < window_end)]
        if chunk.empty:
            break
        windows.append(chunk.copy())
        cursor = window_end

    return windows
