"""Reusable matplotlib/seaborn chart functions for HMATS research notebooks.

All functions return the matplotlib Figure so callers can do:
    fig = plot_equity_drawdown(...)
    plt.show()          # or
    save_fig(fig, path)

Palette
-------
BLUE    #2196F3   — primary strategy line
ACCENT  #FF6F00   — secondary / highlight
GREEN   #2E7D32   — wins / bullish
RED     #C62828   — losses / bearish
GREY    #9E9E9E   — baselines / neutral
PURPLE  #7B1FA2   — third series
SP500   #43A047   — S&P500 benchmark line
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ── Shared palette ─────────────────────────────────────────────────────────────
BLUE   = "#2196F3"
ACCENT = "#FF6F00"
GREEN  = "#2E7D32"
RED    = "#C62828"
GREY   = "#9E9E9E"
PURPLE = "#7B1FA2"
SP500  = "#43A047"

# ── Global style ──────────────────────────────────────────────────────────────
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("seaborn-whitegrid")

mpl.rcParams.update({
    "figure.dpi":        120,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "lines.linewidth":   1.4,
})


# ══════════════════════════════════════════════════════════════════════════════
# Utility
# ══════════════════════════════════════════════════════════════════════════════

def save_fig(fig: plt.Figure, path: str | Path, dpi: int = 150) -> None:
    """Save *fig* to *path*, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")


def _maxdd(equity: np.ndarray) -> float:
    """Maximum drawdown from equity array (starting at 1.0)."""
    peak = np.maximum.accumulate(equity)
    dd   = (equity - peak) / peak
    return float(dd.min())


def _drawdown_series(equity: np.ndarray) -> np.ndarray:
    peak = np.maximum.accumulate(equity)
    return (equity - peak) / peak


# ══════════════════════════════════════════════════════════════════════════════
# Feature-selection charts
# ══════════════════════════════════════════════════════════════════════════════

def plot_correlation_heatmap(
    df: pd.DataFrame,
    title: str = "Feature Correlation (Spearman ρ)",
    figsize: tuple = (14, 12),
    max_features: int = 60,
) -> plt.Figure:
    """Seaborn clustermap of feature×feature Spearman ρ.

    Parameters
    ----------
    df : DataFrame whose columns are features (rows = observations).
    max_features : subsample if the DataFrame has more columns, keeping
                   those with the highest variance.
    """
    if df.shape[1] > max_features:
        top = df.var().nlargest(max_features).index
        df  = df[top]

    corr = df.fillna(0).corr(method="spearman")

    g = sns.clustermap(
        corr,
        cmap="RdBu_r",
        center=0,
        vmin=-1, vmax=1,
        figsize=figsize,
        linewidths=0.3,
        xticklabels=True,
        yticklabels=True,
        cbar_kws={"shrink": 0.6, "label": "Spearman ρ"},
    )
    g.fig.suptitle(title, y=1.01, fontsize=13, fontweight="bold")
    g.ax_heatmap.tick_params(axis="both", labelsize=7)
    return g.fig


def plot_mi_ranking(
    mi_df: pd.DataFrame,
    top_n: int = 40,
    title: str = "Feature Ranking: Mutual Information vs |Spearman ρ|",
    figsize: tuple = (13, 10),
) -> plt.Figure:
    """Horizontal bar chart of MI and |Spearman ρ| side-by-side.

    Parameters
    ----------
    mi_df : DataFrame with columns 'feature', 'MI', and optionally 'spearman'.
    """
    df = mi_df.head(top_n).copy()
    has_spearman = "spearman" in df.columns

    fig, axes = plt.subplots(1, 2 if has_spearman else 1,
                             figsize=figsize, sharey=True)
    if not has_spearman:
        axes = [axes]

    feats = df["feature"].tolist()
    y     = range(len(feats))

    ax = axes[0]
    ax.barh(y, df["MI"], color=ACCENT, edgecolor="white", linewidth=0.4)
    ax.set_yticks(list(y))
    ax.set_yticklabels(feats, fontsize=8)
    ax.set_xlabel("Mutual Information")
    ax.set_title("Mutual Information (non-linear dependence)")
    ax.invert_yaxis()

    if has_spearman:
        ax2 = axes[1]
        ax2.barh(y, df["spearman"].abs(), color=BLUE, edgecolor="white", linewidth=0.4)
        ax2.set_xlabel("|Spearman ρ|")
        ax2.set_title("|Spearman ρ| (monotonic dependence)")
        ax2.invert_yaxis()

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_wf_stability(
    appearance_dict: Dict[str, float],
    threshold: float = 0.50,
    title: str = "Walk-Forward Feature Stability",
    figsize: tuple = (10, 12),
) -> plt.Figure:
    """Horizontal bar chart of feature appearance rates across rolling windows.

    Parameters
    ----------
    appearance_dict : {feature_name: appearance_rate_0_to_1}
    threshold       : features below this are shown in red.
    """
    sorted_items = sorted(appearance_dict.items(), key=lambda x: x[1], reverse=True)
    feats  = [k for k, _ in sorted_items]
    rates  = [v for _, v in sorted_items]
    colors = [GREEN if r >= threshold else RED for r in rates]

    fig, ax = plt.subplots(figsize=figsize)
    y = range(len(feats))
    ax.barh(list(y), rates, color=colors, edgecolor="white", linewidth=0.4)
    ax.axvline(threshold, color=GREY, lw=1.5, ls="--",
               label=f"Threshold ({threshold:.0%})")
    ax.set_yticks(list(y))
    ax.set_yticklabels(feats, fontsize=8)
    ax.set_xlabel("Window Appearance Rate")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend()
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    fig.tight_layout()
    return fig


def plot_permutation_importance(
    perm_df: pd.DataFrame,
    threshold: float = 0.0005,
    title: str = "Permutation Importance (AUC drop)",
    figsize: tuple = (10, 8),
    top_n: int = 40,
) -> plt.Figure:
    """Horizontal bar chart of permutation importance scores.

    Parameters
    ----------
    perm_df   : DataFrame with columns 'feature' and 'importance'.
    threshold : minimum importance to keep; drawn as a vertical line.
    """
    df = perm_df.nlargest(top_n, "importance").copy()
    colors = [GREEN if v >= threshold else RED for v in df["importance"]]

    fig, ax = plt.subplots(figsize=figsize)
    y = range(len(df))
    ax.barh(list(y), df["importance"], color=colors, edgecolor="white", linewidth=0.4)
    ax.axvline(threshold, color=GREY, lw=1.5, ls="--",
               label=f"Threshold ({threshold})")
    ax.set_yticks(list(y))
    ax.set_yticklabels(df["feature"].tolist(), fontsize=8)
    ax.set_xlabel("AUC Drop (higher = more important)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend()
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Backtest / equity charts
# ══════════════════════════════════════════════════════════════════════════════

def plot_wfo_schemes(
    oos_index: pd.DatetimeIndex,
    wfo_metrics: Dict,
    bh_pct: Optional[np.ndarray] = None,
    sp500_pct: Optional[np.ndarray] = None,
    ath_start: Optional[pd.Timestamp] = None,
    figsize: tuple = (14, 9),
) -> plt.Figure:
    """2×2 grid of WFO scheme equity curves (% returns, base 0).

    Parameters
    ----------
    wfo_metrics : {scheme_key: {'equity': np.ndarray, 'auc': float,
                                'sharpe': float, 'total_ret': float,
                                'name': str, 'color': str}}
    bh_pct      : BTC B&H return series as % (starts at 0).
    sp500_pct   : S&P 500 return series as % (starts at 0).
    ath_start   : if given, draw a vertical line at ATH entry date.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True)
    axes = axes.flatten()

    for ax, (sk, m) in zip(axes, wfo_metrics.items()):
        eq_pct = (m["equity"] - 1) * 100
        ax.plot(oos_index, eq_pct, color=m.get("color", BLUE), lw=1.4, label=sk)
        if bh_pct is not None:
            ax.plot(oos_index, bh_pct, color=GREY, lw=1.0, ls="--", alpha=0.7,
                    label="BTC B&H")
        if sp500_pct is not None:
            ax.plot(oos_index, sp500_pct, color=SP500, lw=1.0, ls=":", alpha=0.8,
                    label="S&P 500")
        ax.axhline(0, color=GREY, lw=0.6, ls=":", alpha=0.5)
        if ath_start is not None and ath_start in oos_index:
            ax.axvline(ath_start, color=ACCENT, lw=1.0, ls="--", alpha=0.6,
                       label="ATH entry")
        ax.set_title(
            f"{sk} — {m.get('name', '')}\n"
            f"AUC={m['auc']:.4f}  Sharpe={m['sharpe']:.3f}  "
            f"Return={m['total_ret']:+.1%}",
            fontsize=10, fontweight="bold",
        )
        ax.set_ylabel("Return (%)")
        ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.0f%%"))
        ax.legend(fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    fig.suptitle("WFO Schemes — OOS Performance (% Return)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_equity_drawdown(
    oos_index: pd.DatetimeIndex,
    equity: np.ndarray,
    trades_df: pd.DataFrame,
    bh_pct: Optional[np.ndarray] = None,
    sp500_pct: Optional[np.ndarray] = None,
    ath_start: Optional[pd.Timestamp] = None,
    label: str = "Strategy",
    threshold: float = 0.55,
    color: str = ACCENT,
    figsize: tuple = (14, 9),
) -> plt.Figure:
    """Equity curve (top) + drawdown (bottom) with trade markers.

    All equity / benchmark lines shown as % returns (base 0).
    """
    eq_pct = (equity - 1) * 100
    dd     = _drawdown_series(equity) * 100  # as %

    fig, axes = plt.subplots(
        2, 1, figsize=figsize,
        gridspec_kw={"height_ratios": [3, 1.2], "hspace": 0.06},
    )

    # ── Top: equity curves ────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(oos_index, eq_pct, color=color, lw=1.5,
            label=f"{label} (p>{threshold})")
    if bh_pct is not None:
        ax.plot(oos_index, bh_pct, color=GREY, lw=1.2, ls="--", label="BTC B&H")
    if sp500_pct is not None:
        ax.plot(oos_index, sp500_pct, color=SP500, lw=1.2, ls=":", label="S&P 500")
    ax.axhline(0, color=GREY, lw=0.7, ls=":", alpha=0.6)
    if ath_start is not None:
        ax.axvline(ath_start, color=ACCENT, lw=1.0, ls="--", alpha=0.5,
                   label="ATH entry")

    # Trade markers
    if len(trades_df) > 0 and "pnl_pct" in trades_df.columns:
        wins  = trades_df[trades_df["pnl_pct"] > 0]
        loses = trades_df[trades_df["pnl_pct"] <= 0]
        et_col = "entry_time" if "entry_time" in trades_df.columns else trades_df.index.name
        if et_col and et_col in trades_df.columns:
            ax.scatter(wins[et_col],  [eq_pct[oos_index.get_indexer([t], method="nearest")[0]]
                                        for t in wins[et_col]],
                       color=GREEN, s=18, zorder=4, alpha=0.7, marker="^")
            ax.scatter(loses[et_col], [eq_pct[oos_index.get_indexer([t], method="nearest")[0]]
                                        for t in loses[et_col]],
                       color=RED,   s=18, zorder=4, alpha=0.7, marker="v")

    total_ret = eq_pct[-1]
    mdd       = dd.min()
    n_trades  = len(trades_df)
    _pnl_col  = "pnl_pct" if "pnl_pct" in trades_df.columns else "net"
    wr        = (trades_df[_pnl_col] > 0).mean() if n_trades > 0 else float("nan")
    ax.set_title(
        f"{label} | Return={total_ret:+.1f}%  MaxDD={mdd:.1f}%  "
        f"Trades={n_trades}  WR={wr:.1%}",
        fontsize=11, fontweight="bold",
    )
    ax.set_ylabel("Return (%)")
    ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.0f%%"))
    ax.legend(fontsize=9)

    # ── Bottom: drawdown ──────────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.fill_between(oos_index, dd, 0, color=RED, alpha=0.35, label="Drawdown")
    ax2.plot(oos_index, dd, color=RED, lw=0.8, alpha=0.7)
    ax2.axhline(0, color=GREY, lw=0.5)
    ax2.set_ylabel("Drawdown (%)")
    ax2.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.0f%%"))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")

    fig.tight_layout()
    return fig


def plot_fee_comparison(
    oos_index: pd.DatetimeIndex,
    eq_0fee: np.ndarray,
    eq_fee: np.ndarray,
    threshold: float = 0.55,
    eq_0fee_alt: Optional[np.ndarray] = None,
    eq_fee_alt: Optional[np.ndarray] = None,
    threshold_alt: float = 0.60,
    bh_pct: Optional[np.ndarray] = None,
    sp500_pct: Optional[np.ndarray] = None,
    figsize: tuple = (14, 9),
) -> plt.Figure:
    """2-column comparison: 0-fee vs fee-adjusted equity + drawdown."""
    height_ratios = [3, 1.2]
    fig, axes = plt.subplots(
        2, 2, figsize=figsize,
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.08},
    )

    pairs = [
        (threshold, eq_0fee, eq_fee),
    ]
    if eq_0fee_alt is not None:
        pairs.append((threshold_alt, eq_0fee_alt, eq_fee_alt))

    for col_idx, (thr, e0, ef) in enumerate(pairs):
        ax_eq = axes[0][col_idx]
        ax_dd = axes[1][col_idx]

        e0_pct = (e0 - 1) * 100
        ef_pct = (ef - 1) * 100

        ax_eq.plot(oos_index, e0_pct, color=ACCENT, lw=1.4, label=f"0-fee  (p>{thr})")
        ax_eq.plot(oos_index, ef_pct, color=RED,    lw=1.4, label=f"Spot fees (p>{thr})")
        if bh_pct is not None:
            ax_eq.plot(oos_index, bh_pct, color=GREY, lw=1.0, ls="--", alpha=0.7,
                       label="BTC B&H")
        if sp500_pct is not None:
            ax_eq.plot(oos_index, sp500_pct, color=SP500, lw=1.0, ls=":", alpha=0.8,
                       label="S&P 500")
        ax_eq.axhline(0, color=GREY, lw=0.5, ls=":")
        ax_eq.set_ylabel("Return (%)")
        ax_eq.set_title(
            f"Threshold p>{thr}\n"
            f"0-fee={e0_pct[-1]:+.1f}%  Fees={ef_pct[-1]:+.1f}%  "
            f"Fee drag={(ef_pct[-1]-e0_pct[-1]):+.1f}%",
            fontsize=10, fontweight="bold",
        )
        ax_eq.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.0f%%"))
        ax_eq.legend(fontsize=8)

        dd0 = _drawdown_series(e0) * 100
        ddf = _drawdown_series(ef) * 100
        ax_dd.fill_between(oos_index, ddf, 0, color=RED,   alpha=0.30)
        ax_dd.fill_between(oos_index, dd0, 0, color=ACCENT, alpha=0.20)
        ax_dd.axhline(0, color=GREY, lw=0.5)
        ax_dd.set_ylabel("Drawdown (%)")
        ax_dd.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.0f%%"))
        ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax_dd.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax_dd.xaxis.get_majorticklabels(), rotation=30, ha="right")

    fig.suptitle("Fee Impact on Strategy Performance", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_prob_distribution(
    probs: np.ndarray,
    labels: np.ndarray,
    title: str = "Predicted Probability Distribution",
    figsize: tuple = (13, 5),
    n_oos_bars: Optional[int] = None,
) -> plt.Figure:
    """Side-by-side histograms: class 0 vs class 1 predicted probabilities."""
    valid = ~np.isnan(probs)
    half  = (n_oos_bars or len(probs)) // 2

    splits = [
        ("Full OOS Period", valid),
        ("Second Half OOS",
         np.concatenate([np.zeros(half, bool),
                         np.ones(len(probs) - half, bool)]) & valid),
    ]

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, (split_title, mask) in zip(axes, splits):
        y_sub = labels[mask]
        p_sub = probs[mask]
        if len(p_sub) < 10 or len(np.unique(y_sub)) < 2:
            ax.set_title(f"{split_title}\n(insufficient data)")
            continue

        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_sub, p_sub)

        ax.hist(p_sub[y_sub == 0], bins=40, alpha=0.55, color=RED,
                density=True, label="Class 0 (Down/Flat)")
        ax.hist(p_sub[y_sub == 1], bins=40, alpha=0.55, color=GREEN,
                density=True, label="Class 1 (Up)")
        ax.set_title(f"{split_title}\nAUC={auc:.4f}  N={mask.sum():,}", fontsize=10)
        ax.set_xlabel("P(Up)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_feature_importance_bar(
    importance_df: pd.DataFrame,
    top_n: int = 30,
    title: str = "Feature Importance (LGBM gain)",
    color: str = ACCENT,
    figsize: tuple = (10, 10),
) -> plt.Figure:
    """Horizontal bar chart of LightGBM feature importances."""
    df = importance_df.nlargest(top_n, "importance").copy()

    fig, ax = plt.subplots(figsize=figsize)
    y = range(len(df))
    ax.barh(list(y), df["importance"], color=color, edgecolor="white", linewidth=0.4)
    ax.set_yticks(list(y))
    ax.set_yticklabels(df["feature"].tolist(), fontsize=8)
    ax.set_xlabel("Importance (gain)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


def plot_trade_scatter(
    trades_df: pd.DataFrame,
    title: str = "Trade P&L Scatter",
    figsize: tuple = (13, 5),
) -> plt.Figure:
    """Scatter plot of entry_time vs pnl_pct, coloured by outcome."""
    if len(trades_df) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No trades", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return fig

    et_col = "entry_time" if "entry_time" in trades_df.columns else trades_df.index.name
    wins   = trades_df[trades_df["pnl_pct"] > 0]
    loses  = trades_df[trades_df["pnl_pct"] <= 0]

    fig, ax = plt.subplots(figsize=figsize)
    if et_col and et_col in trades_df.columns:
        ax.scatter(wins[et_col],  wins["pnl_pct"]  * 100, color=GREEN, alpha=0.70,
                   s=30, label="Win",  zorder=3)
        ax.scatter(loses[et_col], loses["pnl_pct"] * 100, color=RED,   alpha=0.70,
                   s=30, label="Loss", zorder=3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    ax.axhline(0, color=GREY, lw=0.8, ls="--")
    wr      = (trades_df["pnl_pct"] > 0).mean()
    avg_win = wins["pnl_pct"].mean() * 100 if len(wins) > 0 else 0
    avg_los = loses["pnl_pct"].mean() * 100 if len(loses) > 0 else 0
    ax.set_title(
        f"{title}\n"
        f"Win rate={wr:.1%}  Avg win={avg_win:+.2f}%  Avg loss={avg_los:+.2f}%  "
        f"N={len(trades_df)}",
        fontsize=10, fontweight="bold",
    )
    ax.set_ylabel("Trade P&L (%)")
    ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.1f%%"))
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_prob_timeseries(
    oos_index: pd.DatetimeIndex,
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.55,
    title: str = "OOS Predicted Probability",
    figsize: tuple = (14, 4),
) -> plt.Figure:
    """Probability time series with green background shading where label==1."""
    valid = ~np.isnan(probs)
    fig, ax = plt.subplots(figsize=figsize)

    # Shade actual up-bars
    chunk = 24
    for i in range(0, len(oos_index) - chunk, chunk):
        if labels[i] == 1:
            ax.axvspan(
                oos_index[i],
                oos_index[min(i + chunk, len(oos_index) - 1)],
                alpha=0.06, color=GREEN, linewidth=0,
            )

    ax.plot(oos_index[valid], probs[valid], color=ACCENT, lw=0.6,
            alpha=0.85, label="P(Up)")
    ax.axhline(threshold, color=RED,  lw=1.0, ls="--",
               label=f"Threshold ({threshold})")
    ax.axhline(0.5,       color=GREY, lw=0.7, ls=":",  alpha=0.7,
               label="P=0.5 (no-edge)")

    hit_rate = (probs[valid] >= threshold).mean()
    ax.set_title(
        f"{title}  |  Signal rate={hit_rate:.1%}  "
        f"(green bg = actual Up bars)",
        fontsize=10, fontweight="bold",
    )
    ax.set_ylabel("P(Up)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    return fig


def plot_feature_group_pie(
    groups_dict: Dict[str, List[str]],
    title: str = "Selected Feature Group Breakdown",
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """Pie chart + grouped horizontal bar chart of feature counts per group.

    Parameters
    ----------
    groups_dict : {'V1 Technical': [feat1, ...], 'V4 New': [...], ...}
    """
    palette = [BLUE, ACCENT, GREEN, PURPLE, RED]
    labels  = list(groups_dict.keys())
    sizes   = [len(v) for v in groups_dict.values()]
    colors  = palette[:len(labels)]
    total   = sum(sizes)

    if total == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No features selected", ha="center", va="center",
                transform=ax.transAxes)
        return fig

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Pie
    axes[0].pie(
        sizes,
        labels=[f"{l}\n({n})" for l, n in zip(labels, sizes)],
        colors=colors,
        autopct="%1.0f%%",
        startangle=90,
        wedgeprops=dict(edgecolor="white", lw=1.5),
    )
    axes[0].set_title(f"Total: {total} features", fontsize=10)

    # Bar
    y = range(len(labels))
    axes[1].barh(list(y), sizes, color=colors, edgecolor="white", linewidth=0.5)
    axes[1].set_yticks(list(y))
    axes[1].set_yticklabels(labels)
    axes[1].set_xlabel("Number of selected features")
    axes[1].invert_yaxis()
    for i, (n, c) in enumerate(zip(sizes, colors)):
        axes[1].text(n + 0.1, i, str(n), va="center", fontsize=9, color=c,
                     fontweight="bold")

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig
