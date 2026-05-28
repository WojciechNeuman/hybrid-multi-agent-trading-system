"""
reporting.py — Institutional-grade strategy tearsheet.

Design lineage: lab/notebooks/07_lgbm_grid_v8.ipynb
Adapted for fixed-horizon, directional-only (long) WFO signals.

Exported
--------
generate_tearsheet(trades_df, price_df, prob_series, params, metrics, title, ...)
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Union

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── House style (mirrors v8) ─────────────────────────────────────────────────
_RCPARAMS: dict = {
    'font.family':       'serif',
    'font.serif':        ['DejaVu Serif'],
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.labelsize':    10,
    'axes.titlesize':    11,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
    'legend.fontsize':   9,
    'legend.framealpha': 0.85,
    'figure.dpi':        120,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
}

ACCENT = '#F7931A'   # Bitcoin orange — strategy line
BLUE   = '#2962FF'   # B&H reference
GREY   = '#9E9E9E'
RED    = '#EF5350'
GREEN  = '#26A69A'
PURPLE = '#7B1FA2'


# ── Internal helpers ─────────────────────────────────────────────────────────

def _equity_from_trades(
    trades_df: pd.DataFrame,
    price_index: pd.DatetimeIndex,
    fh_bars: int = 72,
) -> pd.Series:
    """
    Build a time-indexed equity curve from a trade log.

    Uses mark-to-market linear interpolation: each trade's P&L is spread
    linearly from entry bar to exit bar via np.linspace, producing a smooth
    curve instead of a step function.  Between trades the equity is flat.
    """
    n = len(price_index)
    equity_arr = np.full(n, np.nan)
    equity_arr[0] = 1.0
    cum = 1.0
    for entry_time, row in trades_df.sort_index().iterrows():
        dur   = int(row['duration_bars']) if 'duration_bars' in row.index else fh_bars
        pnl   = float(row['pnl_pct'])
        i_entry = int(price_index.searchsorted(entry_time, side='left'))
        i_exit  = min(i_entry + dur, n - 1)
        entry_eq = cum
        exit_eq  = cum * (1.0 + pnl)
        equity_arr[i_entry: i_exit + 1] = np.linspace(
            entry_eq, exit_eq, i_exit - i_entry + 1
        )
        cum = exit_eq
        if i_exit + 1 < n:
            equity_arr[i_exit + 1] = cum
    return pd.Series(equity_arr, index=price_index).ffill().fillna(1.0)


def _drawdown_series(equity: np.ndarray) -> np.ndarray:
    peak = np.maximum.accumulate(equity)
    return (equity - peak) / (peak + 1e-12)


def _stats(trades_df: pd.DataFrame, equity_arr: np.ndarray) -> dict:
    """Compute all KPIs from trades and equity array."""
    if len(trades_df) == 0:
        return {}
    pnl = trades_df['pnl_pct'].values.astype(float)
    n   = len(pnl)

    tot_ret  = float(equity_arr[-1] - 1)
    pk       = np.maximum.accumulate(equity_arr)
    max_dd   = float(((equity_arr - pk) / (pk + 1e-12)).min())
    win_rate = float((pnl > 0).mean())

    gp = pnl[pnl > 0].sum() if (pnl > 0).any() else 0.0
    gl = abs(pnl[pnl < 0].sum()) if (pnl < 0).any() else 1e-9
    pf = float(gp / gl)

    t0 = trades_df.index[0]; t1 = trades_df.index[-1]
    n_days = max((t1 - t0).total_seconds() / 86400.0, 1.0)
    years  = n_days / 365.25
    ann    = float(equity_arr[-1] ** (1.0 / max(years, 1.0 / 365.0)) - 1)
    calmar = float(ann / (abs(max_dd) + 1e-9))
    tpd    = n / n_days
    sd     = float(pnl.std(ddof=1)) if n > 1 else 1e-9
    sharpe = float(pnl.mean() / sd * math.sqrt(tpd * 252)) if sd > 1e-12 else 0.0

    dir_col = trades_df.get('direction', pd.Series(['long'] * n, index=trades_df.index))
    long_n  = int((dir_col == 'long').sum())
    short_n = int((dir_col == 'short').sum())

    return dict(
        tot_ret=tot_ret, ann_ret=ann, sharpe=sharpe,
        max_dd=max_dd, calmar=calmar,
        win_rate=win_rate, pf=pf, ev=float(pnl.mean()),
        n=n, tpd=tpd, long_n=long_n, short_n=short_n,
    )


def _fmt_xaxis(ax: plt.Axes, ts: pd.DatetimeIndex) -> None:
    """Adaptive date formatting based on OOS span."""
    span = (ts[-1] - ts[0]).days
    if span <= 180:
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    elif span <= 730:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    else:
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=8)


def _nan(v: float) -> bool:
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return True


def _fmt_pct(v, prefix='') -> str:
    return 'N/A' if _nan(v) else f'{prefix}{float(v):.2%}'


def _fmt_f(v, dec=3) -> str:
    return 'N/A' if _nan(v) else f'{float(v):.{dec}f}'


# ── Public API ───────────────────────────────────────────────────────────────

def generate_tearsheet(
    trades_df: pd.DataFrame,
    price_df: pd.DataFrame,
    prob_series: Optional[pd.Series] = None,
    params: Optional[dict] = None,
    metrics: Optional[dict] = None,
    title: str = 'Strategy Tearsheet',
    fig_size: tuple = (14, 17),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Generate an institutional-grade multi-panel strategy tearsheet.

    Parameters
    ----------
    trades_df : pd.DataFrame indexed by entry_time (DatetimeIndex).
        Required columns:
          pnl_pct       — trade return as decimal (0.01 = 1%)
          duration_bars — bars held until exit (int)
        Optional columns:
          direction     — 'long' | 'short'  (default: all 'long')
          outcome       — 'FH' | 'TP' | 'SL'
          prob          — entry signal probability

    price_df : pd.DataFrame with DatetimeIndex and 'close' column.
        Must span the full OOS evaluation period.

    prob_series : pd.Series indexed by DatetimeIndex.
        P(TP) at every OOS bar. Displayed in the probability panel.
        Pass None to skip that panel.

    params : dict — execution metadata for the parameter banner.
        Keys used:
          wfo_scheme        str   — e.g. 'Expanding (all history)'
          fh_horizon_bars   int   — e.g. 72
          fh_threshold      float — e.g. 0.003
          prob_threshold    float — e.g. 0.75
          embargo_bars      int   — e.g. 72
          n_signals         int   — total bars where p > threshold
          n_muted           int   — regime-blocked signals (0 if no filter)

    metrics : dict — pre-computed OOS metrics for the banner.
        Keys used: auc, oos_sharpe, oos_return

    title : str — figure suptitle.

    fig_size : (width, height) in inches.

    save_path : if provided, saves PNG at 300 dpi.

    Returns
    -------
    matplotlib.figure.Figure
    """
    params  = params  or {}
    metrics = metrics or {}

    with mpl.rc_context(_RCPARAMS):
        fig = plt.figure(figsize=fig_size)

        # ── Graceful empty-trades handling ───────────────────────────────
        if trades_df is None or len(trades_df) == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, 'No trades above threshold — nothing to display.',
                    ha='center', va='center', fontsize=16, transform=ax.transAxes)
            fig.suptitle(title, fontsize=14, fontweight='bold')
            if save_path:
                fig.savefig(save_path)
            return fig

        # ── Derived arrays ───────────────────────────────────────────────
        fh_bars  = int(params.get('fh_horizon_bars', 72))
        ts       = price_df.index
        eq_ts    = _equity_from_trades(trades_df, ts, fh_bars)
        bh       = (price_df['close'] / price_df['close'].iloc[0]).values
        dd_strat = _drawdown_series(eq_ts.values)
        dd_bh    = _drawdown_series(bh)
        s        = _stats(trades_df, eq_ts.values)
        pnl      = trades_df['pnl_pct'].values.astype(float)

        # ── Layout ───────────────────────────────────────────────────────
        has_probs = prob_series is not None
        n_rows    = 5 if has_probs else 4
        h_ratios  = ([3.5, 1.0, 1.2, 1.8, 1.2] if has_probs
                     else [3.5, 1.2, 1.8, 1.2])
        gs = gridspec.GridSpec(
            n_rows, 2, figure=fig,
            height_ratios=h_ratios,
            hspace=0.10 if has_probs else 0.12,
            wspace=0.30,
            top=0.93, bottom=0.06, left=0.08, right=0.97,
        )
        row = 0

        # ── 1. Equity Curve ──────────────────────────────────────────────
        ax_eq = fig.add_subplot(gs[row, :])
        eq_pct = (eq_ts.values - 1) * 100
        bh_pct = (bh - 1) * 100

        ax_eq.plot(ts, eq_pct, color=ACCENT, lw=1.6,
                   label=f'Strategy  ({s.get("tot_ret", 0):+.2%})')
        ax_eq.fill_between(ts, eq_pct, 0,
                           where=eq_pct >= 0, alpha=0.08, color=ACCENT)
        ax_eq.fill_between(ts, eq_pct, 0,
                           where=eq_pct < 0,  alpha=0.08, color=RED)
        ax_eq.plot(ts, bh_pct, color=BLUE, lw=1.2, ls='--',
                   label=f'Buy & Hold  ({bh[-1]-1:+.2%})')
        ax_eq.axhline(0, color=GREY, lw=0.7, ls=':')
        ax_eq.set_ylabel('Cumulative Return %')
        ax_eq.set_title(title, fontweight='bold', pad=8)
        ax_eq.legend(ncol=2)
        ax_eq.grid(axis='y', alpha=0.25)
        ax_eq.grid(axis='x', alpha=0.12)
        _fmt_xaxis(ax_eq, ts)
        plt.setp(ax_eq.xaxis.get_majorticklabels(), visible=has_probs)
        row += 1

        # ── 2. Model Probabilities (optional) ────────────────────────────
        ax_pr = None
        if has_probs:
            ax_pr = fig.add_subplot(gs[row, :], sharex=ax_eq)
            prob_s = prob_series.reindex(ts)
            prob_arr = prob_s.values.astype(float)
            # Rolling mean window: ~1 day of 5m bars = 288 bars
            roll_win = min(288, max(24, len(prob_s) // 30))
            roll_mean = prob_s.rolling(roll_win, center=True, min_periods=1).mean()
            ax_pr.plot(ts, prob_arr, color=GREEN, lw=0.40, alpha=0.45, label='P(TP)')
            ax_pr.plot(ts, roll_mean.values, color=GREEN, lw=1.4, alpha=0.90,
                       label=f'P(TP) rolling mean ({roll_win}b)')
            thr = float(params.get('prob_threshold', 0.75))
            ax_pr.axhspan(thr, 1.0, alpha=0.06, color=ACCENT,
                          label=f'Signal zone  p > {thr:.2f}')
            ax_pr.axhline(thr, color=ACCENT, lw=1.2, ls='--', alpha=0.9)
            ax_pr.axhline(0.50, color=GREY, lw=0.7, ls=':', label='Prior 0.5')
            ax_pr.set_ylim(0, 1)
            ax_pr.set_ylabel('P(TP)')
            ax_pr.set_title('Model P(TP) — OOS bar-level predictions', fontsize=10)
            ax_pr.legend(ncol=4, fontsize=8)
            ax_pr.grid(axis='y', alpha=0.25)
            plt.setp(ax_pr.xaxis.get_majorticklabels(), visible=False)
            row += 1

        # ── 3. Drawdown ──────────────────────────────────────────────────
        ax_dd = fig.add_subplot(gs[row, :], sharex=ax_eq)
        ax_dd.fill_between(ts, dd_strat * 100, 0,
                           color=ACCENT, alpha=0.50, label='Strategy')
        ax_dd.fill_between(ts, dd_bh * 100, 0,
                           color=BLUE, alpha=0.25, label='Buy & Hold')
        ax_dd.plot(ts, dd_strat * 100, color=ACCENT, lw=0.7)
        max_dd_v = s.get('max_dd', 0) or 0
        ax_dd.set_ylabel('Drawdown %')
        ax_dd.set_title(f'Drawdown Underwater  (Strategy max: {max_dd_v:.1%})',
                        fontsize=10)
        ax_dd.legend(ncol=2)
        ax_dd.grid(axis='y', alpha=0.25)
        plt.setp(ax_dd.xaxis.get_majorticklabels(), visible=False)
        row += 1

        # ── 4a. Trade PnL Distribution ───────────────────────────────────
        ax_pnl = fig.add_subplot(gs[row, 0])
        pnl_pct = pnl * 100
        qs = np.percentile(pnl_pct, [5, 25, 50, 75, 95])

        if len(pnl_pct) >= 30:
            # Enough points for violin
            try:
                vp = ax_pnl.violinplot([pnl_pct], positions=[0], widths=0.55,
                                       showmedians=True, showextrema=False)
                for body in vp['bodies']:
                    body.set_facecolor(ACCENT)
                    body.set_alpha(0.50)
                vp['cmedians'].set_color(BLUE)
                vp['cmedians'].set_linewidth(2)
                ax_pnl.set_xticks([])
                ax_pnl.set_ylabel('PnL per Trade %')
            except Exception:
                ax_pnl.boxplot([pnl_pct], positions=[0], widths=0.5,
                               patch_artist=True,
                               boxprops=dict(facecolor=ACCENT, alpha=0.5),
                               medianprops=dict(color=BLUE, linewidth=2))
                ax_pnl.set_xticks([])
                ax_pnl.set_ylabel('PnL per Trade %')
        else:
            # Small-n: horizontal histogram (rotated bar chart)
            bins = max(10, len(pnl_pct) // 4)
            counts, edges = np.histogram(pnl_pct, bins=bins)
            centers = (edges[:-1] + edges[1:]) / 2
            colors = [GREEN if c >= 0 else RED for c in centers]
            ax_pnl.barh(centers, counts, height=(edges[1] - edges[0]) * 0.85,
                        color=colors, alpha=0.65, edgecolor='white')
            ax_pnl.axhline(0, color=GREY, lw=1.2)
            ax_pnl.set_xlabel('# Trades')
            ax_pnl.set_ylabel('PnL per Trade %')

        ax_pnl.axhline(0, color=GREY, lw=1.0)
        for qv, qn in zip(qs, ['p5', 'p25', 'p50', 'p75', 'p95']):
            c = GREEN if qv >= 0 else RED
            ax_pnl.axhline(qv, color=c, lw=0.7, ls=':', alpha=0.9)
            ax_pnl.text(0.55, qv, f'{qn}: {qv:+.3f}%',
                        fontsize=7.5, va='center', color=c,
                        transform=ax_pnl.get_yaxis_transform())
        ax_pnl.set_title('Trade PnL Distribution', fontweight='bold', fontsize=10)
        ax_pnl.grid(axis='y', alpha=0.25)

        # ── 4b. Monthly Trade Count ──────────────────────────────────────
        ax_mc = fig.add_subplot(gs[row, 1])
        try:
            monthly_n  = trades_df['pnl_pct'].resample('ME').count()
            monthly_wr = trades_df['pnl_pct'].resample('ME').apply(
                lambda x: float((x > 0).mean()) if len(x) > 0 else 0.5
            ).fillna(0.5)
            bar_cols = [GREEN if w >= 0.5 else RED for w in monthly_wr.values]
            n_months = max(len(monthly_n), 1)
            bar_width = max(5, min(25, 600 // n_months))
            ax_mc.bar(monthly_n.index, monthly_n.values,
                      color=bar_cols, alpha=0.78, width=bar_width, edgecolor='white')
            ax_mc.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
            plt.setp(ax_mc.xaxis.get_majorticklabels(),
                     rotation=40, ha='right', fontsize=7)
        except Exception as exc:
            ax_mc.text(0.5, 0.5, f'Resample error:\n{exc}',
                       ha='center', va='center', transform=ax_mc.transAxes,
                       fontsize=8, color=RED)
        ax_mc.set_ylabel('# Trades')
        ax_mc.set_title('Monthly Trade Count\n(green ≥ 50% win rate)',
                        fontweight='bold', fontsize=10)
        ax_mc.grid(axis='y', alpha=0.25)
        row += 1

        # ── 5. Performance Summary Table ─────────────────────────────────
        ax_tbl = fig.add_subplot(gs[row, :])
        ax_tbl.axis('off')

        fill_rate = (
            params.get('n_trades', len(trades_df)) /
            max(int(params.get('n_signals', len(trades_df))), 1)
        )
        long_n  = s.get('long_n', len(trades_df))
        short_n = s.get('short_n', 0)

        headers = [
            'Total Return', 'Ann. Return', 'Sharpe (ann.)',
            'Max DD', 'Calmar', 'Win Rate',
            'Profit Factor', 'Trades', 'Long / Short', 'Fill Rate',
        ]
        vals = [
            _fmt_pct(s.get('tot_ret')),
            _fmt_pct(s.get('ann_ret')),
            _fmt_f(s.get('sharpe')),
            _fmt_pct(s.get('max_dd')),
            _fmt_f(s.get('calmar')),
            _fmt_pct(s.get('win_rate')),
            _fmt_f(s.get('pf')),
            str(s.get('n', 0)),
            f'{long_n} / {short_n}',
            f'{fill_rate:.1%}',
        ]

        tbl = ax_tbl.table(
            cellText=[vals],
            colLabels=headers,
            loc='center',
            cellLoc='center',
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9.5)
        tbl.scale(1, 2.4)

        # Header row: Bitcoin orange background
        for j in range(len(headers)):
            tbl[(0, j)].set_facecolor(ACCENT)
            tbl[(0, j)].set_text_props(color='white', fontweight='bold')

        # Colour-code data row
        def _cell_color(j: int, val_str: str) -> None:
            """Apply green/red to directional metrics."""
            try:
                v = float(val_str.replace('%', '').replace('+', '').replace(',', ''))
                colour = GREEN if v > 0 else RED
            except ValueError:
                return
            tbl[(1, j)].set_text_props(color=colour, fontweight='bold')

        for j, (h, v) in enumerate(zip(headers, vals)):
            if h in ('Total Return', 'Ann. Return', 'Sharpe (ann.)', 'Calmar'):
                _cell_color(j, v)
            elif h == 'Max DD':
                tbl[(1, j)].set_text_props(color=RED, fontweight='bold')

        # ── Parameter banner (italic footer, v8 style) ───────────────────
        scheme  = params.get('wfo_scheme', 'N/A')
        fh_t    = params.get('fh_threshold', 0.003)
        p_thr   = params.get('prob_threshold', 0.75)
        emb     = params.get('embargo_bars', fh_bars)
        n_sig   = params.get('n_signals', '?')
        n_muted = params.get('n_muted', 0)
        auc_v   = metrics.get('auc', float('nan'))
        auc_str = f'{auc_v:.4f}' if not _nan(auc_v) else 'N/A'

        banner = (
            f'WFO: {scheme}  |  '
            f'FH: {fh_bars}b ({fh_bars * 5}min), fwd-ret > {fh_t:.1%}  |  '
            f'p > {p_thr}  |  embargo: {emb}b  |  '
            f'AUC: {auc_str}  |  '
            f'Signals above thr: {n_sig}  |  '
            f'Regime-muted: {n_muted}'
        )
        fig.text(0.5, 0.004, banner,
                 ha='center', va='bottom',
                 fontsize=7.5, color=GREY, style='italic')

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def generate_fee_comparison(
    trades_nofee: pd.DataFrame,
    trades_fee: pd.DataFrame,
    price_df: pd.DataFrame,
    scheme_label: str = 'Strategy',
    fee_label: str = 'Fee-adjusted (0.05% taker)',
    params: Optional[dict] = None,
    fig_size: tuple = (14, 10),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Side-by-side zero-fee vs fee-adjusted comparison tearsheet.

    Layout (2×2 grid):
      Top-left:  Equity curves both regimes on same axis
      Top-right: Trade PnL distribution comparison (violin / hist)
      Bottom-left: Per-trade EV scatter (zero-fee x, fee-adj y)
      Bottom-right: Performance delta table

    Parameters
    ----------
    trades_nofee : DataFrame — trade log with pnl_pct computed at zero fees
    trades_fee   : DataFrame — same trades with fee-adjusted pnl_pct
    price_df     : DataFrame with 'close' and DatetimeIndex (OOS period)
    """
    params = params or {}

    with mpl.rc_context(_RCPARAMS):
        fig, axes = plt.subplots(2, 2, figsize=fig_size,
                                 gridspec_kw={'hspace': 0.38, 'wspace': 0.35})
        ts = price_df.index
        fh_bars = int(params.get('fh_horizon_bars', 72))

        def _eq(t):
            if t is None or len(t) == 0:
                return pd.Series(1.0, index=ts)
            return _equity_from_trades(t, ts, fh_bars)

        eq_nf = _eq(trades_nofee)
        eq_f  = _eq(trades_fee)
        bh    = (price_df['close'] / price_df['close'].iloc[0]).values

        s_nf = _stats(trades_nofee, eq_nf.values) if len(trades_nofee) else {}
        s_f  = _stats(trades_fee,   eq_f.values)  if len(trades_fee)  else {}

        # ── Panel A: Equity curves ────────────────────────────────────────
        ax = axes[0, 0]
        eq_nf_pct = (eq_nf.values - 1) * 100
        eq_f_pct  = (eq_f.values  - 1) * 100
        bh_pct    = (bh - 1) * 100

        ax.plot(ts, eq_nf_pct, color=ACCENT, lw=1.6, label=f'Zero-fee  ({s_nf.get("tot_ret",0):+.2%})')
        ax.plot(ts, eq_f_pct,  color=GREEN,  lw=1.6, label=f'Fee-adj   ({s_f.get("tot_ret",0):+.2%})')
        ax.plot(ts, bh_pct,    color=BLUE,   lw=1.0, ls='--', alpha=0.75,
                label=f'B&H  ({bh[-1]-1:+.2%})')
        ax.axhline(0, color=GREY, lw=0.7, ls=':')
        ax.fill_between(ts, eq_f_pct, eq_nf_pct, alpha=0.10, color=RED, label='Fee drag')
        ax.set_ylabel('Cumulative Return %')
        ax.set_title(f'{scheme_label} — Equity: Zero-fee vs Fee-adj', fontweight='bold', fontsize=10)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(axis='y', alpha=0.25)
        _fmt_xaxis(ax, ts)

        # ── Panel B: PnL distributions ────────────────────────────────────
        ax = axes[0, 1]
        pnl_nf = (trades_nofee['pnl_pct'].values * 100
                  if len(trades_nofee) else np.array([0.0]))
        pnl_f  = (trades_fee['pnl_pct'].values * 100
                  if len(trades_fee)  else np.array([0.0]))

        positions = [1, 2]
        data_list = [pnl_nf, pnl_f]
        colors_vp  = [ACCENT, GREEN]
        labels_vp  = ['Zero-fee', 'Fee-adj']
        for pos, data, col in zip(positions, data_list, colors_vp):
            if len(data) >= 5:
                try:
                    vp = ax.violinplot([data], positions=[pos], widths=0.5,
                                       showmedians=True, showextrema=False)
                    for body in vp['bodies']:
                        body.set_facecolor(col)
                        body.set_alpha(0.55)
                    vp['cmedians'].set_color(BLUE)
                    vp['cmedians'].set_linewidth(2)
                except Exception:
                    pass
        ax.set_xticks(positions)
        ax.set_xticklabels(labels_vp)
        ax.axhline(0, color=GREY, lw=1.0)
        ax.set_ylabel('PnL per Trade %')
        ax.set_title('PnL Distribution Comparison', fontweight='bold', fontsize=10)
        ax.grid(axis='y', alpha=0.25)

        # ── Panel C: Per-trade EV scatter ──────────────────────────────────
        ax = axes[1, 0]
        if len(trades_nofee) > 0 and len(trades_fee) > 0:
            x = trades_nofee['pnl_pct'].values * 100
            y = trades_fee['pnl_pct'].values * 100
            min_n = min(len(x), len(y))
            x, y = x[:min_n], y[:min_n]
            cols_sc = [GREEN if yi >= 0 else RED for yi in y]
            ax.scatter(x, y, c=cols_sc, s=18, alpha=0.60, edgecolors='none')
            # Diagonal: fee drag line
            lim = max(abs(x).max(), abs(y).max()) * 1.05
            ax.plot([-lim, lim], [-lim, lim], color=GREY, lw=0.8, ls='--', label='No drag')
            # Constant fee drag line
            fee_drag = float(np.mean(x - y))
            ax.plot([-lim, lim], [-lim - fee_drag, lim - fee_drag],
                    color=RED, lw=1.0, ls=':', label=f'Avg drag: {fee_drag:.3f}%')
            ax.axhline(0, color=GREY, lw=0.5)
            ax.axvline(0, color=GREY, lw=0.5)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
        ax.set_xlabel('Zero-fee PnL %')
        ax.set_ylabel('Fee-adj PnL %')
        ax.set_title('Per-Trade: Zero-fee vs Fee-adj', fontweight='bold', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.20)

        # ── Panel D: Performance delta table ──────────────────────────────
        ax = axes[1, 1]
        ax.axis('off')

        def _pct(d, k): return _fmt_pct(d.get(k)) if d else 'N/A'
        def _flt(d, k): return _fmt_f(d.get(k))   if d else 'N/A'

        row_labels = ['Total Return', 'Ann. Return', 'Sharpe', 'Max DD',
                      'Win Rate', 'Avg EV', 'Trades']
        col_headers = ['Metric', 'Zero-fee', 'Fee-adj']
        rows = [
            ['Total Return', _pct(s_nf, 'tot_ret'), _pct(s_f, 'tot_ret')],
            ['Ann. Return',  _pct(s_nf, 'ann_ret'), _pct(s_f, 'ann_ret')],
            ['Sharpe',       _flt(s_nf, 'sharpe'),  _flt(s_f, 'sharpe')],
            ['Max DD',       _pct(s_nf, 'max_dd'),  _pct(s_f, 'max_dd')],
            ['Win Rate',     _pct(s_nf, 'win_rate'),_pct(s_f, 'win_rate')],
            ['Avg EV',       _fmt_pct(s_nf.get('ev')), _fmt_pct(s_f.get('ev'))],
            ['Trades',       str(s_nf.get('n', 0)),    str(s_f.get('n', 0))],
        ]
        tbl = ax.table(
            cellText=rows,
            colLabels=col_headers,
            loc='center',
            cellLoc='center',
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.1, 2.0)
        # Header: orange
        for j in range(3):
            tbl[(0, j)].set_facecolor(ACCENT)
            tbl[(0, j)].set_text_props(color='white', fontweight='bold')
        # Zero-fee / fee-adj columns: colour by sign
        for i, row_data in enumerate(rows, start=1):
            for j_col, j_data in [(1, 1), (2, 2)]:
                v_str = row_data[j_data]
                try:
                    v = float(v_str.replace('%', '').replace('+', ''))
                    if row_data[0] == 'Max DD':
                        tbl[(i, j_col)].set_text_props(color=RED)
                    else:
                        tbl[(i, j_col)].set_text_props(color=GREEN if v > 0 else RED)
                except ValueError:
                    pass
        ax.set_title('Performance Metrics', fontweight='bold', fontsize=10, pad=12)

        # Footer
        scheme = params.get('wfo_scheme', 'N/A')
        p_thr  = params.get('prob_threshold', 0.75)
        emb    = params.get('embargo_bars', fh_bars)
        banner = (f'WFO: {scheme}  |  p > {p_thr}  |  embargo: {emb}b  |  '
                  f'{fee_label}')
        fig.text(0.5, 0.005, banner, ha='center', va='bottom',
                 fontsize=7.5, color=GREY, style='italic')

        fig.suptitle(f'{scheme_label} — Zero-fee vs Fee-adjusted', fontsize=13,
                     fontweight='bold', y=0.995)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig
