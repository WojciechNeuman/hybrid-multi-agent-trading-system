"""Explainable-AI / case-study core for notebook 07.

Loads the committed agent signals and the final fund, reconstructs every agent's risk-managed
position path with the same ``bracket_run`` engine used in production (so nothing is recomputed
in a way that could disagree with the headline result), and exposes:

* :func:`weekly_table`     — per-ISO-week BTC vs fund return over the OOS window;
* :func:`select_weeks`     — three illustrative weeks under transparent criteria;
* :func:`plot_case_week`   — a six-panel anatomy of one week (price + exposure, capital state,
                             per-agent trade events, learned/rule signals,
                             cumulative fund-vs-BTC);
* :func:`plot_btc_overview`— a BTC price/volume figure for the data chapter.

The module is import-only; the notebook and the figure-build script both call into it so the
university-facing notebook and the saved thesis figures cannot drift.
"""
from __future__ import annotations

from pathlib import Path

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

import hmats.mas.mas07 as m

ART = m.repo_root() / "artifacts" / "notebooks_v2"
OUT = ART / "07_xai"
OUT.mkdir(parents=True, exist_ok=True)

# Learned agents (in the final fund) whose probability traces are interpretable on a single price
# panel. PatchTST was screened out of the fund (no OOS skill) and is no longer shown here.
PROB_AGENTS = ["lgbm", "mamba", "tcn"]
RULE_SIGNAL_AGENTS = ["trend", "dominance_rotation"]
AGENT_LABEL = {
    "lgbm": "LightGBM", "mamba": "Mamba", "tcn": "TCN", "patch": "PatchTST",
    "trend": "Trend", "volbreak": "VolBreak", "dominance_rotation": "DomRot",
}
AGENT_COLORS = {
    "lgbm": "#1b7837", "mamba": "#5aae61", "tcn": "#2166ac", "patch": "#b2182b",
    "trend": "#7b3294", "volbreak": "#008837", "dominance_rotation": "#c2a500",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_everything():
    """Return (panel, agents, fund_eq, weights) all aligned on the OOS hourly index."""
    panel = m.load_panel()
    agents = m.build_agents(panel, ART)

    fund_eq = pd.Series(
        np.load(OUT.parent / "06_mas" / "final_equity.npy"),
        index=pd.to_datetime(np.load(OUT.parent / "06_mas" / "oos_index.npy")),
        name="fund",
    )
    weights = pd.read_csv(OUT.parent / "06_mas" / "capped_inverse_vol_weights_oos.csv",
                          parse_dates=["open_time"]).set_index("open_time")
    return panel, agents, fund_eq, weights


def fund_net_exposure(agents, weights) -> pd.Series:
    """Capital-weighted net directional exposure of the fund in [-1, 1] over the OOS window."""
    pos = pd.DataFrame({a: agents[a].position for a in weights.columns}).reindex(weights.index)
    return (pos * weights).sum(axis=1)


def fund_capital_state(agents, weights, idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Capital share allocated to agents that are long, short, or flat/cash at each timestamp."""
    w = weights.reindex(idx).ffill().fillna(0.0)
    pos = pd.DataFrame({a: agents[a].position for a in w.columns}).reindex(idx).fillna(0.0)
    long_cap = w.where(pos > 0, 0.0).sum(axis=1)
    short_cap = w.where(pos < 0, 0.0).sum(axis=1)
    flat_cap = (1.0 - long_cap - short_cap).clip(lower=0.0)
    return pd.DataFrame({"long": long_cap, "flat": flat_cap, "short": short_cap}, index=idx)


# ---------------------------------------------------------------------------
# Week selection
# ---------------------------------------------------------------------------
def weekly_table(panel, fund_eq) -> pd.DataFrame:
    """Per-week (Mon-anchored) BTC return, fund return, and fund-minus-BTC divergence on OOS."""
    oos = (panel.index >= m.OOS_START) & (panel.index <= m.OOS_END)
    close = panel.loc[oos, "close"]
    fund = fund_eq.reindex(close.index).ffill()
    wk = pd.DataFrame({"close": close, "fund": fund})
    # Label weeks by their true Monday start so case titles and plot windows use
    # the same seven-day interval.
    g = wk.resample("W-MON", closed="left", label="left")
    out = pd.DataFrame({
        "btc_ret": g["close"].last() / g["close"].first() - 1.0,
        "fund_ret": g["fund"].last() / g["fund"].first() - 1.0,
        "btc_maxdd": g["close"].apply(lambda s: float(((s / s.cummax()) - 1).min())),
        "n": g["close"].count(),
    })
    out = out[out["n"] >= 120]  # near-full weeks only
    out["divergence"] = out["fund_ret"] - out["btc_ret"]
    return out


def select_weeks(wt: pd.DataFrame) -> dict:
    """Three transparent, non-overlapping illustrative weeks.

    * ``divergence`` — BTC clearly down (< -4%) while the fund itself closes *up*: the headline
      argument that the short-capable agents and the allocator add value precisely when
      buy-and-hold suffers.
    * ``trend``      — among weeks where BTC rallies hard (> +5%), the one with the strongest fund
      gain: the system riding a clean bull leg rather than fighting it.
    * ``defence``    — the week of BTC's worst intra-week drawdown: capital preservation under the
      single most violent sell-off, whatever the close-to-close number.
    """
    up_fund_down_btc = wt[(wt["btc_ret"] < -0.04) & (wt["fund_ret"] > 0)]
    div = up_fund_down_btc["divergence"].idxmax() if len(up_fund_down_btc) \
        else wt["divergence"].idxmax()
    bull = wt[wt["btc_ret"] > 0.05]
    trend = bull["fund_ret"].idxmax() if len(bull) else wt[wt["btc_ret"] > 0]["fund_ret"].idxmax()
    defence = wt["btc_maxdd"].idxmin()
    return {"divergence": div, "trend": trend, "defence": defence}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _candles(ax, o, h, l, c, idx):
    w = (mdates.date2num(idx[1]) - mdates.date2num(idx[0])) * 0.7
    x = mdates.date2num(idx)
    up = c >= o
    ax.vlines(x, l, h, color="#555", linewidth=0.6, zorder=1)
    for xi, oi, ci, u in zip(x, o, c, up):
        ax.add_patch(Rectangle((xi - w / 2, min(oi, ci)), w, abs(ci - oi) or 1e-9,
                               facecolor="#1a9850" if u else "#d73027",
                               edgecolor="none", alpha=0.9, zorder=2))
    ax.xaxis_date()


def _agent_trade_events(agent, panel: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct entries and TP/SL/time exits from the same ATR-bracket rules as ``bracket_run``."""
    bp = dict(agent.best_params)
    prob = agent.prob.reindex(panel.index).values
    prob_dn = agent.prob_dn.reindex(panel.index).values if agent.prob_dn is not None else None
    prob_neutral = (
        agent.prob_neutral.reindex(panel.index).values if agent.prob_neutral is not None else None
    )
    close = panel["close"].values
    high = panel["high"].values
    low = panel["low"].values
    atr = panel["atr_14_pct"].values

    long_threshold = bp["long_threshold"]
    short_threshold = bp["short_threshold"]
    entry_atr_mult = bp["entry_atr_mult"]
    sl_atr_mult = bp["sl_atr_mult"]
    tp_atr_mult = bp["tp_atr_mult"]
    min_hold = bp["min_hold"]
    max_hold = bp["max_hold"]
    cooldown = bp["cooldown"]
    min_sl = bp.get("min_sl", 0.01)
    trade_direction = bp.get("trade_direction", "both")
    neutral_max = bp.get("neutral_max", 1.0)
    edge_margin = bp.get("edge_margin", 0.0)

    events = []
    in_pos = False
    direction = None
    entry_px = sl_px = tp_px = 0.0
    hold = cd = 0
    pend = None

    for i, ts in enumerate(panel.index):
        lo, hi, px = low[i], high[i], close[i]
        if in_pos:
            hold += 1
            if hold >= min_hold:
                reason = None
                xpx = 0.0
                if direction == "long":
                    if lo <= sl_px:
                        xpx, reason = sl_px, "SL"
                    elif hi >= tp_px:
                        xpx, reason = tp_px, "TP"
                    elif hold >= max_hold:
                        xpx, reason = px, "time"
                else:
                    if hi >= sl_px:
                        xpx, reason = sl_px, "SL"
                    elif lo <= tp_px:
                        xpx, reason = tp_px, "TP"
                    elif hold >= max_hold:
                        xpx, reason = px, "time"
                if reason is not None:
                    events.append({
                        "time": ts, "agent": agent.name, "event": "exit",
                        "direction": direction, "reason": reason, "price": xpx,
                    })
                    in_pos = False
                    cd = cooldown
        elif pend is not None:
            d, lim, ps, pt = pend
            fill = lo <= lim + m.BUFFER if d == "long" else hi >= lim - m.BUFFER
            entry_px = lim if fill else px
            sl_px, tp_px = ps, pt
            direction = d
            in_pos = True
            hold = 0
            events.append({
                "time": ts, "agent": agent.name, "event": "entry",
                "direction": d, "reason": "limit" if fill else "market", "price": entry_px,
            })
            pend = None
        elif cd > 0:
            cd -= 1
        elif not np.isnan(prob[i]) and i + 1 < len(panel):
            a = max(atr[i], min_sl)
            if prob_dn is None:
                go_long = trade_direction in ("both", "long_only") and prob[i] > long_threshold
                go_short = trade_direction in ("both", "short_only") and prob[i] < short_threshold
            else:
                if np.isnan(prob_dn[i]):
                    continue
                neutral_ok = True
                if prob_neutral is not None:
                    if np.isnan(prob_neutral[i]):
                        continue
                    neutral_ok = prob_neutral[i] <= neutral_max
                go_long = (
                    trade_direction in ("both", "long_only")
                    and prob[i] > long_threshold
                    and neutral_ok
                    and (prob[i] - prob_dn[i]) >= edge_margin
                )
                go_short = (
                    trade_direction in ("both", "short_only")
                    and prob_dn[i] > short_threshold
                    and neutral_ok
                    and (prob_dn[i] - prob[i]) >= edge_margin
                )
            if go_long:
                pend = ("long", px * (1 - entry_atr_mult * a),
                        px * (1 - sl_atr_mult * a), px * (1 + tp_atr_mult * a))
            elif go_short:
                pend = ("short", px * (1 + entry_atr_mult * a),
                        px * (1 + sl_atr_mult * a), px * (1 - tp_atr_mult * a))

    return pd.DataFrame(events).set_index("time") if events else pd.DataFrame(
        columns=["agent", "event", "direction", "reason", "price"]
    )


def plot_case_week(week_start, panel, agents, fund_eq, weights, title, fname):
    s = pd.Timestamp(week_start)
    e = s + pd.Timedelta(days=7)
    win = (panel.index >= s) & (panel.index < e)
    idx = panel.index[win]
    px = panel.loc[win]
    net = fund_net_exposure(agents, weights).reindex(idx)
    cap_state = fund_capital_state(agents, weights, idx) * 100.0

    fig, ax = plt.subplots(6, 1, figsize=(11.4, 13.2), sharex=True,
                           gridspec_kw={"height_ratios": [2.35, 0.9, 0.85, 1.15, 0.95, 1.55]})

    # --- Panel A: candles + fund net exposure shading + entries/exits ---
    a0 = ax[0]
    _candles(a0, px["open"].values if "open" in px else px["close"].values,
             px["high"].values, px["low"].values, px["close"].values, idx)
    a0.set_ylabel("BTC price (USDT)")
    a0.set_title(title, loc="left", fontsize=12, fontweight="bold")
    twin = a0.twinx()
    twin.fill_between(idx, 0, net.values, where=net.values >= 0, color="#1a9850", alpha=0.12,
                      step="post", label="net long")
    twin.fill_between(idx, 0, net.values, where=net.values < 0, color="#d73027", alpha=0.12,
                      step="post", label="net short")
    twin.set_ylabel("fund net exposure")
    twin.set_ylim(-1.05, 1.05)
    twin.axhline(0, color="grey", lw=0.5)

    entry_style = {
        "long": {"marker": "^", "color": "#1a9850", "label": "long entry"},
        "short": {"marker": "v", "color": "#762a83", "label": "short entry"},
    }
    exit_style = {
        "TP": {"marker": "o", "color": "#2166ac", "label": "TP exit"},
        "SL": {"marker": "X", "color": "#d73027", "label": "SL exit"},
        "time": {"marker": "s", "color": "#333333", "label": "time exit"},
    }

    all_events = {}
    # Entry / exit markers from the same ATR-bracket rules that generated each position path.
    for a in PROB_AGENTS:
        ev = _agent_trade_events(agents[a], panel)
        ev = ev[(ev.index >= s) & (ev.index < e)]
        all_events[a] = ev
        if ev.empty:
            continue
        for direction, st in entry_style.items():
            rows = ev[(ev["event"] == "entry") & (ev["direction"] == direction)]
            if len(rows):
                a0.scatter(rows.index, rows["price"], marker=st["marker"], color=st["color"],
                           edgecolor="white", linewidth=0.45, s=34, zorder=5)
        for reason, st in exit_style.items():
            rows = ev[(ev["event"] == "exit") & (ev["reason"] == reason)]
            if len(rows):
                a0.scatter(rows.index, rows["price"], marker=st["marker"], color=st["color"],
                           edgecolor="white", linewidth=0.45, s=32, zorder=5)

    exposure_handles, exposure_labels = twin.get_legend_handles_labels()
    event_handles = [
        Line2D([0], [0], marker=st["marker"], color="none", markerfacecolor=st["color"],
               markeredgecolor="white", markeredgewidth=0.45, markersize=6, label=st["label"])
        for st in [entry_style["long"], entry_style["short"],
                   exit_style["TP"], exit_style["SL"], exit_style["time"]]
    ]
    a0.legend(exposure_handles + event_handles, exposure_labels + [h.get_label() for h in event_handles],
              loc="upper right", fontsize=8, framealpha=0.68, ncol=2,
              borderpad=0.45, handletextpad=0.5, columnspacing=0.9)

    # --- Panel B: fund capital state ---
    acap = ax[1]
    acap.stackplot(
        idx,
        cap_state["long"].values,
        cap_state["flat"].values,
        cap_state["short"].values,
        colors=["#1a9850", "#bdbdbd", "#d73027"],
        alpha=0.72,
        labels=["long capital", "flat / cash", "short capital"],
    )
    acap.set_ylabel("capital (%)")
    acap.set_ylim(0, 100)
    acap.legend(loc="upper left", ncol=3, fontsize=8, framealpha=0.68,
                borderpad=0.35, handlelength=1.4, columnspacing=0.9)

    # --- Panel C: clean per-agent event strip ---
    aev = ax[2]
    agent_y = {a: len(PROB_AGENTS) - i for i, a in enumerate(PROB_AGENTS)}
    for a in PROB_AGENTS:
        aev.axhline(agent_y[a], color="#e5e5e5", lw=0.7, zorder=0)
        ev = all_events.get(a)
        if ev is None or ev.empty:
            continue
        y = agent_y[a]
        for direction, st in entry_style.items():
            rows = ev[(ev["event"] == "entry") & (ev["direction"] == direction)]
            if len(rows):
                aev.scatter(rows.index, [y] * len(rows), marker=st["marker"], color=st["color"],
                            edgecolor="white", linewidth=0.45, s=42, zorder=3)
        for reason, st in exit_style.items():
            rows = ev[(ev["event"] == "exit") & (ev["reason"] == reason)]
            if len(rows):
                aev.scatter(rows.index, [y] * len(rows), marker=st["marker"], color=st["color"],
                            edgecolor="white", linewidth=0.45, s=42, zorder=3)
                for t in rows.index:
                    aev.annotate(reason, (t, y), xytext=(0, 7), textcoords="offset points",
                                 ha="center", va="bottom", fontsize=6.5, color=st["color"])
    aev.set_yticks([agent_y[a] for a in PROB_AGENTS], [AGENT_LABEL[a] for a in PROB_AGENTS])
    aev.set_ylim(0.45, len(PROB_AGENTS) + 0.55)
    aev.set_ylabel("agent")
    aev.tick_params(axis="y", labelsize=8)

    # --- Panel D: learned-agent probabilities + thresholds ---
    a1 = ax[3]
    for a in PROB_AGENTS:
        a1.plot(idx, panel.loc[win, a].values, color=AGENT_COLORS[a], lw=1.25,
                label=f"{AGENT_LABEL[a]} P(up)")
    a1.axhline(0.5, color="grey", ls=":", lw=0.8)
    a1.set_ylabel("learned\nP(up)")
    a1.set_ylim(0, 1)
    a1.legend(loc="upper left", ncol=4, fontsize=8, framealpha=0.6)

    # --- Panel E: rule-agent directional signals ---
    arule = ax[4]
    for a in RULE_SIGNAL_AGENTS:
        if a in panel:
            arule.plot(idx, panel.loc[win, a].values, color=AGENT_COLORS[a], lw=1.25,
                       label=f"{AGENT_LABEL[a]} signal")
    arule.axhline(0.5, color="grey", ls=":", lw=0.8)
    arule.set_ylabel("rule\nsignal")
    arule.set_ylim(0, 1)
    arule.legend(loc="upper left", ncol=3, fontsize=8, framealpha=0.6)

    # --- Panel F: cumulative fund, BTC and individual agent returns over the week ---
    a2 = ax[5]
    fund = fund_eq.reindex(idx).ffill()
    for a in weights.columns:
        eq = agents[a].eq.reindex(idx).ffill()
        if eq.notna().any() and eq.iloc[0] != 0:
            a2.plot(idx, (eq / eq.iloc[0] - 1) * 100, color=AGENT_COLORS.get(a, "#777"),
                    lw=0.95, alpha=0.48, label=AGENT_LABEL.get(a, a))
    a2.plot(idx, (px["close"] / px["close"].iloc[0] - 1).values * 100, color="#f1a340", lw=1.5,
            label="BTC buy & hold")
    a2.plot(idx, (fund / fund.iloc[0] - 1) * 100, color="#000", lw=2.0, label="MAS fund")
    a2.axhline(0, color="grey", lw=0.5)
    a2.set_ylabel("cum. return (%)")
    a2.legend(loc="upper left", fontsize=7.4, framealpha=0.66, ncol=4,
              borderpad=0.35, handlelength=1.4, columnspacing=0.75)
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%a %d"))

    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return OUT / fname


def plot_btc_overview(panel, fname="btc_price_volume.png"):
    """Full-history BTC price (log) + daily volume with the OOS window shaded — data-chapter figure.

    Hourly data is resampled to daily candles. The price line is shaded under to read as an
    area chart; daily volume bars are coloured green on up-days and red on down-days so the
    distribution of activity across the cycle is legible. The two BTC halvings inside the sample
    and the out-of-sample window are annotated.
    """
    df = panel
    # Resample to daily so the volume bars are not an unreadable hourly smear.
    daily = pd.DataFrame({
        "close": df["close"].resample("1D").last(),
        "volume": (df["volume"] if "volume" in df else df["ret"].abs()).resample("1D").sum(),
    }).dropna()
    up = daily["close"].diff().fillna(0.0) >= 0

    price_c, oos_c = "#08519c", "#4575b4"
    up_c, dn_c = "#2ca25f", "#d6604d"
    halvings = [pd.Timestamp("2020-05-11"), pd.Timestamp("2024-04-20")]

    fig, ax = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True,
                           gridspec_kw={"height_ratios": [3, 1]})

    # --- Price panel --------------------------------------------------------
    ax[0].plot(daily.index, daily["close"], color=price_c, lw=1.1, zorder=3)
    ax[0].fill_between(daily.index, daily["close"], daily["close"].min(),
                       color=price_c, alpha=0.07, zorder=1)
    ax[0].set_yscale("log")
    ax[0].set_ylabel("BTC close (USDT, log scale)")
    ax[0].set_ylim(daily["close"].min() * 0.85, daily["close"].max() * 1.25)
    oos_band = ax[0].axvspan(m.OOS_START, m.OOS_END, color=oos_c, alpha=0.12, zorder=0)
    for h in halvings:
        ax[0].axvline(h, color="#444", lw=1.0, ls="--", alpha=0.7, zorder=2)
        ax[0].annotate("halving", xy=(h, ax[0].get_ylim()[0]), xytext=(4, 6),
                       textcoords="offset points", rotation=90, va="bottom", ha="left",
                       fontsize=8, color="#444")
    ax[0].grid(True, which="major", axis="y", ls=":", lw=0.5, color="#bbb", alpha=0.6)

    # --- Volume panel -------------------------------------------------------
    ax[1].bar(daily.index[up.values], daily["volume"][up.values], width=1.0,
              color=up_c, alpha=0.85, linewidth=0)
    ax[1].bar(daily.index[~up.values], daily["volume"][~up.values], width=1.0,
              color=dn_c, alpha=0.85, linewidth=0)
    ax[1].set_ylabel("daily volume (BTC)")
    ax[1].axvspan(m.OOS_START, m.OOS_END, color=oos_c, alpha=0.12, zorder=0)
    for h in halvings:
        ax[1].axvline(h, color="#444", lw=1.0, ls="--", alpha=0.7)
    ax[1].grid(True, axis="y", ls=":", lw=0.5, color="#bbb", alpha=0.6)
    ax[1].margins(x=0.005)

    legend_handles = [
        Line2D([0], [0], color=price_c, lw=1.5, label="BTC close (daily)"),
        Rectangle((0, 0), 1, 1, fc=up_c, alpha=0.85, label="up-day volume"),
        Rectangle((0, 0), 1, 1, fc=dn_c, alpha=0.85, label="down-day volume"),
        Rectangle((0, 0), 1, 1, fc=oos_c, alpha=0.12, label="out-of-sample window"),
    ]
    ax[0].legend(handles=legend_handles, loc="upper left", fontsize=8.5,
                 framealpha=0.9, ncol=2)

    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return OUT / fname
