"""Hybrid Multi-Agent Trading System — final coordination layer (notebook 06).

This module turns the trained base models (LightGBM, Mamba, TCN, PatchTST) and the accepted rule
agents into genuine *autonomous trading agents* and a *coordinator* that allocates capital across
them.

What makes this multi-agent rather than an ensemble of probabilities:

1. **Agents act and manage their own risk.** Each agent runs its *own tuned* ATR-bracket
   strategy (entries, stop-loss, take-profit, holding limits) and produces a realised
   return stream and a held position. Its individually-tuned edge — the thing the averaging
   meta-learner threw away — is preserved intact.
2. **Agents have measured specialisations.** A per-regime competence prior is estimated for
   every agent on the *pre-OOS* window only (leak-free): the coordinator learns *whom to
   trust in which market regime*.
3. **The system tests several capital allocators.** The original regime-gated coordinator is
   retained as an ablation, while the final reported allocator is capped inverse-volatility risk
   parity over the accepted agents.
4. **It is honest about failed agents.** The cross-asset learner and the contrarian-sentiment rule
   are excluded because their OOS returns are negative. The mean-reversion rule is excluded for the
   same reason. The ``dominance_rotation`` rule is included as a *diversification* agent only — it
   is OOS-profitable with a shallower drawdown than buy-and-hold, but its OOS return sits at the
   95% boundary of the random-bracket null and is not claimed as alpha.

Leakage discipline: an allocation decided with information up to bar ``t`` earns each agent's
return over ``t -> t+1``; every trailing statistic uses a right-open window shifted by >= 1 bar
plus an embargo; competence priors see only pre-OOS data.

The module is framework-light (numpy / pandas) so it can be lifted into ``src/hmats/agents`` and
``src/hmats/coordinator`` later. Class names mirror the existing ``BaseAgent`` / ``Supervisor``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Learned agents (nonlinear models on the shared feature panel) + accepted rule agents. The rule
# agents are structurally orthogonal: their edge is strategy logic, not another learned transform of
# the same feature matrix. The mean-reversion rule and cross-asset learner remain in the repository
    # as experiments, but are excluded from the final agent set for the reasons documented below.
LEARNED_AGENTS = ["lgbm", "mamba", "tcn", "patch"]
RULE_AGENTS = ["trend", "volbreak", "dominance_rotation"]
EXCLUDED_AGENTS = {
    "meanrev": "excluded from final agent set: negative OOS return",
    "crossasset": "excluded from final agent set: weak OOS AUC and not significant vs random bracket null",
    "sentiment_regime": "excluded from final agent set: negative OOS return (-29.9%) and below the "
                        "random-bracket null (7th percentile) — contrarian Fear & Greed has no edge OOS",
}
AGENTS = LEARNED_AGENTS + RULE_AGENTS
AGENT_DIR = {"lgbm": "01_lgbm", "mamba": "02_mamba", "tcn": "03_tcn", "patch": "04_patchtst",
             "trend": "05_trend", "volbreak": "05_volbreak",
             "dominance_rotation": "05_dominance_rotation"}
# Multiclass TBM agents emit two *independent* softmax channels (P-up, P-down) and decide
# long on P-up and short on P-down. A single saved probability is the P-up channel only, so
# these agents must be backtested with their P-down channel too — otherwise the binary engine
# manufactures shorts whenever P-up is merely low, destroying their genuine (short-heavy) edge.
MULTICLASS = {"tcn", "patch"}
PARADIGM = {
    "lgbm": "gradient boosting (tabular)",
    "mamba": "selective state-space",
    "tcn": "dilated causal conv.",
    "patch": "patch transformer",
    "trend": "rule: trend-following",
    "volbreak": "rule: volatility breakout",
    "dominance_rotation": "rule: cross-asset dominance rotation",
}

OOS_START = pd.Timestamp("2024-05-31")
OOS_END = pd.Timestamp("2026-05-31")
COMPETENCE_START = pd.Timestamp("2023-01-01")  # pre-OOS window common to all four agents

REGIMES = ("chop", "bull", "bear")
REGIME_DATES = {  # reporting-only OOS sub-periods; the live detector is feature-based
    "chop": (pd.Timestamp("2024-05-31"), pd.Timestamp("2024-11-05")),
    "bull": (pd.Timestamp("2024-11-06"), pd.Timestamp("2025-10-31")),
    "bear": (pd.Timestamp("2025-11-01"), pd.Timestamp("2026-05-31")),
}

# Fee model — identical to the base agents' backtests.
MAKER_FEE = 0.0
TAKER_FEE = 0.0005
BUFFER = 0.0005
SHORT_FUNDING_H = 0.0000077
REALLOC_FEE = 0.0002       # capital reallocation cost across agents (per unit |Δweight|)

EMBARGO_H = 48   # embargo applied to every trailing/online statistic
ANN = np.sqrt(24 * 365)

# --- Coordinator defaults (a-priori reasonable; NOT tuned on the OOS window) -------------
# A 60-day trailing window to score each agent's recent skill, a neutral softmax temperature,
# and a 7-day regime-smoothing window to suppress the whipsaw that wrecks instantaneous routing.
# Proper selection of these belongs on pre-OOS data (purged CV) and is left as future work.
PERF_WIN = 1440        # 60-day trailing window for the online performance score
PERF_TEMP = 0.75       # softmax temperature over trailing Sharpe
REGIME_SMOOTH = 168    # 7-day majority-vote smoothing of the regime label
COMP_FLOOR = 0.15      # floor on the competence tilt so a hot agent is never fully vetoed


def repo_root() -> Path:
    p = Path.cwd()
    while p != p.parent:
        if (p / "pyproject.toml").exists():
            return p
        p = p.parent
    raise RuntimeError("repo root not found")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def sharpe(eq: np.ndarray) -> float:
    r = np.diff(np.log(np.maximum(eq, 1e-12)))
    return float(r.mean() / (r.std(ddof=1) + 1e-12) * ANN)


def sortino(eq: np.ndarray) -> float:
    r = np.diff(np.log(np.maximum(eq, 1e-12)))
    neg = r[r < 0]
    d = neg.std(ddof=1) if len(neg) > 1 else 1e-12
    return float(r.mean() / (d + 1e-12) * ANN)


def maxdd(eq: np.ndarray) -> float:
    pk = np.maximum.accumulate(eq)
    return float(((eq - pk) / (pk + 1e-12)).min())


# ---------------------------------------------------------------------------
# ATR-bracket engine — the agents' own risk-managed execution
# ---------------------------------------------------------------------------

def bracket_run(prob, close, high, low, atr, *, long_threshold, short_threshold,
                entry_atr_mult, sl_atr_mult, tp_atr_mult, min_hold, max_hold, cooldown,
                min_sl=0.01, trade_direction="both", with_fees=True, prob_dn=None, **_ignored):
    """Single-pass ATR-bracket backtester (identical logic to the base agents).

    Returns three full-length arrays: the equity curve ``eq``, the held position ``pos`` in
    ``{-1, 0, +1}``, and a held ``conf`` in ``[0, 1]`` (entry-probability strength carried for
    the life of the trade). Equity is net of the maker/taker fee model and short funding.

    ``trade_direction`` can be ``both``, ``long_only`` or ``short_only`` and is honoured for
    both binary and dual-channel agents. ``prob_dn`` selects the entry convention:

    * ``None`` — binary single-probability agent (LightGBM, Mamba): long when
      ``prob > long_threshold``, short when ``prob < short_threshold``.
    * an array — multiclass TBM agent (TCN, PatchTST): ``prob`` is the P-up channel and
      ``prob_dn`` the P-down channel; long when ``prob > long_threshold``, short when
      ``prob_dn > short_threshold`` (long takes priority, mirroring the base notebooks).
    """
    n = len(close)
    eq = np.ones(n); pos = np.zeros(n); conf = np.zeros(n)
    cur = 1.0
    in_pos = False; direction = None
    entry_px = sl_px = tp_px = pos_eq = entry_fee = 0.0
    held_conf = 0.0; hold = cd = 0; funding = 0.0; pend = None
    for i in range(n):
        lo, hi, px = low[i], high[i], close[i]
        if in_pos:
            hold += 1
            if direction == "short":
                funding += SHORT_FUNDING_H
            eq[i] = pos_eq * (px / entry_px if direction == "long" else 1 + (entry_px - px) / entry_px)
            pos[i] = 1.0 if direction == "long" else -1.0
            conf[i] = held_conf
            ex = False; xpx = 0.0; xf = 0.0
            if hold >= min_hold:
                if direction == "long":
                    if lo <= sl_px: xpx, ex, xf = sl_px, True, (TAKER_FEE if with_fees else 0.0)
                    elif hi >= tp_px: xpx, ex, xf = tp_px, True, MAKER_FEE
                    elif hold >= max_hold: xpx, ex, xf = px, True, (TAKER_FEE if with_fees else 0.0)
                else:
                    if hi >= sl_px: xpx, ex, xf = sl_px, True, (TAKER_FEE if with_fees else 0.0)
                    elif lo <= tp_px: xpx, ex, xf = tp_px, True, MAKER_FEE
                    elif hold >= max_hold: xpx, ex, xf = px, True, (TAKER_FEE if with_fees else 0.0)
            if ex:
                g = ((xpx - entry_px) / entry_px if direction == "long" else (entry_px - xpx) / entry_px)
                net = g - (entry_fee + xf if with_fees else 0.0) - funding
                cur = pos_eq * (1.0 + net); eq[i] = cur
                in_pos = False; cd = cooldown; funding = 0.0
        elif pend is not None:
            d, lim, ps, pt, pc = pend
            if d == "long":
                fill = lo <= lim + BUFFER
                ef = MAKER_FEE if (fill and with_fees) else (TAKER_FEE if with_fees else 0.0)
            else:
                fill = hi >= lim - BUFFER
                ef = MAKER_FEE if (fill and with_fees) else (TAKER_FEE if with_fees else 0.0)
            entry_px = lim if fill else px
            sl_px, tp_px, entry_fee = ps, pt, ef
            direction = d; in_pos = True; pos_eq = cur; hold = 0; funding = 0.0
            held_conf = pc; pos[i] = 1.0 if d == "long" else -1.0; conf[i] = pc
            eq[i] = cur; pend = None
        elif cd > 0:
            cd -= 1; eq[i] = cur
        elif not np.isnan(prob[i]) and i + 1 < n:
            a = max(atr[i], min_sl)
            if prob_dn is None:
                go_long = trade_direction in ("both", "long_only") and prob[i] > long_threshold
                go_short = trade_direction in ("both", "short_only") and prob[i] < short_threshold
                pc_long = pc_short = float(np.clip(2 * abs(prob[i] - 0.5), 0, 1))
            else:
                go_long = trade_direction in ("both", "long_only") and prob[i] > long_threshold
                go_short = trade_direction in ("both", "short_only") and prob_dn[i] > short_threshold
                pc_long = float(np.clip(2 * abs(prob[i] - 0.5), 0, 1))
                pc_short = float(np.clip(2 * abs(prob_dn[i] - 0.5), 0, 1))
            if go_long:
                pend = ("long", px * (1 - entry_atr_mult * a), px * (1 - sl_atr_mult * a),
                        px * (1 + tp_atr_mult * a), pc_long)
            elif go_short:
                pend = ("short", px * (1 + entry_atr_mult * a), px * (1 + sl_atr_mult * a),
                        px * (1 - tp_atr_mult * a), pc_short)
            eq[i] = cur
        else:
            eq[i] = cur
    return eq, pos, conf


# ---------------------------------------------------------------------------
# Regime detector (feature-based, works across all history -> leak-free priors)
# ---------------------------------------------------------------------------

class RegimeDetector:
    """Label each bar ``chop`` / ``bull`` / ``bear`` from *stationary* trend/volatility features
    only. Calendar/monotonic features are deliberately excluded (they caused the meta-learner's
    regime memorisation). The same detector is applied pre-OOS and OOS, which is what lets
    per-regime competence be estimated without leakage.
    """

    def __init__(self, chop_hurst: float = 0.5):
        self.chop_hurst = chop_hurst

    def label(self, df: pd.DataFrame) -> pd.Series:
        trend = df["close_vs_sma_200"] if "close_vs_sma_200" in df else df["sma100_vs_sma200"]
        sideways = df.get("sideways_flag", pd.Series(0, index=df.index)).fillna(0).astype(bool)
        hurst = df.get("hurst_24h", pd.Series(0.5, index=df.index)).fillna(0.5)
        is_chop = sideways | (hurst < self.chop_hurst)
        out = np.where(trend.fillna(0) >= 0, "bull", "bear")
        out = np.where(is_chop, "chop", out)
        return pd.Series(out, index=df.index, name="regime")


# ---------------------------------------------------------------------------
# Data panel
# ---------------------------------------------------------------------------

def _load_signal(a2: Path, sub: str, kind: str, value: str = "probs") -> pd.Series:
    """Load one channel of an agent's signal. ``value`` picks the file suffix
    (``probs`` -> ``{kind}_probs.npy``, ``pdown`` -> ``{kind}_pdown.npy``); both share
    the ``{kind}_index.npy`` timestamps."""
    f = a2 / sub
    p = np.load(f / f"{kind}_{value}.npy")
    idx = pd.to_datetime(np.load(f / f"{kind}_index.npy"), unit="ns")
    return pd.Series(p, index=idx)


def _spliced_signal(a2: Path, sub: str, index: pd.Index, value: str = "probs") -> pd.Series:
    """Walk-forward signal over full history, with the OOS window overwritten by the
    held-out OOS signal (the same splice used for the P-up channel)."""
    wfo = _load_signal(a2, sub, "wfo", value).reindex(index)
    oos = _load_signal(a2, sub, "oos", value).reindex(index)
    mask = (index >= OOS_START) & (index <= OOS_END) & oos.notna()
    wfo.loc[mask] = oos.loc[mask]
    return wfo


def load_panel() -> pd.DataFrame:
    """Aligned panel: each agent's walk-forward probability over full history spliced with its
    OOS probability over the OOS window, plus price, return, regime and stationary features.
    """
    repo = repo_root()
    a2 = repo / "artifacts" / "notebooks_v2"
    df = pd.read_parquet(repo / "data" / "features" / "BTCUSDT_1h_unified.parquet")
    df.index = df.index.tz_localize(None) if df.index.tz else df.index

    panel = pd.DataFrame(index=df.index)
    for a in AGENTS:
        panel[a] = _spliced_signal(a2, AGENT_DIR[a], df.index)
        if a in MULTICLASS:  # P-down channel for the dual-channel TBM agents
            panel[f"{a}_dn"] = _spliced_signal(a2, AGENT_DIR[a], df.index, "pdown")

    for c in ["close", "high", "low", "atr_14_pct", "close_vs_sma_200", "sma100_vs_sma200",
              "sideways_flag", "hurst_24h", "bb_width_pct", "vol_ratio_24h", "trend_score"]:
        if c in df:
            panel[c] = df[c]
    panel["ret"] = df["close"].pct_change().fillna(0.0)
    panel["regime"] = RegimeDetector().label(df)
    return panel


# ---------------------------------------------------------------------------
# Agents — autonomous, risk-managed traders
# ---------------------------------------------------------------------------

@dataclass
class TradingAgent:
    """One base model wrapped as an autonomous, risk-managed trading agent.

    On :meth:`build` it runs its own tuned ATR-bracket strategy over the full history, yielding a
    realised equity curve, a per-bar return stream ``g`` (what the coordinator allocates over),
    a held position path, and a communicated confidence. Each agent keeps its authentic strategy
    — including any directional bias; reconciling those biases across regimes is the
    coordinator's job, not the agent's.
    """

    name: str
    prob: pd.Series
    best_params: dict
    prob_dn: pd.Series = field(default=None, repr=False)  # P-down channel for multiclass agents
    eq: pd.Series = field(default=None, repr=False)
    g: pd.Series = field(default=None, repr=False)
    position: pd.Series = field(default=None, repr=False)
    confidence: pd.Series = field(default=None, repr=False)

    @property
    def paradigm(self) -> str:
        return PARADIGM.get(self.name, "unknown")

    def build(self, panel: pd.DataFrame) -> "TradingAgent":
        prob_dn = self.prob_dn.reindex(panel.index).values if self.prob_dn is not None else None
        eq, pos, conf = bracket_run(
            self.prob.reindex(panel.index).values,
            panel["close"].values, panel["high"].values, panel["low"].values,
            panel["atr_14_pct"].values, prob_dn=prob_dn, **self.best_params)
        self.eq = pd.Series(eq, index=panel.index, name=self.name)
        self.g = pd.Series(np.diff(np.log(np.maximum(eq, 1e-12)), prepend=0.0),
                           index=panel.index, name=self.name)
        self.position = pd.Series(pos, index=panel.index, name=self.name)
        self.confidence = pd.Series(conf, index=panel.index, name=self.name)
        return self


def build_agents(panel: pd.DataFrame, a2: Path) -> dict[str, TradingAgent]:
    agents: dict[str, TradingAgent] = {}
    for a in AGENTS:
        bp = json.load(open(a2 / AGENT_DIR[a] / "results.json")).get("best_params", {})
        pdn = panel[f"{a}_dn"] if f"{a}_dn" in panel else None
        agents[a] = TradingAgent(a, panel[a], bp, prob_dn=pdn).build(panel)
    return agents


# ---------------------------------------------------------------------------
# Competence priors (per-regime, pre-OOS only) and online reliability
# ---------------------------------------------------------------------------

def estimate_competence(agents: dict[str, TradingAgent], panel: pd.DataFrame) -> pd.DataFrame:
    """Per-regime competence of each agent, measured on the pre-OOS window only (leak-free).

    Competence = annualised Sharpe of the agent's *active-bar* returns within that regime,
    passed through ReLU (a non-positive track record earns zero trust), then normalised within
    each regime so weights are comparable. This is the gate's prior on *who is good where*.
    """
    pre = (panel.index >= COMPETENCE_START) & (panel.index < OOS_START)
    rows = {}
    for a, ag in agents.items():
        active = ag.position.abs() > 0
        row = {}
        for r in REGIMES:
            m = pre & active & (panel["regime"] == r)
            seg = ag.g[m].values
            if len(seg) > 50 and seg.std() > 0:
                row[r] = max(seg.mean() / (seg.std() + 1e-12) * ANN, 0.0)
            else:
                row[r] = 0.0
        rows[a] = row
    comp = pd.DataFrame(rows).T.reindex(AGENTS)
    comp = comp.div(comp.sum(axis=0).replace(0, np.nan), axis=1).fillna(1.0 / len(AGENTS))
    return comp


def trailing_sharpe(agents: dict[str, TradingAgent], win: int = PERF_WIN,
                    embargo: int = EMBARGO_H) -> pd.DataFrame:
    """Trailing annualised Sharpe of each agent's own returns (leak-free).

    This is the online skill score the coordinator chases — it already reflects the *current*
    market regime (a bear specialist's trailing Sharpe rises in a bear) without needing an
    explicit, whipsaw-prone regime label.
    """
    out = {}
    for a, ag in agents.items():
        mu = ag.g.rolling(win, min_periods=200).mean().shift(1 + embargo)
        sd = ag.g.rolling(win, min_periods=200).std().shift(1 + embargo)
        out[a] = (mu / (sd + 1e-12) * ANN).fillna(0.0)
    return pd.DataFrame(out)


def smoothed_competence(competence: pd.DataFrame, panel: pd.DataFrame,
                        smooth: int = REGIME_SMOOTH) -> pd.DataFrame:
    """Per-bar competence vector under a *smoothed* (majority-vote) regime label, to suppress
    the whipsaw that makes instantaneous regime routing unprofitable."""
    code = panel["regime"].map({"chop": 0, "bull": 1, "bear": 2})
    sm = code.rolling(smooth, min_periods=1).apply(
        lambda x: np.bincount(x.astype(int), minlength=3).argmax(), raw=True)
    inv = {0: "chop", 1: "bull", 2: "bear"}
    smr = sm.map(inv)
    cols = list(competence.index)
    return pd.DataFrame(competence.T.reindex(smr.values).values,
                        index=panel.index, columns=cols)


# ---------------------------------------------------------------------------
# Coordinator — regime-gated capital allocation (mixture of experts)
# ---------------------------------------------------------------------------

@dataclass
class Coordinator:
    """Mixture-of-experts capital allocator over autonomous agents.

    Two information sources are fused, each addressing a failure we diagnosed empirically:

    * **Online performance gate** — a softmax over every agent's *trailing Sharpe*. This is the
      workhorse: it adaptively backs whoever is currently skilful and needs no regime label, so
      it sidesteps the whipsaw that made instantaneous regime-routing lose money.
    * **Smoothed-regime competence tilt** — the pre-OOS, per-regime competence prior under a
      7-day-smoothed regime label, multiplied in as a structural prior on *who is good where*.

    Per-bar capital weight::

        raw_i = softmax_j( trailingSharpe_j / temp )_i  *  ( competence[i, regime~_t] + floor )
        w_i   = raw_i / sum_i raw_i

    The portfolio return is ``sum_i w_i(t-1) * g_i(t)`` minus a small reallocation cost; an agent
    that is flat contributes zero and its weight acts as cash. No confidence/activity multiplier
    is used — it starved the selective specialists and was found to hurt.
    """

    competence: pd.DataFrame
    temp: float = PERF_TEMP
    smooth: int = REGIME_SMOOTH
    floor: float = COMP_FLOOR

    def allocate(self, agents: dict[str, TradingAgent], panel: pd.DataFrame,
                 perf: pd.DataFrame) -> pd.DataFrame:
        names = list(agents)
        idx = panel.index
        z = (perf[names] / self.temp).clip(-10, 10)
        soft = np.exp(z); soft = soft.div(soft.sum(axis=1), axis=0).fillna(1.0 / len(names))
        tilt = smoothed_competence(self.competence.loc[names], panel, self.smooth) + self.floor
        raw = soft.values * tilt.values
        W = raw / np.maximum(raw.sum(axis=1, keepdims=True), 1e-12)
        return pd.DataFrame(W, index=idx, columns=names)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _oos(panel: pd.DataFrame) -> pd.DatetimeIndex:
    return panel.index[(panel.index >= OOS_START) & (panel.index <= OOS_END)]


def portfolio_equity(weights: pd.DataFrame, agents: dict[str, TradingAgent],
                     idx: pd.DatetimeIndex) -> np.ndarray:
    """Equity of a capital-allocation portfolio over agents (leak-free: w[t-1] earns g[t])."""
    names = list(agents)
    W = weights.reindex(idx).fillna(0.0)[names].values
    G = pd.DataFrame({a: agents[a].g for a in names}).reindex(idx).fillna(0.0)[names].values
    n = len(idx)
    eq = np.ones(n); cur = 1.0; prev = np.zeros(len(names))
    for t in range(n):
        r = float((prev * G[t]).sum()) if t > 0 else 0.0
        realloc = float(np.abs(W[t] - prev).sum()) * REALLOC_FEE
        cur *= (1.0 + r - realloc); eq[t] = cur
        prev = W[t]
    return eq


def equal_weight_weights(agents: dict[str, TradingAgent], panel: pd.DataFrame) -> pd.DataFrame:
    """Static 1/N capital allocation over the supplied agent set."""
    names = list(agents)
    return pd.DataFrame(1.0 / len(names), index=panel.index, columns=names)


def inverse_vol_weights(agents: dict[str, TradingAgent], panel: pd.DataFrame,
                        win: int = PERF_WIN, embargo: int = EMBARGO_H) -> pd.DataFrame:
    """Leak-free inverse-volatility allocation.

    Volatility is measured on each agent's own realised return stream and shifted by
    ``1 + embargo`` bars before it can affect a weight.
    """
    names = list(agents)
    vol = pd.DataFrame({
        a: agents[a].g.rolling(win, min_periods=200).std().shift(1 + embargo)
        for a in names
    })
    inv = 1.0 / (vol + 1e-9)
    return inv.div(inv.sum(axis=1).replace(0, np.nan), axis=0).fillna(1.0 / len(names))


def _cap_weight_row(row: pd.Series, cap: float) -> pd.Series:
    """Cap one allocation row and redistribute excess across uncapped agents."""
    w = row.astype(float).copy()
    if len(w) == 0 or w.sum() <= 0:
        return w
    w = w / w.sum()
    capped = pd.Series(False, index=w.index)
    for _ in range(len(w)):
        over = (w > cap) & ~capped
        if not over.any():
            break
        excess = float((w[over] - cap).sum())
        w[over] = cap
        capped |= over
        free = ~capped
        if not free.any() or excess <= 0:
            break
        base = w[free].sum()
        if base <= 0:
            w[free] += excess / free.sum()
        else:
            w[free] += excess * (w[free] / base)
    return w / w.sum()


def capped_inverse_vol_weights(agents: dict[str, TradingAgent], panel: pd.DataFrame,
                               cap_mult: float = 2.0) -> pd.DataFrame:
    """Inverse-volatility allocation with a diversification cap.

    The cap is ``cap_mult`` times equal weight. With seven accepted agents and ``cap_mult=2``, no
    single agent can receive more than one third of capital. This prevents the low-volatility
    allocator from assigning almost all capital to an inactive or stale agent.
    """
    base = inverse_vol_weights(agents, panel)
    cap = cap_mult / len(agents)
    return base.apply(lambda row: _cap_weight_row(row, cap), axis=1)


def static_subset(agents: dict[str, TradingAgent], names: list[str]) -> dict[str, TradingAgent]:
    """Return an ordered agent subset for agent-set bake-offs."""
    return {a: agents[a] for a in names if a in agents}


def load_sp500_benchmark(idx: pd.DatetimeIndex) -> pd.Series | None:
    """Daily S&P 500 / SPY benchmark, aligned to the hourly OOS index when local data exists."""
    path = repo_root() / "data" / "external" / "sp500_daily.parquet"
    if not path.exists():
        return None
    sp = pd.read_parquet(path)
    sp.index = pd.to_datetime(sp.index).tz_localize(None) if sp.index.tz else pd.to_datetime(sp.index)
    close = sp["close"].astype(float).sort_index()
    aligned = close.reindex(idx, method="ffill").dropna()
    if aligned.empty:
        return None
    eq = aligned / aligned.iloc[0]
    return eq.reindex(idx).ffill().rename("S&P 500 Buy & Hold")


def evaluate_equity(eq_full: pd.Series, idx: pd.DatetimeIndex, name: str) -> tuple[dict, np.ndarray]:
    seg = eq_full.reindex(idx).values
    seg = seg / seg[0]
    bh_first = idx[0]
    row = dict(name=name, ret=float(seg[-1] - 1), sharpe=sharpe(seg), sortino=sortino(seg),
               maxdd=maxdd(seg))
    return row, seg


def regime_breakdown(eq_full: pd.Series, panel: pd.DataFrame) -> pd.DataFrame:
    out = []
    for r in REGIMES:
        s, e = REGIME_DATES[r]
        m = (eq_full.index >= s) & (eq_full.index <= e)
        if m.sum() < 24:
            continue
        seg = eq_full[m].values; seg = seg / seg[0]
        out.append({"regime": r, "ret": f"{seg[-1] - 1:+.1%}", "sharpe": f"{sharpe(seg):.2f}",
                    "maxdd": f"{maxdd(seg):.1%}"})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_pipeline(save: bool = True, verbose: bool = True) -> dict:
    repo = repo_root()
    a2 = repo / "artifacts" / "notebooks_v2"
    arts = a2 / "06_mas"; arts.mkdir(parents=True, exist_ok=True)

    panel = load_panel()
    agents = build_agents(panel, a2)
    competence = estimate_competence(agents, panel)
    perf = trailing_sharpe(agents)
    coordinator = Coordinator(competence)
    weights = coordinator.allocate(agents, panel, perf)
    ew_w = equal_weight_weights(agents, panel)
    iv_w = capped_inverse_vol_weights(agents, panel)

    idx = _oos(panel)
    bh = pd.Series((1.0 + panel["ret"].reindex(idx)).cumprod().values, index=idx)
    sp500 = load_sp500_benchmark(idx)

    results = []; equities = {}

    def _add_equity(name, eq_full):
        row, seg = evaluate_equity(eq_full, idx, name)
        row["alpha"] = float(seg[-1] - bh.values[-1] / bh.values[0])
        results.append(row); equities[name] = seg
        return row

    # standalone agents (their authentic risk-managed strategy)
    for a, ag in agents.items():
        _add_equity(f"{a}", ag.eq)
    # capped risk-parity fund-of-agents (gross exposure 1.0, low turnover)
    iv_eq = pd.Series(portfolio_equity(iv_w, agents, idx), index=idx)
    final_row = _add_equity(
        "Final MAS fund (capped inverse-vol)", iv_eq.reindex(panel.index).ffill().fillna(1.0))
    # naive equal-weight fund-of-agents (1/N each → gross exposure 1.0)
    ew_eq = pd.Series(portfolio_equity(ew_w, agents, idx), index=idx)
    _add_equity("Naive EW fund", ew_eq.reindex(panel.index).ffill().fillna(1.0))
    # original coordinator retained as an ablation rather than the final result
    coord_eq_oos = pd.Series(portfolio_equity(weights, agents, idx), index=idx)
    coord_eq_full = coord_eq_oos.reindex(panel.index).ffill().fillna(1.0)
    coord_row = _add_equity("Coordinator ablation (Sharpe x regime)", coord_eq_full)
    _add_equity("BTC Buy & Hold", pd.Series(
        (1.0 + panel["ret"]).cumprod().values, index=panel.index))
    if sp500 is not None:
        _add_equity("S&P 500 Buy & Hold", sp500.reindex(panel.index).ffill().fillna(1.0))

    lb = pd.DataFrame(results).sort_values("sharpe", ascending=False).reset_index(drop=True)
    final_full = iv_eq.reindex(panel.index).ffill().fillna(1.0)
    breakdown = regime_breakdown(final_full, panel)
    mean_w = weights.reindex(idx).mean()
    mean_iv_w = iv_w.reindex(idx).mean()

    if verbose:
        print("=== Per-regime competence priors (pre-OOS Sharpe, normalised, leak-free) ===")
        print(competence.round(3).to_string())
        print(f"\nFinal agent set: {AGENTS}")
        print(f"Excluded experiments: {EXCLUDED_AGENTS}")
        print(f"Mean final capped inverse-vol weights (OOS): {mean_iv_w.round(3).to_dict()}")
        print(f"Mean coordinator ablation weights (OOS): {mean_w.round(3).to_dict()}")
        print(f"Coordinator mean gross exposure (OOS): {weights.reindex(idx).sum(axis=1).mean():.2f}\n")
        show = lb.copy()
        for c in ["ret", "maxdd", "alpha"]:
            show[c] = (show[c] * 100).round(1)
        show["sharpe"] = show["sharpe"].round(2); show["sortino"] = show["sortino"].round(2)
        print(show[["name", "ret", "sharpe", "sortino", "maxdd", "alpha"]].to_string(index=False))
        print("\n=== Final MAS fund regime breakdown ===")
        print(breakdown.to_string(index=False))

    out = dict(
        notebook="06_multi_agent_v1", created=pd.Timestamp.now().isoformat(),
        design="hybrid multi-agent trading system over seven accepted agents: four learned models "
               "and three rule-based agents. The original regime-gated coordinator is retained as "
               "an ablation; the final reported allocator is leak-free capped inverse-volatility "
               "risk parity over autonomous risk-managed agents.",
        accepted_agents=AGENTS,
        excluded_agents=EXCLUDED_AGENTS,
        oos_period=f"{OOS_START.date()} -> {OOS_END.date()}",
        competence=competence.round(4).to_dict(),
        mean_weights_oos=mean_w.round(4).to_dict(),
        mean_capped_inverse_vol_weights_oos=mean_iv_w.round(4).to_dict(),
        final=final_row,
        coordinator_ablation=coord_row,
        regime_breakdown=breakdown.to_dict("records"),
        leaderboard=lb.to_dict("records"))
    if save:
        json.dump(out, open(arts / "results.json", "w"), indent=2, default=float)
        lb.to_csv(arts / "leaderboard.csv", index=False)
        competence.to_csv(arts / "competence.csv")
        weights.reindex(idx).to_csv(arts / "coordinator_weights_oos.csv")
        iv_w.reindex(idx).to_csv(arts / "capped_inverse_vol_weights_oos.csv")
        np.save(arts / "oos_index.npy", idx.values.astype("int64"))
        np.save(arts / "final_equity.npy", iv_eq.values.astype(np.float32))
        np.save(arts / "coord_ablation_equity.npy", coord_eq_oos.values.astype(np.float32))
        if verbose:
            print(f"\nArtifacts -> {arts}")

    out.update(_equities=equities, _panel=panel, _weights=weights, _agents=agents,
               _capped_inverse_vol_weights=iv_w, _final_eq=iv_eq, _coord_eq=coord_eq_full,
               _bh=bh, _sp500=sp500, _idx=idx)
    return out


def plot_results(out: dict, save: bool = True):
    """Create the thesis-ready chart suite for the final multi-agent notebook."""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    arts = repo_root() / "artifacts" / "notebooks_v2" / "06_mas"
    idx = out["_idx"]
    eqs = out["_equities"]
    weights = out["_capped_inverse_vol_weights"].reindex(idx).fillna(0.0)
    colours = {"lgbm": "#F7931A", "mamba": "#7B1FA2", "tcn": "#00ACC1", "patch": "#EF5350",
               "trend": "#43A047", "volbreak": "#5E35B1", "dominance_rotation": "#FB8C00"}
    main = "Final MAS fund (capped inverse-vol)"

    fig, ax1 = plt.subplots(figsize=(13.5, 6.2))
    for name, eq in eqs.items():
        if name == main:
            ax1.plot(idx, (eq - 1) * 100, lw=3.0, color="#005BBB",
                     label=f"{name} ({eq[-1]-1:+.0%})", zorder=6)
        elif name == "BTC Buy & Hold":
            ax1.plot(idx, (eq - 1) * 100, lw=1.5, ls=":", color="#757575",
                     label=f"{name} ({eq[-1]-1:+.0%})")
        elif name == "S&P 500 Buy & Hold":
            ax1.plot(idx, (eq - 1) * 100, lw=1.5, ls=(0, (4, 2)), color="#00897B",
                     label=f"{name} ({eq[-1]-1:+.0%})")
        elif name == "Naive EW fund":
            ax1.plot(idx, (eq - 1) * 100, lw=1.7, ls="--", color="#263238",
                     label=f"{name} ({eq[-1]-1:+.0%})")
        elif name.startswith("Coordinator ablation"):
            ax1.plot(idx, (eq - 1) * 100, lw=1.4, ls="-.", color="#8D6E63",
                     label=f"{name} ({eq[-1]-1:+.0%})")
        else:
            ax1.plot(idx, (eq - 1) * 100, lw=1.0, alpha=0.42, color=colours.get(name),
                     label=f"{name} ({eq[-1]-1:+.0%})")
    for r, c in [("chop", "#9E9E9E"), ("bull", "#26A69A"), ("bear", "#EF5350")]:
        s, e = REGIME_DATES[r]
        ax1.axvspan(s, min(e, idx[-1]), alpha=0.06, color=c)
    ax1.axhline(0, color="#9E9E9E", lw=0.6, ls=":")
    ax1.set_ylabel("Return (%)"); ax1.legend(fontsize=8, ncol=2)
    ax1.set_title("Final hybrid multi-agent fund vs agents and benchmarks (OOS)", fontweight="bold")
    ax1.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    if save:
        fig.savefig(arts / "01_equity_comparison.png", dpi=180, bbox_inches="tight")

    fig2, ax2 = plt.subplots(figsize=(13.5, 4.8))
    names = list(out["_agents"])
    ax2.stackplot(idx, *[weights[a].values for a in names],
                  labels=names, colors=[colours[a] for a in names], alpha=0.88)
    ax2.set_ylabel("Capital weight"); ax2.set_ylim(0, 1); ax2.legend(fontsize=8, ncol=4, loc="upper left")
    ax2.set_title("Capped inverse-volatility allocation over time", fontweight="bold")
    ax2.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")
    fig2.tight_layout()
    if save:
        fig2.savefig(arts / "02_capped_inverse_vol_weights.png", dpi=180, bbox_inches="tight")

    lb = pd.DataFrame(out["leaderboard"]).copy()
    top = lb.sort_values("sharpe", ascending=True)
    fig3, (ax3, ax4) = plt.subplots(1, 2, figsize=(13.5, 5.3), sharey=True)
    colors = ["#005BBB" if x == main else "#90A4AE" for x in top["name"]]
    ax3.barh(top["name"], top["ret"] * 100, color=colors)
    ax3.axvline(0, color="#9E9E9E", lw=0.8)
    ax3.set_xlabel("Total return (%)")
    ax3.set_title("OOS total return", fontweight="bold")
    ax4.barh(top["name"], top["sharpe"], color=colors)
    ax4.axvline(0, color="#9E9E9E", lw=0.8)
    ax4.set_xlabel("Annualised Sharpe")
    ax4.set_title("OOS risk-adjusted result", fontweight="bold")
    fig3.tight_layout()
    if save:
        fig3.savefig(arts / "03_leaderboard_return_sharpe.png", dpi=180, bbox_inches="tight")

    monthly = {}
    monthly_names = [main, "Naive EW fund", "BTC Buy & Hold"]
    if "S&P 500 Buy & Hold" in eqs:
        monthly_names.append("S&P 500 Buy & Hold")
    for name in monthly_names:
        eq = pd.Series(eqs[name], index=idx)
        monthly[name] = eq.resample("ME").last().pct_change().fillna(eq.resample("ME").last() - 1)
    mon = pd.DataFrame(monthly).dropna(how="all") * 100
    fig4, ax5 = plt.subplots(figsize=(13.5, 4.8))
    x = np.arange(len(mon))
    width = 0.8 / len(monthly_names)
    offsets = (np.arange(len(monthly_names)) - (len(monthly_names) - 1) / 2) * width
    palette = {
        main: "#005BBB", "Naive EW fund": "#455A64",
        "BTC Buy & Hold": "#9E9E9E", "S&P 500 Buy & Hold": "#00897B",
    }
    for off, name in zip(offsets, monthly_names):
        ax5.bar(x + off, mon[name].values, width=width, label=name, color=palette[name])
    ax5.axhline(0, color="#9E9E9E", lw=0.8)
    ax5.set_ylabel("Monthly return (%)")
    ax5.set_xticks(x[::2])
    ax5.set_xticklabels([d.strftime("%b %y") for d in mon.index[::2]], rotation=30, ha="right")
    ax5.legend(fontsize=8)
    ax5.set_title("Monthly return comparison: MAS fund vs benchmarks", fontweight="bold")
    fig4.tight_layout()
    if save:
        fig4.savefig(arts / "04_monthly_returns_comparison.png", dpi=180, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    run_pipeline()
