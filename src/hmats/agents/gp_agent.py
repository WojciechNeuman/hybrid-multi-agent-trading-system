"""Genetic Programming Trading Agent (DEAP).

Evolves interpretable symbolic rule trees over a stationary indicator diet.
M1Y Walk-Forward Optimization produces OOS discrete actions {-1, 0, 1}.

AFML defenses
─────────────
A. Feature stationarity (Ch. 2):  bounded oscillator diet, no raw prices.
B. Purged fitness evaluation (Ch. 7): last ``embargo`` bars excluded from
   fitness computation, preventing leakage into the adjacent OOS fold.
C. Parsimony pressure (Ch. 14):  fitness = Sharpe − α × n_nodes, which
   penalises complex trees that likely overfit via selection bias.

Output
──────
pd.Series named ``gp_action`` with values in {−1, 0, 1} (Short/Flat/Long),
index-aligned to the OOS DatetimeIndex.  Compatible with MetaSupervisoryAgent.
"""

from __future__ import annotations

import operator
import warnings
from typing import Any

import numpy as np
import pandas as pd
from deap import algorithms, base, creator, gp, tools

warnings.filterwarnings("ignore")

# ── GP-specific feature diet ──────────────────────────────────────────────────
# Stationary, bounded indicators distinct from LGBM's 11 features and DRL's 19.
# All are bounded oscillators or normalised volume metrics — no raw price levels.
GP_FEATURES: list[str] = [
    "rsi_7",           # short-term momentum oscillator  [0, 100]
    "rsi_21",          # medium-term momentum            [0, 100]
    "stoch_k_21",      # slow stochastic %K 21-period    [0, 100]
    "bb_position_50",  # close position in 50-period BB  [-1, 1] approx
    "bb_squeeze_20",   # Bollinger squeeze flag           {0, 1}
    "mfi_14",          # Money Flow Index                 [0, 100]
    "williams_r",      # Williams %R                      [-100, 0]
    "cmf_20",          # Chaikin Money Flow               [-1, 1]
    "macd_hist_12_26", # MACD histogram (12/26, not 5/13) unbounded
    "hl_position_24h", # close in 24h H-L range           [0, 1]
    "obv_z_72",        # OBV volume momentum z-score      unbounded
    "vol_z_24h",       # volume spike z-score             unbounded
]

# ── Numpy-vectorised primitives ───────────────────────────────────────────────

def _protdiv(a: Any, b: Any) -> Any:
    """Protected division — returns 1.0 where denominator is near zero."""
    if isinstance(b, np.ndarray):
        safe = np.where(np.abs(b) > 1e-8, b, 1.0)
        return a / safe
    return a / b if abs(float(b)) > 1e-8 else (np.ones_like(a) if isinstance(a, np.ndarray) else 1.0)


def _ifpos(cond: Any, true_val: Any, false_val: Any) -> Any:
    """If cond > 0 return true_val else false_val — vectorised via np.where."""
    if isinstance(cond, np.ndarray):
        return np.where(cond > 0, true_val, false_val)
    return true_val if float(cond) > 0 else false_val


def _gt(a: Any, b: Any) -> Any:
    """Greater-than returning float {1.0, 0.0} for use in arithmetic trees."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return (np.asarray(a) > np.asarray(b)).astype(float)
    return 1.0 if float(a) > float(b) else 0.0


def _lt(a: Any, b: Any) -> Any:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return (np.asarray(a) < np.asarray(b)).astype(float)
    return 1.0 if float(a) < float(b) else 0.0


def _and(a: Any, b: Any) -> Any:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return ((np.asarray(a) > 0) & (np.asarray(b) > 0)).astype(float)
    return 1.0 if (float(a) > 0 and float(b) > 0) else 0.0


def _or(a: Any, b: Any) -> Any:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return ((np.asarray(a) > 0) | (np.asarray(b) > 0)).astype(float)
    return 1.0 if (float(a) > 0 or float(b) > 0) else 0.0


def _not(a: Any) -> Any:
    if isinstance(a, np.ndarray):
        return (np.asarray(a) <= 0).astype(float)
    return 1.0 if float(a) <= 0 else 0.0


# ── Primitive set builder ─────────────────────────────────────────────────────

def _build_pset(features: list[str]) -> gp.PrimitiveSet:
    pset = gp.PrimitiveSet("MAIN", len(features))
    # Arithmetic
    pset.addPrimitive(np.add,      2, name="Add")
    pset.addPrimitive(np.subtract, 2, name="Sub")
    pset.addPrimitive(np.multiply, 2, name="Mul")
    pset.addPrimitive(_protdiv,    2, name="Div")
    # Logical / relational
    pset.addPrimitive(_gt,         2, name="GT")
    pset.addPrimitive(_lt,         2, name="LT")
    pset.addPrimitive(_and,        2, name="And")
    pset.addPrimitive(_or,         2, name="Or")
    pset.addPrimitive(_not,        1, name="Not")
    # Conditional
    pset.addPrimitive(_ifpos,      3, name="IfPos")
    # Ephemeral constants — small bounded range to stay stationary
    pset.addEphemeralConstant("RC", lambda: round(np.random.uniform(-3.0, 3.0), 2))
    # Rename ARG0..N to feature names
    for i, feat in enumerate(features):
        pset.renameArguments(**{f"ARG{i}": feat})
    return pset


def _build_pset_logic_only(features: list[str]) -> gp.PrimitiveSet:
    """Logic-gate-only primitive set: no arithmetic scaling, no ephemeral constants.

    Allowed primitives produce outputs in {0.0, 1.0}.  Sub(bool_a, bool_b) extends
    this to {-1, 0, 1}, giving Short/Flat/Long signals without allowing the
    coefficient-inflation attack seen in v1 (where Sub(rsi_7, rsi_21) × large
    constant dominated every fold).

    Banned: Add, Mul, Div, and ephemeral constants.
    """
    pset = gp.PrimitiveSet("MAIN", len(features))
    # Sub only — allows -1 via Sub(GT(a,b), GT(c,d)) but prevents coefficient scaling
    pset.addPrimitive(np.subtract, 2, name="Sub")
    # Relational (return {0.0, 1.0})
    pset.addPrimitive(_gt,         2, name="GT")
    pset.addPrimitive(_lt,         2, name="LT")
    # Boolean gates (return {0.0, 1.0})
    pset.addPrimitive(_and,        2, name="And")
    pset.addPrimitive(_or,         2, name="Or")
    pset.addPrimitive(_not,        1, name="Not")
    # Conditional selector
    pset.addPrimitive(_ifpos,      3, name="IfPos")
    # No ephemeral constants — prevents tree from inflating outputs to bypass thresholds
    for i, feat in enumerate(features):
        pset.renameArguments(**{f"ARG{i}": feat})
    return pset


# ── DEAP creator setup (module-level, idempotent) ────────────────────────────

def _ensure_creator() -> None:
    if not hasattr(creator, "GPFitnessMax"):
        creator.create("GPFitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "GPIndividual"):
        creator.create("GPIndividual", gp.PrimitiveTree, fitness=creator.GPFitnessMax)


# ── Agent ─────────────────────────────────────────────────────────────────────

class GPTradingAgent:
    """Genetic Programming trading agent with M1Y WFO and AFML defences.

    Parameters
    ----------
    features:
        Input feature columns.  Defaults to the 12-feature GP-specific diet.
    population_size:
        Number of individuals per generation.
    generations:
        Number of evolutionary generations per WFO fold.
    parsimony_coefficient:
        Alpha — penalises tree complexity: fitness = Sharpe − α × n_nodes.
    train_window_h:
        Training window per fold in bars (default 8760 = 1 year of hourly).
    step_size:
        OOS step per fold in bars (default 720 = 1 month).
    embargo:
        Bars excluded from both the fitness evaluation tail and OOS head.
    cx_prob:
        Crossover probability.
    mut_prob:
        Mutation probability.
    max_tree_height:
        Hard limit on tree height — bloat control.
    random_seed:
        For reproducibility.
    """

    AGENT_ID   = "gp_v0"
    SIGNAL_COL = "gp_action"

    def __init__(
        self,
        features: list[str] | None = None,
        population_size: int = 300,
        generations: int = 30,
        parsimony_coefficient: float = 0.01,
        train_window_h: int = 8760,
        step_size: int = 720,
        embargo: int = 12,
        cx_prob: float = 0.70,
        mut_prob: float = 0.25,
        max_tree_height: int = 8,
        flat_threshold: float = 0.0,
        logic_only: bool = False,
        agent_id: str | None = None,
        random_seed: int = 42,
    ) -> None:
        self.features             = features or GP_FEATURES
        self.pop_size             = population_size
        self.generations          = generations
        self.alpha                = parsimony_coefficient
        self.train_window_h       = train_window_h
        self.step_size            = step_size
        self.embargo              = embargo
        self.cx_prob              = cx_prob
        self.mut_prob             = mut_prob
        self.max_tree_height      = max_tree_height
        self.flat_threshold       = flat_threshold
        self.logic_only           = logic_only
        self.AGENT_ID             = agent_id or self.__class__.AGENT_ID
        self.random_seed          = random_seed

        self._pset: gp.PrimitiveSet | None = None
        self._toolbox: base.Toolbox | None = None

        # Populated during generate_signals — enables post-hoc tree inspection
        self.fold_winners: list[dict] = []

    # ── DEAP initialisation (lazy, once per instance) ─────────────────────────

    def _setup_deap(self) -> None:
        if self._pset is not None:
            return

        np.random.seed(self.random_seed)
        _ensure_creator()

        pset = _build_pset_logic_only(self.features) if self.logic_only else _build_pset(self.features)
        self._pset = pset

        tb = base.Toolbox()
        tb.register("expr",       gp.genHalfAndHalf, pset=pset, min_=1, max_=4)
        tb.register("individual", tools.initIterate, creator.GPIndividual, tb.expr)
        tb.register("population", tools.initRepeat, list, tb.individual)
        tb.register("select",     tools.selTournament, tournsize=4)
        tb.register("mate",       gp.cxOnePoint)
        tb.register("expr_mut",   gp.genFull, min_=0, max_=3)
        tb.register("mutate",     gp.mutUniform, expr=tb.expr_mut, pset=pset)

        # Bloat control — static height limit on both operators
        height_limit = gp.staticLimit(
            key=operator.attrgetter("height"), max_value=self.max_tree_height
        )
        tb.decorate("mate",   height_limit)
        tb.decorate("mutate", height_limit)

        self._toolbox = tb

    # ── Fitness evaluation ────────────────────────────────────────────────────

    def _evaluate_fitness(
        self,
        individual: gp.PrimitiveTree,
        X: np.ndarray,
        log_rets: np.ndarray,
    ) -> tuple[float]:
        """Constraint B + C: purged Sharpe with parsimony penalty.

        Training data passed in already excludes the last ``embargo`` bars
        (purging applied upstream in ``_evolve``).
        """
        n = len(X)
        if n < 50:
            return (-999.0,)

        try:
            func = gp.compile(individual, self._pset)
            raw = func(*[X[:, j] for j in range(X.shape[1])])
        except Exception:
            return (-999.0,)

        if np.isscalar(raw):
            raw = np.full(n, float(raw))

        raw = np.nan_to_num(
            np.asarray(raw, dtype=np.float64),
            nan=0.0, posinf=1.0, neginf=-1.0,
        )

        if self.flat_threshold > 0:
            signals = np.where(raw > self.flat_threshold, 1.0,
                      np.where(raw < -self.flat_threshold, -1.0, 0.0))
        else:
            signals = np.sign(raw)
        strategy_rets = signals * log_rets

        std = strategy_rets.std(ddof=1)
        if std < 1e-10:
            sharpe = 0.0
        else:
            sharpe = float(strategy_rets.mean() / std * np.sqrt(24 * 365))

        # Parsimony pressure (Constraint C)
        n_nodes = len(individual)
        fitness = sharpe - self.alpha * n_nodes
        return (fitness,)

    # ── Single-fold evolution ─────────────────────────────────────────────────

    def _evolve(
        self,
        X_tr: np.ndarray,
        log_rets_tr: np.ndarray,
        fold: int,
        verbose: bool,
    ) -> tools.HallOfFame:
        """Run the evolutionary loop on one training fold.

        Constraint B (purging): the last ``embargo`` bars are stripped from
        the fitness evaluation data before evolution begins.
        """
        # Purge: exclude last embargo bars so fitness cannot see bars adjacent
        # to the OOS window.
        purged_end = max(len(X_tr) - self.embargo, 100)
        X_fit      = X_tr[:purged_end]
        lr_fit     = log_rets_tr[:purged_end]

        tb = self._toolbox

        def _eval(ind: gp.PrimitiveTree) -> tuple[float]:
            return self._evaluate_fitness(ind, X_fit, lr_fit)

        tb.register("evaluate", _eval)

        pop = tb.population(n=self.pop_size)
        hof = tools.HallOfFame(3)

        stats = tools.Statistics(lambda ind: ind.fitness.values[0] if ind.fitness.valid else float("-inf"))
        stats.register("max", lambda v: float(np.nanmax(v)))
        stats.register("avg", lambda v: float(np.nanmean(v)))

        pop, logbook = algorithms.eaSimple(
            pop, tb,
            cxpb=self.cx_prob, mutpb=self.mut_prob,
            ngen=self.generations,
            stats=stats, halloffame=hof,
            verbose=False,
        )

        if verbose:
            best = hof[0]
            best_fit = best.fitness.values[0]
            print(
                f"  [{self.AGENT_ID}] fold {fold:>3}  "
                f"fitness={best_fit:+.4f}  nodes={len(best):>3}  "
                f"height={best.height}  "
                f"gen_max={logbook[-1]['max']:+.4f}"
            )

        return hof

    # ── Inference ────────────────────────────────────────────────────────────

    def _infer(
        self,
        individual: gp.PrimitiveTree,
        X_oos: np.ndarray,
    ) -> np.ndarray:
        """Apply the best evolved tree to OOS features; returns int8 signals."""
        n = len(X_oos)
        try:
            func = gp.compile(individual, self._pset)
            raw = func(*[X_oos[:, j] for j in range(X_oos.shape[1])])
        except Exception:
            return np.zeros(n, dtype=np.int8)

        if np.isscalar(raw):
            raw = np.full(n, float(raw))

        raw = np.nan_to_num(
            np.asarray(raw, dtype=np.float64),
            nan=0.0, posinf=1.0, neginf=-1.0,
        )
        if self.flat_threshold > 0:
            out = np.where(raw > self.flat_threshold, 1.0,
                  np.where(raw < -self.flat_threshold, -1.0, 0.0))
        else:
            out = np.sign(raw)
        return out.astype(np.int8)

    # ── Public API ────────────────────────────────────────────────────────────

    def _compile_tree(
        self,
        individual: gp.PrimitiveTree,
        data: pd.DataFrame,
    ) -> np.ndarray:
        """Convert a symbolic tree to its output array over ``data``.

        Utility for external inspection / notebook visualisation.
        """
        self._setup_deap()
        X = data[self.features].fillna(0).values
        return self._infer(individual, X)

    def tree_to_str(self, individual: gp.PrimitiveTree) -> str:
        """Return a human-readable infix string of the evolved rule tree."""
        return str(individual)

    def generate_signals(
        self,
        df: pd.DataFrame,
        oos_start: pd.Timestamp,
        verbose: bool = True,
        full_history: bool = False,
    ) -> pd.Series:
        """Run M1Y WFO; return OOS discrete position signals.

        Parameters
        ----------
        df:
            Full DataFrame (training history + OOS).  Must contain all columns
            in ``self.features`` and ``close``.
        oos_start:
            First bar of the true OOS hold-out window.
        verbose:
            Print per-fold progress.
        full_history:
            If True, return valid signals for the entire period (meta-learning).
        """
        self._setup_deap()
        self.fold_winners.clear()

        feats_present = [f for f in self.features if f in df.columns]
        missing = [f for f in self.features if f not in df.columns]
        if missing and verbose:
            print(f"[{self.AGENT_ID}] WARNING: missing features (will be 0): {missing}")

        # Build feature matrix — fill missing columns with 0
        X_raw = df[feats_present].fillna(0).values
        if missing:
            X_full = np.zeros((len(df), len(self.features)), dtype=np.float64)
            idx_map = {f: i for i, f in enumerate(self.features)}
            for j, f in enumerate(feats_present):
                X_full[:, idx_map[f]] = X_raw[:, j]
        else:
            X_full = X_raw.astype(np.float64)

        log_rets = np.log(
            df["close"] / df["close"].shift(1)
        ).fillna(0).values.astype(np.float64)

        n = len(df)
        all_signals = np.zeros(n, dtype=np.int8)
        fold = 0

        i = self.train_window_h
        while i < n:
            tr_start     = max(0, i - self.train_window_h)
            tr_end       = i
            oos_emb      = min(tr_end + self.embargo, n)
            oos_end      = min(i + self.step_size, n)

            if oos_emb >= oos_end:
                i += self.step_size
                continue

            if tr_end - tr_start < 200:
                i += self.step_size
                continue

            X_tr   = X_full[tr_start:tr_end]
            lr_tr  = log_rets[tr_start:tr_end]
            X_oos  = X_full[oos_emb:oos_end]

            fold += 1
            hof = self._evolve(X_tr, lr_tr, fold, verbose)

            best = hof[0]
            oos_sigs = self._infer(best, X_oos)
            all_signals[oos_emb:oos_end] = oos_sigs

            # Record for notebook inspection
            self.fold_winners.append({
                "fold":          fold,
                "train_start":   df.index[tr_start],
                "train_end":     df.index[tr_end - 1],
                "oos_start":     df.index[oos_emb],
                "oos_end":       df.index[oos_end - 1],
                "tree_str":      str(best),
                "n_nodes":       len(best),
                "height":        best.height,
                "fitness":       float(best.fitness.values[0]),
            })

            i += self.step_size

        if verbose:
            oos_mask = df.index >= oos_start
            a = all_signals[oos_mask]
            print(
                f"[{self.AGENT_ID}] WFO done: {fold} folds  "
                f"OOS Long={( a==1).sum()}  Short={(a==-1).sum()}  Flat={(a==0).sum()}"
            )

        full_series = pd.Series(all_signals, index=df.index, name=self.SIGNAL_COL)
        if full_history:
            return full_series
        return full_series[df.index >= oos_start].copy()
