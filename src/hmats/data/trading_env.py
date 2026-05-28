"""Gymnasium trading environment for training RL / neuroevolution agents."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass
class TradingEnvConfig:
    fee: float = 0.0005
    start_cash: float = 1.0
    max_episode_steps: int | None = None


class TradingEnv(gym.Env):
    """Binary-position trading environment.

    Actions: 0 = hold, 1 = buy (all in), 2 = sell (all out).
    Reward: log equity return per step.
    Observation: ``features + [position_flag, cash_frac]``.
    """

    metadata: ClassVar[dict] = {"render_modes": []}

    def __init__(
        self,
        features: np.ndarray,
        prices: np.ndarray,
        cfg: TradingEnvConfig,
        seed: int = 0,
    ) -> None:
        super().__init__()
        assert len(features) == len(prices)
        assert len(features) >= 10, "Not enough data."

        self.features = features.astype(np.float32)
        self.prices = prices.astype(np.float32)
        self.cfg = cfg

        self.n_steps_data = len(self.features)
        self.max_steps = cfg.max_episode_steps or (self.n_steps_data - 1)

        self.obs_dim = self.features.shape[1] + 2

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

        self._rng = np.random.default_rng(seed)
        self._seed = seed

        # Instance attributes initialised in reset()
        self.t: int = 0
        self.cash: float = 0.0
        self.units: float = 0.0
        self.num_trades: int = 0
        self.steps_in_pos: int = 0
        self.pos_history: list[int] = []
        self.equity_curve: list[float] = []
        self.returns: list[float] = []

        self.reset(seed=seed)

    def _equity(self, price: float) -> float:
        return float(self.cash + self.units * price)

    def _get_obs(self) -> np.ndarray:
        price = float(self.prices[self.t])
        eq = self._equity(price)
        pos_flag = 1.0 if self.units > 0 else 0.0
        cash_frac = float(self.cash / (eq + 1e-12))
        return np.concatenate(
            [self.features[self.t], np.array([pos_flag, cash_frac], dtype=np.float32)]
        ).astype(np.float32)

    def reset(  # type: ignore[override]
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        self.steps_in_pos = 0
        self.pos_history = []
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._seed = seed

        self.t = 0
        self.cash = float(self.cfg.start_cash)
        self.units = 0.0
        self.num_trades = 0

        price0 = float(self.prices[self.t])
        self.equity_curve = [self._equity(price0)]
        self.returns = []

        obs = self._get_obs()
        return obs, {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self.action_space.contains(action)
        fee = float(self.cfg.fee)
        price = float(self.prices[self.t])
        eq_before = self._equity(price)

        if action == 1 and self.units == 0.0 and self.cash > 0.0:
            spend = self.cash * (1.0 - fee)
            self.units = spend / price
            self.cash = 0.0
            self.num_trades += 1
        elif action == 2 and self.units > 0.0:
            proceeds = (self.units * price) * (1.0 - fee)
            self.cash = proceeds
            self.units = 0.0
            self.num_trades += 1

        in_pos = 1 if self.units > 0.0 else 0
        self.steps_in_pos += in_pos
        self.pos_history.append(in_pos)

        self.t += 1
        terminated = self.t >= self.n_steps_data - 1 or self.t >= self.max_steps
        truncated = False

        next_price = float(self.prices[self.t])
        eq_after = self._equity(next_price)

        step_log_ret = float(np.log((eq_after + 1e-12) / (eq_before + 1e-12)))

        self.equity_curve.append(eq_after)
        self.returns.append(step_log_ret)

        obs = self._get_obs()
        info = {"equity": eq_after, "num_trades": self.num_trades, "t": self.t}
        return obs, step_log_ret, terminated, truncated, info


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def max_drawdown(equity: np.ndarray) -> float:
    equity = np.asarray(equity, dtype=np.float64)
    peak = np.maximum.accumulate(equity)
    dd = equity / (peak + 1e-12) - 1.0
    return float(dd.min())


def annualization_factor(interval: str) -> float:
    if interval.endswith("m"):
        minutes = int(interval[:-1])
        return (24 * 60 / minutes) * 365
    if interval.endswith("h"):
        hours = int(interval[:-1])
        return (24 / hours) * 365
    if interval.endswith("d"):
        days = int(interval[:-1])
        return 365 / days
    return 365.0


def sharpe_annualized(log_returns: np.ndarray, ann_factor: float) -> float:
    r = np.asarray(log_returns, dtype=np.float64)
    if len(r) < 2:
        return 0.0
    mu = r.mean()
    sd = r.std(ddof=1) + 1e-12
    return float((mu / sd) * math.sqrt(ann_factor))


def evaluate_policy(
    env: TradingEnv,
    act_fn,
    seed: int = 42,
) -> dict:
    """Run *act_fn* through *env* and return summary metrics."""
    obs, _ = env.reset(seed=seed)
    done = False
    while not done:
        action = act_fn(obs)
        obs, _, terminated, truncated, _ = env.step(int(action))
        done = terminated or truncated

    equity = np.array(env.equity_curve, dtype=np.float64)
    log_rets = np.array(env.returns, dtype=np.float64)
    ann = annualization_factor("1h")

    return {
        "final_equity": float(equity[-1]),
        "total_return": float(equity[-1] - equity[0]),
        "sharpe": sharpe_annualized(log_rets, ann),
        "max_drawdown": max_drawdown(equity),
        "num_trades": int(env.num_trades),
        "equity_curve": equity,
        "log_returns": log_rets,
    }


def buy_and_hold_metrics(prices: np.ndarray, fee: float = 0.0005) -> dict:
    prices = np.asarray(prices, dtype=np.float64)
    equity0 = 1.0
    units = (equity0 * (1.0 - fee)) / prices[0]
    equity_end = (units * prices[-1]) * (1.0 - fee)
    eq_curve = units * prices
    log_rets = np.log((eq_curve[1:] + 1e-12) / (eq_curve[:-1] + 1e-12))
    ann = annualization_factor("1h")

    return {
        "final_equity": float(equity_end),
        "total_return": float(equity_end - equity0),
        "sharpe": sharpe_annualized(log_rets, ann),
        "max_drawdown": max_drawdown(eq_curve),
        "num_trades": 2,
        "equity_curve": eq_curve,
        "log_returns": log_rets,
    }


def neat_fitness_from_episode(env: TradingEnv) -> float:
    """Balanced fitness function for NEAT genome evaluation."""
    equity = np.array(env.equity_curve, dtype=np.float64)
    log_rets = np.array(env.returns, dtype=np.float64)

    steps = max(1, len(log_rets))
    mean_r = float(np.mean(log_rets)) if len(log_rets) else 0.0
    vol_r = float(np.std(log_rets)) if len(log_rets) else 0.0
    mdd = abs(max_drawdown(equity))

    trades = env.num_trades
    trades_per_1000 = trades / steps * 1000.0

    target = 30.0
    tol = 25.0
    activity_score = math.exp(-(((trades_per_1000 - target) / tol) ** 2))

    exposure = getattr(env, "steps_in_pos", 0) / (steps + 1e-12)
    exposure_target = 0.5
    exposure_tol = 0.4
    exposure_score = math.exp(-(((exposure - exposure_target) / exposure_tol) ** 2))

    fitness = (
        400.0 * mean_r - 80.0 * vol_r - 2.0 * mdd + 2.0 * activity_score + 1.0 * exposure_score
    )

    if trades == 0:
        fitness -= 0.5
    if exposure < 0.01:
        fitness -= 0.5

    if not np.isfinite(fitness):
        return -1e9
    return float(fitness)
