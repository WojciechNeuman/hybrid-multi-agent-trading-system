"""DRL Agent using PPO (Stable-Baselines3).

v1 changes vs v0
----------------
* Randomised episode start in reset() — agent can no longer memorise the fixed
  8 760-bar training sequence.  Each reset draws a random start index so that
  different episodes cover different sub-windows of the training data.
* Capped episode length (default 1 000 bars) — forces the policy to generalise
  across the regime landscape rather than overfit one contiguous year.
* Direct-flip churn penalty — an extra fee is charged whenever the agent goes
  Long→Short or Short→Long without passing through Flat.  This is *on top of*
  the taker fee and discourages the exploitative "perfect prediction every bar"
  strategy that produced the v0 training Sharpe of +15.

M1Y Walk-Forward: trains a fresh PPO agent on each 1-year window, generates
discrete actions for the subsequent OOS month.

Actions: -1=Short, 0=Flat, 1=Long.
Output: pd.Series named ``drl_action`` with integer actions.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO
    _SB3_AVAILABLE = True
except ImportError:
    _SB3_AVAILABLE = False

# ── Fee constants (match LGBM/TCN notebooks) ──────────────────────────────────
_TAKER_FEE = 0.0005       # 0.05% on position changes
_FUNDING_H  = 0.0000077   # +0.00077%/h received on short positions


class MarketEnv(gym.Env):
    """Three-action (Short/Flat/Long) trading environment for RL.

    Observation:
        Rolling window of ``window_size`` bars × ``n_features`` features,
        plus two scalar state features: current position (−1/0/1) and
        normalised unrealised PnL.
        Shape: (window_size * n_features + 2,)

    Action space:
        Discrete(3):  0 = Short, 1 = Flat, 2 = Long

    Reward:
        step_pnl     = position × log_return_t
        fee          = taker_fee   if position changed this step  else 0
        churn_pen    = churn_penalty if direct Long↔Short flip    else 0
        funding      = funding_h   if position == Short           else 0
        reward       = step_pnl + funding - fee - churn_pen
    """

    metadata: dict = {"render_modes": []}

    ACTION_TO_POS = {0: -1, 1: 0, 2: 1}   # discrete action → position multiplier

    def __init__(
        self,
        features: np.ndarray,
        log_returns: np.ndarray,
        window_size: int = 24,
        episode_len: int = 1000,
        churn_penalty: float = 0.001,
        step_penalty: float = 0.0,
        taker_fee: float = _TAKER_FEE,
        funding_h: float = _FUNDING_H,
        seed: int = 42,
        min_hold: int = 0,
    ) -> None:
        super().__init__()
        assert len(features) == len(log_returns), "features and log_returns must align"
        assert len(features) > window_size + 1, "Too few bars for the given window_size"

        self.features = features.astype(np.float32)
        self.log_returns = log_returns.astype(np.float64)
        self.window_size = window_size
        self.episode_len = min(episode_len, len(features) - window_size - 2)
        self.churn_penalty = churn_penalty
        self.step_penalty = step_penalty
        self.taker_fee = taker_fee
        self.funding_h = funding_h
        self.min_hold = min_hold
        self.n_features = features.shape[1]

        obs_dim = window_size * self.n_features + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        # episode state (reset in reset())
        self.t: int = 0
        self.episode_start: int = window_size
        self.position: int = 0
        self.hold_counter: int = 0
        self.entry_log_price: float = 0.0
        self.equity: float = 1.0
        self.eq_history: list[float] = []
        self.reward_history: list[float] = []
        self._rng = np.random.default_rng(seed)

    def _get_obs(self) -> np.ndarray:
        window = self.features[self.t - self.window_size : self.t].flatten()
        unreal_pnl = float(
            self.position * (self.log_returns[self.t - 1] if self.t > 0 else 0.0)
        )
        return np.concatenate(
            [window, np.array([float(self.position) / 1.0, unreal_pnl], dtype=np.float32)]
        ).astype(np.float32)

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Randomised start — the core fix that prevents sequence memorisation.
        # Draw uniformly from all valid start positions in the training window.
        max_start = len(self.features) - self.episode_len - 2
        if max_start > self.window_size:
            self.episode_start = int(
                self._rng.integers(self.window_size, max_start + 1)
            )
        else:
            self.episode_start = self.window_size

        self.t = self.episode_start
        self.position = 0
        self.hold_counter = 0
        self.entry_log_price = 0.0
        self.equity = 1.0
        self.eq_history = [1.0]
        self.reward_history = []
        return self._get_obs(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        new_pos = self.ACTION_TO_POS[int(action)]

        # Hard min-hold: block position changes until hold_counter reaches min_hold.
        # PPO still receives clean PnL rewards — only its physical ability to flip is gated.
        if self.min_hold > 0 and self.position != 0 and self.hold_counter < self.min_hold:
            new_pos = self.position  # override to hold

        pos_changed = new_pos != self.position

        log_ret = float(self.log_returns[self.t]) if self.t < len(self.log_returns) else 0.0
        step_pnl = self.position * log_ret
        fee = self.taker_fee if pos_changed else 0.0
        # Extra penalty for a direct Long↔Short flip (abs diff == 2).
        # This is the "churn" that let v0 achieve astronomical training returns
        # by perfectly timing every reversal.
        churn = (
            self.churn_penalty
            if pos_changed and abs(new_pos - self.position) == 2
            else 0.0
        )
        funding = self.funding_h if self.position == -1 else 0.0
        step_pen = self.step_penalty if pos_changed else 0.0
        reward = float(step_pnl + funding - fee - churn - step_pen)

        # Start at 1 on entry so hold_counter reaches min_hold after exactly min_hold bars held.
        self.hold_counter = 1 if pos_changed else (self.hold_counter + 1 if new_pos != 0 else 0)
        self.position = new_pos
        self.equity *= (1.0 + reward)
        self.eq_history.append(self.equity)
        self.reward_history.append(reward)

        self.t += 1
        # Terminate when the capped episode length is exhausted.
        terminated = self.t >= self.episode_start + self.episode_len
        truncated = False
        obs = self._get_obs() if not terminated else np.zeros(
            self.observation_space.shape, dtype=np.float32
        )
        return obs, reward, terminated, truncated, {"equity": self.equity, "t": self.t}

    def final_sharpe(self) -> float:
        r = np.array(self.reward_history, dtype=np.float64)
        if len(r) < 2:
            return 0.0
        return float(r.mean() / (r.std(ddof=1) + 1e-12) * np.sqrt(24 * 365))


# ── DRL Agent ─────────────────────────────────────────────────────────────────

class DRLAgent:
    """PPO-based trading agent with M1Y walk-forward training.

    Parameters
    ----------
    features:
        Column names to use from the input DataFrame.
    window_size:
        Number of lookback bars fed as observation (default 24 = 1 day).
    episode_len:
        Number of bars per training episode.  Randomised start + short episodes
        force the policy to generalise rather than memorise the training window.
        Default 1000 (≈6 weeks of hourly data).
    churn_penalty:
        Extra fee charged on direct Long↔Short flips (on top of taker_fee).
        Discourages constant direction reversal that exploits the fixed sequence.
        Default 0.001 (0.1 %).
    train_window_h:
        Training window per WFO fold in bars (default 8760 = 1 year).
    step_size:
        OOS step size per fold in bars (default 720 = 1 month).
    total_timesteps:
        PPO training steps per fold (default 300_000).
    ppo_kwargs:
        Extra kwargs forwarded to ``stable_baselines3.PPO``.
    """

    AGENT_ID = "drl_ppo_v1"
    SIGNAL_COL = "drl_action"

    def __init__(
        self,
        features: list[str] | None = None,
        window_size: int = 24,
        episode_len: int = 1000,
        churn_penalty: float = 0.001,
        step_penalty: float = 0.0,
        min_hold: int = 0,
        train_window_h: int = 8760,
        step_size: int = 720,
        total_timesteps: int = 300_000,
        agent_id: str | None = None,
        ppo_kwargs: dict | None = None,
    ) -> None:
        if not _SB3_AVAILABLE:
            raise ImportError(
                "stable-baselines3 and gymnasium are required for DRLAgent. "
                "Run: pip install stable-baselines3 gymnasium"
            )
        self.features = features or _DEFAULT_DRL_FEATURES
        self.window_size = window_size
        self.episode_len = episode_len
        self.churn_penalty = churn_penalty
        self.step_penalty = step_penalty
        self.min_hold = min_hold
        self.train_window_h = train_window_h
        self.step_size = step_size
        self.total_timesteps = total_timesteps
        self.AGENT_ID = agent_id or self.__class__.AGENT_ID
        self.ppo_kwargs = ppo_kwargs or {}

    def _normalise(self, X: np.ndarray) -> np.ndarray:
        """Robust z-score normalisation per feature (fit on train, apply to all)."""
        med = np.nanmedian(X, axis=0)
        mad = np.nanmedian(np.abs(X - med), axis=0)
        scale = np.where(mad > 1e-8, mad * 1.4826, 1.0)
        return np.clip((X - med) / scale, -5.0, 5.0).astype(np.float32)

    def _train_fold(
        self,
        X_train: np.ndarray,
        log_rets_train: np.ndarray,
        verbose: bool,
        fold: int,
    ) -> PPO:
        """Train a fresh PPO agent on one WFO fold."""
        env = MarketEnv(
            features=X_train,
            log_returns=log_rets_train,
            window_size=self.window_size,
            episode_len=self.episode_len,
            churn_penalty=self.churn_penalty,
            step_penalty=self.step_penalty,
            min_hold=self.min_hold,
        )

        default_ppo = dict(
            policy="MlpPolicy",
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=0,
            seed=42,
        )
        default_ppo.update(self.ppo_kwargs)

        model = PPO(env=env, **default_ppo)
        model.learn(
            total_timesteps=self.total_timesteps,
            progress_bar=False,
        )
        if verbose:
            sharpe = env.final_sharpe()
            ret = env.equity - 1.0
            print(
                f"  [{self.AGENT_ID}] fold {fold:>3}  "
                f"train_sharpe={sharpe:+.3f}  train_ret={ret:+.3%}  "
                f"train_bars={len(X_train):,}  episode_len={env.episode_len}"
            )
        return model

    def _infer_fold(
        self,
        model: PPO,
        X_oos: np.ndarray,
        log_rets_oos: np.ndarray,
    ) -> np.ndarray:
        """Run deterministic inference on one OOS fold; returns action array.

        During inference the environment runs sequentially from the start (no
        random restart) so the agent sees the full context prefix before the
        OOS slice begins.
        """
        env = MarketEnv(
            features=X_oos,
            log_returns=log_rets_oos,
            window_size=self.window_size,
            episode_len=len(X_oos),  # capped internally; we override below
            churn_penalty=0.0,       # penalty only during training
            min_hold=self.min_hold,
        )
        env.reset()
        # Force sequential inference from bar window_size through the full array.
        # Override episode_start and episode_len so terminated = (t >= len(X_oos)).
        env.episode_start = self.window_size
        env.episode_len = len(X_oos) - self.window_size
        env.t = self.window_size
        obs = env._get_obs()

        actions: list[int] = []
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(int(action))
            actions.append(int(action))
            done = terminated or truncated
        # Map 0→-1 (short), 1→0 (flat), 2→1 (long)
        return np.array([MarketEnv.ACTION_TO_POS[a] for a in actions], dtype=np.int8)

    def generate_signals(
        self,
        df: pd.DataFrame,
        oos_start: pd.Timestamp,
        verbose: bool = True,
        full_history: bool = False,
    ) -> pd.Series:
        """Run M1Y WFO and return discrete position signals.

        Parameters
        ----------
        full_history:
            If True, return WFO signals for the entire valid period (for meta-learning).
        """
        feats_present = [f for f in self.features if f in df.columns]
        if not feats_present:
            raise ValueError(f"None of the requested features found in df: {self.features[:5]}...")

        X_raw = df[feats_present].fillna(0).values
        log_rets = np.log(df["close"] / df["close"].shift(1)).fillna(0).values

        n = len(df)
        all_actions = np.full(n, 0, dtype=np.int8)
        fold = 0

        # Walk-forward: train on [i - train_window_h, i), predict on [i, i + step_size)
        i = self.train_window_h
        while i < n:
            tr_start = max(0, i - self.train_window_h)
            tr_end = i
            oos_end = min(i + self.step_size, n)

            X_train_fold = self._normalise(X_raw[tr_start:tr_end])
            log_rets_train = log_rets[tr_start:tr_end]
            X_oos_fold = self._normalise(X_raw[tr_start:oos_end])  # include context prefix
            log_rets_oos = log_rets[tr_start:oos_end]

            if len(X_train_fold) < self.window_size + 50:
                i += self.step_size
                continue

            fold += 1
            model = self._train_fold(X_train_fold, log_rets_train, verbose, fold)

            oos_actions = self._infer_fold(model, X_oos_fold, log_rets_oos)

            # The first train_window_h actions correspond to train context; keep only OOS
            n_oos = oos_end - tr_end
            if len(oos_actions) >= n_oos:
                all_actions[tr_end:oos_end] = oos_actions[-n_oos:]
            else:
                all_actions[tr_end : tr_end + len(oos_actions)] = oos_actions

            i += self.step_size

        if verbose:
            oos_mask = df.index >= oos_start
            a_oos = all_actions[oos_mask]
            longs  = (a_oos ==  1).sum()
            shorts = (a_oos == -1).sum()
            flats  = (a_oos ==  0).sum()
            print(
                f"[{self.AGENT_ID}] WFO done: {fold} folds  "
                f"OOS: Long={longs}  Short={shorts}  Flat={flats}"
            )

        full_series = pd.Series(all_actions, index=df.index, name=self.SIGNAL_COL)
        if full_history:
            return full_series
        return full_series[df.index >= oos_start].copy()


# ── Default DRL features (regime + portfolio-state context) ───────────────────
# Different from LGBM's tabular features — DRL benefits from raw, unpruned signals
# that describe market state and volatility regime.
_DEFAULT_DRL_FEATURES: list[str] = [
    # Momentum / mean-reversion
    "ret_1h", "ret_2h", "ret_3h", "ret_6h",
    "rsi_14", "stoch_k_14", "macd_hist_5_13",
    # Volatility / regime
    "atr_14_pct", "vol_ratio_24h", "bb_width_pct", "bb_position_20",
    # Trend
    "close_vs_sma_7", "close_vs_sma_50", "trend_score", "ma_bull_score",
    # Microstructure
    "close_vs_true_vwap", "hurst_24h",
    # Temporal
    "hour_sin", "hour_cos",
]
