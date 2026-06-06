"""DRL Bracket Agent — PPO entry-timer with mechanical ATR-bracket exits (v4).

Why this exists
───────────────
DRL v0–v3 lose money *even at zero fees* (−17% OOS) because the training
environment (``drl_agent.MarketEnv``) lets the policy pick a new position every
bar.  On an hourly series with Hurst ≈ 0.52 that produces a per-bar churner; the
v3 backtest then wraps ATR brackets around that churn and re-enters constantly
(2 484 trades).  The reward the agent optimised (per-step PnL) is dominated by
microstructure noise, so there is no stable gradient toward a real edge.

The reframe
───────────
Decouple *direction* from *holding*, exactly like the LGBM agent that works:

  • The agent only ever decides **WHEN to enter** (long / short / stay flat).
  • Once entered, the trade is **locked**: exit is mechanical (TP / SL / max-hold),
    identical brackets in training and evaluation, so the agent physically cannot
    flip bar-to-bar.  Trade count is bounded by the number of entries.
  • Reward is the **bracketed trade's net return** (fees + funding included),
    distributed as per-bar mark-to-market.  The agent is graded on trade outcomes,
    not on guessing the next bar's tick — i.e. it learns a meta-label
    ("is entering here likely to hit TP before SL?").

The environment is the single source of truth: ``BracketMarketEnv.rollout`` runs
the trained policy sequentially and returns the realised position path, equity,
and trade log, so training and OOS evaluation use the same mechanics (no second,
divergent backtest layer).

Output: ``pd.Series`` named ``drl_action`` ∈ {−1, 0, 1} = position held each bar
(meta-compatible with ``build_signal_df``).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO
    _SB3_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SB3_AVAILABLE = False

_TAKER_FEE = 0.0005
_FUNDING_H = 0.0000077


class BracketMarketEnv(gym.Env):
    """Entry-timer env: agent opens bracketed trades, exits are mechanical.

    Action space — Discrete(3):
        0 = stay flat / no-op   (also the only action that matters mid-trade: ignored)
        1 = enter LONG  (only honoured when flat and cooldown elapsed)
        2 = enter SHORT (only honoured when flat and cooldown elapsed)

    A trade, once opened at ``close[t]``, is held until one of:
        • SL touched intra-bar (low/high vs sl_px)        → realise at sl_px
        • TP touched intra-bar (high/low vs tp_px)        → realise at tp_px
        • hold_cnt >= max_hold                            → realise at close
    subject to ``min_hold`` (no exit before it) then a ``cooldown`` of flat bars.

    Reward (per bar) = position · (close_t/close_{t-1} − 1) + funding − fees.
    Summed over a trade this equals its net return, so the discounted return PPO
    maximises *is* risk-adjusted trade quality.
    """

    metadata: dict = {"render_modes": []}
    ACTION_ENTER = {1: 1, 2: -1}  # action → direction

    def __init__(
        self,
        features: np.ndarray,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        atr_pct: np.ndarray,
        window_size: int = 24,
        episode_len: int = 1000,
        sl_mult: float = 2.0,
        tp_mult: float = 3.0,
        min_hold: int = 4,
        max_hold: int = 24,
        cooldown: int = 2,
        min_sl: float = 0.010,
        taker_fee: float = _TAKER_FEE,
        funding_h: float = _FUNDING_H,
        idle_penalty: float = 0.0,
        seed: int = 42,
    ) -> None:
        super().__init__()
        n = len(features)
        assert len(close) == n == len(high) == len(low) == len(atr_pct)
        assert n > window_size + 2

        self.features = features.astype(np.float32)
        self.close = close.astype(np.float64)
        self.high = high.astype(np.float64)
        self.low = low.astype(np.float64)
        self.atr = atr_pct.astype(np.float64)
        self.window_size = window_size
        self.episode_len = min(episode_len, n - window_size - 2)
        self.sl_mult, self.tp_mult = sl_mult, tp_mult
        self.min_hold, self.max_hold, self.cooldown = min_hold, max_hold, cooldown
        self.min_sl = min_sl
        self.taker_fee, self.funding_h = taker_fee, funding_h
        self.idle_penalty = idle_penalty
        self.n_features = features.shape[1]

        obs_dim = window_size * self.n_features + 3  # +position, +hold frac, +unreal pnl
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        self.action_space = spaces.Discrete(3)
        self._rng = np.random.default_rng(seed)
        self._reset_trade_state()
        self.t = window_size
        self.episode_start = window_size

    # ── trade state ───────────────────────────────────────────────────────────
    def _reset_trade_state(self) -> None:
        self.position = 0          # -1 / 0 / +1
        self.direction = 0
        self.entry_px = 0.0
        self.sl_px = self.tp_px = 0.0
        self.hold_cnt = 0
        self.cd_cnt = 0
        self.unreal = 0.0

    def _get_obs(self) -> np.ndarray:
        w = self.features[self.t - self.window_size:self.t].flatten()
        hold_frac = self.hold_cnt / max(self.max_hold, 1)
        return np.concatenate([
            w, np.array([float(self.position), hold_frac, self.unreal], np.float32)
        ]).astype(np.float32)

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        max_start = len(self.features) - self.episode_len - 2
        self.episode_start = (
            int(self._rng.integers(self.window_size, max_start + 1))
            if max_start > self.window_size else self.window_size
        )
        self.t = self.episode_start
        self._reset_trade_state()
        return self._get_obs(), {}

    # ── core transition (shared by training step and rollout) ─────────────────
    def _advance(self, action: int) -> float:
        """Advance one bar; return this bar's reward. Mutates trade state and self.t."""
        t = self.t
        px = self.close[t]
        prev_px = self.close[t - 1]
        reward = 0.0

        if self.position != 0:
            # mark-to-market for this bar
            bar_ret = (px / prev_px - 1.0) * self.position
            self.hold_cnt += 1
            funding = self.funding_h if self.position < 0 else 0.0
            reward += bar_ret + funding
            self.unreal = (px / self.entry_px - 1.0) * self.position

            exited = False
            if self.hold_cnt >= self.min_hold:
                if self.position == 1:
                    if self.low[t] <= self.sl_px:
                        reward += (self.sl_px / px - 1.0)      # adjust MTM to SL fill
                        exited = True
                    elif self.high[t] >= self.tp_px:
                        reward += (self.tp_px / px - 1.0)
                        exited = True
                else:
                    if self.high[t] >= self.sl_px:
                        reward += -(self.sl_px / px - 1.0)
                        exited = True
                    elif self.low[t] <= self.tp_px:
                        reward += -(self.tp_px / px - 1.0)
                        exited = True
                if not exited and self.hold_cnt >= self.max_hold:
                    exited = True
            if exited:
                reward -= self.taker_fee                       # exit fee
                self._reset_trade_state()
                self.cd_cnt = self.cooldown
        elif self.cd_cnt > 0:
            self.cd_cnt -= 1
            reward -= self.idle_penalty
        else:
            direction = self.ACTION_ENTER.get(int(action), 0)
            if direction != 0 and t + 1 < len(self.close):
                atr = max(float(self.atr[t]) if np.isfinite(self.atr[t]) else self.min_sl,
                          self.min_sl)
                self.position = self.direction = direction
                self.entry_px = px
                if direction == 1:
                    self.sl_px = px * (1 - self.sl_mult * atr)
                    self.tp_px = px * (1 + self.tp_mult * atr)
                else:
                    self.sl_px = px * (1 + self.sl_mult * atr)
                    self.tp_px = px * (1 - self.tp_mult * atr)
                self.hold_cnt = 0
                self.unreal = 0.0
                reward -= self.taker_fee                       # entry fee
            else:
                reward -= self.idle_penalty

        self.t += 1
        return float(reward)

    def step(self, action: int):
        reward = self._advance(action)
        terminated = self.t >= self.episode_start + self.episode_len or self.t >= len(self.close) - 1
        obs = self._get_obs() if not terminated else np.zeros(self.observation_space.shape, np.float32)
        return obs, reward, terminated, False, {"t": self.t}

    # ── deterministic full-sequence rollout for inference ─────────────────────
    def rollout(self, model) -> tuple[np.ndarray, np.ndarray, list[dict]]:
        """Run ``model`` sequentially over the whole array from window_size.

        Returns
        -------
        position_path : int8 array, position held at each bar (len == n).
        step_returns  : float array, per-bar net return (fees+funding incl.).
        trades        : list of dict (one per closed trade).
        """
        n = len(self.close)
        self.t = self.window_size
        self.episode_start = self.window_size
        self._reset_trade_state()

        pos_path = np.zeros(n, dtype=np.int8)
        step_ret = np.zeros(n, dtype=np.float64)
        trades: list[dict] = []
        cur_trade: dict | None = None
        obs = self._get_obs()

        while self.t < n - 1:
            t = self.t
            act, _ = model.predict(obs, deterministic=True)
            held = self.position            # position active DURING bar t (earns MTM)
            r = self._advance(int(act))
            step_ret[t] = r
            pos_path[t] = held              # 0 on entry bar, dir on held & exit bars

            opened = held == 0 and self.position != 0
            closed = held != 0 and self.position == 0
            if opened:
                cur_trade = {"entry_bar": t, "direction": "long" if self.position == 1 else "short",
                             "ret": r, "hold": 0}
            elif cur_trade is not None:
                cur_trade["ret"] += r
                cur_trade["hold"] += 1
                if closed:
                    cur_trade["exit_bar"] = t
                    trades.append(cur_trade)
                    cur_trade = None
            obs = self._get_obs()

        return pos_path, step_ret, trades


# ── Agent (M1Y walk-forward, mirrors DRLAgent API) ────────────────────────────

_DEFAULT_BRACKET_FEATURES: list[str] = [
    "ret_1h", "ret_2h", "ret_3h", "ret_6h",
    "rsi_14", "stoch_k_14", "macd_hist_5_13",
    "atr_14_pct", "vol_ratio_24h", "bb_width_pct", "bb_position_20",
    "close_vs_sma_7", "close_vs_sma_50", "trend_score", "ma_bull_score",
    "close_vs_true_vwap", "hurst_24h",
    "hour_sin", "hour_cos",
    # microstructure / complexity context (from features/microstructure.py)
    "roll_measure_50", "amihud_50", "vol_imbalance_50", "sampen_48",
]


class DRLBracketAgent:
    """PPO entry-timer with mechanical ATR-bracket exits and M1Y WFO."""

    AGENT_ID = "drl_bracket_v4"
    SIGNAL_COL = "drl_action"

    def __init__(
        self,
        features: list[str] | None = None,
        window_size: int = 24,
        episode_len: int = 1000,
        sl_mult: float = 2.0,
        tp_mult: float = 3.0,
        min_hold: int = 4,
        max_hold: int = 24,
        cooldown: int = 2,
        idle_penalty: float = 0.0,
        train_window_h: int = 8760,
        step_size: int = 720,
        total_timesteps: int = 500_000,
        agent_id: str | None = None,
        ppo_kwargs: dict | None = None,
    ) -> None:
        if not _SB3_AVAILABLE:
            raise ImportError("stable-baselines3 and gymnasium are required for DRLBracketAgent.")
        self.features = features or _DEFAULT_BRACKET_FEATURES
        self.window_size = window_size
        self.episode_len = episode_len
        self.sl_mult, self.tp_mult = sl_mult, tp_mult
        self.min_hold, self.max_hold, self.cooldown = min_hold, max_hold, cooldown
        self.idle_penalty = idle_penalty
        self.train_window_h = train_window_h
        self.step_size = step_size
        self.total_timesteps = total_timesteps
        self.AGENT_ID = agent_id or self.__class__.AGENT_ID
        self.ppo_kwargs = ppo_kwargs or {}
        # populated by generate_signals — used by the notebook for equity/trades
        self.step_returns_: pd.Series | None = None
        self.trades_: list[dict] = []

    @staticmethod
    def _normalise(X: np.ndarray) -> np.ndarray:
        med = np.nanmedian(X, axis=0)
        mad = np.nanmedian(np.abs(X - med), axis=0)
        scale = np.where(mad > 1e-8, mad * 1.4826, 1.0)
        return np.clip((X - med) / scale, -5.0, 5.0).astype(np.float32)

    def _make_env(self, feats, close, high, low, atr, episode_len) -> BracketMarketEnv:
        return BracketMarketEnv(
            feats, close, high, low, atr,
            window_size=self.window_size, episode_len=episode_len,
            sl_mult=self.sl_mult, tp_mult=self.tp_mult,
            min_hold=self.min_hold, max_hold=self.max_hold, cooldown=self.cooldown,
            idle_penalty=self.idle_penalty,
        )

    def generate_signals(
        self,
        df: pd.DataFrame,
        oos_start: pd.Timestamp,
        verbose: bool = True,
        full_history: bool = False,
    ) -> pd.Series:
        feats_present = [f for f in self.features if f in df.columns]
        if not feats_present:
            raise ValueError(f"No requested features present: {self.features[:5]}...")

        X_raw = df[feats_present].fillna(0).values
        close = df["close"].values.astype(np.float64)
        high = df["high"].values.astype(np.float64) if "high" in df else close
        low = df["low"].values.astype(np.float64) if "low" in df else close
        atr = (df["atr_14_pct"].values.astype(np.float64)
               if "atr_14_pct" in df else np.full(len(df), 0.01))

        n = len(df)
        pos_all = np.zeros(n, dtype=np.int8)
        ret_all = np.zeros(n, dtype=np.float64)
        trades_all: list[dict] = []
        fold = 0

        i = self.train_window_h
        while i < n:
            tr_start = max(0, i - self.train_window_h)
            tr_end = i
            oos_end = min(i + self.step_size, n)
            if tr_end - tr_start < self.window_size + 50:
                i += self.step_size
                continue

            fold += 1
            Xtr = self._normalise(X_raw[tr_start:tr_end])
            env = self._make_env(Xtr, close[tr_start:tr_end], high[tr_start:tr_end],
                                  low[tr_start:tr_end], atr[tr_start:tr_end], self.episode_len)
            ppo = dict(policy="MlpPolicy", learning_rate=3e-4, n_steps=2048, batch_size=64,
                       n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                       ent_coef=0.01, verbose=0, seed=42)
            ppo.update(self.ppo_kwargs)
            model = PPO(env=env, **ppo)
            model.learn(total_timesteps=self.total_timesteps, progress_bar=False)

            # rollout on [tr_start : oos_end] so the policy sees a context prefix
            Xev = self._normalise(X_raw[tr_start:oos_end])
            ev = self._make_env(Xev, close[tr_start:oos_end], high[tr_start:oos_end],
                                low[tr_start:oos_end], atr[tr_start:oos_end], oos_end - tr_start)
            pos, sret, trades = ev.rollout(model)

            n_oos = oos_end - tr_end
            off = tr_end - tr_start
            pos_all[tr_end:oos_end] = pos[off:off + n_oos]
            ret_all[tr_end:oos_end] = sret[off:off + n_oos]
            for tr in trades:
                if tr["entry_bar"] >= off:           # entry falls in this fold's OOS slice
                    tr_abs = dict(tr)
                    tr_abs["entry_bar"] += tr_start   # local → absolute df index
                    if "exit_bar" in tr_abs:
                        tr_abs["exit_bar"] += tr_start
                    trades_all.append(tr_abs)

            if verbose:
                a = pos[off:off + n_oos]
                print(f"  [{self.AGENT_ID}] fold {fold:>3}  "
                      f"entries={sum(1 for tr in trades if tr['entry_bar']>=off):>3}  "
                      f"L={(a==1).sum():>4} S={(a==-1).sum():>4} F={(a==0).sum():>4}")
            i += self.step_size

        self.step_returns_ = pd.Series(ret_all, index=df.index, name="drl_step_ret")
        self.trades_ = trades_all

        series = pd.Series(pos_all, index=df.index, name=self.SIGNAL_COL)
        if verbose:
            m = df.index >= oos_start
            a = pos_all[m]
            print(f"[{self.AGENT_ID}] WFO done: {fold} folds  "
                  f"OOS entries={sum(1 for t in trades_all if df.index[t['entry_bar']] >= oos_start)}  "
                  f"Long={(a==1).sum()} Short={(a==-1).sum()} Flat={(a==0).sum()}")
        return series if full_history else series[df.index >= oos_start].copy()
