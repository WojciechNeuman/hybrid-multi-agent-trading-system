"""NEAT agent — loads a pre-trained genome and runs inference."""

from __future__ import annotations

import pickle
from pathlib import Path

import neat
import numpy as np

from hmats.agents.base import AgentSignal, BaseAgent, MarketSnapshot


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


_ACTION_MAP = {0: "hold", 1: "buy", 2: "sell"}


class NEATAgent(BaseAgent):
    """Wraps a serialised NEAT winner genome into a ``BaseAgent``.

    The genome must have been trained externally (e.g. via notebook 03) and
    saved with :func:`pickle.dump`.  At init the genome is deserialised and
    converted to a :class:`neat.nn.FeedForwardNetwork` for inference.

    ``compute()`` expects ``snapshot.indicators`` to contain the 12 standardised
    features produced by :func:`hmats.data.features.make_features` plus two
    extra observation dimensions (position_flag, cash_frac) that the caller
    may inject via ``snapshot.indicators["position_flag"]`` and
    ``snapshot.indicators["cash_frac"]``.  If those keys are absent the agent
    defaults to position_flag=0 and cash_frac=1 (flat / fully in cash).
    """

    def __init__(
        self,
        genome_path: str | Path,
        config_path: str | Path,
    ) -> None:
        super().__init__(agent_id="neat")
        self.genome_path = Path(genome_path)
        self.config_path = Path(config_path)

        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            str(self.config_path),
        )

        with open(self.genome_path, "rb") as f:
            genome = pickle.load(f)

        self.net = neat.nn.FeedForwardNetwork.create(genome, config)

    def compute(self, snapshot: MarketSnapshot) -> AgentSignal:
        obs = self._build_observation(snapshot)
        raw_output = np.array(self.net.activate(obs.tolist()), dtype=np.float64)
        probs = _softmax(raw_output)
        action_idx = int(np.argmax(probs))
        confidence = float(probs[action_idx])
        action = _ACTION_MAP[action_idx]

        return AgentSignal(
            agent_id=self.agent_id,
            timestamp=snapshot.timestamp,
            ticker=snapshot.ticker,
            action=action,  # type: ignore[arg-type]
            confidence=confidence,
            horizon="1h",
            metadata={"strategy": "neat", "raw_output": raw_output.tolist()},
        )

    @staticmethod
    def _build_observation(snapshot: MarketSnapshot) -> np.ndarray:
        feature_keys = [
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
        feats = [snapshot.indicators.get(k, 0.0) for k in feature_keys]
        pos_flag = snapshot.indicators.get("position_flag", 0.0)
        cash_frac = snapshot.indicators.get("cash_frac", 1.0)
        feats.extend([pos_flag, cash_frac])
        return np.array(feats, dtype=np.float32)
