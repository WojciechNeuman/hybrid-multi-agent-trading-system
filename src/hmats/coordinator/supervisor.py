"""Supervisor — orchestrates agents and aggregates signals.

Supports two modes:
- **Equal-weight majority vote** (cold start / default).
- **Performance-weighted voting** — weights derived from each agent's rolling
  Sharpe ratio, updated via :meth:`update_outcome`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from loguru import logger

from hmats.agents.base import AgentSignal, BaseAgent, MarketSnapshot, TradingDecision


@dataclass
class AgentRecord:
    """Tracks one agent's signal history and retrospective performance."""

    agent_id: str
    signals: list[AgentSignal] = field(default_factory=list)
    outcomes: list[float] = field(default_factory=list)
    sharpe: float = 0.0
    weight: float = 1.0


class Supervisor:
    """Collects signals from all registered agents and produces a single decision.

    After each :meth:`run` call the emitted signals are stored.  When the next
    candle's close is observed the caller should invoke :meth:`update_outcome`
    so that per-agent weights can be re-computed.
    """

    def __init__(
        self,
        agents: list[BaseAgent],
        *,
        sharpe_window: int = 30,
        risk_tolerance: float = 1.0,
    ) -> None:
        if not agents:
            raise ValueError("Supervisor requires at least one agent")
        self.agents = agents
        self.sharpe_window = sharpe_window
        self.risk_tolerance = risk_tolerance

        self._records: dict[str, AgentRecord] = {
            a.agent_id: AgentRecord(agent_id=a.agent_id) for a in agents
        }
        self._last_signals: dict[str, AgentSignal] = {}
        self._initialise_weights()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, snapshot: MarketSnapshot) -> TradingDecision:
        signals: dict[str, AgentSignal] = {}

        for agent in self.agents:
            signal = agent.compute(snapshot)
            signals[signal.agent_id] = signal
            self._records[signal.agent_id].signals.append(signal)
            logger.info(
                "Agent {id} → {action} (confidence={conf:.2f}, weight={w:.3f})",
                id=signal.agent_id,
                action=signal.action,
                conf=signal.confidence,
                w=self._records[signal.agent_id].weight,
            )

        action = self._weighted_vote(signals)
        avg_confidence = self._weighted_confidence(signals, action)
        position_size = self._position_size(action, avg_confidence)

        decision = TradingDecision(
            timestamp=snapshot.timestamp,
            ticker=snapshot.ticker,
            action=action,
            confidence=avg_confidence,
            position_size=position_size,
            reasoning=signals,
        )

        self._last_signals = signals

        logger.info(
            "Decision → {action} | confidence={conf:.2f} | size={size:.2f}",
            action=decision.action,
            conf=decision.confidence,
            size=decision.position_size,
        )
        return decision

    def update_outcome(self, ticker: str, realised_return: float) -> None:
        """Score the previous decision against the observed price change.

        Call this after the next candle closes.  Each agent's signal is scored:
        correct direction → ``+confidence``, wrong → ``-confidence``,
        hold → 0.

        After scoring, rolling Sharpe ratios and weights are recomputed.
        """
        for agent_id, signal in self._last_signals.items():
            if signal.ticker != ticker:
                continue

            record = self._records[agent_id]
            score = self._score_signal(signal, realised_return)
            record.outcomes.append(score)

        self._recompute_weights()

    @property
    def weights(self) -> dict[str, float]:
        return {aid: rec.weight for aid, rec in self._records.items()}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _initialise_weights(self) -> None:
        n = len(self.agents)
        for rec in self._records.values():
            rec.weight = 1.0 / n

    def _recompute_weights(self) -> None:
        sharpes: dict[str, float] = {}
        for aid, rec in self._records.items():
            recent = rec.outcomes[-self.sharpe_window :]
            if len(recent) < 2:
                rec.sharpe = 0.0
            else:
                arr = np.array(recent, dtype=np.float64)
                mu = float(arr.mean())
                sd = float(arr.std(ddof=1)) + 1e-12
                rec.sharpe = mu / sd
            sharpes[aid] = rec.sharpe

        vals = np.array(list(sharpes.values()), dtype=np.float64)
        exp = np.exp(vals - np.max(vals))
        softmax = exp / exp.sum()

        for i, aid in enumerate(sharpes):
            self._records[aid].weight = float(softmax[i])

    @staticmethod
    def _score_signal(signal: AgentSignal, realised_return: float) -> float:
        if signal.action == "hold":
            return 0.0
        direction = 1.0 if signal.action == "buy" else -1.0
        correct = math.copysign(1.0, realised_return) == direction
        return signal.confidence if correct else -signal.confidence

    def _weighted_vote(self, signals: dict[str, AgentSignal]) -> str:
        action_weights: dict[str, float] = {}
        for aid, sig in signals.items():
            w = self._records[aid].weight * sig.confidence
            action_weights[sig.action] = action_weights.get(sig.action, 0.0) + w
        return max(action_weights, key=action_weights.get)  # type: ignore[arg-type]

    def _weighted_confidence(self, signals: dict[str, AgentSignal], action: str) -> float:
        total_w = 0.0
        conf_sum = 0.0
        for aid, sig in signals.items():
            if sig.action == action:
                w = self._records[aid].weight
                conf_sum += w * sig.confidence
                total_w += w
        return conf_sum / total_w if total_w > 0 else 0.0

    def _position_size(self, action: str, confidence: float) -> float:
        if action == "hold":
            return 0.0
        return round(min(confidence * self.risk_tolerance, 1.0), 4)
