"""Supervisor — orchestrates agents and aggregates signals via majority vote."""

from __future__ import annotations

from collections import Counter

from loguru import logger

from hmats.agents.base import AgentSignal, BaseAgent, MarketSnapshot, TradingDecision


class Supervisor:
    """Collects signals from all registered agents and produces a single decision."""

    def __init__(self, agents: list[BaseAgent]) -> None:
        if not agents:
            raise ValueError("Supervisor requires at least one agent")
        self.agents = agents

    def run(self, snapshot: MarketSnapshot) -> TradingDecision:
        signals: dict[str, AgentSignal] = {}

        for agent in self.agents:
            signal = agent.compute(snapshot)
            signals[signal.agent_id] = signal
            logger.info(
                "Agent {id} → {action} (confidence={conf:.2f})",
                id=signal.agent_id,
                action=signal.action,
                conf=signal.confidence,
            )

        action = self._majority_vote(list(signals.values()))
        avg_confidence = sum(s.confidence for s in signals.values() if s.action == action) / max(
            sum(1 for s in signals.values() if s.action == action), 1
        )
        position_size = self._position_size(action, avg_confidence)

        decision = TradingDecision(
            timestamp=snapshot.timestamp,
            ticker=snapshot.ticker,
            action=action,
            confidence=avg_confidence,
            position_size=position_size,
            reasoning=signals,
        )

        logger.info(
            "Decision → {action} | confidence={conf:.2f} | size={size:.2f}",
            action=decision.action,
            conf=decision.confidence,
            size=decision.position_size,
        )
        return decision

    @staticmethod
    def _majority_vote(signals: list[AgentSignal]) -> str:  # type: ignore[return]
        """Simple majority vote; ties broken by confidence-weighted sum."""
        vote_counts: Counter[str] = Counter(s.action for s in signals)
        max_votes = vote_counts.most_common(1)[0][1]

        candidates = [action for action, count in vote_counts.items() if count == max_votes]

        if len(candidates) == 1:
            return candidates[0]

        best_action = max(
            candidates,
            key=lambda a: sum(s.confidence for s in signals if s.action == a),
        )
        return best_action

    @staticmethod
    def _position_size(action: str, confidence: float) -> float:
        if action == "hold":
            return 0.0
        return round(min(confidence, 1.0), 4)
