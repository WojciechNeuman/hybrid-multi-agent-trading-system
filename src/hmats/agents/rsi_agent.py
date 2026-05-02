"""RSI mean-reversion agent — buys oversold, sells overbought."""

from __future__ import annotations

from hmats.agents.base import AgentSignal, BaseAgent, MarketSnapshot


class RSIAgent(BaseAgent):
    OVERSOLD = 30.0
    OVERBOUGHT = 70.0

    def __init__(self) -> None:
        super().__init__(agent_id="rsi_mean_reversion")

    def compute(self, snapshot: MarketSnapshot) -> AgentSignal:
        rsi = snapshot.indicators.get("RSI_14")

        if rsi is None:
            return self._signal(snapshot, "hold", 0.0)

        confidence = abs(rsi - 50.0) / 50.0
        confidence = min(max(confidence, 0.0), 1.0)

        if rsi < self.OVERSOLD:
            action = "buy"
        elif rsi > self.OVERBOUGHT:
            action = "sell"
        else:
            action = "hold"

        return self._signal(snapshot, action, confidence)

    def _signal(
        self,
        snapshot: MarketSnapshot,
        action: str,
        confidence: float,
    ) -> AgentSignal:
        return AgentSignal(
            agent_id=self.agent_id,
            timestamp=snapshot.timestamp,
            ticker=snapshot.ticker,
            action=action,  # type: ignore[arg-type]
            confidence=confidence,
            horizon="1d",
            metadata={"strategy": "rsi_mean_reversion"},
        )
