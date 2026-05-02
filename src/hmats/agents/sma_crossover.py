"""SMA Crossover agent — buys/sells on SMA(20) vs SMA(50) crossovers."""

from __future__ import annotations

from hmats.agents.base import AgentSignal, BaseAgent, MarketSnapshot


class SMACrossoverAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(agent_id="sma_crossover")

    def compute(self, snapshot: MarketSnapshot) -> AgentSignal:
        ohlcv = snapshot.ohlcv
        if len(ohlcv) < 2:
            return self._signal(snapshot, "hold", 0.0)

        sma20_col = "SMA_20" if "SMA_20" in ohlcv.columns else None
        sma50_col = "SMA_50" if "SMA_50" in ohlcv.columns else None

        if sma20_col and sma50_col:
            sma20_now = ohlcv[sma20_col].iloc[-1]
            sma50_now = ohlcv[sma50_col].iloc[-1]
            sma20_prev = ohlcv[sma20_col].iloc[-2]
            sma50_prev = ohlcv[sma50_col].iloc[-2]
        else:
            sma20_now = snapshot.indicators.get("SMA_20")
            sma50_now = snapshot.indicators.get("SMA_50")
            if sma20_now is None or sma50_now is None:
                return self._signal(snapshot, "hold", 0.0)
            return self._from_indicators(snapshot, sma20_now, sma50_now)

        if _any_nan(sma20_now, sma50_now, sma20_prev, sma50_prev):
            return self._signal(snapshot, "hold", 0.0)

        crossed_above = sma20_prev <= sma50_prev and sma20_now > sma50_now
        crossed_below = sma20_prev >= sma50_prev and sma20_now < sma50_now

        gap = abs(sma20_now - sma50_now) / sma50_now if sma50_now != 0 else 0.0
        confidence = min(gap, 1.0)

        if crossed_above:
            action = "buy"
        elif crossed_below:
            action = "sell"
        else:
            action = "hold"

        return self._signal(snapshot, action, confidence)

    # ------------------------------------------------------------------

    def _from_indicators(self, snapshot: MarketSnapshot, sma20: float, sma50: float) -> AgentSignal:
        gap = abs(sma20 - sma50) / sma50 if sma50 != 0 else 0.0
        confidence = min(gap, 1.0)
        action = "buy" if sma20 > sma50 else ("sell" if sma20 < sma50 else "hold")
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
            metadata={"strategy": "sma_crossover"},
        )


def _any_nan(*values: float) -> bool:
    import math

    return any(math.isnan(v) for v in values)
