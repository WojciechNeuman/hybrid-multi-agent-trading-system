from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import pandas as pd


@dataclass
class MarketSnapshot:
    """Input consumed by every agent — a point-in-time view of the market."""

    ticker: str
    timestamp: datetime
    ohlcv: pd.DataFrame
    indicators: dict[str, float]
    interval: str = "1h"
    sentiment: float | None = None


@dataclass
class AgentSignal:
    """Output produced by every agent."""

    agent_id: str
    timestamp: datetime
    ticker: str
    action: Literal["buy", "sell", "hold"]
    confidence: float
    horizon: str = "1d"
    interval: str = "1h"
    metadata: dict = field(default_factory=dict)


@dataclass
class TradingDecision:
    timestamp:     datetime
    ticker:        str
    action:        Literal["buy", "sell", "hold"]
    confidence:    float
    position_size: float
    horizon:       str = "1h"      # add this
    reasoning:     dict[str, AgentSignal] = field(default_factory=dict)


class BaseAgent(ABC):
    """Abstract base class that every trading agent must extend."""

    agent_id: str

    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id

    @abstractmethod
    def compute(self, snapshot: MarketSnapshot) -> AgentSignal:
        """Analyse a snapshot and return a trading signal."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(agent_id={self.agent_id!r})"
