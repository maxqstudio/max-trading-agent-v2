from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from backend.core.models import Bar, MarketRegime, TradeSignal
from backend.features.indicators import IndicatorSnapshot


class BaseScalpingStrategy(ABC):
    strategy_id: str = "base"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params: dict[str, Any] = dict(params or {})

    @property
    def enabled(self) -> bool:
        return bool(self.params.get("enabled", True))

    @abstractmethod
    def evaluate(
        self,
        symbol: str,
        bars: list[Bar],
        snap: IndicatorSnapshot,
        regime: MarketRegime,
    ) -> TradeSignal | None:
        """Return a causal TradeSignal evaluated strictly on closed bars, or None."""
