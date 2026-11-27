from __future__ import annotations

from typing import Any

from backend.core.config import TradingConfig
from backend.core.models import Bar, MarketRegime, TradeSignal
from backend.features.indicators import IndicatorSnapshot
from backend.strategies.band_reversion import BandReversionStrategy
from backend.strategies.base import BaseScalpingStrategy
from backend.strategies.day_range_reversal import DayRangeReversalStrategy
from backend.strategies.ema_vwap_pullback import EmaVwapPullbackStrategy
from backend.strategies.liquidity_sweep import LiquiditySweepStrategy
from backend.strategies.micro_breakout import MicroBreakoutStrategy
from backend.strategies.structure_shift import StructureShiftStrategy
from backend.strategies.tick_imbalance import TickImbalanceStrategy

STRATEGY_CLASSES: dict[str, type[BaseScalpingStrategy]] = {
    "liquidity_sweep": LiquiditySweepStrategy,
    "micro_breakout": MicroBreakoutStrategy,
    "ema_vwap_pullback": EmaVwapPullbackStrategy,
    "tick_imbalance": TickImbalanceStrategy,
    "structure_shift": StructureShiftStrategy,
    "day_range_reversal": DayRangeReversalStrategy,
    "band_reversion": BandReversionStrategy,
}


class StrategySuite:
    """Instantiates and runs the configured scalping micro-strategy ensemble."""

    def __init__(self, strategy_configs: dict[str, dict[str, Any]]) -> None:
        self.strategies: dict[str, BaseScalpingStrategy] = {}
        for sid, cls in STRATEGY_CLASSES.items():
            self.strategies[sid] = cls(strategy_configs.get(sid, {}))

    @classmethod
    def from_config(cls, config: TradingConfig) -> StrategySuite:
        return cls(config.strategies)

    def scan_all(
        self,
        symbol: str,
        bars: list[Bar],
        snap: IndicatorSnapshot,
        regime: MarketRegime,
        context: dict[str, Any] | None = None,
    ) -> list[TradeSignal]:
        """``context``: market facts some strategies need beyond the bars, such as the previous
        day's range ("prev_day") and the M15/H1 direction ("htf_m15", "htf_h1")."""
        signals: list[TradeSignal] = []
        for strat in self.strategies.values():
            if getattr(strat, "uses_context", False):
                sig = strat.evaluate(symbol, bars, snap, regime, context=context)
            else:
                sig = strat.evaluate(symbol, bars, snap, regime)
            if sig is not None:
                signals.append(sig)
        return signals
