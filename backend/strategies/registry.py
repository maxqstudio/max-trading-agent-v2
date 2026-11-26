from __future__ import annotations

from typing import Any

from backend.core.config import TradingConfig
from backend.core.models import Bar, MarketRegime, TradeSignal
from backend.features.indicators import IndicatorSnapshot
from backend.strategies.base import BaseScalpingStrategy
from backend.strategies.ema_vwap_pullback import EmaVwapPullbackStrategy
from backend.strategies.liquidity_sweep import LiquiditySweepStrategy
from backend.strategies.micro_breakout import MicroBreakoutStrategy
from backend.strategies.tick_imbalance import TickImbalanceStrategy

STRATEGY_CLASSES: dict[str, type[BaseScalpingStrategy]] = {
    "liquidity_sweep": LiquiditySweepStrategy,
    "micro_breakout": MicroBreakoutStrategy,
    "ema_vwap_pullback": EmaVwapPullbackStrategy,
    "tick_imbalance": TickImbalanceStrategy,
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
    ) -> list[TradeSignal]:
        signals: list[TradeSignal] = []
        for strat in self.strategies.values():
            sig = strat.evaluate(symbol, bars, snap, regime)
            if sig is not None:
                signals.append(sig)
        return signals
