from __future__ import annotations

from backend.core.models import Bar, MarketRegime, Side, TradeSignal
from backend.features.indicators import IndicatorSnapshot
from backend.strategies.base import BaseScalpingStrategy


class TickImbalanceStrategy(BaseScalpingStrategy):
    """Microstructure impulse proxy detecting consecutive directional order-flow bars."""

    strategy_id = "tick_imbalance"

    def evaluate(
        self,
        symbol: str,
        bars: list[Bar],
        snap: IndicatorSnapshot,
        regime: MarketRegime,
    ) -> TradeSignal | None:
        if not self.enabled or len(bars) < 20:
            return None
        if regime == MarketRegime.LOW_VOL_CHOP:
            return None

        consec = int(self.params.get("consecutive_bars", 3))
        min_mom = float(self.params.get("min_momentum_atr", 0.55))
        sl_mult = float(self.params.get("sl_atr_mult", 1.25))
        tp_mult = float(self.params.get("tp_atr_mult", 2.00))

        recent = bars[-consec:]
        atr_val = max(snap.atr14, 1e-9)
        net_move = recent[-1].close - recent[0].open

        all_bull = all(b.close > b.open for b in recent)
        all_bear = all(b.close < b.open for b in recent)

        if all_bull and (net_move / atr_val) >= min_mom and snap.rsi7 < 82.0:
            entry = recent[-1].close
            sl = entry - sl_mult * atr_val
            tp = entry + tp_mult * atr_val
            conf = min(0.89, 0.52 + 0.14 * min(net_move / atr_val, 1.8) + 0.06 * min(snap.rel_volume, 2.0))
            return TradeSignal(
                symbol=symbol,
                strategy_id=self.strategy_id,
                side=Side.BUY,
                entry_price=entry,
                sl_price=sl,
                tp_price=tp,
                atr=atr_val,
                confidence=round(conf, 4),
                regime=regime,
                reason=f"{consec}x bullish impulse bars ({net_move / atr_val:.2f} ATR, vol {snap.rel_volume:.1f}x)",
                timestamp=recent[-1].timestamp,
            )

        if all_bear and (-net_move / atr_val) >= min_mom and snap.rsi7 > 18.0:
            entry = recent[-1].close
            sl = entry + sl_mult * atr_val
            tp = entry - tp_mult * atr_val
            conf = min(0.89, 0.52 + 0.14 * min(-net_move / atr_val, 1.8) + 0.06 * min(snap.rel_volume, 2.0))
            return TradeSignal(
                symbol=symbol,
                strategy_id=self.strategy_id,
                side=Side.SELL,
                entry_price=entry,
                sl_price=sl,
                tp_price=tp,
                atr=atr_val,
                confidence=round(conf, 4),
                regime=regime,
                reason=f"{consec}x bearish impulse bars ({-net_move / atr_val:.2f} ATR, vol {snap.rel_volume:.1f}x)",
                timestamp=recent[-1].timestamp,
            )
        return None
