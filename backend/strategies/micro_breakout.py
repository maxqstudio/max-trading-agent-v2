from __future__ import annotations

from backend.core.models import Bar, MarketRegime, Side, TradeSignal
from backend.features.indicators import IndicatorSnapshot
from backend.strategies.base import BaseScalpingStrategy


class MicroBreakoutStrategy(BaseScalpingStrategy):
    """Captures volatility squeeze / micro Donchian breakouts confirmed by candle body expansion."""

    strategy_id = "micro_breakout"

    def evaluate(
        self,
        symbol: str,
        bars: list[Bar],
        snap: IndicatorSnapshot,
        regime: MarketRegime,
    ) -> TradeSignal | None:
        if not self.enabled or len(bars) < 25:
            return None
        if regime == MarketRegime.LOW_VOL_CHOP:
            return None

        brk_win = int(self.params.get("breakout_window", 10))
        min_body = float(self.params.get("min_body_atr", 0.45))
        sl_mult = float(self.params.get("sl_atr_mult", 1.35))
        tp_mult = float(self.params.get("tp_atr_mult", 2.40))

        last = bars[-1]
        prior = bars[-(brk_win + 1) : -1]
        donch_hi = max(b.high for b in prior)
        donch_lo = min(b.low for b in prior)
        atr_val = max(snap.atr14, 1e-9)

        if snap.body_atr_ratio < min_body:
            return None

        # Bullish breakout with expansion
        if last.close > donch_hi and last.close > last.open and snap.close > snap.ema21:
            entry = last.close
            sl = entry - sl_mult * atr_val
            tp = entry + tp_mult * atr_val
            conf = min(0.92, 0.54 + 0.15 * min(snap.body_atr_ratio, 1.5) + (0.10 if regime == MarketRegime.TRENDING_BULL else 0.0))
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
                reason=f"Bullish micro breakout above {donch_hi:.5f} (body {snap.body_atr_ratio:.2f} ATR)",
                timestamp=last.timestamp,
            )

        # Bearish breakout with expansion
        if last.close < donch_lo and last.close < last.open and snap.close < snap.ema21:
            entry = last.close
            sl = entry + sl_mult * atr_val
            tp = entry - tp_mult * atr_val
            conf = min(0.92, 0.54 + 0.15 * min(snap.body_atr_ratio, 1.5) + (0.10 if regime == MarketRegime.TRENDING_BEAR else 0.0))
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
                reason=f"Bearish micro breakout below {donch_lo:.5f} (body {snap.body_atr_ratio:.2f} ATR)",
                timestamp=last.timestamp,
            )
        return None
