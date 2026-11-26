from __future__ import annotations

from backend.core.models import Bar, MarketRegime, Side, TradeSignal
from backend.features.indicators import IndicatorSnapshot
from backend.strategies.base import BaseScalpingStrategy


class EmaVwapPullbackStrategy(BaseScalpingStrategy):
    """Trend-aligned pullback scalper to EMA9/VWAP zone with ADX & RSI confirmation."""

    strategy_id = "ema_vwap_pullback"

    def evaluate(
        self,
        symbol: str,
        bars: list[Bar],
        snap: IndicatorSnapshot,
        regime: MarketRegime,
    ) -> TradeSignal | None:
        if not self.enabled or len(bars) < 35:
            return None

        min_adx = float(self.params.get("min_adx", 20.0))
        rsi_low = float(self.params.get("rsi_low", 38.0))
        rsi_high = float(self.params.get("rsi_high", 62.0))
        sl_mult = float(self.params.get("sl_atr_mult", 1.30))
        tp_mult = float(self.params.get("tp_atr_mult", 2.10))

        if snap.adx14 < min_adx:
            return None

        last = bars[-1]
        prev = bars[-2]
        atr_val = max(snap.atr14, 1e-9)

        # Bullish trend pullback: EMA9 > EMA21, price touched near EMA9/VWAP and resumed bullish close
        if snap.ema9 > snap.ema21 and last.close > snap.vwap:
            touched_zone = (last.low <= snap.ema9 + 0.25 * atr_val) or (prev.low <= snap.ema9 + 0.25 * atr_val)
            if touched_zone and last.close > last.open and rsi_low <= snap.rsi14 <= 66.0:
                entry = last.close
                sl = entry - sl_mult * atr_val
                tp = entry + tp_mult * atr_val
                conf = min(0.90, 0.56 + min((snap.adx14 - min_adx) / 60.0, 0.20) + (0.08 if snap.close > snap.ema50 else 0.0))
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
                    reason=f"Bullish EMA9/VWAP pullback (ADX {snap.adx14:.1f}, RSI {snap.rsi14:.1f})",
                    timestamp=last.timestamp,
                )

        # Bearish trend pullback
        if snap.ema9 < snap.ema21 and last.close < snap.vwap:
            touched_zone = (last.high >= snap.ema9 - 0.25 * atr_val) or (prev.high >= snap.ema9 - 0.25 * atr_val)
            if touched_zone and last.close < last.open and 34.0 <= snap.rsi14 <= rsi_high:
                entry = last.close
                sl = entry + sl_mult * atr_val
                tp = entry - tp_mult * atr_val
                conf = min(0.90, 0.56 + min((snap.adx14 - min_adx) / 60.0, 0.20) + (0.08 if snap.close < snap.ema50 else 0.0))
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
                    reason=f"Bearish EMA9/VWAP pullback (ADX {snap.adx14:.1f}, RSI {snap.rsi14:.1f})",
                    timestamp=last.timestamp,
                )
        return None
