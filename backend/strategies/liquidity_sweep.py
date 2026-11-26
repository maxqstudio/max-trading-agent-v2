from __future__ import annotations

from backend.core.models import Bar, MarketRegime, Side, TradeSignal
from backend.features.indicators import IndicatorSnapshot
from backend.strategies.base import BaseScalpingStrategy


class LiquiditySweepStrategy(BaseScalpingStrategy):
    """Detects M1-M5 stop-hunts (sweep of recent swing extreme followed by strong rejection wick)."""

    strategy_id = "liquidity_sweep"

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

        lookback = int(self.params.get("lookback_bars", 16))
        min_wick = float(self.params.get("min_wick_ratio", 0.55))
        sl_mult = float(self.params.get("sl_atr_mult", 1.45))
        tp_mult = float(self.params.get("tp_atr_mult", 2.20))

        last = bars[-1]
        prior = bars[-(lookback + 1) : -1]
        prior_low = min(b.low for b in prior)
        prior_high = max(b.high for b in prior)
        atr_val = max(snap.atr14, 1e-9)

        # Bullish liquidity sweep: wicked below prior swing low, closed back inside with strong lower wick
        if last.low < prior_low and last.close > prior_low and snap.lower_wick_ratio >= min_wick:
            if snap.rsi7 <= 48.0:
                entry = last.close
                sl = min(last.low - 0.15 * atr_val, entry - sl_mult * atr_val)
                risk_dist = max(entry - sl, 0.4 * atr_val)
                tp = entry + (tp_mult / max(sl_mult, 0.5)) * risk_dist
                conf = min(0.92, 0.55 + (snap.lower_wick_ratio - min_wick) * 0.9 + (0.08 if regime == MarketRegime.RANGING else 0.0))
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
                    reason=f"Bullish sweep below {prior_low:.5f} with {snap.lower_wick_ratio:.0%} lower wick",
                    timestamp=last.timestamp,
                )

        # Bearish liquidity sweep: wicked above prior swing high, closed back inside with strong upper wick
        if last.high > prior_high and last.close < prior_high and snap.upper_wick_ratio >= min_wick:
            if snap.rsi7 >= 52.0:
                entry = last.close
                sl = max(last.high + 0.15 * atr_val, entry + sl_mult * atr_val)
                risk_dist = max(sl - entry, 0.4 * atr_val)
                tp = entry - (tp_mult / max(sl_mult, 0.5)) * risk_dist
                conf = min(0.92, 0.55 + (snap.upper_wick_ratio - min_wick) * 0.9 + (0.08 if regime == MarketRegime.RANGING else 0.0))
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
                    reason=f"Bearish sweep above {prior_high:.5f} with {snap.upper_wick_ratio:.0%} upper wick",
                    timestamp=last.timestamp,
                )
        return None
