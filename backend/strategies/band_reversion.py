from __future__ import annotations

from typing import Any

import numpy as np

from backend.core.models import Bar, MarketRegime, Side, TradeSignal
from backend.features.indicators import IndicatorSnapshot, atr, bollinger, stochastic
from backend.strategies.base import BaseScalpingStrategy


class BandReversionStrategy(BaseScalpingStrategy):
    """Pullback against the move, taken in the direction of the bigger trend.

    Buy (sell is the mirror):
      1. One of the last ``outside_bars`` candles closed below the lower Bollinger band (20, 2),
         and the current candle closes back inside, bullish.
      2. Stochastic (14,3,3) %K crossed back above ``stoch_low`` on this candle or the one before.
      3. The H1 direction (``context["htf_h1"]``) is not down. With ``trend_filter: with`` it
         must be up.
      4. SL below the lowest low of the pullback plus a buffer; TP at the opposite band
         (``target: opposite``) or the middle band (``target: middle``).
    """

    strategy_id = "band_reversion"
    uses_context = True

    def evaluate(
        self, symbol: str, bars: list[Bar], snap: IndicatorSnapshot, regime: MarketRegime, context: dict[str, Any] | None = None,
    ) -> TradeSignal | None:
        p = self.params
        outside = int(p.get("outside_bars", 3))
        lo_lvl, hi_lvl = float(p.get("stoch_low", 20.0)), float(p.get("stoch_high", 80.0))
        buf = float(p.get("sl_buffer_atr", 0.3))
        max_sl = float(p.get("max_sl_atr", 3.0))
        trend_filter = str(p.get("trend_filter", "not_against"))
        target = str(p.get("target", "opposite"))
        if not self.enabled or len(bars) < 60:
            return None
        bias = int((context or {}).get("htf_h1", 0))
        win = bars[-60:]
        o = np.array([b.open for b in win])
        h = np.array([b.high for b in win])
        lo = np.array([b.low for b in win])
        c = np.array([b.close for b in win])
        t = len(c) - 1
        a = max(float(atr(h, lo, c, 14)[t]), 1e-9)
        up, mid, dn = bollinger(c, 20, 2.0)
        k, _ = stochastic(h, lo, c, 14, 3, 3)
        prior = range(t - outside, t)
        crossed_up = any(k[j - 1] <= lo_lvl < k[j] for j in (t - 1, t))
        crossed_dn = any(k[j - 1] >= hi_lvl > k[j] for j in (t - 1, t))

        buy_ok = bias > 0 if trend_filter == "with" else bias >= 0
        sell_ok = bias < 0 if trend_filter == "with" else bias <= 0
        if buy_ok and crossed_up and any(c[j] < dn[j] for j in prior) and dn[t] < c[t] < mid[t] and c[t] > o[t]:
            sl = float(lo[t - outside : t + 1].min()) - buf * a
            tp = float(up[t] if target == "opposite" else mid[t])
            return self._signal(symbol, regime, Side.BUY, float(c[t]), sl, tp, a, max_sl, win[-1].timestamp, float(k[t]))
        if sell_ok and crossed_dn and any(c[j] > up[j] for j in prior) and mid[t] < c[t] < up[t] and c[t] < o[t]:
            sl = float(h[t - outside : t + 1].max()) + buf * a
            tp = float(dn[t] if target == "opposite" else mid[t])
            return self._signal(symbol, regime, Side.SELL, float(c[t]), sl, tp, a, max_sl, win[-1].timestamp, float(k[t]))
        return None

    def _signal(self, symbol, regime, side, entry, sl, tp, a, max_sl, ts, stoch_k) -> TradeSignal | None:
        risk = abs(entry - sl)
        if risk <= 0 or risk > max_sl * a or (tp - entry) * (1 if side == Side.BUY else -1) <= 0:
            return None
        conf = 0.57 + (0.05 if regime == MarketRegime.RANGING else 0.0)
        return TradeSignal(
            symbol=symbol, strategy_id=self.strategy_id, side=side, entry_price=entry, sl_price=sl, tp_price=tp, atr=a,
            confidence=round(conf, 4), regime=regime,
            reason=f"Band reversion {side.value}: back inside the Bollinger band, Stochastic {stoch_k:.0f}",
            features={"risk_atr": round(risk / a, 3)}, timestamp=ts,
        )
