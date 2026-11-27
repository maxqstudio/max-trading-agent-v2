from __future__ import annotations

from typing import Any

import numpy as np

from backend.core.models import Bar, MarketRegime, Side, TradeSignal
from backend.features.indicators import IndicatorSnapshot, atr, stochastic
from backend.strategies.base import BaseScalpingStrategy


class DayRangeReversalStrategy(BaseScalpingStrategy):
    """Counter-trend: price pokes through the previous day's low (high) and turns back.

    Buy (sell is the mirror):
      1. Within ``sweep_bars`` candles, a low went below the previous day's low.
      2. Stochastic (14,3,3) %K was at or below ``stoch_low`` during that time and now crosses
         back above it.
      3. The candle closes above the 3-period SMA shifted 3 bars and above the previous
         candle's high (in place of a buy stop above that high).
      4. SL beyond the lowest point of the poke plus a buffer; TP at ``tp_rr`` times the risk.
    Needs the previous day's range in ``context["prev_day"]``; without it, it stays quiet.
    """

    strategy_id = "day_range_reversal"
    uses_context = True

    def evaluate(
        self, symbol: str, bars: list[Bar], snap: IndicatorSnapshot, regime: MarketRegime, context: dict[str, Any] | None = None,
    ) -> TradeSignal | None:
        p = self.params
        rng = (context or {}).get("prev_day")
        sweep_bars = int(p.get("sweep_bars", 30))
        lo_lvl, hi_lvl = float(p.get("stoch_low", 20.0)), float(p.get("stoch_high", 80.0))
        sma_n, shift = int(p.get("sma_period", 3)), int(p.get("sma_shift", 3))
        buf = float(p.get("sl_buffer_atr", 0.2))
        max_sl = float(p.get("max_sl_atr", 3.0))
        tp_rr = float(p.get("tp_rr", 1.5))
        if not self.enabled or not rng or len(bars) < sweep_bars + 30:
            return None
        win = bars[-(sweep_bars + 40) :]
        o = np.array([b.open for b in win])
        h = np.array([b.high for b in win])
        lo = np.array([b.low for b in win])
        c = np.array([b.close for b in win])
        t = len(c) - 1
        a = max(float(atr(h, lo, c, 14)[t]), 1e-9)
        k, _ = stochastic(h, lo, c, 14, 3, 3)
        sma_shifted = float(np.mean(c[t - shift - sma_n + 1 : t - shift + 1]))
        day_hi, day_lo = rng
        s = t - sweep_bars + 1

        if lo[s : t + 1].min() < day_lo and k[t - 1] <= lo_lvl < k[t] and k[s:t].min() <= lo_lvl:
            if c[t] > sma_shifted and c[t] > h[t - 1] and c[t] > o[t]:
                return self._signal(symbol, regime, Side.BUY, float(c[t]), float(lo[s : t + 1].min()) - buf * a, a, max_sl, tp_rr,
                                    win[-1].timestamp, f"Back above yesterday's low {day_lo:.5f}", float(k[t]))
        if h[s : t + 1].max() > day_hi and k[t - 1] >= hi_lvl > k[t] and k[s:t].max() >= hi_lvl:
            if c[t] < sma_shifted and c[t] < lo[t - 1] and c[t] < o[t]:
                return self._signal(symbol, regime, Side.SELL, float(c[t]), float(h[s : t + 1].max()) + buf * a, a, max_sl, tp_rr,
                                    win[-1].timestamp, f"Back below yesterday's high {day_hi:.5f}", float(k[t]))
        return None

    def _signal(self, symbol, regime, side, entry, sl, a, max_sl, tp_rr, ts, why, stoch_k) -> TradeSignal | None:
        risk = abs(entry - sl)
        if risk <= 0 or risk > max_sl * a:
            return None
        tp = entry + tp_rr * risk if side == Side.BUY else entry - tp_rr * risk
        conf = 0.58 + (0.06 if regime in (MarketRegime.RANGING, MarketRegime.LOW_VOL_CHOP) else 0.0)
        return TradeSignal(
            symbol=symbol, strategy_id=self.strategy_id, side=side, entry_price=entry, sl_price=sl, tp_price=tp, atr=a,
            confidence=round(conf, 4), regime=regime, reason=f"Day range reversal {side.value}: {why}, Stochastic {stoch_k:.0f}",
            features={"risk_atr": round(risk / a, 3)}, timestamp=ts,
        )
