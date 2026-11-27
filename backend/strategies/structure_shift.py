from __future__ import annotations

import numpy as np

from backend.core.models import Bar, MarketRegime, Side, TradeSignal
from backend.features.indicators import IndicatorSnapshot, atr
from backend.features.structure import engulfing_at, swings
from backend.strategies.base import BaseScalpingStrategy

EXTENSIONS = (1.618, 2.618, 4.236)


class StructureShiftStrategy(BaseScalpingStrategy):
    """Engulfing, break of the nearest swing level, then a Fibonacci retest.

    Buy (sell is the mirror):
      1. A bullish engulfing candle gives the early warning.
      2. Within ``max_break_bars`` a candle closes above the last confirmed swing high that was
         above the engulfing close (the nearest resistance).
      3. The Fibonacci leg runs from the low of the move to the CLOSE of that breaking candle
         (not its wick).
      4. Price pulls back to at least ``retest_min`` of the leg (38.2%; the 38.2 to 61.8 golden
         zone and down to 78.6 count) without going past ``invalid_level`` and without first
         running to the 1.618 target.
      5. Entry on the first candle that turns back up (bullish and closing above the previous
         high). SL below the retest low plus a buffer.
      6. Targets are leg extensions 1.618 / 2.618 / 4.236. The final TP is ``target_stage``;
         the earlier ones are stages where the stop is stepped up.

    Supply/demand zones, the Ichimoku cloud, Bollinger position and engulfing context are not
    rules here: they are ML features, so the robot learns how much each one helps.
    """

    strategy_id = "structure_shift"

    def evaluate(self, symbol: str, bars: list[Bar], snap: IndicatorSnapshot, regime: MarketRegime) -> TradeSignal | None:
        p = self.params
        k = int(p.get("swing_bars", 3))
        setup_window = int(p.get("setup_window", 60))
        max_break = int(p.get("max_break_bars", 15))
        max_retest = int(p.get("max_retest_bars", 25))
        min_body = float(p.get("min_engulf_body_atr", 0.5))
        r_min = float(p.get("retest_min", 0.382))
        invalid = float(p.get("invalid_level", 0.786))
        buf = float(p.get("sl_buffer_atr", 0.2))
        max_sl = float(p.get("max_sl_atr", 3.0))
        min_leg = float(p.get("min_leg_atr", 1.5))
        stage = min(max(int(p.get("target_stage", 2)), 1), len(EXTENSIONS))

        need = setup_window + max_retest + 4 * k + 20
        if not self.enabled or len(bars) < need:
            return None
        win = bars[-(need + 60) :]
        o = np.array([b.open for b in win], dtype=np.float64)
        h = np.array([b.high for b in win], dtype=np.float64)
        lo = np.array([b.low for b in win], dtype=np.float64)
        c = np.array([b.close for b in win], dtype=np.float64)
        a = atr(h, lo, c, 14)
        n = len(c)
        t = n - 1
        cur_atr = max(float(a[t]), 1e-9)
        sw_hi, sw_lo = swings(h, lo, k)

        for sign in (1, -1):
            for e in range(t - 2, max(t - setup_window, 1), -1):
                if engulfing_at(o, c, e) != sign or abs(c[e] - o[e]) < min_body * float(a[e]):
                    continue
                plan = self._plan(sign, e, t, o, h, lo, c, a, sw_hi, sw_lo, k, max_break, max_retest,
                                  r_min, invalid, min_leg)
                if plan is None:
                    continue
                leg_lo, leg_hi, brk, retest_px = plan
                rng = leg_hi - leg_lo
                entry = float(c[t])
                if sign > 0:
                    sl = retest_px - buf * cur_atr
                    targets = [leg_lo + x * rng for x in EXTENSIONS[:stage]]
                else:
                    sl = retest_px + buf * cur_atr
                    targets = [leg_hi - x * rng for x in EXTENSIONS[:stage]]
                risk = abs(entry - sl)
                if risk <= 0 or risk > max_sl * cur_atr:
                    return None
                depth = abs(leg_hi - retest_px) / rng if sign > 0 else abs(retest_px - leg_lo) / rng
                conf = 0.56 + 0.10 * min(abs(targets[0] - entry) / risk, 2.0) / 2.0
                if (sign > 0 and regime == MarketRegime.TRENDING_BULL) or (sign < 0 and regime == MarketRegime.TRENDING_BEAR):
                    conf += 0.06
                side = Side.BUY if sign > 0 else Side.SELL
                return TradeSignal(
                    symbol=symbol,
                    strategy_id=self.strategy_id,
                    side=side,
                    entry_price=entry,
                    sl_price=sl,
                    tp_price=targets[-1],
                    atr=cur_atr,
                    confidence=round(min(conf, 0.9), 4),
                    regime=regime,
                    reason=(
                        f"Structure shift {side.value}: engulfing, break of {'swing high' if sign > 0 else 'swing low'} "
                        f"closed at {leg_hi if sign > 0 else leg_lo:.5f}, retest {depth * 100:.0f}% of the leg"
                    ),
                    features={"retest_depth": round(depth, 4), "leg_atr": round(rng / cur_atr, 3), "bars_since_break": float(t - brk)},
                    tp_stages=[round(x, 8) for x in targets[:-1]],
                    timestamp=win[-1].timestamp,
                )
        return None

    @staticmethod
    def _plan(sign, e, t, o, h, lo, c, a, sw_hi, sw_lo, k, max_break, max_retest, r_min, invalid, min_leg):
        """Return (leg_low, leg_high, break_bar, retest_extreme) when bar t is the entry."""
        # Nearest level: the last swing known at the engulfing bar on the far side of its close.
        level = None
        for s in reversed(sw_hi if sign > 0 else sw_lo):
            if s.confirmed_at > e:
                continue
            if (sign > 0 and s.price > c[e]) or (sign < 0 and s.price < c[e]):
                level = s.price
                break
        if level is None:
            return None
        brk = None
        for b in range(e + 1, min(e + max_break, t - 1) + 1):
            if (sign > 0 and c[b] > level) or (sign < 0 and c[b] < level):
                brk = b
                break
        if brk is None or t - brk > max_retest or t - brk < 2:
            return None
        start = max(0, e - k)
        if sign > 0:
            leg_lo, leg_hi = float(lo[start : brk + 1].min()), float(c[brk])
        else:
            leg_lo, leg_hi = float(c[brk]), float(h[start : brk + 1].max())
        rng = leg_hi - leg_lo
        if rng < min_leg * float(a[brk]):
            return None
        if sign > 0:
            lvl = lambda r: leg_hi - r * rng
            if h[brk + 1 : t + 1].max() >= leg_lo + 1.618 * rng:
                return None  # the move already reached the first target without us
            if c[brk + 1 : t + 1].min() < lvl(invalid):
                return None
            touch = next((j for j in range(brk + 1, t + 1) if lo[j] <= lvl(r_min)), None)
            if touch is None or touch == t:
                return None
            deepest = float(lo[touch : t + 1].min())
            if deepest < lvl(invalid):
                return None
            if not (c[t] > o[t] and c[t] > h[t - 1] and c[t] < leg_hi):
                return None
            for j in range(touch + 1, t):
                if c[j] > o[j] and c[j] > h[j - 1]:
                    return None  # an earlier candle already turned up: not the first trigger
            return leg_lo, leg_hi, brk, deepest
        lvl = lambda r: leg_lo + r * rng
        if lo[brk + 1 : t + 1].min() <= leg_hi - 1.618 * rng:
            return None
        if c[brk + 1 : t + 1].max() > lvl(invalid):
            return None
        touch = next((j for j in range(brk + 1, t + 1) if h[j] >= lvl(r_min)), None)
        if touch is None or touch == t:
            return None
        highest = float(h[touch : t + 1].max())
        if highest > lvl(invalid):
            return None
        if not (c[t] < o[t] and c[t] < lo[t - 1] and c[t] > leg_lo):
            return None
        for j in range(touch + 1, t):
            if c[j] < o[j] and c[j] < lo[j - 1]:
                return None
        return leg_lo, leg_hi, brk, highest
