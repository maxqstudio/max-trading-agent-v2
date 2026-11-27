"""Previous trading day's high and low, from M1 bars.

MT5 bar times are broker server time and MT5 daily candles are cut on that same clock, so the
day is simply ``timestamp // 86400``. The previous day is the last day with bars before the
current one (Friday on a Monday). It only counts when the buffer holds it from its first hour,
otherwise its range would be cut short.
"""

from __future__ import annotations

from collections.abc import Sequence

from backend.core.models import Bar

DAY = 86400
FIRST_HOUR = 3600


def prev_day_range(bars: Sequence[Bar]) -> tuple[float, float] | None:
    """(high, low) of the previous day as known at the last bar, or None."""
    if not bars:
        return None
    today = bars[-1].timestamp // DAY
    hi = lo = None
    day = None
    i = len(bars) - 1
    while i >= 0 and bars[i].timestamp // DAY == today:
        i -= 1
    if i < 0:
        return None
    day = bars[i].timestamp // DAY
    first_ts = bars[i].timestamp
    while i >= 0 and bars[i].timestamp // DAY == day:
        b = bars[i]
        hi = b.high if hi is None else max(hi, b.high)
        lo = b.low if lo is None else min(lo, b.low)
        first_ts = b.timestamp
        i -= 1
    if i < 0 and first_ts % DAY >= FIRST_HOUR:
        return None  # the buffer starts in the middle of that day
    return (hi, lo) if hi is not None and lo is not None else None


def prev_day_series(bars: Sequence[Bar]) -> list[tuple[float, float] | None]:
    """``prev_day_range`` as known at every bar, in one pass (for the backtester)."""
    out: list[tuple[float, float] | None] = []
    cur_day = None
    cur_hi = cur_lo = 0.0
    cur_complete = False
    prev: tuple[float, float] | None = None
    for idx, b in enumerate(bars):
        d = b.timestamp // DAY
        if d != cur_day:
            if cur_day is not None:
                prev = (cur_hi, cur_lo) if cur_complete else None
            cur_day = d
            cur_hi, cur_lo = b.high, b.low
            cur_complete = idx > 0 or b.timestamp % DAY < FIRST_HOUR
        else:
            cur_hi = max(cur_hi, b.high)
            cur_lo = min(cur_lo, b.low)
        out.append(prev)
    return out


def range_position(price: float, rng: tuple[float, float] | None) -> float:
    """-1 at the previous day's low, +1 at its high, beyond that clipped to -1.5..+1.5; 0 if unknown."""
    if not rng:
        return 0.0
    hi, lo = rng
    if hi - lo <= 0:
        return 0.0
    return max(-1.5, min(1.5, (price - lo) / (hi - lo) * 2.0 - 1.0))
