"""Higher-timeframe direction (M15 / H1) and trading session, built from the M1 bars we already have.

Bias for one timeframe, using completed higher-timeframe bars only (no look-ahead):
- EMA(20) slope over the last 3 bars, ignored when smaller than 10% of the average bar range;
- structure: the last 6 bars made a higher high and a higher low than the 6 before (up),
  or a lower high and a lower low (down).
+1 (up) or -1 (down) when both agree, 0 otherwise.

Bar timestamps from MT5 are in the broker's server time; ``server_offset_sec`` (server minus
UTC, reported by the EA) converts them to UTC for the session.
"""

from __future__ import annotations

from typing import Sequence

from backend.core.models import Bar

EMA_PERIOD = 20
SLOPE_LOOKBACK = 3
STRUCT_BARS = 6
MIN_HTF_BARS = EMA_PERIOD + SLOPE_LOOKBACK
SESSIONS = ("asia", "london", "overlap", "newyork", "late")


class _HtfState:
    """Completed higher-timeframe bars and an incremental EMA."""

    def __init__(self, minutes: int) -> None:
        self.sec = minutes * 60
        self.bucket: int | None = None
        self.cur: list[float] | None = None  # [high, low, close] of the bucket being built
        self.highs: list[float] = []
        self.lows: list[float] = []
        self.closes: list[float] = []
        self.ema: list[float] = []
        self.bias = 0

    def add(self, bar: Bar) -> None:
        b = bar.timestamp // self.sec
        if self.bucket is None or b != self.bucket:
            if self.cur is not None:
                self._complete(*self.cur)
            self.bucket = b
            self.cur = [bar.high, bar.low, bar.close]
        else:
            self.cur[0] = max(self.cur[0], bar.high)
            self.cur[1] = min(self.cur[1], bar.low)
            self.cur[2] = bar.close

    def _complete(self, high: float, low: float, close: float) -> None:
        self.highs.append(high)
        self.lows.append(low)
        self.closes.append(close)
        k = 2.0 / (EMA_PERIOD + 1)
        self.ema.append(close if not self.ema else self.ema[-1] + k * (close - self.ema[-1]))
        self.bias = self._bias()

    def _bias(self) -> int:
        n = len(self.closes)
        if n < MIN_HTF_BARS or n < 2 * STRUCT_BARS:
            return 0
        avg_range = sum(h - l for h, l in zip(self.highs[-EMA_PERIOD:], self.lows[-EMA_PERIOD:])) / EMA_PERIOD
        slope = self.ema[-1] - self.ema[-1 - SLOPE_LOOKBACK]
        slope_dir = 0 if abs(slope) < 0.1 * avg_range else (1 if slope > 0 else -1)
        hi_now, hi_before = max(self.highs[-STRUCT_BARS:]), max(self.highs[-2 * STRUCT_BARS : -STRUCT_BARS])
        lo_now, lo_before = min(self.lows[-STRUCT_BARS:]), min(self.lows[-2 * STRUCT_BARS : -STRUCT_BARS])
        if hi_now > hi_before and lo_now > lo_before:
            struct_dir = 1
        elif hi_now < hi_before and lo_now < lo_before:
            struct_dir = -1
        else:
            struct_dir = 0
        return slope_dir if slope_dir != 0 and slope_dir == struct_dir else 0


def htf_bias_series(bars: Sequence[Bar], minutes: int) -> list[int]:
    """Bias of the ``minutes`` timeframe as known at each M1 bar (completed HTF bars only)."""
    st = _HtfState(minutes)
    out: list[int] = []
    for bar in bars:
        st.add(bar)
        out.append(st.bias)
    return out


def htf_bias(bars: Sequence[Bar], minutes: int) -> int:
    """Bias right now (after the last M1 bar)."""
    st = _HtfState(minutes)
    for bar in bars:
        st.add(bar)
    return st.bias


def session_of(ts_sec: int, server_offset_sec: int = 0) -> str:
    """Trading session for a bar timestamp (UTC hours, no daylight saving adjustment)."""
    h = ((ts_sec - server_offset_sec) % 86400) // 3600
    if 7 <= h < 12:
        return "london"
    if 12 <= h < 16:
        return "overlap"  # London and New York both open
    if 16 <= h < 21:
        return "newyork"
    if 21 <= h < 23:
        return "late"
    return "asia"


def alignment(side_sign: float, bias: int) -> int:
    """+1 when the trade goes with the bias, -1 against it, 0 when there is no bias."""
    return int(bias * (1 if side_sign > 0 else -1))
