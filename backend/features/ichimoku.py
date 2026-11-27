"""Ichimoku Kinko Hyo (9 / 26 / 52), read the way a checklist reads it. Causal: values at bar t
only use bars up to t.

Lines at bar t:
- Tenkan  = midpoint of the last 9 highs/lows; Kijun = midpoint of the last 26.
- Current cloud = Span A ((Tenkan + Kijun) / 2) and Span B (midpoint of 52) computed 26 bars ago.
- Future cloud  = the same spans computed now (they are drawn 26 bars ahead).
- Chikou = today's close compared with the candle (and the cloud) of 26 bars ago.

Checklist, each +1 bullish / -1 bearish / 0:
  1. price against the current cloud, 2. Tenkan against Kijun, 3. Chikou against the price and the
  cloud of 26 bars ago, 4. price against Kijun, 5. future cloud colour (Span A above or below B).
The score is their sum (-5 to +5). The engine does not trade on it directly: it is an ML feature
and an entry filter whose "auto" mode only blocks once paper results show it helps.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from backend.core.models import Bar

TENKAN, KIJUN, SENKOU = 9, 26, 52
SHIFT = KIJUN
MIN_BARS = SENKOU + 2 * SHIFT + 1  # Chikou needs the cloud of 26 bars ago, itself shifted 26
CROSS_LOOKBACK = 10


@dataclass(frozen=True, slots=True)
class IchimokuRead:
    score: int = 0  # checklist sum, -5..+5
    price_cloud: int = 0
    tk: int = 0
    chikou: int = 0
    price_kijun: int = 0
    future_cloud: int = 0
    tk_cross: float = 0.0  # recent Tenkan/Kijun cross: +-1 strong, +-0.5 inside the cloud, +-0.25 weak
    cloud_thick_atr: float = 0.0
    kijun_dist_atr: float = 0.0  # (close - Kijun) / ATR: how stretched price is from its "magnet"
    cloud_dist_atr: float = 0.0  # distance from the nearest edge when outside the cloud, signed


def _mid(h: np.ndarray, lo: np.ndarray, period: int, end: int) -> float:
    s = end - period + 1
    return (float(h[s : end + 1].max()) + float(lo[s : end + 1].min())) / 2.0


def _spans(h: np.ndarray, lo: np.ndarray, end: int) -> tuple[float, float]:
    a = (_mid(h, lo, TENKAN, end) + _mid(h, lo, KIJUN, end)) / 2.0
    return a, _mid(h, lo, SENKOU, end)


def _side(x: float, lo_edge: float, hi_edge: float) -> int:
    return 1 if x > hi_edge else -1 if x < lo_edge else 0


def read(h: np.ndarray, lo: np.ndarray, c: np.ndarray, atr_now: float, t: int | None = None) -> IchimokuRead:
    t = len(c) - 1 if t is None else t
    if t + 1 < MIN_BARS:
        return IchimokuRead()
    a = max(atr_now, 1e-9)
    tenkan, kijun = _mid(h, lo, TENKAN, t), _mid(h, lo, KIJUN, t)
    span_a, span_b = _spans(h, lo, t - SHIFT)
    top, bottom = max(span_a, span_b), min(span_a, span_b)
    fut_a, fut_b = _spans(h, lo, t)
    px = float(c[t])

    price_cloud = _side(px, bottom, top)
    tk = 1 if tenkan > kijun else -1 if tenkan < kijun else 0
    # Chikou: today's close plotted 26 bars back, against that candle and that bar's cloud.
    then = t - SHIFT
    pa, pb = _spans(h, lo, then - SHIFT)
    vs_candle = 1 if px > h[then] else -1 if px < lo[then] else 0
    vs_cloud = _side(px, min(pa, pb), max(pa, pb))
    chikou = vs_candle if vs_candle == vs_cloud else 0
    price_kijun = 1 if px > kijun else -1 if px < kijun else 0
    future_cloud = 1 if fut_a > fut_b else -1 if fut_a < fut_b else 0

    cross = 0.0
    for j in range(t, max(t - CROSS_LOOKBACK, SENKOU + SHIFT), -1):
        d_now = _mid(h, lo, TENKAN, j) - _mid(h, lo, KIJUN, j)
        d_prev = _mid(h, lo, TENKAN, j - 1) - _mid(h, lo, KIJUN, j - 1)
        if d_now == 0 or np.sign(d_now) == np.sign(d_prev):
            continue
        direction = 1 if d_now > 0 else -1
        ca, cb = _spans(h, lo, j - SHIFT)
        where = _side(_mid(h, lo, KIJUN, j), min(ca, cb), max(ca, cb))
        quality = 1.0 if where == direction else 0.5 if where == 0 else 0.25
        cross = direction * quality
        break

    dist = (px - top) if price_cloud > 0 else (px - bottom) if price_cloud < 0 else 0.0
    return IchimokuRead(
        score=price_cloud + tk + chikou + price_kijun + future_cloud,
        price_cloud=price_cloud,
        tk=tk,
        chikou=chikou,
        price_kijun=price_kijun,
        future_cloud=future_cloud,
        tk_cross=cross,
        cloud_thick_atr=min((top - bottom) / a, 5.0),
        kijun_dist_atr=float(np.clip((px - kijun) / a, -5.0, 5.0)),
        cloud_dist_atr=float(np.clip(dist / a, -5.0, 5.0)),
    )


def _resample(bars: Sequence[Bar], minutes: int) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
    """Completed higher-timeframe candles: (bucket ids, highs, lows, closes)."""
    sec = minutes * 60
    ids: list[int] = []
    hs: list[float] = []
    ls: list[float] = []
    cs: list[float] = []
    for b in bars:
        k = b.timestamp // sec
        if ids and ids[-1] == k:
            hs[-1] = max(hs[-1], b.high)
            ls[-1] = min(ls[-1], b.low)
            cs[-1] = b.close
        else:
            ids.append(k)
            hs.append(b.high)
            ls.append(b.low)
            cs.append(b.close)
    return ids, np.array(hs), np.array(ls), np.array(cs)


def _atr_simple(h: np.ndarray, lo: np.ndarray, c: np.ndarray, end: int, period: int = 14) -> float:
    s = max(1, end - period + 1)
    tr = [max(h[i] - lo[i], abs(h[i] - c[i - 1]), abs(lo[i] - c[i - 1])) for i in range(s, end + 1)]
    return float(np.mean(tr)) if tr else 0.0


def htf_score(bars: Sequence[Bar], minutes: int = 15) -> int:
    """Checklist score on completed ``minutes`` candles built from M1 bars (the last, still
    forming candle is left out). 0 while there is not enough history."""
    ids, h, lo, c = _resample(bars, minutes)
    t = len(ids) - 2  # last completed candle
    if t + 1 < MIN_BARS:
        return 0
    return read(h, lo, c, _atr_simple(h, lo, c, t), t).score


def htf_score_series(bars: Sequence[Bar], minutes: int = 15) -> list[int]:
    """``htf_score`` as known at every M1 bar, for the backtester."""
    ids, h, lo, c = _resample(bars, minutes)
    per_candle = [0] * len(ids)
    for t in range(MIN_BARS - 1, len(ids)):
        per_candle[t] = read(h, lo, c, _atr_simple(h, lo, c, t), t).score
    out: list[int] = []
    sec = minutes * 60
    pos = 0
    for b in bars:
        k = b.timestamp // sec
        while pos < len(ids) and ids[pos] < k:
            pos += 1
        out.append(per_candle[pos - 1] if pos >= 1 else 0)  # the candle before the one forming
    return out


def score_bucket(aligned_score: int, threshold: int = 3) -> str:
    """with / mixed / against, for a score already signed with the trade."""
    return "with" if aligned_score >= threshold else "against" if aligned_score <= -threshold else "mixed"
