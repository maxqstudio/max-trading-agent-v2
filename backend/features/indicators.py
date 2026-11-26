from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backend.core.models import Bar


@dataclass(slots=True)
class IndicatorSnapshot:
    close: float
    open: float
    high: float
    low: float
    ema9: float
    ema21: float
    ema50: float
    vwap: float
    atr14: float
    atr_baseline: float
    rsi7: float
    rsi14: float
    adx14: float
    plus_di: float
    minus_di: float
    bb_upper: float
    bb_lower: float
    bb_width: float
    bb_width_pct_rank: float
    upper_wick_ratio: float
    lower_wick_ratio: float
    body_atr_ratio: float
    rel_volume: float
    ema_slope_atr: float


def ema(values: np.ndarray, period: int) -> np.ndarray:
    n = len(values)
    out = np.empty(n, dtype=np.float64)
    if n == 0:
        return out
    alpha = 2.0 / (period + 1.0)
    out[0] = values[0]
    for i in range(1, n):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
    n = len(closes)
    if n == 0:
        return np.array([], dtype=np.float64)
    tr = np.empty(n, dtype=np.float64)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr[i] = max(hl, hc, lc)
    out = np.empty(n, dtype=np.float64)
    out[0] = tr[0]
    alpha = 1.0 / period
    for i in range(1, n):
        out[i] = alpha * tr[i] + (1.0 - alpha) * out[i - 1]
    return np.maximum(out, 1e-9)


def rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    n = len(closes)
    if n == 0:
        return np.array([], dtype=np.float64)
    deltas = np.diff(closes, prepend=closes[0])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.empty(n, dtype=np.float64)
    avg_loss = np.empty(n, dtype=np.float64)
    avg_gain[0] = gains[0]
    avg_loss[0] = losses[0]
    alpha = 1.0 / period
    for i in range(1, n):
        avg_gain[i] = alpha * gains[i] + (1.0 - alpha) * avg_gain[i - 1]
        avg_loss[i] = alpha * losses[i] + (1.0 - alpha) * avg_loss[i - 1]
    rs = np.divide(avg_gain, np.maximum(avg_loss, 1e-12))
    return 100.0 - (100.0 / (1.0 + rs))


def adx(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 14,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(closes)
    if n < 2:
        zeros = np.zeros(n, dtype=np.float64)
        return zeros, zeros, zeros
    up_move = np.diff(highs, prepend=highs[0])
    down_move = np.diff(-lows, prepend=-lows[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr_vals = atr(highs, lows, closes, period)
    alpha = 1.0 / period
    sm_plus = np.empty(n, dtype=np.float64)
    sm_minus = np.empty(n, dtype=np.float64)
    sm_plus[0] = plus_dm[0]
    sm_minus[0] = minus_dm[0]
    for i in range(1, n):
        sm_plus[i] = alpha * plus_dm[i] + (1.0 - alpha) * sm_plus[i - 1]
        sm_minus[i] = alpha * minus_dm[i] + (1.0 - alpha) * sm_minus[i - 1]

    plus_di = 100.0 * sm_plus / np.maximum(atr_vals, 1e-9)
    minus_di = 100.0 * sm_minus / np.maximum(atr_vals, 1e-9)
    dx = 100.0 * np.abs(plus_di - minus_di) / np.maximum(plus_di + minus_di, 1e-9)

    adx_vals = np.empty(n, dtype=np.float64)
    adx_vals[0] = dx[0]
    for i in range(1, n):
        adx_vals[i] = alpha * dx[i] + (1.0 - alpha) * adx_vals[i - 1]
    return adx_vals, plus_di, minus_di


def session_vwap(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, volumes: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    n = len(closes)
    out = np.empty(n, dtype=np.float64)
    cum_pv = 0.0
    cum_v = 0.0
    last_day = -1
    for i in range(n):
        day = int(timestamps[i] // 86400)
        if day != last_day:
            cum_pv = 0.0
            cum_v = 0.0
            last_day = day
        typical = (highs[i] + lows[i] + closes[i]) / 3.0
        vol = max(float(volumes[i]), 1.0)
        cum_pv += typical * vol
        cum_v += vol
        out[i] = cum_pv / cum_v
    return out


def bollinger(closes: np.ndarray, period: int = 20, num_std: float = 2.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(closes)
    mid = np.empty(n, dtype=np.float64)
    upper = np.empty(n, dtype=np.float64)
    lower = np.empty(n, dtype=np.float64)
    for i in range(n):
        start = max(0, i - period + 1)
        window = closes[start : i + 1]
        m = float(np.mean(window))
        s = float(np.std(window))
        mid[i] = m
        upper[i] = m + num_std * s
        lower[i] = m - num_std * s
    return upper, mid, lower


def compute_indicator_snapshot(bars: list[Bar]) -> IndicatorSnapshot | None:
    """Compute strictly causal technical indicators on closed bars up to bars[-1]."""
    if len(bars) < 35:
        return None
    opens = np.array([b.open for b in bars], dtype=np.float64)
    highs = np.array([b.high for b in bars], dtype=np.float64)
    lows = np.array([b.low for b in bars], dtype=np.float64)
    closes = np.array([b.close for b in bars], dtype=np.float64)
    volumes = np.array([max(b.volume, 1.0) for b in bars], dtype=np.float64)
    timestamps = np.array([b.timestamp for b in bars], dtype=np.int64)

    e9 = ema(closes, 9)
    e21 = ema(closes, 21)
    e50 = ema(closes, 50)
    vw = session_vwap(highs, lows, closes, volumes, timestamps)
    a14 = atr(highs, lows, closes, 14)
    r7 = rsi(closes, 7)
    r14 = rsi(closes, 14)
    adx_v, p_di, m_di = adx(highs, lows, closes, 14)
    bb_u, bb_m, bb_l = bollinger(closes, 20, 2.0)

    bb_widths = (bb_u - bb_l) / np.maximum(bb_m, 1e-9)
    recent_w = bb_widths[-min(len(bb_widths), 50) :]
    cur_w = float(bb_widths[-1])
    bb_rank = float(np.mean(recent_w <= cur_w))

    cur_atr = float(a14[-1])
    base_atr = float(np.median(a14[-min(len(a14), 60) :]))

    o, h, l, c = float(opens[-1]), float(highs[-1]), float(lows[-1]), float(closes[-1])
    bar_range = max(h - l, 1e-9)
    body = abs(c - o)
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l

    vol_base = float(np.mean(volumes[-min(len(volumes), 20) :]))
    ema_slope = (float(e9[-1]) - float(e9[-4])) / max(cur_atr, 1e-9)

    return IndicatorSnapshot(
        close=c,
        open=o,
        high=h,
        low=l,
        ema9=float(e9[-1]),
        ema21=float(e21[-1]),
        ema50=float(e50[-1]),
        vwap=float(vw[-1]),
        atr14=cur_atr,
        atr_baseline=max(base_atr, 1e-9),
        rsi7=float(r7[-1]),
        rsi14=float(r14[-1]),
        adx14=float(adx_v[-1]),
        plus_di=float(p_di[-1]),
        minus_di=float(m_di[-1]),
        bb_upper=float(bb_u[-1]),
        bb_lower=float(bb_l[-1]),
        bb_width=cur_w,
        bb_width_pct_rank=bb_rank,
        upper_wick_ratio=upper_wick / bar_range,
        lower_wick_ratio=lower_wick / bar_range,
        body_atr_ratio=body / max(cur_atr, 1e-9),
        rel_volume=float(volumes[-1]) / max(vol_base, 1.0),
        ema_slope_atr=ema_slope,
    )
