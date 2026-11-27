"""Market-structure building blocks: swings, engulfing, supply/demand bases, Ichimoku cloud and
Bollinger position. Everything here is causal: a value at bar i only uses bars up to i.

Ichimoku itself lives in ``ichimoku.py``. These feed the Structure Shift strategy and are appended to the ML meta features, so the models
learn how much each one matters instead of the robot treating them as hard rules.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backend.core.models import Bar
from backend.features import ichimoku
from backend.features.indicators import atr, bollinger


@dataclass(frozen=True, slots=True)
class Swing:
    index: int  # bar of the swing point
    price: float
    confirmed_at: int  # first bar at which the swing is known (index + k)


def swings(highs: np.ndarray, lows: np.ndarray, k: int = 3) -> tuple[list[Swing], list[Swing]]:
    """Fractal swing highs and lows: a bar higher (lower) than the k bars on each side.
    A swing is only known k bars later, which ``confirmed_at`` records."""
    n = len(highs)
    sh: list[Swing] = []
    sl: list[Swing] = []
    for i in range(k, n - k):
        h, lo = highs[i], lows[i]
        if h > highs[i - k : i].max() and h >= highs[i + 1 : i + k + 1].max():
            sh.append(Swing(i, float(h), i + k))
        if lo < lows[i - k : i].min() and lo <= lows[i + 1 : i + k + 1].min():
            sl.append(Swing(i, float(lo), i + k))
    return sh, sl


def engulfing_at(opens: np.ndarray, closes: np.ndarray, i: int, max_run: int = 3) -> int:
    """+1 bullish / -1 bearish / 0. The candle at i closes beyond the open of the opposite run
    before it (one to ``max_run`` candles), so it "swallows" that move, not only one body."""
    if i < 1:
        return 0
    body = closes[i] - opens[i]
    if body > 0:
        j = i - 1
        top = None
        while j >= 0 and i - j <= max_run and closes[j] < opens[j]:
            top = opens[j] if top is None else max(top, opens[j])
            j -= 1
        if top is not None and closes[i] > top and opens[i] <= closes[i - 1] + 1e-12:
            return 1
    elif body < 0:
        j = i - 1
        bottom = None
        while j >= 0 and i - j <= max_run and closes[j] > opens[j]:
            bottom = opens[j] if bottom is None else min(bottom, opens[j])
            j -= 1
        if bottom is not None and closes[i] < bottom and opens[i] >= closes[i - 1] - 1e-12:
            return -1
    return 0


@dataclass(frozen=True, slots=True)
class Zone:
    kind: str  # "demand" (rally out: RBR / DBR) or "supply" (drop out: DBD / RBD)
    pattern: str  # RBR, DBR, DBD or RBD
    low: float
    high: float
    formed_at: int  # bar at which the leg out confirmed the zone


def sd_zones(
    opens: np.ndarray, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, atr_vals: np.ndarray,
    base_max: int = 4, base_body_atr: float = 0.5, leg_bars: int = 3, leg_atr: float = 1.5,
) -> list[Zone]:
    """Fresh supply/demand bases. A base is 1 to ``base_max`` small candles (body at most
    ``base_body_atr`` ATR). The leg out over the next ``leg_bars`` candles must move at least
    ``leg_atr`` ATR away from the base. The leg in tells the pattern (rally or drop into the
    base). A zone stops being fresh once a candle closes through it."""
    n = len(closes)
    bodies = np.abs(closes - opens)
    zones: list[Zone] = []
    i = leg_bars
    while i < n - leg_bars:
        a = max(float(atr_vals[i]), 1e-9)
        if bodies[i] > base_body_atr * a:
            i += 1
            continue
        j = i
        while j + 1 < n and j - i + 1 < base_max and bodies[j + 1] <= base_body_atr * float(atr_vals[j + 1]):
            j += 1
        if j + leg_bars >= n:
            break
        b_lo, b_hi = float(lows[i : j + 1].min()), float(highs[i : j + 1].max())
        out_close = float(closes[j + leg_bars])
        in_move = float(closes[i - 1] - closes[max(0, i - 1 - leg_bars)])
        rally_in = "R" if in_move > 0 else "D"
        if out_close - b_hi >= leg_atr * a:
            zones.append(Zone("demand", f"{rally_in}BR", b_lo, b_hi, j + leg_bars))
        elif b_lo - out_close >= leg_atr * a:
            zones.append(Zone("supply", f"{rally_in}BD", b_lo, b_hi, j + leg_bars))
        i = j + 1
    fresh: list[Zone] = []
    for z in zones:
        after = closes[z.formed_at + 1 :]
        broken = bool(np.any(after < z.low)) if z.kind == "demand" else bool(np.any(after > z.high))
        if not broken:
            fresh.append(z)
    return fresh


CONTEXT_KEYS = (
    "demand_near", "supply_near", "kumo", "bb_pos", "engulf",
    "ichi_score", "ichi_m15", "ichi_tk_cross", "ichi_chikou", "ichi_future_cloud", "ichi_thick_atr", "ichi_kijun_dist_atr",
)


def structure_context(bars: list[Bar], ichi_m15: int | None = None) -> dict[str, float]:
    """Direction-neutral context at the last bar. ``extract_meta_features`` turns it into
    with-the-trade / against-the-trade values. ``ichi_m15`` is the M15 Ichimoku checklist score
    (computed from a longer M1 history than ``bars``); None leaves it at 0."""
    out = dict.fromkeys(CONTEXT_KEYS, 0.0)
    out["ichi_m15"] = float(ichi_m15 or 0)
    if len(bars) < 40:
        return out
    win = bars[-300:]
    o = np.array([b.open for b in win], dtype=np.float64)
    h = np.array([b.high for b in win], dtype=np.float64)
    lo = np.array([b.low for b in win], dtype=np.float64)
    c = np.array([b.close for b in win], dtype=np.float64)
    a = atr(h, lo, c, 14)
    cur_atr = max(float(a[-1]), 1e-9)
    px = float(c[-1])
    near = 0.5 * cur_atr
    for z in sd_zones(o, h, lo, c, a):
        if z.kind == "demand" and z.low - near <= px <= z.high + near:
            out["demand_near"] = 1.0
        if z.kind == "supply" and z.low - near <= px <= z.high + near:
            out["supply_near"] = 1.0
    ichi = ichimoku.read(h, lo, c, cur_atr)
    out.update(
        kumo=float(ichi.price_cloud),
        ichi_score=float(ichi.score),
        ichi_tk_cross=ichi.tk_cross,
        ichi_chikou=float(ichi.chikou),
        ichi_future_cloud=float(ichi.future_cloud),
        ichi_thick_atr=min(ichi.cloud_thick_atr, 3.0),
        ichi_kijun_dist_atr=float(np.clip(ichi.kijun_dist_atr, -3.0, 3.0)),
    )
    up, mid, _ = bollinger(c[-40:], 20, 2.0)
    half = max(float(up[-1] - mid[-1]), 1e-9)
    out["bb_pos"] = float(np.clip((px - float(mid[-1])) / half, -1.5, 1.5))
    for i in range(len(c) - 1, len(c) - 4, -1):
        e = engulfing_at(o, c, i)
        if e:
            out["engulf"] = float(e)
            break
    return out


ALIGNED_NAMES = (
    "sd_zone_align", "kumo_align", "bb_pos_align", "engulf_align",
    "ichi_score_align", "ichi_m15_align", "ichi_tk_cross_align", "ichi_chikou_align", "ichi_future_cloud_align",
    "ichi_cloud_thick_atr", "ichi_kijun_dist_align",
)


def aligned_context(ctx: dict[str, float] | None, side_sign: float) -> list[float]:
    """The context signed with a trade in ``side_sign`` (+1 = with the trade), in the order of
    ``ALIGNED_NAMES``. Checklist scores are scaled to -1..+1; cloud thickness has no side."""
    if not ctx:
        return [0.0] * len(ALIGNED_NAMES)
    with_zone = ctx.get("demand_near", 0.0) if side_sign > 0 else ctx.get("supply_near", 0.0)
    against_zone = ctx.get("supply_near", 0.0) if side_sign > 0 else ctx.get("demand_near", 0.0)
    return [
        float(with_zone - against_zone),
        float(ctx.get("kumo", 0.0) * side_sign),
        float(ctx.get("bb_pos", 0.0) * side_sign),
        float(ctx.get("engulf", 0.0) * side_sign),
        float(ctx.get("ichi_score", 0.0) * side_sign / 5.0),
        float(ctx.get("ichi_m15", 0.0) * side_sign / 5.0),
        float(ctx.get("ichi_tk_cross", 0.0) * side_sign),
        float(ctx.get("ichi_chikou", 0.0) * side_sign),
        float(ctx.get("ichi_future_cloud", 0.0) * side_sign),
        float(ctx.get("ichi_thick_atr", 0.0)),
        float(ctx.get("ichi_kijun_dist_atr", 0.0) * side_sign),
    ]
