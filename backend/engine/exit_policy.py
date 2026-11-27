"""Exit rules for protecting open profit. Pure functions, shared by MT5 positions and paper books.

ProfitLock (existing system): once a trade has been +be_at_r in profit, the stop goes to breakeven
(+ costs); from +trail_start_r it trails trail_dist_r behind the best price. This fixes trades
that were in profit and then came all the way back to a full -1R loss.

HitRun: take profit fast. When the trade reaches +take_r and the profit is at least
min_spread_mult x spread, either close at once (trail_dist_r = 0) or lock breakeven and trail very
tight. The original stop loss always stays active.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from backend.core.models import Side

Action = Literal["HOLD", "MOVE_SL", "CLOSE"]


@dataclass(frozen=True, slots=True)
class ProfitLock:
    be_at_r: float = 0.5
    be_offset_r: float = 0.05
    trail_start_r: float = 1.0
    trail_dist_r: float = 0.5


@dataclass(frozen=True, slots=True)
class HitRun:
    take_r: float = 0.5
    min_spread_mult: float = 2.0
    trail_dist_r: float = 0.15  # 0 = close as soon as take_r is reached
    be_offset_r: float = 0.05


def _price(side: Side, entry: float, r: float, risk: float) -> float:
    return entry + r * risk if side == Side.BUY else entry - r * risk


def _better(side: Side, new_sl: float, cur_sl: float) -> bool:
    if cur_sl <= 0:
        return True
    return new_sl > cur_sl if side == Side.BUY else new_sl < cur_sl


def evaluate(
    policy: ProfitLock | HitRun,
    *,
    side: Side,
    entry: float,
    risk: float,
    price: float,
    peak_r: float,
    sl: float,
    spread_px: float = 0.0,
) -> tuple[Action, float]:
    """Return (action, new_sl). ``peak_r`` is the best open profit so far in R (MFE)."""
    risk = max(risk, 1e-12)
    cost = spread_px  # breakeven must cover the spread paid on entry
    if isinstance(policy, HitRun):
        profit_px = (price - entry) if side == Side.BUY else (entry - price)
        armed = peak_r >= policy.take_r and (peak_r * risk) >= policy.min_spread_mult * spread_px
        if not armed:
            return "HOLD", sl
        cur_r = profit_px / risk
        if policy.trail_dist_r <= 0 and cur_r >= policy.take_r:
            return "CLOSE", sl
        lock_r = max(policy.be_offset_r + cost / risk, peak_r - policy.trail_dist_r)
        new_sl = _price(side, entry, lock_r, risk)
        return ("MOVE_SL", new_sl) if _better(side, new_sl, sl) else ("HOLD", sl)

    if peak_r < policy.be_at_r:
        return "HOLD", sl
    lock_r = policy.be_offset_r + cost / risk
    if peak_r >= policy.trail_start_r:
        lock_r = max(lock_r, peak_r - policy.trail_dist_r)
    new_sl = _price(side, entry, lock_r, risk)
    return ("MOVE_SL", new_sl) if _better(side, new_sl, sl) else ("HOLD", sl)
