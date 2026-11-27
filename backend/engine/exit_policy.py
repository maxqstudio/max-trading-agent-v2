"""Exit rules for protecting open profit. Pure functions, shared by MT5 positions and paper books.

ProfitLock (existing system): once a trade has been +be_at_r in profit, the stop goes to breakeven
(+ costs); from +trail_start_r it trails trail_dist_r behind the best price. This fixes trades
that were in profit and then came all the way back to a full -1R loss.

NetTake (Hit & Run): close as soon as the trade is in net profit, i.e. after spread, commission
and swap, once the net profit reaches min_net_profit (default $1). No profit target in R: a small
net win is taken. The original stop loss stays as the loss limit.
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
class NetTake:
    min_net_profit: float = 1.0  # close once the net profit (after all costs) reaches this much money
    commission_per_lot: float = 7.0  # round-trip estimate, used when the real commission is unknown
    min_cost_multiple: float = 0.0  # when > 0, require net profit >= min_cost_multiple * (commission + spread)


def net_take(
    policy: NetTake,
    *,
    gross_money: float,
    volume: float,
    swap: float = 0.0,
    commission: float | None = None,
    spread_money: float = 0.0,
) -> tuple[bool, float]:
    """(take_now, net_money). ``gross_money`` must already be valued at the closing side
    (bid for a buy, ask for a sell), so the spread is included. ``commission`` is the round-trip
    commission in money (positive); None uses the per-lot estimate."""
    costs = abs(commission) if commission is not None else policy.commission_per_lot * volume
    net = round(gross_money + swap - costs, 2)
    total_friction = costs + max(spread_money, 0.0)
    floor = max(policy.min_net_profit, policy.min_cost_multiple * total_friction, 0.01)
    return net >= floor, net


def _price(side: Side, entry: float, r: float, risk: float) -> float:
    return entry + r * risk if side == Side.BUY else entry - r * risk


def _better(side: Side, new_sl: float, cur_sl: float) -> bool:
    if cur_sl <= 0:
        return True
    return new_sl > cur_sl if side == Side.BUY else new_sl < cur_sl


def stage_stop(*, side: Side, entry: float, stages: list[float], peak_price: float, sl: float, cost_px: float = 0.0) -> float | None:
    """Staged targets: once the best price has passed stage i, the stop goes to breakeven
    (+ costs) for the first stage, and to the previous stage after that. Returns the new stop
    when it is better than ``sl``, else None."""
    passed = [i for i, lvl in enumerate(stages) if (peak_price >= lvl if side == Side.BUY else peak_price <= lvl)]
    if not passed:
        return None
    i = passed[-1]
    if i == 0:
        new_sl = entry + cost_px if side == Side.BUY else entry - cost_px
    else:
        new_sl = stages[i - 1]
    return new_sl if _better(side, new_sl, sl) else None


def evaluate(
    policy: ProfitLock,
    *,
    side: Side,
    entry: float,
    risk: float,
    price: float,
    peak_r: float,
    sl: float,
    spread_px: float = 0.0,
    commission_px: float = 0.0,
) -> tuple[Action, float]:
    """Return (action, new_sl). ``peak_r`` is the best open profit so far in R (MFE)."""
    risk = max(risk, 1e-12)
    cost = max(spread_px, 0.0) + max(commission_px, 0.0)  # breakeven must cover spread + round-trip commission
    if peak_r < policy.be_at_r:
        return "HOLD", sl
    lock_r = policy.be_offset_r + cost / risk
    if peak_r >= policy.trail_start_r:
        lock_r = max(lock_r, peak_r - policy.trail_dist_r)
    new_sl = _price(side, entry, lock_r, risk)
    return ("MOVE_SL", new_sl) if _better(side, new_sl, sl) else ("HOLD", sl)
