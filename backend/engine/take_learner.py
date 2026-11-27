"""Adaptive Hit & Run take level, learned per symbol and market regime.

Every Hit & Run paper trade is replayed on the same price path for each candidate take level
(``ARMS``, in R). The real paper position uses one of them; the others are "ghost" exits that
follow the same ticks, the same original SL and the same spread rule. So each trade gives a
result for every arm, with no guessing about what the other levels would have done.

The learner keeps the default (``default_take_r``) until a group has ``min_samples`` finished
trades and another arm's average R beats the default by more than ``z`` standard errors.
Groups fall back from symbol+regime to symbol to all trades, like the exit-style selector.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from backend.core.models import Side
from backend.engine.exit_policy import HitRun, evaluate

ARMS: tuple[float, ...] = (0.3, 0.5, 0.7, 1.0)


@dataclass
class _Stat:
    n: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    wins: int = 0

    def add(self, r: float) -> None:
        self.n += 1
        self.total += r
        self.total_sq += r * r
        self.wins += r > 0

    @property
    def mean(self) -> float:
        return self.total / self.n if self.n else 0.0

    @property
    def var(self) -> float:
        if self.n < 2:
            return 0.0
        return max(0.0, (self.total_sq - self.total * self.total / self.n) / (self.n - 1))


@dataclass
class GhostArm:
    take_r: float
    sl: float
    peak_r: float = 0.0
    result_r: float | None = None


@dataclass
class GhostTrade:
    """One Hit & Run setup replayed for every take level."""

    symbol: str
    regime: str
    side: str
    entry: float
    risk: float
    tp: float
    open_time: int
    arms: list[GhostArm] = field(default_factory=list)

    @property
    def done(self) -> bool:
        return all(a.result_r is not None for a in self.arms)

    def step(self, price: float, spread_px: float, base: HitRun, timed_out: bool = False) -> bool:
        """Advance every open arm on one price. Returns True once all arms have finished."""
        side = Side(self.side)
        cur_r = ((price - self.entry) if side == Side.BUY else (self.entry - price)) / max(self.risk, 1e-12)
        for a in self.arms:
            if a.result_r is not None:
                continue
            a.peak_r = max(a.peak_r, cur_r)
            pol = HitRun(take_r=a.take_r, min_spread_mult=base.min_spread_mult, trail_dist_r=base.trail_dist_r, be_offset_r=base.be_offset_r)
            action, new_sl = evaluate(pol, side=side, entry=self.entry, risk=self.risk, price=price, peak_r=a.peak_r, sl=a.sl, spread_px=spread_px)
            if action == "MOVE_SL":
                a.sl = new_sl
            hit_sl = a.sl > 0 and ((side == Side.BUY and price <= a.sl) or (side == Side.SELL and price >= a.sl))
            hit_tp = self.tp > 0 and ((side == Side.BUY and price >= self.tp) or (side == Side.SELL and price <= self.tp))
            if action == "CLOSE" or hit_sl or hit_tp or timed_out:
                a.result_r = round(cur_r, 4)
        return self.done

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GhostTrade | None:
        try:
            arms = [GhostArm(**a) for a in d.get("arms", [])]
            return cls(**{**d, "arms": arms})
        except (KeyError, TypeError, ValueError):
            return None


@dataclass
class TakeChoice:
    take_r: float
    scope: str
    reason: str


class TakeLevelLearner:
    def __init__(self, default_take_r: float = 0.5, min_samples: int = 20, z: float = 1.0) -> None:
        self.default_take_r = default_take_r
        self.min_samples = min_samples
        self.z = z
        self.groups: dict[str, dict[float, _Stat]] = {}

    def arms(self) -> tuple[float, ...]:
        return tuple(sorted(set(ARMS) | {round(self.default_take_r, 2)}))

    def new_ghost(self, *, symbol: str, regime: str, side: Side, entry: float, risk: float, sl: float, tp: float, open_time: int) -> GhostTrade:
        return GhostTrade(
            symbol=symbol, regime=regime, side=side.value, entry=entry, risk=risk, tp=tp, open_time=open_time,
            arms=[GhostArm(take_r=t, sl=sl) for t in self.arms()],
        )

    @staticmethod
    def _keys(symbol: str, regime: str) -> list[str]:
        return [f"{symbol}|{regime}", symbol, "*"]

    def record(self, ghost: GhostTrade) -> None:
        for key in self._keys(ghost.symbol, ghost.regime):
            g = self.groups.setdefault(key, {})
            for a in ghost.arms:
                if a.result_r is not None:
                    g.setdefault(round(a.take_r, 2), _Stat()).add(a.result_r)

    def _verdict(self, g: dict[float, _Stat]) -> tuple[float, str] | None:
        base = g.get(round(self.default_take_r, 2))
        if base is None or base.n < self.min_samples:
            return None
        best_t, best = round(self.default_take_r, 2), base
        for t, s in g.items():
            if s.n >= self.min_samples and s.mean > best.mean:
                best_t, best = t, s
        if best_t == round(self.default_take_r, 2):
            return best_t, f"{best_t}R tetap terbaik ({base.mean:+.2f}R rata-rata, {base.n} trade)"
        se = math.sqrt(best.var / best.n + base.var / base.n)
        if best.mean - base.mean > self.z * se:
            return best_t, f"{best_t}R {best.mean:+.2f}R vs {self.default_take_r}R {base.mean:+.2f}R ({best.n} trade)"
        return round(self.default_take_r, 2), f"{best_t}R sedikit lebih baik tapi belum meyakinkan ({best.n} trade)"

    def choose(self, symbol: str, regime: str) -> TakeChoice:
        for key, scope in zip(self._keys(symbol, regime), ("symbol|regime", "symbol", "all")):
            g = self.groups.get(key)
            v = self._verdict(g) if g else None
            if v:
                return TakeChoice(v[0], scope, v[1])
        return TakeChoice(self.default_take_r, "default", f"Belum cukup data (minimal {self.min_samples} trade)")

    def table(self) -> list[dict[str, Any]]:
        rows = []
        for key, g in sorted(self.groups.items()):
            v = self._verdict(g)
            base = g.get(round(self.default_take_r, 2))
            rows.append({
                "key": "Semua" if key == "*" else key.replace("|", " · "),
                "arms": [
                    {"take_r": t, "n": s.n, "avg_r": round(s.mean, 3), "wr": round(s.wins / s.n * 100, 1) if s.n else 0.0}
                    for t, s in sorted(g.items())
                ],
                "pick": v[0] if v else None,
                "reason": v[1] if v else "",
                "need": max(0, self.min_samples - (base.n if base else 0)),
            })
        return rows

    def export(self) -> dict[str, Any]:
        return {k: {str(t): asdict(s) for t, s in g.items()} for k, g in self.groups.items()}

    def load(self, data: dict[str, Any] | None) -> None:
        self.groups = {}
        for k, g in (data or {}).items():
            try:
                self.groups[k] = {round(float(t), 2): _Stat(**s) for t, s in g.items()}
            except (TypeError, ValueError):
                continue
