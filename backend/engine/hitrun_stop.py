"""Hit & Run's own stop loss, separate from (and tighter than) the Normal pattern's.

Hit & Run takes small wins (net profit >= ``min_net_profit``), so a full Normal-sized loss wipes
out dozens of them. It therefore gets three limits of its own:

1. Price stop: ``sl_atr_mult`` x ATR(M1), never closer than ``sl_min_spread_mult`` x spread or the
   broker's stops level, never wider than the Normal SL.
2. Money cap: the lot is sized so a stop-out costs at most ``max_loss_mult`` x ``min_net_profit``,
   commission included. Never larger than the Normal lot. If even the minimum lot would lose
   more than that, Hit & Run skips the trade.
3. Time stop: close if the trade is still not in net profit after ``time_stop_bars`` M1 bars.

``HitRunTuner`` learns which (ATR multiple, time stop) pair works best per symbol and regime.
For every setup the "No Veto" book takes, it follows one ghost trade per pair on live ticks,
all with the same money cap, and keeps the net result of each. Only market results count.
It moves away from the configured default only with enough trades and a clear edge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

from backend.core.models import Side, SymbolSpec

SL_ATR_MULTS = (0.6, 0.9, 1.2, 1.6)
TIME_STOPS = (3, 5, 0)  # M1 bars; 0 = no time stop
GHOST_MAX_HOLD_SEC = 4 * 3600


@dataclass(frozen=True)
class HitRunArm:
    sl_atr_mult: float
    time_stop_bars: int

    @property
    def key(self) -> str:
        return f"{self.sl_atr_mult:g}|{self.time_stop_bars}"

    @classmethod
    def from_key(cls, key: str) -> "HitRunArm":
        m, t = key.split("|")
        return cls(float(m), int(t))

    def label(self) -> str:
        ts = f"{self.time_stop_bars} bars" if self.time_stop_bars else "no time stop"
        return f"SL {self.sl_atr_mult:g}x ATR, {ts}"


@dataclass
class HitRunPlan:
    sl: float
    volume: float
    risk_dist: float  # price distance entry -> SL
    max_loss: float  # money lost at the SL, commission included
    arm: HitRunArm


def money_per_price(spec: SymbolSpec) -> float:
    """Account money per 1.0 price move per 1.0 lot."""
    return spec.tick_value / spec.tick_size if spec.tick_size > 0 else 0.0


def floor_volume(volume: float, spec: SymbolSpec) -> float:
    step = spec.volume_step if spec.volume_step > 0 else 0.01
    v = math.floor(volume / step + 1e-9) * step
    return round(min(v, spec.volume_max), 8)


def plan_stop(
    *,
    side: Side,
    entry: float,
    atr: float,
    spread_px: float,
    spec: SymbolSpec,
    normal_sl: float,
    normal_volume: float,
    commission_per_lot: float,
    min_net_profit: float,
    max_loss_mult: float,
    sl_min_spread_mult: float,
    arm: HitRunArm,
) -> HitRunPlan | None:
    """SL and lot for a Hit & Run entry, or None when the money cap cannot be met."""
    k = money_per_price(spec)
    if k <= 0 or entry <= 0:
        return None
    dist = max(arm.sl_atr_mult * max(atr, 0.0), sl_min_spread_mult * max(spread_px, 0.0),
               spec.stops_level * spec.point, spec.point)
    if normal_sl > 0:
        dist = min(dist, max(abs(entry - normal_sl), spec.point))
    per_lot = dist * k + max(commission_per_lot, 0.0)
    budget = max(max_loss_mult, 0.0) * max(min_net_profit, 0.01)
    volume = floor_volume(min(budget / per_lot, normal_volume), spec)
    if volume < spec.volume_min - 1e-12:
        return None
    sl = entry - dist if side == Side.BUY else entry + dist
    return HitRunPlan(
        sl=round(sl, spec.digits), volume=volume, risk_dist=dist, max_loss=round(volume * per_lot, 2), arm=arm,
    )


def time_stop_due(open_sec: int, now_sec: int, bars: int, net: float) -> bool:
    """Time stop: ``bars`` M1 bars have passed and the trade is still not in net profit."""
    return bars > 0 and now_sec - open_sec >= bars * 60 and net <= 0


# --------------------------------------------------------------------------------------------
# Learning the best arm per symbol and regime from ghost trades
# --------------------------------------------------------------------------------------------


@dataclass
class _Stat:
    n: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    wins: int = 0

    def add(self, x: float) -> None:
        self.n += 1
        self.total += x
        self.total_sq += x * x
        self.wins += x > 0

    @property
    def mean(self) -> float:
        return self.total / self.n if self.n else 0.0

    @property
    def var(self) -> float:
        if self.n < 2:
            return 0.0
        return max(0.0, (self.total_sq - self.total * self.total / self.n) / (self.n - 1))


@dataclass
class Ghost:
    arm: HitRunArm
    symbol: str
    regime: str
    side: Side
    entry: float
    sl: float
    volume: float
    open_sec: int


@dataclass
class TunerChoice:
    arm: HitRunArm
    scope: str
    reason: str


@dataclass
class HitRunTuner:
    min_samples: int = 30
    z: float = 1.0
    stats: dict[str, dict[str, _Stat]] = field(default_factory=dict)
    ghosts: dict[str, list[Ghost]] = field(default_factory=dict)

    @staticmethod
    def arms(default: HitRunArm | None = None) -> list[HitRunArm]:
        out = [HitRunArm(m, t) for m in SL_ATR_MULTS for t in TIME_STOPS]
        if default is not None and default not in out:
            out.append(default)
        return out

    @staticmethod
    def _keys(symbol: str, regime: str) -> list[str]:
        return [f"{symbol}|{regime}", symbol, "*"]

    def spawn(self, symbol: str, regime: str, side: Side, entry: float, open_sec: int,
              plans: list[HitRunPlan]) -> None:
        """Start one ghost per arm for a new setup (replaces ghosts still open on this symbol)."""
        self.ghosts[symbol] = [
            Ghost(arm=p.arm, symbol=symbol, regime=regime, side=side, entry=entry, sl=p.sl, volume=p.volume, open_sec=open_sec)
            for p in plans
        ]

    def on_tick(
        self,
        symbol: str,
        bid: float,
        ask: float,
        now_sec: int,
        min_net_profit: float,
        net_at: Callable[[Ghost, float], float],
        *,
        min_cost_multiple: float = 0.0,
        friction_at: Callable[[Ghost], float] | None = None,
    ) -> int:
        """Advance this symbol's ghosts. ``net_at(ghost, price)`` is the net money if closed at
        ``price`` (bid for buys, ask for sells). Returns how many ghosts closed."""
        live = self.ghosts.get(symbol)
        if not live:
            return 0
        keep: list[Ghost] = []
        closed = 0
        for g in live:
            px = bid if g.side == Side.BUY else ask
            net = net_at(g, px)
            friction = friction_at(g) if friction_at is not None else 0.0
            take_floor = max(min_net_profit, min_cost_multiple * max(friction, 0.0), 0.01)
            stopped = (px <= g.sl) if g.side == Side.BUY else (px >= g.sl)
            if (
                net >= take_floor
                or stopped
                or time_stop_due(g.open_sec, now_sec, g.arm.time_stop_bars, net)
                or now_sec - g.open_sec >= GHOST_MAX_HOLD_SEC
            ):
                self.record(g.symbol, g.regime, g.arm, net)
                closed += 1
            else:
                keep.append(g)
        self.ghosts[symbol] = keep
        return closed

    def record(self, symbol: str, regime: str, arm: HitRunArm, net: float) -> None:
        for k in self._keys(symbol, regime):
            self.stats.setdefault(k, {}).setdefault(arm.key, _Stat()).add(round(net, 2))

    def choose(self, symbol: str, regime: str, default: HitRunArm) -> TunerChoice:
        for key in self._keys(symbol, regime):
            group = self.stats.get(key)
            if not group:
                continue
            base = group.get(default.key)
            ready = {k: s for k, s in group.items() if s.n >= self.min_samples}
            if base is None or base.n < self.min_samples or len(ready) < 2:
                continue
            best_key, best = max(ready.items(), key=lambda kv: kv[1].mean)
            if best_key == default.key:
                return TunerChoice(default, key, "default is best so far")
            se = math.sqrt(best.var / best.n + base.var / base.n)
            if best.mean - base.mean > self.z * se:
                arm = HitRunArm.from_key(best_key)
                return TunerChoice(arm, key, f"${best.mean:+.2f}/trade vs default ${base.mean:+.2f}")
            return TunerChoice(default, key, "no clear edge over the default yet")
        return TunerChoice(default, "default", f"not enough ghost trades yet (min {self.min_samples} per setting)")

    def table(self, default: HitRunArm) -> list[dict[str, Any]]:
        rows = []
        for key, group in self.stats.items():
            n = max((s.n for s in group.values()), default=0)
            sym, _, regime = key.partition("|")
            choice = self.choose(sym if key != "*" else "", regime, default) if key != "*" else None
            best_key, best = max(group.items(), key=lambda kv: kv[1].mean)
            rows.append({
                "key": "All" if key == "*" else key.replace("|", " · "),
                "trades": n,
                "best": HitRunArm.from_key(best_key).label(),
                "best_avg": round(best.mean, 2),
                "best_wr": round(best.wins / best.n * 100, 1) if best.n else 0.0,
                "default_avg": round(group[default.key].mean, 2) if default.key in group else None,
                "using": (choice.arm.label() if choice and choice.scope == key else None),
            })
        rows.sort(key=lambda r: (r["key"] != "All", -r["trades"]))
        return rows

    def to_dict(self) -> dict[str, Any]:
        return {k: {a: [s.n, round(s.total, 4), round(s.total_sq, 4), s.wins] for a, s in g.items()} for k, g in self.stats.items()}

    def load(self, data: dict[str, Any] | None) -> None:
        self.stats = {}
        for k, g in (data or {}).items():
            try:
                self.stats[k] = {a: _Stat(int(v[0]), float(v[1]), float(v[2]), int(v[3])) for a, v in g.items()}
            except (TypeError, ValueError, IndexError):
                continue
