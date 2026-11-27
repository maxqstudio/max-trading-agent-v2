"""Learns which exit style (Normal or Hit & Run) works better, per symbol and market regime.

Both paper books open on exactly the same setups, so every closed paper trade is a direct
comparison of the two exit styles. The selector keeps the R result of each one, grouped by
(symbol, regime), and picks Hit & Run for a group only when:

- each style has at least ``min_samples`` closed trades in that group, and
- Hit & Run's average R beats Normal's by more than ``z`` standard errors.

Otherwise it falls back to the symbol as a whole, then to all trades, then to Normal.
This is deliberately slow to switch, so a lucky streak does not change how real positions exit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

STYLES = ("normal", "hitrun")


@dataclass
class _Arm:
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
class _Group:
    arms: dict[str, _Arm] = field(default_factory=lambda: {s: _Arm() for s in STYLES})


@dataclass
class Choice:
    style: str  # "normal" or "hitrun"
    scope: str  # which group decided it: "symbol|regime", "symbol", "all" or "default"
    reason: str


class ExitStyleSelector:
    def __init__(self, min_samples: int = 20, z: float = 1.0) -> None:
        self.min_samples = min_samples
        self.z = z
        self.groups: dict[str, _Group] = {}

    @staticmethod
    def _keys(symbol: str, regime: str) -> list[str]:
        return [f"{symbol}|{regime}", symbol, "*"]

    def record(self, style: str, symbol: str, regime: str, r: float) -> None:
        if style not in STYLES:
            return
        for k in self._keys(symbol, regime):
            self.groups.setdefault(k, _Group()).arms[style].add(r)

    def _verdict(self, g: _Group) -> tuple[str, str] | None:
        n, h = g.arms["normal"], g.arms["hitrun"]
        if min(n.n, h.n) < self.min_samples:
            return None
        diff = h.mean - n.mean
        se = math.sqrt(n.var / n.n + h.var / h.n)
        style = "hitrun" if diff > self.z * se and diff > 0 else "normal"
        return style, f"Hit & Run {h.mean:+.2f}R vs Normal {n.mean:+.2f}R ({min(n.n, h.n)} trade)"

    def choose(self, symbol: str, regime: str) -> Choice:
        labels = ["symbol|regime", "symbol", "all"]
        for key, scope in zip(self._keys(symbol, regime), labels):
            g = self.groups.get(key)
            v = self._verdict(g) if g else None
            if v:
                return Choice(v[0], scope, v[1])
        return Choice("normal", "default", f"Not enough data yet (at least {self.min_samples} trades per style)")

    def table(self) -> list[dict[str, Any]]:
        rows = []
        for key, g in sorted(self.groups.items()):
            n, h = g.arms["normal"], g.arms["hitrun"]
            v = self._verdict(g)
            rows.append({
                "key": "All" if key == "*" else key.replace("|", " · "),
                "normal_n": n.n, "normal_avg_r": round(n.mean, 3), "normal_wr": round(n.wins / n.n * 100, 1) if n.n else 0.0,
                "hitrun_n": h.n, "hitrun_avg_r": round(h.mean, 3), "hitrun_wr": round(h.wins / h.n * 100, 1) if h.n else 0.0,
                "pick": v[0] if v else None,
                "need": max(0, self.min_samples - min(n.n, h.n)),
            })
        return rows
