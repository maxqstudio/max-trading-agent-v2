"""Entry filters on direction (M15/H1 bias), trading session and the M15 Ichimoku checklist, and
how well they would have worked.

Each filter has three modes:
- "off":  never blocks;
- "on":   always blocks (trades against the bias, or the listed sessions);
- "auto": blocks only once the "No Veto" book shows that those setups do clearly worse than
          the rest (enough trades on both sides and a gap of more than ``z`` standard errors).

The "No Veto" book opens before these filters, so it keeps seeing every setup and the
comparison never runs dry. Only market results (R of closed trades) are counted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from backend.features.htf import SESSIONS
from backend.features.ichimoku import score_bucket

HTF_BUCKETS = ("with", "neutral", "against")
HTF_TIMEFRAMES = ("M15", "H1", "BOTH")
ICHI_BUCKETS = ("with", "mixed", "against")
ICHI_THRESHOLDS = (2, 3, 4)  # stats are kept for each, so changing the setting keeps its history
TREND_STRATEGIES = frozenset({"micro_breakout", "ema_vwap_pullback", "tick_imbalance"})


def htf_bucket(m15: int, h1: int, timeframe: str) -> str:
    """Bucket of a trade from its M15 and H1 alignment (+1 with, -1 against, 0 none)."""
    if timeframe == "M15":
        a = m15
    elif timeframe == "H1":
        a = h1
    else:  # BOTH: against if either timeframe is against, with if any is with and none against
        a = -1 if -1 in (m15, h1) else (1 if 1 in (m15, h1) else 0)
    return "with" if a > 0 else "against" if a < 0 else "neutral"


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

    def merge(self, o: _Stat) -> _Stat:
        return _Stat(self.n + o.n, self.total + o.total, self.total_sq + o.total_sq, self.wins + o.wins)

    @property
    def mean(self) -> float:
        return self.total / self.n if self.n else 0.0

    @property
    def var(self) -> float:
        if self.n < 2:
            return 0.0
        return max(0.0, (self.total_sq - self.total * self.total / self.n) / (self.n - 1))


class FilterLearner:
    def __init__(self, min_samples: int = 30, z: float = 1.0) -> None:
        self.min_samples = min_samples
        self.z = z
        self.stats: dict[str, dict[str, _Stat]] = {}

    def record(self, m15: int, h1: int, session: str, r: float, ichi: int | None = None) -> None:
        for tf in HTF_TIMEFRAMES:
            self._add(f"htf:{tf}", htf_bucket(m15, h1, tf), r)
        if session:
            self._add("session", session, r)
        if ichi is not None:
            for th in ICHI_THRESHOLDS:
                self._add(f"ichi:{th}", score_bucket(ichi, th), r)

    def _add(self, key: str, bucket: str, r: float) -> None:
        self.stats.setdefault(key, {}).setdefault(bucket, _Stat()).add(r)

    def clearly_worse(self, key: str, bucket: str) -> bool:
        group = self.stats.get(key, {})
        b = group.get(bucket)
        rest = _Stat()
        for k, s in group.items():
            if k != bucket:
                rest = rest.merge(s)
        if b is None or b.n < self.min_samples or rest.n < self.min_samples:
            return False
        se = math.sqrt(b.var / b.n + rest.var / rest.n)
        return rest.mean - b.mean > self.z * se

    def table(self, key: str, buckets: tuple[str, ...]) -> list[dict[str, Any]]:
        group = self.stats.get(key, {})
        rows = []
        for bucket in buckets:
            s = group.get(bucket, _Stat())
            rows.append({
                "bucket": bucket,
                "trades": s.n,
                "win_rate": round(s.wins / s.n * 100, 1) if s.n else 0.0,
                "avg_r": round(s.mean, 3),
                "clearly_worse": self.clearly_worse(key, bucket),
            })
        return rows

    def to_dict(self) -> dict[str, Any]:
        return {k: {b: [s.n, round(s.total, 4), round(s.total_sq, 4), s.wins] for b, s in g.items()} for k, g in self.stats.items()}

    def load(self, data: dict[str, Any] | None) -> None:
        self.stats = {}
        for k, g in (data or {}).items():
            try:
                self.stats[k] = {b: _Stat(int(v[0]), float(v[1]), float(v[2]), int(v[3])) for b, v in g.items()}
            except (TypeError, ValueError, IndexError):
                continue


@dataclass
class FilterDecision:
    code: str  # "" = allowed
    detail: str = ""


def decide(
    learner: FilterLearner,
    *,
    htf_mode: str,
    htf_timeframe: str,
    session_mode: str,
    blocked_sessions: list[str],
    m15: int,
    h1: int,
    session: str,
    ichimoku_mode: str = "off",
    ichi: int = 0,
    ichi_threshold: int = 3,
    strategy_id: str = "",
    regime: str = "",
    trend_strategies_require_htf: bool = False,
    block_breakout_in_ranging: bool = False,
) -> FilterDecision:
    if block_breakout_in_ranging and strategy_id == "micro_breakout" and regime == "RANGING":
        return FilterDecision("BREAKOUT_IN_RANGE", "Micro Breakout blocked inside RANGING chop")
    bucket = htf_bucket(m15, h1, htf_timeframe)
    if trend_strategies_require_htf and strategy_id in TREND_STRATEGIES and bucket == "against":
        return FilterDecision(
            "HTF_TREND_AGAINST",
            f"Trend strategy {strategy_id} blocked against the {htf_timeframe} direction",
        )
    if bucket == "against" and htf_mode != "off":
        if htf_mode == "on":
            return FilterDecision("HTF_AGAINST", f"Against the {htf_timeframe} direction")
        if learner.clearly_worse(f"htf:{htf_timeframe}", "against"):
            return FilterDecision("HTF_AGAINST", f"Against the {htf_timeframe} direction (learned: these trades do clearly worse)")
    if session and session_mode != "off":
        if session_mode == "on" and session in blocked_sessions:
            return FilterDecision("SESSION_FILTER", f"{session} session is blocked")
        if session_mode == "auto" and learner.clearly_worse("session", session):
            return FilterDecision("SESSION_FILTER", f"{session} session (learned: trades here do clearly worse)")
    if ichimoku_mode != "off" and score_bucket(ichi, ichi_threshold) == "against":
        if ichimoku_mode == "on":
            return FilterDecision("ICHIMOKU_AGAINST", f"M15 Ichimoku checklist against the trade ({ichi:+d} of 5)")
        if learner.clearly_worse(f"ichi:{ichi_threshold}", "against"):
            return FilterDecision(
                "ICHIMOKU_AGAINST", f"M15 Ichimoku checklist against the trade ({ichi:+d} of 5; learned: these trades do clearly worse)"
            )
    return FilterDecision("")


__all__ = ["HTF_BUCKETS", "HTF_TIMEFRAMES", "ICHI_BUCKETS", "SESSIONS", "FilterDecision", "FilterLearner", "decide", "htf_bucket"]
