from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from backend.core.config import ShieldsConfig
from backend.core.models import Bar


@dataclass(slots=True)
class ShieldVerdict:
    allowed: bool
    reason_code: str = "CLEAR"
    detail: str = ""
    should_flatten: bool = False


def _parse_hhmm(hhmm: str) -> int:
    h, m = hhmm.strip().split(":")
    return int(h) * 60 + int(m)


class MarketShields:
    """Multi-gate microstructure shield protecting scalping against rollover, news, spread spikes, shocks & Friday gaps."""

    def __init__(self, config: ShieldsConfig) -> None:
        self.config = config
        self._spread_history: dict[str, deque[float]] = {}
        self._shock_until_ts: dict[str, int] = {}
        self._news_events: list[dict] = []
        self.manual_shock_override: bool = False
        self.reload_news_calendar()

    def reload_news_calendar(self) -> None:
        path = Path(self.config.news_file)
        if not path.exists():
            self._news_events = []
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            self._news_events = payload.get("events", [])
        except Exception:
            self._news_events = []

    def record_spread(self, symbol: str, spread_points: float) -> None:
        buf = self._spread_history.setdefault(symbol, deque(maxlen=120))
        buf.append(float(spread_points))

    def median_spread(self, symbol: str) -> float:
        buf = self._spread_history.get(symbol)
        if not buf:
            return 0.0
        ordered = sorted(buf)
        return ordered[len(ordered) // 2]

    def record_bar_for_shock(self, symbol: str, bar: Bar, atr_val: float, now_ts: int) -> bool:
        bar_range = bar.high - bar.low
        if atr_val > 0 and bar_range >= self.config.shock_bar_atr_multiplier * atr_val:
            self._shock_until_ts[symbol] = now_ts + self.config.shock_cooldown_minutes * 60
            return True
        return False

    def evaluate(
        self,
        symbol: str,
        spread_points: float,
        now_utc: datetime | None = None,
    ) -> ShieldVerdict:
        now = now_utc or datetime.now(timezone.utc)
        now_ts = int(now.timestamp())

        if self.manual_shock_override:
            return ShieldVerdict(False, "MANUAL_SHIELD", "Manual shield freeze enabled")

        # 1. Friday pre-close auto-flatten (avoid weekend gap exposure)
        if self.config.friday_flatten_enabled and now.weekday() == 4:
            cur_min = now.hour * 60 + now.minute
            flat_min = _parse_hhmm(self.config.friday_flatten_utc)
            if cur_min >= flat_min:
                return ShieldVerdict(
                    False,
                    "FRIDAY_CLOSE_FLATTEN",
                    f"Friday pre-close window (>= {self.config.friday_flatten_utc} UTC)",
                    should_flatten=True,
                )

        # 2. NY Rollover blackout (extreme swap spreads)
        if self.config.rollover_blackout_enabled:
            cur_min = now.hour * 60 + now.minute
            start_m = _parse_hhmm(self.config.rollover_start_utc)
            end_m = _parse_hhmm(self.config.rollover_end_utc)
            in_rollover = (start_m <= cur_min <= end_m) if start_m <= end_m else (cur_min >= start_m or cur_min <= end_m)
            if in_rollover:
                return ShieldVerdict(
                    False,
                    "ROLLOVER_BLACKOUT",
                    f"NY Rollover blackout ({self.config.rollover_start_utc}-{self.config.rollover_end_utc} UTC)",
                )

        # 3. Scheduled high-impact economic news blackout
        if self.config.news_blackout_enabled and self._news_events:
            buf_sec = self.config.news_buffer_minutes * 60
            for ev in self._news_events:
                currencies = ev.get("currencies", [])
                if currencies and not any(c in symbol for c in currencies):
                    continue
                try:
                    ev_ts = int(datetime.fromisoformat(ev["timestamp_utc"].replace("Z", "+00:00")).timestamp())
                except Exception:
                    continue
                if abs(now_ts - ev_ts) <= buf_sec:
                    return ShieldVerdict(
                        False,
                        "NEWS_BLACKOUT",
                        f"High-impact news window: {ev.get('title', 'Macro Event')}",
                    )

        # 4. Real-time bar shock cooldown
        shock_until = self._shock_until_ts.get(symbol, 0)
        if now_ts < shock_until:
            rem = shock_until - now_ts
            return ShieldVerdict(False, "SHOCK_COOLDOWN", f"Post-spike shock cooldown ({rem}s left)")

        # 5. Rolling spread anomaly spike detector
        med = self.median_spread(symbol)
        if med > 0 and len(self._spread_history.get(symbol, ())) >= 15:
            if spread_points > self.config.spread_anomaly_multiplier * med:
                return ShieldVerdict(
                    False,
                    "SPREAD_SPIKE",
                    f"Spread {spread_points:.1f}pt > {self.config.spread_anomaly_multiplier:.1f}x median ({med:.1f}pt)",
                )

        # 6. Optional London/NY prime session gate (07:00 - 20:00 UTC)
        if self.config.preferred_sessions_only and not (7 <= now.hour < 20):
            return ShieldVerdict(False, "OFF_SESSION", f"Outside London/NY session ({now.hour:02d}:00 UTC)")

        return ShieldVerdict(True, "CLEAR", "All market shields green")
