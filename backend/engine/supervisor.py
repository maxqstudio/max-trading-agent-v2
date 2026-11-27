"""Claude as the trade supervisor: a fast veto layer on top of the quant engine.

The quant stack (strategies -> bandit -> ML meta-labeler -> risk manager) proposes a fully
sized trade. The supervisor may APPROVE it, REDUCE its size, or REJECT it. It can never
increase size, move stops, or open a trade the quant stack did not propose.

Built for M1 scalping:
- one LLM call, no tool round-trips, compact context, fast model + low effort by default
- hard timeout; on timeout / error / over budget the policy decides (default: skip the trade)
- runs off the bridge loop, so the EA never waits on the model
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from backend.core.models import Bar, LivePosition, Side, SymbolSpec, TradeRecord, TradeSignal
from backend.llm.base import LLMRequest
from backend.llm.router import LLMRouter

log = logging.getLogger("max_agent.supervisor")

Decision = Literal["APPROVE", "REDUCE", "REJECT"]

SUPERVISOR_SYSTEM = """You are the final pre-trade supervisor for an aggressive automated M1 scalping system on MT5 (XAUUSD, EURUSD, GBPUSD, USDJPY).

Every setup you see has ALREADY passed: the strategy's own rules, a regime filter, a 1D-CNN direction model, a calibrated ML win-probability filter, spread/news/rollover shields, and a portfolio risk manager that sized the position. Your job is to catch what those numeric filters miss, not to redo them.

The operator wants an aggressive, high-frequency scalper. A healthy veto rate is roughly 10-30%. If you would reject most setups, you are probably being too cautious. Approve by default; veto only for a concrete reason you can name.

Decisions:
- APPROVE: take the trade at the proposed size.
- REDUCE: take it smaller. Set size_multiplier between 0.25 and 0.9. Use it when the setup is valid but one factor is uncomfortable (spread near the limit, mildly against `change_last_25_bars_atr`, two losses in a row on this strategy).
- REJECT: skip it. Use it for clear red flags.

Red flags worth a REJECT (name the one you see):
- Chasing: `bars_in_trade_direction_streak` >= 5 and the entry is at the edge of the range (`dist_to_20bar_high_atr` < 0.2 for buys, `dist_to_20bar_low_atr` < 0.2 for sells).
- Stop inside noise: `setup.sl_atr` < 1.0 while `avg_bar_range_last_12_atr` > `setup.sl_atr`.
- Spread: `spread_atr` >= 0.30, or `spread_points` more than ~1.8x `avg_spread_last_20_points`.
- Direction conflict: the DL probability for the opposite side exceeds the trade side by 0.2+ AND `change_last_25_bars_atr` (the M5 trend proxy) points the other way by more than 1 ATR.
- `last_trades_same_strategy_symbol_r` shows 3+ losses in a row.
- `daily_dd_pct` above two-thirds of `daily_dd_limit_pct`: only take clean setups, otherwise REDUCE or REJECT.

You cannot change SL, TP, or direction, and you cannot increase size; the code ignores any attempt.

Answer fields:
- decision: APPROVE | REDUCE | REJECT
- size_multiplier: 1.0 for APPROVE, 0.25-0.9 for REDUCE, 0.0 for REJECT
- confidence: 0.0-1.0, how sure you are about the decision
- reason: one short sentence (max ~20 words) naming the deciding factor. It is shown on the operator's dashboard.
"""

SUPERVISOR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["decision", "size_multiplier", "confidence", "reason"],
    "properties": {
        "decision": {"type": "string", "enum": ["APPROVE", "REDUCE", "REJECT"]},
        "size_multiplier": {"type": "number"},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
}


class SupervisorAnswer(BaseModel):
    model_config = ConfigDict(extra="ignore")
    decision: Decision
    size_multiplier: float = 1.0
    confidence: float = 0.5
    reason: str = ""


@dataclass(slots=True)
class SupervisorSettings:
    enabled: bool = True
    timeout_s: float = 6.0
    use_fast_model: bool = True
    effort: str = "low"
    on_unavailable: Literal["block", "allow"] = "block"  # what to do on timeout / error / budget
    max_calls_per_hour: int = 120
    daily_budget_usd: float = 5.0
    max_price_drift_atr: float = 0.25  # re-check after the answer: skip if price ran away
    min_reduce_multiplier: float = 0.25


@dataclass(slots=True)
class SupervisorVerdict:
    decision: Decision
    size_multiplier: float
    reason: str
    confidence: float = 0.0
    provider: str = ""
    latency_ms: float = 0.0
    cost_usd: float | None = None
    error: str = ""

    @property
    def allows_entry(self) -> bool:
        return self.decision != "REJECT" and self.size_multiplier > 0


@dataclass(slots=True)
class _Budget:
    calls: list[float] = field(default_factory=list)
    day: str = ""
    spent_usd: float = 0.0


def _fmt(x: float, digits: int) -> float:
    return round(float(x), digits)


def build_context(
    *,
    signal: TradeSignal,
    spec: SymbolSpec,
    volume: float,
    sl: float,
    tp: float,
    spread_points: float,
    bars: list[Bar],
    ind: Any,
    dl_up: float,
    dl_down: float,
    ml_prob: float,
    bandit_weight: float,
    open_positions: list[LivePosition],
    recent_trades: list[TradeRecord],
    daily_dd_pct: float,
    max_daily_dd_pct: float,
    consecutive_losses: int,
    hour_utc: int,
) -> str:
    d = spec.digits
    entry = signal.entry_price
    sl_dist = abs(entry - sl)
    tp_dist = abs(tp - entry)
    atr = max(signal.atr, 1e-12)
    last = bars[-20:]
    recent_hi = max(b.high for b in last) if last else entry
    recent_lo = min(b.low for b in last) if last else entry
    closes = [b.close for b in bars]

    def change(n: int) -> float:
        return _fmt((closes[-1] - closes[-n]) / atr, 2) if len(closes) >= n else 0.0

    run = 0
    for b in reversed(last):
        up = b.close > b.open
        if (up and signal.side == Side.BUY) or (not up and signal.side == Side.SELL and b.close < b.open):
            run += 1
        else:
            break
    spreads = [b.spread_points for b in last if b.spread_points > 0]
    same = [t for t in recent_trades if t.symbol == signal.symbol and t.strategy_id == signal.strategy_id][-5:]
    ctx = {
        "setup": {
            "symbol": signal.symbol,
            "side": signal.side.value,
            "strategy": signal.strategy_id,
            "strategy_reason": signal.reason[:160],
            "regime": signal.regime.value,
            "entry": _fmt(entry, d),
            "sl": _fmt(sl, d),
            "tp": _fmt(tp, d),
            "sl_atr": _fmt(sl_dist / atr, 2),
            "tp_atr": _fmt(tp_dist / atr, 2),
            "rr": _fmt(tp_dist / max(sl_dist, 1e-12), 2),
            "lots": volume,
        },
        "market": {
            "atr14": _fmt(atr, d),
            "spread_points": _fmt(spread_points, 1),
            "spread_atr": _fmt(spread_points * spec.point / atr, 3),
            "avg_spread_last_20_points": _fmt(sum(spreads) / len(spreads), 1) if spreads else None,
            "avg_bar_range_last_12_atr": _fmt(sum(b.high - b.low for b in last[-12:]) / max(len(last[-12:]), 1) / atr, 2),
            "bars_in_trade_direction_streak": run,
            "change_last_25_bars_atr": change(25),  # ~5 x M5 candles: the M5 trend proxy
            "change_last_60_bars_atr": change(60),
            "dist_to_20bar_high_atr": _fmt((recent_hi - entry) / atr, 2),
            "dist_to_20bar_low_atr": _fmt((entry - recent_lo) / atr, 2),
            "ema9_vs_ema21": "above" if ind.ema9 > ind.ema21 else "below",
            "price_vs_vwap_atr": _fmt((ind.close - ind.vwap) / atr, 2),
            "rsi7": _fmt(ind.rsi7, 1),
            "adx14": _fmt(ind.adx14, 1),
            "hour_utc": hour_utc,
            # Oldest -> newest. Prices relative to entry in ATR units keep this short and scale-free.
            "last_bars_atr": [
                [_fmt((b.open - entry) / atr, 2), _fmt((b.high - entry) / atr, 2),
                 _fmt((b.low - entry) / atr, 2), _fmt((b.close - entry) / atr, 2)]
                for b in last[-12:]
            ],
        },
        "models": {
            "dl_prob_up": _fmt(dl_up, 2),
            "dl_prob_down": _fmt(dl_down, 2),
            "ml_win_prob": _fmt(ml_prob, 2),
            "bandit_weight": _fmt(bandit_weight, 2),
            "strategy_confidence": _fmt(signal.confidence, 2),
        },
        "account": {
            "daily_dd_pct": _fmt(daily_dd_pct, 2),
            "daily_dd_limit_pct": _fmt(max_daily_dd_pct, 2),
            "consecutive_losses": consecutive_losses,
            "open_positions": [f"{p.symbol} {p.side.value} {p.profit:+.0f}$" for p in open_positions][:5],
            "last_trades_same_strategy_symbol_r": [_fmt(t.pnl_r, 2) for t in same],
        },
    }
    return (
        "Proposed trade. Price fields ending in _atr are distances in ATR units; last_bars_atr rows are"
        " [open, high, low, close] minus entry, divided by ATR, oldest first:\n"
        + json.dumps(ctx, separators=(",", ":"))
    )


class TradeSupervisor:
    def __init__(self, router: LLMRouter, settings: SupervisorSettings | None = None) -> None:
        self.router = router
        self.settings = settings or SupervisorSettings()
        self._budget = _Budget()
        self.stats = {"calls": 0, "approve": 0, "reduce": 0, "reject": 0, "unavailable": 0}

    @property
    def active(self) -> bool:
        return self.settings.enabled and self.router.enabled

    def _over_budget(self) -> str:
        now = time.time()
        day = time.strftime("%Y-%m-%d", time.gmtime(now))
        if day != self._budget.day:
            self._budget.day, self._budget.spent_usd = day, 0.0
        self._budget.calls = [t for t in self._budget.calls if now - t < 3600]
        if len(self._budget.calls) >= self.settings.max_calls_per_hour:
            return "hourly call limit reached"
        if self._budget.spent_usd >= self.settings.daily_budget_usd:
            return f"daily budget ${self.settings.daily_budget_usd:.2f} used"
        return ""

    def _unavailable(self, why: str) -> SupervisorVerdict:
        self.stats["unavailable"] += 1
        if self.settings.on_unavailable == "allow":
            return SupervisorVerdict("APPROVE", 1.0, f"Supervisor unavailable ({why}); policy=allow", error=why)
        return SupervisorVerdict("REJECT", 0.0, f"Supervisor unavailable ({why}); policy=block", error=why)

    async def review(self, context: str) -> SupervisorVerdict:
        why = self._over_budget()
        if why:
            return self._unavailable(why)
        self._budget.calls.append(time.time())
        self.stats["calls"] += 1
        req = LLMRequest(
            task="supervisor",
            system=SUPERVISOR_SYSTEM,
            user=context,
            schema=SUPERVISOR_SCHEMA,
            timeout_s=self.settings.timeout_s,
            fast=self.settings.use_fast_model,
            max_tokens=4000,
            effort=self.settings.effort,
        )
        try:
            res, ans = await self.router.complete(req, validate=SupervisorAnswer.model_validate)
        except Exception as exc:
            log.warning("supervisor unavailable: %s", exc)
            return self._unavailable(str(exc)[:160])
        if res.cost_usd:
            self._budget.spent_usd += res.cost_usd

        mult = ans.size_multiplier
        if not math.isfinite(mult):
            mult = 0.0
        if ans.decision == "APPROVE":
            mult = 1.0
        elif ans.decision == "REDUCE":
            mult = min(0.9, max(self.settings.min_reduce_multiplier, mult))
        else:
            mult = 0.0
        self.stats[ans.decision.lower()] += 1
        return SupervisorVerdict(
            decision=ans.decision,
            size_multiplier=mult,
            reason=ans.reason[:200],
            confidence=max(0.0, min(1.0, ans.confidence)),
            provider=f"{res.provider}:{res.model}",
            latency_ms=res.latency_ms,
            cost_usd=res.cost_usd,
        )

    def status(self) -> dict[str, Any]:
        s = self.settings
        return {
            "active": self.active,
            "enabled": s.enabled,
            "timeout_s": s.timeout_s,
            "fast_model": s.use_fast_model,
            "effort": s.effort,
            "on_unavailable": s.on_unavailable,
            "daily_budget_usd": s.daily_budget_usd,
            "spent_today_usd": round(self._budget.spent_usd, 4),
            **self.stats,
        }


def scale_volume(volume: float, mult: float, spec: SymbolSpec) -> float:
    """Shrink a lot size by ``mult`` (<= 1) on the broker's step; 0 if below the minimum lot."""
    mult = max(0.0, min(1.0, mult))
    step = max(spec.volume_step, 0.001)
    scaled = math.floor((volume * mult) / step + 1e-9) * step
    if scaled < spec.volume_min:
        return 0.0
    return round(min(scaled, volume), 4)


def price_drifted(side: Side, proposed_entry: float, current_price: float, atr: float, max_drift_atr: float) -> bool:
    """True if price moved against the fill (worse entry) by more than max_drift_atr * ATR."""
    adverse = (current_price - proposed_entry) if side == Side.BUY else (proposed_entry - current_price)
    return adverse > max_drift_atr * max(atr, 1e-12)
