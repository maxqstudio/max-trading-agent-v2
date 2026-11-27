"""Claude trade supervisor: veto/reduce/approve, fail-closed, budget, price drift, engine wiring."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from backend.core.config import EnvSettings, load_trading_config
from backend.core.models import Bar, MarketRegime, OrderAction, Side, TickSnapshot, TradeSignal, TradingMode
from backend.engine.orchestrator import TradingOrchestrator
from backend.engine.supervisor import (
    SUPERVISOR_SCHEMA,
    SUPERVISOR_SYSTEM,
    SupervisorSettings,
    TradeSupervisor,
    build_context,
    price_drifted,
    scale_volume,
)
from backend.features.indicators import IndicatorSnapshot
from backend.llm.base import LLMError, LLMRequest, LLMResult
from backend.llm.router import LLMRouter
from backend.llm.settings import load_chain
from backend.risk.manager import DEFAULT_SPECS, RiskDecision
from backend.storage.database import Database


class _Fake:
    def __init__(self, answer: dict[str, Any] | Exception, delay: float = 0.0) -> None:
        self.name, self.answer, self.delay, self.requests = "anthropic", answer, delay, []

    async def complete(self, req: LLMRequest) -> LLMResult:
        self.requests.append(req)
        if self.delay:
            await asyncio.sleep(self.delay)
        if isinstance(self.answer, Exception):
            raise self.answer
        return LLMResult(data=self.answer, provider="anthropic", model="claude-haiku-5-5", latency_ms=420.0, cost_usd=0.0004)


def _router(fake: _Fake | None) -> LLMRouter:
    if fake is None:
        return LLMRouter(resolved=[])
    chain = load_chain(None, overrides={"LLM_PROVIDER_CHAIN": "anthropic", "ANTHROPIC_API_KEY": "k" * 20})
    return LLMRouter(resolved=chain, factory=lambda rp: fake)


def _sup(answer, **kw) -> tuple[TradeSupervisor, _Fake]:
    fake = _Fake(answer, delay=kw.pop("delay", 0.0))
    return TradeSupervisor(_router(fake), SupervisorSettings(**kw)), fake


# ---------- verdict rules ----------

def test_approve_uses_fast_model_low_effort_and_tight_timeout() -> None:
    sup, fake = _sup({"decision": "APPROVE", "size_multiplier": 3.0, "confidence": 0.8, "reason": "clean"})
    v = asyncio.run(sup.review("ctx"))
    assert v.decision == "APPROVE" and v.size_multiplier == 1.0  # can never scale up
    req = fake.requests[0]
    assert req.fast and req.effort == "low" and req.timeout_s == 6.0 and not req.tools
    assert req.system == SUPERVISOR_SYSTEM and req.schema == SUPERVISOR_SCHEMA


@pytest.mark.parametrize("raw,expected", [(0.5, 0.5), (0.01, 0.25), (1.0, 0.9), (float("nan"), 0.25)])
def test_reduce_multiplier_is_clamped(raw: float, expected: float) -> None:
    sup, _ = _sup({"decision": "REDUCE", "size_multiplier": raw, "confidence": 0.6, "reason": "spread high"})
    v = asyncio.run(sup.review("ctx"))
    assert v.decision == "REDUCE" and v.size_multiplier == pytest.approx(expected)


def test_reject() -> None:
    sup, _ = _sup({"decision": "REJECT", "size_multiplier": 1.0, "confidence": 0.9, "reason": "chasing into high"})
    v = asyncio.run(sup.review("ctx"))
    assert not v.allows_entry and v.size_multiplier == 0.0 and "chasing" in v.reason


def test_fail_closed_on_error_and_timeout_by_default() -> None:
    sup, _ = _sup(LLMError("down"))
    assert asyncio.run(sup.review("ctx")).allows_entry is False
    sup_allow, _ = _sup(LLMError("down"), on_unavailable="allow")
    v = asyncio.run(sup_allow.review("ctx"))
    assert v.allows_entry and "policy=allow" in v.reason


def test_invalid_answer_counts_as_unavailable() -> None:
    sup, _ = _sup({"decision": "YOLO", "size_multiplier": 2, "confidence": 1, "reason": "x"})
    v = asyncio.run(sup.review("ctx"))
    assert v.decision == "REJECT" and v.error


def test_budget_and_rate_limit() -> None:
    sup, fake = _sup({"decision": "APPROVE", "size_multiplier": 1, "confidence": 1, "reason": ""}, max_calls_per_hour=2)
    for _ in range(2):
        assert asyncio.run(sup.review("c")).allows_entry
    v = asyncio.run(sup.review("c"))
    assert not v.allows_entry and "hourly" in v.reason and len(fake.requests) == 2
    sup2, _ = _sup({"decision": "APPROVE", "size_multiplier": 1, "confidence": 1, "reason": ""}, daily_budget_usd=0.0)
    assert "budget" in asyncio.run(sup2.review("c")).reason


def test_inactive_without_provider() -> None:
    assert TradeSupervisor(_router(None)).active is False


def test_scale_volume_and_drift() -> None:
    spec = DEFAULT_SPECS["XAUUSD"]
    assert scale_volume(0.40, 0.5, spec) == pytest.approx(0.20)
    assert scale_volume(0.40, 2.0, spec) == pytest.approx(0.40)
    assert scale_volume(0.01, 0.5, spec) == 0.0
    assert price_drifted(Side.BUY, 2600.0, 2600.6, atr=2.0, max_drift_atr=0.25) is True
    assert price_drifted(Side.BUY, 2600.0, 2599.0, atr=2.0, max_drift_atr=0.25) is False  # better price is fine
    assert price_drifted(Side.SELL, 2600.0, 2599.4, atr=2.0, max_drift_atr=0.25) is True


# ---------- context the model actually sees ----------

def _ind(close: float) -> IndicatorSnapshot:
    fields = {f: 0.0 for f in IndicatorSnapshot.__dataclass_fields__}
    fields.update(close=close, open=close, high=close + 1, low=close - 1, ema9=close + 0.2, ema21=close, vwap=close - 0.5,
                  atr14=2.0, rsi7=61.0, adx14=27.0)
    return IndicatorSnapshot(**fields)


def _bars(n: int = 30, base: float = 2600.0) -> list[Bar]:
    return [Bar(symbol="XAUUSD", timestamp=60 * i, open=base + i * 0.1, high=base + i * 0.1 + 1, low=base + i * 0.1 - 1,
                close=base + i * 0.1 + 0.3) for i in range(n)]


def _signal(entry: float = 2603.0) -> TradeSignal:
    return TradeSignal(symbol="XAUUSD", strategy_id="micro_breakout", side=Side.BUY, entry_price=entry, sl_price=entry - 2.7,
                       tp_price=entry + 4.8, atr=2.0, confidence=0.71, regime=MarketRegime.TRENDING_BULL, reason="squeeze break")


def test_context_is_compact_and_complete() -> None:
    ctx = build_context(
        signal=_signal(), spec=DEFAULT_SPECS["XAUUSD"], volume=0.12, sl=2600.3, tp=2607.8, spread_points=25,
        bars=_bars(), ind=_ind(2603.0), dl_up=0.58, dl_down=0.21, ml_prob=0.63, bandit_weight=0.34,
        open_positions=[], recent_trades=[], daily_dd_pct=0.8, max_daily_dd_pct=3.0, consecutive_losses=1, hour_utc=14,
    )
    payload = json.loads(ctx.split("\n", 1)[1])
    assert set(payload) == {"setup", "market", "models", "account"}
    assert payload["setup"]["rr"] == pytest.approx(1.78, abs=0.01)
    assert payload["market"]["spread_atr"] == pytest.approx(0.125)
    assert len(payload["market"]["last_bars_atr"]) == 12
    assert len(ctx) < 2500  # ~600 tokens: keeps latency low for M1


# ---------- engine wiring ----------

def _orch(tmp_path: Path, fake: _Fake | None, mode: TradingMode = TradingMode.MT5_DEMO) -> TradingOrchestrator:
    cfg = load_trading_config()
    cfg.app.database_path = str(tmp_path / "t.db")
    cfg.app.snapshots_dir = str(tmp_path / "snaps")
    cfg.app.kill_switch_path = str(tmp_path / "kill.json")
    cfg.app.trading_mode = mode
    env = EnvSettings(bridge_auth_token="b" * 30, control_api_token="c" * 30, _env_file=None)
    return TradingOrchestrator(cfg, env, Database(cfg.app.database_path), llm_router=_router(fake))


async def _run(orch: TradingOrchestrator, price: float = 2603.0) -> None:
    await orch.db.connect()
    try:
        orch.ticks["XAUUSD"] = TickSnapshot(symbol="XAUUSD", bid=price - 0.2, ask=price, spread_points=20, time_msc=0)
        rd = RiskDecision(True, volume=0.40, adjusted_sl=2600.3, adjusted_tp=2607.8, risk_pct_used=0.5)
        assess = SimpleNamespace(regime=MarketRegime.TRENDING_BULL)
        await orch._supervise_then_finalize("ctx", "XAUUSD", _signal(), rd, assess, DEFAULT_SPECS["XAUUSD"], 0.63, 0.58, 0.34, 20.0, 1.0)
    finally:
        await orch.db.close()


def _queued_opens(orch: TradingOrchestrator) -> list[Any]:
    return [c for c in orch.order_sm._queue if c.action == OrderAction.OPEN]


def test_engine_approve_dispatches(tmp_path: Path) -> None:
    orch = _orch(tmp_path, _Fake({"decision": "APPROVE", "size_multiplier": 1, "confidence": 0.8, "reason": "clean break"}))
    asyncio.run(_run(orch))
    opens = _queued_opens(orch)
    assert len(opens) == 1 and opens[0].volume == pytest.approx(0.40)
    assert any("Supervisor APPROVE" in e.message for e in orch.recent_decisions)


def test_engine_reduce_scales_lots(tmp_path: Path) -> None:
    orch = _orch(tmp_path, _Fake({"decision": "REDUCE", "size_multiplier": 0.5, "confidence": 0.6, "reason": "spread wide"}))
    asyncio.run(_run(orch))
    assert _queued_opens(orch)[0].volume == pytest.approx(0.20)


def test_engine_veto_and_outage_block(tmp_path: Path) -> None:
    orch = _orch(tmp_path, _Fake({"decision": "REJECT", "size_multiplier": 0, "confidence": 0.9, "reason": "chasing"}))
    asyncio.run(_run(orch))
    assert not _queued_opens(orch)
    assert orch.recent_decisions[0].tag == "SUPERVISOR VETO"
    orch2 = _orch(tmp_path / "b", _Fake(LLMError("529 overloaded")))
    asyncio.run(_run(orch2))
    assert not _queued_opens(orch2) and orch2.recent_decisions[0].tag == "SUPERVISOR DOWN"


def test_engine_skips_if_price_ran_away(tmp_path: Path) -> None:
    orch = _orch(tmp_path, _Fake({"decision": "APPROVE", "size_multiplier": 1, "confidence": 0.8, "reason": "ok"}))
    asyncio.run(_run(orch, price=2603.0 + 0.8))  # 0.4 ATR worse than proposed
    assert not _queued_opens(orch) and orch.recent_decisions[0].tag == "STALE AFTER REVIEW"


def test_engine_respects_kill_switch_after_review(tmp_path: Path) -> None:
    orch = _orch(tmp_path, _Fake({"decision": "APPROVE", "size_multiplier": 1, "confidence": 0.8, "reason": "ok"}))
    orch.risk.killed = True
    asyncio.run(_run(orch))
    assert not _queued_opens(orch)


def test_shadow_mode_review_logs_without_dispatch(tmp_path: Path) -> None:
    orch = _orch(tmp_path, _Fake({"decision": "APPROVE", "size_multiplier": 1, "confidence": 0.8, "reason": "ok"}), TradingMode.SHADOW_ONLY)
    asyncio.run(_run(orch))
    assert not _queued_opens(orch)
    assert orch.recent_decisions[0].tag == "SHADOW ONLY" and "Supervisor APPROVE" in orch.recent_decisions[0].message
