from __future__ import annotations

import time

import pytest

from backend.core.config import EnvSettings, RiskConfig, TradingConfig
from backend.core.models import (
    AccountState,
    Bar,
    BridgeSyncRequest,
    MarketRegime,
    Side,
    SymbolSpec,
    TickSnapshot,
    TradeRecord,
    TradeSignal,
)
from backend.engine.exit_policy import NetTake, ProfitLock, evaluate, net_take
from backend.engine.filters import FilterLearner, decide
from backend.engine.orchestrator import PAPER_VERSION, TradingOrchestrator
from backend.engine.supervisor import SupervisorSettings, TradeSupervisor
from backend.llm.base import LLMRequest, LLMResult
from backend.llm.presets import PRESETS
from backend.llm.router import LLMRouter
from backend.llm.settings import ResolvedProvider
from backend.risk.manager import PortfolioRiskManager
from backend.storage.database import Database


def _eurusd_spec() -> SymbolSpec:
    return SymbolSpec(
        symbol="EURUSD",
        broker_symbol="EURUSD",
        digits=5,
        point=0.00001,
        tick_size=0.00001,
        tick_value=1.0,
        volume_min=0.01,
        volume_max=50.0,
        volume_step=0.01,
        stops_level=10,
        commission_per_lot=14.0,
    )


def _xau_spec() -> SymbolSpec:
    return SymbolSpec(
        symbol="XAUUSD",
        broker_symbol="XAUUSD",
        digits=2,
        point=0.01,
        tick_size=0.01,
        tick_value=1.0,
        volume_min=0.01,
        volume_max=50.0,
        volume_step=0.01,
        stops_level=20,
        commission_per_lot=7.0,
    )


def test_cost_to_risk_gate_blocks_tight_sl_high_commission():
    cfg = RiskConfig(risk_per_trade_pct=0.5, max_spread_atr_ratio=0.35, max_cost_to_risk_ratio=0.18)
    rm = PortfolioRiskManager(cfg)
    spec = _eurusd_spec()
    acct = AccountState(balance=10000, equity=10000, day_start_equity=10000)

    # Real EURUSD failure case: 2.4 pip SL + $14/lot commission (= 1.4 pip) + 0.3 pip spread = 1.7 / 2.4 = 71% > 18%
    tight_sig = TradeSignal(
        symbol="EURUSD",
        strategy_id="micro_breakout",
        side=Side.BUY,
        entry_price=1.08500,
        sl_price=1.08476,
        tp_price=1.08550,
        atr=0.00020,
        regime=MarketRegime.TRENDING_BULL,
        confidence=0.75,
        reason="tight scalp",
    )
    dec_tight = rm.evaluate_entry(
        tight_sig,
        acct,
        open_positions=[],
        spec=spec,
        spread_points=3.0,
        ml_probability=0.65,
        bandit_weight=0.5,
        now_ts=1700000000,
        commission_per_lot=14.0,
    )
    assert not dec_tight.approved
    assert dec_tight.reason_code == "COST_TO_RISK_GATE"

    # Healthy 12-pip SL: (0.3 pip spread + 1.4 pip comm) / 12 pip = 14.2% <= 18% -> approved!
    wide_sig = TradeSignal(
        symbol="EURUSD",
        strategy_id="structure_shift",
        side=Side.BUY,
        entry_price=1.08500,
        sl_price=1.08380,
        tp_price=1.08700,
        atr=0.00080,
        regime=MarketRegime.TRENDING_BULL,
        confidence=0.75,
        reason="healthy structural stop",
    )
    dec_wide = rm.evaluate_entry(
        wide_sig,
        acct,
        open_positions=[],
        spec=spec,
        spread_points=3.0,
        ml_probability=0.65,
        bandit_weight=0.5,
        now_ts=1700000000,
        commission_per_lot=14.0,
    )
    assert dec_wide.approved
    assert dec_wide.volume > 0


def test_commission_aware_profit_lock_breakeven_never_nets_negative():
    policy = ProfitLock(be_at_r=0.85, be_offset_r=0.08, trail_start_r=1.25, trail_dist_r=0.55)
    entry = 1.08500
    risk = 0.00100  # 10 pips
    spread_px = 0.00004  # 0.4 pip
    comm_px = 0.00014  # $14/lot on EURUSD = 1.4 pips

    # Before +0.85R: HOLD (no premature choke at +0.5R!)
    act_hold, _ = evaluate(
        policy,
        side=Side.BUY,
        entry=entry,
        risk=risk,
        price=entry + 0.65 * risk,
        peak_r=0.65,
        sl=entry - risk,
        spread_px=spread_px,
        commission_px=comm_px,
    )
    assert act_hold == "HOLD"

    # At +0.90R: BE stop must clear entry + spread_px + comm_px + offset
    act_be, new_sl = evaluate(
        policy,
        side=Side.BUY,
        entry=entry,
        risk=risk,
        price=entry + 0.90 * risk,
        peak_r=0.90,
        sl=entry - risk,
        spread_px=spread_px,
        commission_px=comm_px,
    )
    assert act_be == "MOVE_SL"
    assert new_sl >= entry + spread_px + comm_px + 0.08 * risk - 1e-9

    # Mirror check for SELL
    act_sell, sell_sl = evaluate(
        policy,
        side=Side.SELL,
        entry=entry,
        risk=risk,
        price=entry - 0.90 * risk,
        peak_r=0.90,
        sl=entry + risk,
        spread_px=spread_px,
        commission_px=comm_px,
    )
    assert act_sell == "MOVE_SL"
    assert sell_sl <= entry - (spread_px + comm_px + 0.08 * risk) + 1e-9


def test_hitrun_cost_multiple_profit_floor():
    rule = NetTake(min_net_profit=1.0, commission_per_lot=14.0, min_cost_multiple=2.0)
    # 0.10 lot -> comm = $1.40, spread = $1.10 -> total friction = $2.50 -> required net >= $5.00
    take_small, net_small = net_take(rule, gross_money=3.0, volume=0.10, spread_money=1.10)
    assert not take_small
    assert round(net_small, 2) == 1.60

    take_big, net_big = net_take(rule, gross_money=6.60, volume=0.10, spread_money=1.10)
    assert take_big
    assert round(net_big, 2) == 5.20


def test_htf_trend_guard_and_ranging_breakout_guard():
    learner = FilterLearner(min_samples=30)  # 0 samples so auto mode wouldn't block on its own

    # 1. Trend strategy against M15 HTF -> immediately blocked by HTF_TREND_AGAINST
    d_trend_against = decide(
        learner,
        htf_mode="auto",
        htf_timeframe="M15",
        session_mode="auto",
        blocked_sessions=["asia"],
        m15=-1,
        h1=-1,
        session="london",
        strategy_id="ema_vwap_pullback",
        regime="TRENDING_UP",
        trend_strategies_require_htf=True,
        block_breakout_in_ranging=True,
    )
    assert d_trend_against.code == "HTF_TREND_AGAINST"

    # 2. Reversal strategy against HTF -> allowed (reversal is exempt from trend guard!)
    d_rev = decide(
        learner,
        htf_mode="auto",
        htf_timeframe="M15",
        session_mode="auto",
        blocked_sessions=["asia"],
        m15=-1,
        h1=-1,
        session="london",
        strategy_id="liquidity_sweep",
        regime="RANGING",
        trend_strategies_require_htf=True,
        block_breakout_in_ranging=True,
    )
    assert d_rev.code == ""

    # 3. Micro breakout in RANGING regime -> blocked by BREAKOUT_IN_RANGE even when HTF agrees
    d_range_bo = decide(
        learner,
        htf_mode="auto",
        htf_timeframe="M15",
        session_mode="auto",
        blocked_sessions=["asia"],
        m15=1,
        h1=1,
        session="london",
        strategy_id="micro_breakout",
        regime="RANGING",
        trend_strategies_require_htf=True,
        block_breakout_in_ranging=True,
    )
    assert d_range_bo.code == "BREAKOUT_IN_RANGE"


class _FakeDeskBriefProvider:
    name = "fake"

    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls = 0

    async def complete(self, req: LLMRequest) -> LLMResult:
        self.calls += 1
        return LLMResult(
            data=self.payload,
            provider="fake",
            model="fake-fast",
            latency_ms=42.0,
            cost_usd=0.001,
        )


@pytest.mark.asyncio
async def test_periodic_desk_brief_zero_latency_and_hybrid_routing():
    fake = _FakeDeskBriefProvider(
        {
            "market_summary": "Gold bullish impulse, EUR choppy pre-CPI",
            "symbols": [
                {"symbol": "XAUUSD", "bias": "LONG_ONLY", "risk_multiplier": 1.0, "avoid_strategies": [], "note": "Clean uptrend"},
                {"symbol": "EURUSD", "bias": "FLAT", "risk_multiplier": 0.0, "avoid_strategies": [], "note": "High chop"},
                {
                    "symbol": "GBPUSD",
                    "bias": "BOTH",
                    "risk_multiplier": 0.5,
                    "avoid_strategies": ["micro_breakout"],
                    "note": "Half size only",
                },
            ],
        }
    )
    rp = ResolvedProvider(
        preset=PRESETS["anthropic"],
        api_key="test-key",
        model="m",
        fast_model="m-fast",
        base_url="http://localhost",
        fallback_models=(),
    )
    router = LLMRouter(resolved=[rp], factory=lambda _rp: fake)
    sup = TradeSupervisor(router, SupervisorSettings(enabled=True, mode="periodic", brief_interval_min=15))

    # Routing checks
    assert not sup.uses_per_trade_review("micro_breakout")
    assert not sup.uses_per_trade_review("structure_shift")
    sup.settings.mode = "hybrid"
    assert not sup.uses_per_trade_review("micro_breakout")
    assert sup.uses_per_trade_review("structure_shift")
    sup.settings.mode = "periodic"

    # Refresh once (1 LLM call)
    brief = await sup.refresh_brief("Multi-symbol desk digest:{}")
    assert fake.calls == 1
    assert set(brief.keys()) == {"XAUUSD", "EURUSD", "GBPUSD"}
    assert not sup.brief_is_due()

    # Evaluate multiple signals at 0ms with 0 extra LLM calls
    v1 = sup.evaluate_from_brief(symbol="XAUUSD", side=Side.BUY, strategy_id="ema_vwap_pullback")
    assert v1.decision == "APPROVE" and v1.size_multiplier == 1.0

    v2 = sup.evaluate_from_brief(symbol="XAUUSD", side=Side.SELL, strategy_id="ema_vwap_pullback")
    assert v2.decision == "REJECT" and v2.size_multiplier == 0.0

    v3 = sup.evaluate_from_brief(symbol="EURUSD", side=Side.BUY, strategy_id="liquidity_sweep")
    assert v3.decision == "REJECT"

    v4 = sup.evaluate_from_brief(symbol="GBPUSD", side=Side.BUY, strategy_id="liquidity_sweep")
    assert v4.decision == "REDUCE" and v4.size_multiplier == 0.5

    v5 = sup.evaluate_from_brief(symbol="GBPUSD", side=Side.BUY, strategy_id="micro_breakout")
    assert v5.decision == "REJECT"

    assert fake.calls == 1  # Zero LLM calls during entry evaluation!


@pytest.mark.asyncio
async def test_history_bars_backfill_and_clean_paper_reset(tmp_path):
    cfg = TradingConfig()
    cfg.app.database_path = str(tmp_path / "test.db")
    cfg.app.snapshots_dir = str(tmp_path / "snaps")
    cfg.app.kill_switch_path = str(tmp_path / "kill.json")
    cfg.evolution.auto_warmup_on_backfill = False  # don't run full WFO in unit test
    env = EnvSettings(bridge_auth_token="a" * 32, control_api_token="b" * 32)

    db = Database(cfg.app.database_path)
    await db.connect()
    try:
        orch = TradingOrchestrator(cfg, env, db, llm_router=LLMRouter(resolved=[]))
        await orch.initialize_from_storage()

        base_ts = int(time.time()) - 6000
        # Send out-of-order + duplicate historical bars via history_bars
        hist = [
            Bar(
                symbol="XAUUSD",
                timeframe="M1",
                timestamp=base_ts + i * 60,
                open=2650 + i * 0.1,
                high=2650.5 + i * 0.1,
                low=2649.5 + i * 0.1,
                close=2650.2 + i * 0.1,
                volume=100,
                spread_points=18,
            )
            for i in [5, 2, 0, 3, 1, 4, 2]
        ]
        req = BridgeSyncRequest(
            account=AccountState(balance=10000, equity=10000, day_start_equity=10000, is_demo=True),
            symbol_specs=[_xau_spec()],
            ticks=[TickSnapshot(symbol="XAUUSD", bid=2651.0, ask=2651.2, spread_points=20, timestamp_ms=int(time.time() * 1000))],
            history_bars=hist,
        )
        resp = await orch.handle_bridge_sync(req)
        assert resp.commands == []  # backfill never fires live orders
        stored_ts = [b.timestamp for b in orch.bars["XAUUSD"]]
        assert stored_ts == sorted(set(stored_ts))
        assert len(stored_ts) == 6

        # Seed one paper trade and one real MT5 trade, then call reset_paper_books
        now_s = int(time.time())
        paper_tr = TradeRecord(
            is_shadow=False,
            version_id=PAPER_VERSION,
            symbol="XAUUSD",
            strategy_id="micro_breakout",
            regime=MarketRegime.RANGING,
            side=Side.BUY,
            volume=0.1,
            entry_price=2650.0,
            exit_price=2648.0,
            sl_price=2648.0,
            tp_price=2654.0,
            open_time=now_s - 120,
            close_time=now_s - 60,
            pnl_money=-20.0,
            pnl_r=-1.0,
            exit_reason="SL",
        )
        live_tr = paper_tr.model_copy(
            update={"trade_id": "mt5-real-1", "version_id": "v1.0", "pnl_money": 45.0, "pnl_r": 1.5, "exit_reason": "TP"}
        )
        await db.save_trade(paper_tr)
        await db.save_trade(live_tr)
        orch.paper_books["normal"].equity = 9400.0
        orch.veto_stats = {"APPROVE": {"n": 10, "wins": 4, "sum_r": -2.0, "sum_money": -100.0}}

        res = await orch.reset_paper_books(reset_learners=True)
        assert res["cleared_trades"] == 1
        assert orch.paper_books["normal"].equity == cfg.paper.start_equity
        assert orch.veto_stats == {}
        remaining = await db.list_trades(limit=50)
        assert [t.trade_id for t in remaining] == ["mt5-real-1"]
        # Market bars are still preserved in DB!
        db_bars = await db.load_recent_bars("XAUUSD", "M1", limit=100)
        assert len(db_bars) == 6
    finally:
        await db.close()


def test_effective_ml_cutoff_is_rr_aware_and_monotonic():
    from backend.evolution.meta_labeler import effective_ml_cutoff

    # At RR=1.0 (50% BE WR), cutoff respects ml_min_probability
    assert effective_ml_cutoff(0.52, 1.0) == 0.52
    # At RR=1.6 (38.5% BE WR), positive-EV setups at 0.45 pass instead of getting starved at 0.56
    c16 = effective_ml_cutoff(0.52, 1.6)
    assert 0.43 <= c16 <= 0.46
    # At RR=2.0 (33.3% BE WR), cutoff is lower than RR=1.6 yet safely above break-even
    c20 = effective_ml_cutoff(0.52, 2.0)
    assert 0.38 <= c20 < c16
    # Monotonic: raising ml_min_probability always raises the cutoff
    assert effective_ml_cutoff(0.60, 1.6) > c16


def test_hitrun_tuner_ghosts_respect_min_cost_multiple():
    from backend.engine.hitrun_stop import HitRunArm, HitRunPlan, HitRunTuner

    tuner = HitRunTuner(min_samples=5)
    arm = HitRunArm(0.9, 5)
    plan = HitRunPlan(sl=2645.0, volume=0.10, risk_dist=5.0, max_loss=5.70, arm=arm)
    tuner.spawn("XAUUSD", "RANGING", Side.BUY, 2650.0, 1000, [plan])

    # Friction = $2.00, min_cost_multiple = 2.0 -> floor = $4.00. At net=$2.50, ghost must HOLD!
    closed_small = tuner.on_tick(
        "XAUUSD", 2650.32, 2650.42, 1030, 1.0, lambda g, px: 2.50, min_cost_multiple=2.0, friction_at=lambda g: 2.0
    )
    assert closed_small == 0
    assert len(tuner.ghosts["XAUUSD"]) == 1

    # At net=$4.20 >= $4.00, ghost closes and records
    closed_big = tuner.on_tick(
        "XAUUSD", 2650.50, 2650.60, 1040, 1.0, lambda g, px: 4.20, min_cost_multiple=2.0, friction_at=lambda g: 2.0
    )
    assert closed_big == 1
    assert tuner.ghosts["XAUUSD"] == []


@pytest.mark.asyncio
async def test_fourth_book_raw_hitrun_and_net_r_and_champion_ops_preserve(tmp_path):
    from types import SimpleNamespace
    from backend.engine.orchestrator import RAW_BOOKS, RAW_HITRUN_VERSION

    cfg = TradingConfig()
    cfg.app.database_path = str(tmp_path / "test4.db")
    cfg.app.snapshots_dir = str(tmp_path / "snaps4")
    cfg.app.kill_switch_path = str(tmp_path / "kill4.json")
    cfg.evolution.auto_warmup_on_backfill = False
    cfg.supervisor.mode = "hybrid"
    cfg.filters.htf_mode = "on"
    cfg.exits.be_at_r = 0.90
    env = EnvSettings(bridge_auth_token="a" * 32, control_api_token="b" * 32)

    db = Database(cfg.app.database_path)
    await db.connect()
    try:
        orch = TradingOrchestrator(cfg, env, db, llm_router=LLMRouter(resolved=[]))
        await orch.initialize_from_storage()
        orch.specs["XAUUSD"] = _xau_spec()
        orch.ticks["XAUUSD"] = TickSnapshot(symbol="XAUUSD", bid=2650.0, ask=2650.10, spread_points=10, timestamp_ms=int(time.time() * 1000))

        # 1. apply_champion_config updates strategy/AI params while keeping Setup ops (supervisor/filters/exits)
        champ = TradingConfig()
        champ.app.version = "v2.3"
        champ.ai.ml_min_probability = 0.58
        champ.supervisor.mode = "per_trade"  # should NOT overwrite operator's "hybrid"
        champ.filters.htf_mode = "off"  # should NOT overwrite operator's "on"
        champ.exits.be_at_r = 0.50  # should NOT overwrite operator's 0.90
        orch.apply_champion_config(champ)
        assert orch.config.ai.ml_min_probability == 0.58
        assert orch.config.supervisor.mode == "hybrid"
        assert orch.config.filters.htf_mode == "on"
        assert orch.config.exits.be_at_r == 0.90

        # 2. Opening RAW_BOOKS opens both "raw" (Normal) and "raw_hitrun" (Hit & Run with its own tighter stop/lot)
        sig = SimpleNamespace(side=Side.BUY, strategy_id="liquidity_sweep", atr=2.0)
        risk_dec = SimpleNamespace(volume=0.20, adjusted_sl=2640.0, adjusted_tp=2670.0)
        assess = SimpleNamespace(regime=MarketRegime.RANGING)
        await orch._open_paper_position(
            "XAUUSD", sig, risk_dec, assess, orch.specs["XAUUSD"], 0.65, 0.60, 10.0, books=RAW_BOOKS, veto="REJECT"
        )
        assert "XAUUSD" in orch.paper_books["raw"].positions
        assert "XAUUSD" in orch.paper_books["raw_hitrun"].positions
        rh = orch.paper_books["raw_hitrun"].positions["XAUUSD"]
        assert rh.volume <= risk_dec.volume
        assert rh.version_id == RAW_HITRUN_VERSION

        # 3. Price rallies enough for Hit & Run net take (after comm+spread floor) -> closes raw_hitrun with net_r < gross_r!
        now_s = int(time.time()) + 30
        await orch._update_paper_positions_on_tick(
            TickSnapshot(symbol="XAUUSD", bid=2653.0, ask=2653.10, spread_points=10, timestamp_ms=now_s * 1000),
            now_s,
        )
        assert "XAUUSD" not in orch.paper_books["raw_hitrun"].positions
        assert "XAUUSD" in orch.paper_books["raw"].positions  # raw (Normal) stays open waiting for TP!
        rh_trades = await db.list_trades(limit=10, version_id=RAW_HITRUN_VERSION)
        assert len(rh_trades) == 1
        tr = rh_trades[0]
        gross_r = (tr.exit_price - tr.entry_price) / rh.risk_dist
        assert tr.pnl_r < gross_r  # Net R strictly deducts commission!
        assert tr.pnl_money > 0
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_shadow_challenger_respects_ai_supervisor_brief_and_p0_filters(tmp_path):
    from backend.core.models import DLPrediction
    from backend.engine.supervisor import SymbolBrief
    from backend.features.regime import classify_regime

    cfg = TradingConfig()
    cfg.app.database_path = str(tmp_path / "test_sh.db")
    cfg.app.snapshots_dir = str(tmp_path / "snaps_sh")
    cfg.app.kill_switch_path = str(tmp_path / "kill_sh.json")
    cfg.evolution.auto_warmup_on_backfill = False
    cfg.supervisor.enabled = True
    cfg.supervisor.mode = "periodic"
    cfg.supervisor.review_in_shadow = True
    env = EnvSettings(bridge_auth_token="a" * 32, control_api_token="b" * 32)

    fake = _FakeDeskBriefProvider({"market_summary": "ok", "symbols": []})
    rp = ResolvedProvider(
        preset=PRESETS["anthropic"],
        api_key="test-key",
        model="m",
        fast_model="m-fast",
        base_url="http://localhost",
        fallback_models=(),
    )
    router = LLMRouter(resolved=[rp], factory=lambda _rp: fake)

    db = Database(cfg.app.database_path)
    await db.connect()
    try:
        orch = TradingOrchestrator(cfg, env, db, llm_router=router)
        await orch.initialize_from_storage()
        spec = _xau_spec()
        orch.specs["XAUUSD"] = spec
        now_s = int(time.time())
        bars = [
            Bar(
                symbol="XAUUSD",
                timeframe="M1",
                timestamp=now_s - (50 - i) * 60,
                open=2650.0 + (i % 3) * 0.2,
                high=2652.0 + (i % 3) * 0.2,
                low=2648.5 + (i % 3) * 0.2,
                close=2650.5 + (i % 3) * 0.2,
                volume=100,
                spread_points=10,
            )
            for i in range(50)
        ]
        assess = classify_regime(bars)
        assert assess is not None
        snap = assess.indicators
        sig_buy = TradeSignal(
            symbol="XAUUSD",
            strategy_id="liquidity_sweep",
            side=Side.BUY,
            entry_price=2650.5,
            sl_price=2647.5,
            tp_price=2656.5,
            atr=2.0,
            regime=MarketRegime.RANGING,
            confidence=0.80,
            reason="sweep",
        )
        orch.shadow_suite.scan_all = lambda *a, **kw: [sig_buy]
        dl = DLPrediction(prob_up=0.70, prob_down=0.15, prob_flat=0.15, vol_expansion=0.5, trained=True)

        # 1. AI Supervisor brief says SHORT_ONLY -> Challenger (AI Veto) BUY is rejected,
        #    while pure Challenger (Raw) still opens! Champion stats stay 0.
        orch.supervisor.brief["XAUUSD"] = SymbolBrief(symbol="XAUUSD", bias="SHORT_ONLY", risk_multiplier=1.0, avoid_strategies=[], reason="Bearish")
        orch.supervisor.brief_updated_ts = time.time()
        await orch._scan_shadow_challenger("XAUUSD", bars, snap, MarketRegime.RANGING, dl, 10.0, spec, now_s, htf=(1, 1))
        assert "XAUUSD" not in orch.shadow_positions
        assert "XAUUSD" in orch.shadow_raw_positions
        assert orch.shadow_raw_positions["XAUUSD"].version_id.endswith("-raw")
        assert orch.supervisor.stats["reject"] == 0

        # 2. Micro Breakout in RANGING -> Challenger (AI Veto) blocks via P0 guard even when brief allows BOTH!
        orch.supervisor.brief["XAUUSD"] = SymbolBrief(symbol="XAUUSD", bias="BOTH", risk_multiplier=1.0, avoid_strategies=[], reason="Two-way")
        bo_sig = sig_buy.model_copy(update={"strategy_id": "micro_breakout"})
        orch.shadow_suite.scan_all = lambda *a, **kw: [bo_sig]
        await orch._scan_shadow_challenger("XAUUSD", bars, snap, MarketRegime.RANGING, dl, 10.0, spec, now_s, htf=(1, 1))
        assert "XAUUSD" not in orch.shadow_positions

        # 3. Liquidity sweep when brief allows LONG_ONLY with risk_multiplier=0.5 -> Challenger (AI Veto) opens at 0.5x!
        orch.supervisor.brief["XAUUSD"] = SymbolBrief(symbol="XAUUSD", bias="LONG_ONLY", risk_multiplier=0.5, avoid_strategies=[], reason="Half size")
        orch.shadow_suite.scan_all = lambda *a, **kw: [sig_buy]
        await orch._scan_shadow_challenger("XAUUSD", bars, snap, MarketRegime.RANGING, dl, 10.0, spec, now_s, htf=(1, 1))
        assert "XAUUSD" in orch.shadow_positions
        assert orch.shadow_positions["XAUUSD"].volume == 0.5
        assert orch.supervisor.stats["reduce"] == 0  # Champion stats untouched

        # 4. Reset clears both Challenger books cleanly to $10,000
        await orch.reset_paper_books(reset_learners=True)
        assert orch.shadow_positions == {} and orch.shadow_raw_positions == {}
        assert orch.shadow_equity == cfg.paper.start_equity and orch.shadow_raw_equity == cfg.paper.start_equity
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_openai_max_completion_tokens_and_cross_provider_chains():
    import httpx
    from backend.llm.openai_compat_provider import OpenAICompatProvider
    from backend.llm.settings import load_model_chains

    # 1. OpenAI gpt-6-luna must send max_completion_tokens (NOT max_tokens!)
    seen_bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        seen_bodies.append(_json.loads(request.content.decode()))
        return httpx.Response(
            200,
            json={"model": "gpt-6-luna", "choices": [{"message": {"content": '{"ok": true}'}, "finish_reason": "stop"}]},
        )

    rp_oai = ResolvedProvider(
        preset=PRESETS["openai"],
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        model="gpt-6-luna",
        fast_model="gpt-6-luna",
    )
    prov = OpenAICompatProvider(rp_oai, transport=httpx.MockTransport(handler))
    res = await prov.complete(
        LLMRequest(
            task="ping",
            system="sys",
            user="hi",
            schema={"type": "object", "additionalProperties": False, "required": ["ok"], "properties": {"ok": {"type": "boolean"}}},
            max_tokens=256,
        )
    )
    assert res.data == {"ok": True}
    assert "max_completion_tokens" in seen_bodies[0]
    assert "max_tokens" not in seen_bodies[0]

    # 2. Cross-provider Main & Fast chains route independently across providers and fall back cleanly
    mc, fc = load_model_chains(
        env_file=None,
        overrides={
            "OPENAI_API_KEY": "k1",
            "GEMINI_API_KEY": "k2",
            "LLM_MAIN_CHAIN": "openai:gpt-6-luna,gemini:gemini-2.5-pro",
            "LLM_FAST_CHAIN": "gemini:gemini-3.5-flash-lite,openai:gpt-5-mini",
        },
    )
    assert [(r.name, r.model) for r in mc] == [("openai", "gpt-6-luna"), ("gemini", "gemini-2.5-pro")]
    assert [(r.name, r.fast_model) for r in fc] == [("gemini", "gemini-3.5-flash-lite"), ("openai", "gpt-5-mini")]

    called: list[str] = []

    class _MockProv:
        def __init__(self, rp: ResolvedProvider) -> None:
            self.rp = rp
            self.name = rp.name

        async def complete(self, req: LLMRequest) -> LLMResult:
            m = self.rp.fast_model if req.fast else self.rp.model
            called.append(f"{self.rp.name}:{m}")
            if m == "gpt-6-luna":
                raise RuntimeError("simulated outage")
            return LLMResult(data={"ok": True}, provider=self.rp.name, model=m, latency_ms=10.0)

    router = LLMRouter(resolved=[], main_chain=mc, fast_chain=fc, factory=lambda r: _MockProv(r))
    ping_main = LLMRequest(task="reflection", system="s", user="u", schema={}, fast=False)
    ping_fast = LLMRequest(task="supervisor", system="s", user="u", schema={}, fast=True)

    r_main, _ = await router.complete(ping_main)
    r_fast, _ = await router.complete(ping_fast)
    assert r_main.provider == "gemini" and r_main.model == "gemini-2.5-pro"
    assert r_fast.provider == "gemini" and r_fast.model == "gemini-3.5-flash-lite"
    assert called == ["openai:gpt-6-luna", "gemini:gemini-2.5-pro", "gemini:gemini-3.5-flash-lite"]
