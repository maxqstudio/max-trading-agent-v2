"""Hit & Run's own stop loss: tighter price stop, money cap via lot size, time stop, learning."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from backend.core.config import EnvSettings, load_trading_config
from backend.core.models import MarketRegime, Side, TickSnapshot, utc_now_ms
from backend.engine.hitrun_stop import HitRunArm, HitRunTuner, plan_stop, time_stop_due
from backend.engine.orchestrator import HITRUN_VERSION, TradingOrchestrator
from backend.llm.router import LLMRouter
from backend.risk.manager import DEFAULT_SPECS
from backend.storage.database import Database

XAU = DEFAULT_SPECS["XAUUSD"]  # 100 account money per 1.0 price per lot
EUR = DEFAULT_SPECS["EURUSD"]


def plan(**kw):
    args = dict(
        side=Side.BUY, entry=2650.0, atr=1.0, spread_px=0.1, spec=XAU, normal_sl=2645.0, normal_volume=0.5,
        commission_per_lot=7.0, min_net_profit=1.0, max_loss_mult=4.0, sl_min_spread_mult=3.0, arm=HitRunArm(1.2, 4),
    )
    args.update(kw)
    return plan_stop(**args)


def test_stop_is_atr_based_and_lot_caps_the_loss():
    p = plan()
    assert p.sl == 2648.8  # 1.2 x ATR below entry, much tighter than Normal's 2645
    # 1.2 price x 100 + $7 commission = $127 per lot -> $4 cap -> 0.03 lot
    assert p.volume == 0.03 and p.max_loss <= 4.0
    assert p.risk_dist < 2650.0 - 2645.0


def test_stop_never_closer_than_spread_floor_nor_wider_than_normal():
    p = plan(atr=0.05, spread_px=0.3)  # ATR stop 0.06 < 3 x spread 0.9
    assert abs(p.risk_dist - 0.9) < 1e-9
    p = plan(atr=10.0, normal_sl=2647.0)  # ATR stop 12 > Normal's 3: use Normal's distance
    assert abs(p.risk_dist - 3.0) < 1e-9 and p.sl == 2647.0
    p = plan(side=Side.SELL, entry=2650.0, normal_sl=2655.0)
    assert p.sl == 2651.2


def test_skips_when_min_lot_breaks_the_cap_and_never_exceeds_normal_lot():
    assert plan(atr=10.0, normal_sl=2600.0) is None  # 12 x 100 x 0.01 = $12 > $4
    p = plan_stop(side=Side.BUY, entry=1.1, atr=0.00008, spread_px=0.00001, spec=EUR, normal_sl=1.098,
                  normal_volume=0.2, commission_per_lot=7.0, min_net_profit=1.0, max_loss_mult=4.0,
                  sl_min_spread_mult=3.0, arm=HitRunArm(1.2, 4))
    assert p.volume <= 0.2


def test_time_stop():
    assert not time_stop_due(1000, 1000 + 3 * 60, 4, -0.5)
    assert time_stop_due(1000, 1000 + 4 * 60, 4, -0.5)
    assert not time_stop_due(1000, 1000 + 4 * 60, 4, 0.4)  # in net profit: keep waiting for the take
    assert not time_stop_due(1000, 1000 + 99 * 60, 0, -3.0)  # 0 = off


def test_tuner_needs_samples_and_a_clear_edge():
    default = HitRunArm(1.2, 4)
    t = HitRunTuner(min_samples=10)
    better = HitRunArm(0.9, 3)
    for i in range(9):
        t.record("EURUSD", "RANGING", default, -1.0 if i % 3 == 0 else 1.0)
        t.record("EURUSD", "RANGING", better, 1.5)
    assert t.choose("EURUSD", "RANGING", default).arm == default  # too few trades
    for i in range(6):
        t.record("EURUSD", "RANGING", default, -1.0 if i % 3 == 0 else 1.0)
        t.record("EURUSD", "RANGING", better, 1.5)
    c = t.choose("EURUSD", "RANGING", default)
    assert c.arm == better and c.scope == "EURUSD|RANGING"
    assert t.choose("XAUUSD", "RANGING", default).scope == "*"  # falls back to all trades
    t2 = HitRunTuner(min_samples=10)
    t2.load(t.to_dict())
    assert t2.choose("EURUSD", "RANGING", default).arm == better
    assert any(r["key"] == "All" for r in t.table(default))


def _orch(tmp_path: Path):
    cfg = load_trading_config(local_path=None)
    cfg.app.database_path = str(tmp_path / "h.db")
    cfg.app.snapshots_dir = str(tmp_path / "s")
    cfg.app.kill_switch_path = str(tmp_path / "k.json")
    env = EnvSettings(bridge_auth_token="b" * 32, control_api_token="c" * 32, _env_file=None)
    db = Database(cfg.app.database_path)
    return TradingOrchestrator(cfg, env, db, llm_router=LLMRouter(resolved=[])), db


def tk(bid):
    return TickSnapshot(symbol="XAUUSD", bid=bid, ask=bid + 0.1, spread_points=10, timestamp_ms=utc_now_ms())


def test_paper_hitrun_book_uses_its_own_stop_and_time_stop(tmp_path: Path):
    orch, db = _orch(tmp_path)

    async def run():
        await db.connect()
        try:
            await orch.initialize_from_storage()
            orch.ticks["XAUUSD"] = tk(2650.0)
            await orch._open_paper_position(
                "XAUUSD", SimpleNamespace(side=Side.BUY, strategy_id="s", atr=1.0),
                SimpleNamespace(volume=0.5, adjusted_sl=2645.0, adjusted_tp=2660.0),
                SimpleNamespace(regime=MarketRegime.RANGING), orch.specs["XAUUSD"], 0.6, 0.6, 10.0,
                books=("normal", "hitrun", "raw"),
            )
            hr = orch.paper_books["hitrun"].positions["XAUUSD"]
            nm = orch.paper_books["normal"].positions["XAUUSD"]
            assert hr.sl > nm.sl and hr.volume < nm.volume and hr.time_stop_bars == 4
            assert len(orch.hitrun_tuner.ghosts["XAUUSD"]) == len(orch.hitrun_tuner.arms(orch._hitrun_default_arm()))
            now = hr.open_time + 5 * 60  # 5 bars later, still slightly under water
            await orch._update_paper_positions_on_tick(tk(2649.9), now)
            return await db.list_trades(version_id=HITRUN_VERSION), orch.paper_books["normal"].positions
        finally:
            await db.close()

    hitrun, normal_open = asyncio.run(run())
    assert len(hitrun) == 1 and hitrun[0].exit_reason == "TIME_STOP"
    assert -4.0 <= hitrun[0].pnl_money < 0  # small loss, not Normal's full stop
    assert "XAUUSD" in normal_open  # Normal has no time stop


def test_mt5_entry_with_hitrun_style_sends_own_sl_and_lot(tmp_path: Path):
    orch, db = _orch(tmp_path)
    from backend.core.models import TradingMode

    async def run():
        await db.connect()
        try:
            await orch.initialize_from_storage()
            orch.config.app.trading_mode = TradingMode.MT5_DEMO
            orch.config.paper.enabled = False
            orch.ticks["XAUUSD"] = tk(2650.0)
            orch._mt5_exit_style = lambda sym, regime: "hitrun"
            sig = SimpleNamespace(side=Side.BUY, strategy_id="s", atr=1.0, entry_price=2650.1, reason="r")
            risk = SimpleNamespace(volume=0.5, adjusted_sl=2645.0, adjusted_tp=2660.0)
            await orch._finalize_entry("XAUUSD", sig, risk, SimpleNamespace(regime=MarketRegime.RANGING),
                                       orch.specs["XAUUSD"], 0.6, 0.6, 0.3, 10.0, 1.0)
            return list(orch.order_sm._orders.values())
        finally:
            await db.close()

    recs = asyncio.run(run())
    assert len(recs) == 1
    rec = recs[0]
    assert rec.exit_style == "hitrun" and rec.time_stop_bars == 4
    assert rec.command.volume == 0.03 and rec.command.sl > 2645.0
