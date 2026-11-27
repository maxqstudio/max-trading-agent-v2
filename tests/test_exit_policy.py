"""Profit lock (Normal) and Hit & Run exits, plus the two paper books running side by side."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from backend.core.config import EnvSettings, load_trading_config
from backend.core.models import MarketRegime, Side, TickSnapshot, utc_now_ms
from backend.engine.exit_policy import NetTake, ProfitLock, evaluate, net_take
from backend.engine.orchestrator import HITRUN_VERSION, PAPER_VERSION, TradingOrchestrator
from backend.llm.router import LLMRouter
from backend.storage.database import Database


def ev(policy, peak, price, sl=95.0, side=Side.BUY, spread=0.0):
    return evaluate(policy, side=side, entry=100.0, risk=5.0, price=price, peak_r=peak, sl=sl, spread_px=spread)


def test_profit_lock_holds_below_trigger_then_breakeven_then_trails():
    pl = ProfitLock(be_at_r=0.5, be_offset_r=0.05, trail_start_r=1.0, trail_dist_r=0.5)
    assert ev(pl, 0.4, 102.0) == ("HOLD", 95.0)
    act, sl = ev(pl, 0.5, 102.5)
    assert act == "MOVE_SL" and abs(sl - 100.25) < 1e-9  # BE + 0.05R
    act, sl = ev(pl, 1.6, 108.0, sl=100.25)
    assert act == "MOVE_SL" and abs(sl - 105.5) < 1e-9  # peak 1.6R - 0.5R = 1.1R
    assert ev(pl, 1.6, 108.0, sl=106.0)[0] == "HOLD"  # never loosens


def test_profit_lock_sell_side_and_spread_cost():
    pl = ProfitLock()
    act, sl = ev(pl, 0.6, 97.0, sl=105.0, side=Side.SELL, spread=0.5)
    assert act == "MOVE_SL" and abs(sl - (100.0 - 0.25 - 0.5)) < 1e-9


def test_net_take_needs_profit_after_all_costs():
    nt = NetTake(min_net_profit=1.0, commission_per_lot=7.0)
    # 0.10 lot pays $0.70 commission: $1.50 gross is only $0.80 net, not enough yet.
    assert net_take(nt, gross_money=1.5, volume=0.1) == (False, 0.8)
    assert net_take(nt, gross_money=1.7, volume=0.1) == (True, 1.0)
    # Real MT5 costs: swap and the reported commission are used instead of the estimate.
    assert net_take(nt, gross_money=2.0, volume=0.1, swap=-0.5, commission=-0.4) == (True, 1.1)
    assert net_take(nt, gross_money=2.0, volume=0.1, swap=-0.8, commission=0.4)[0] is False


def test_two_books_same_setup_different_exits(tmp_path: Path):
    cfg = load_trading_config(local_path=None)
    cfg.app.database_path = str(tmp_path / "b.db")
    cfg.app.snapshots_dir = str(tmp_path / "s")
    cfg.app.kill_switch_path = str(tmp_path / "k.json")
    env = EnvSettings(bridge_auth_token="b" * 32, control_api_token="c" * 32, _env_file=None)
    db = Database(cfg.app.database_path)
    orch = TradingOrchestrator(cfg, env, db, llm_router=LLMRouter(resolved=[]))

    def tk(bid):
        return TickSnapshot(symbol="XAUUSD", bid=bid, ask=bid + 0.1, spread_points=10, timestamp_ms=utc_now_ms())

    async def run():
        await db.connect()
        try:
            await orch.initialize_from_storage()
            orch.ticks["XAUUSD"] = tk(2650.0)
            await orch._open_paper_position(
                "XAUUSD", SimpleNamespace(side=Side.BUY, strategy_id="s", atr=1.0),
                SimpleNamespace(volume=0.1, adjusted_sl=2645.1, adjusted_tp=2660.1),
                SimpleNamespace(regime=MarketRegime.RANGING), orch.specs["XAUUSD"], 0.6, 0.6, 10.0,
            )
            now = int(utc_now_ms() // 1000)
            await orch._update_paper_positions_on_tick(tk(2650.1), now)  # gross $0, net -$0.70: hold
            assert len(orch.paper_books["hitrun"].positions) == 1
            for bid in (2653.0, 2654.5, 2651.0):  # +0.58R, +0.88R, then back down
                await orch._update_paper_positions_on_tick(tk(bid), now)
            normal = await db.list_trades(version_id=PAPER_VERSION)
            hitrun = await db.list_trades(version_id=HITRUN_VERSION)
            return normal, hitrun, orch.paper_snapshot()
        finally:
            await db.close()

    normal, hitrun, snap = asyncio.run(run())
    # Hit & Run took the first net profit; its P&L is after commission
    assert len(hitrun) == 1 and hitrun[0].exit_reason == "TAKE" and hitrun[0].pnl_money > 0
    assert hitrun[0].exit_price == 2653.0
    # Normal is still open, but its stop is at breakeven or better now (no full -1R anymore)
    assert normal == []
    pos = snap["books"]["normal"]["positions"][0]
    assert pos["sl"] >= 2650.1


def test_summarize_stats():
    from backend.api.paper_routes import summarize
    from backend.core.models import TradeRecord

    def tr(pnl, mfe, t):
        return TradeRecord(symbol="X", strategy_id="s", regime=MarketRegime.RANGING, side=Side.BUY, volume=0.1,
                           entry_price=1, exit_price=1, sl_price=1, tp_price=1, open_time=t, close_time=t,
                           pnl_money=pnl, pnl_r=pnl / 50, mfe_r=mfe)

    s = summarize([tr(100, 2, 1), tr(-50, 0.6, 2), tr(-50, 0.1, 3), tr(80, 1.5, 4)], 10000)
    assert s["trades"] == 4 and s["win_rate"] == 50.0 and s["profit_factor"] == 1.8
    assert s["losers_after_half_r"] == 1 and s["net_pnl"] == 80
    assert 0 < s["max_dd_pct"] < 1.0


def test_selector_needs_evidence_before_switching():
    from backend.engine.exit_selector import ExitStyleSelector

    sel = ExitStyleSelector(min_samples=20, z=1.0)
    for i in range(10):
        sel.record("hitrun", "EURUSD", "RANGING", 0.4)
        sel.record("normal", "EURUSD", "RANGING", -0.5 if i % 2 else 0.3)
    assert sel.choose("EURUSD", "RANGING").style == "normal"  # too few trades
    for i in range(20):
        sel.record("hitrun", "EURUSD", "RANGING", 0.4 if i % 4 else -1.0)
        sel.record("normal", "EURUSD", "RANGING", -0.5 if i % 2 else 0.3)
    c = sel.choose("EURUSD", "RANGING")
    assert c.style == "hitrun" and c.scope == "symbol|regime"
    # Trending XAUUSD: Normal lets winners run, so it stays Normal.
    for i in range(25):
        sel.record("hitrun", "XAUUSD", "TRENDING_UP", 0.4 if i % 3 else -1.0)
        sel.record("normal", "XAUUSD", "TRENDING_UP", 2.0 if i % 3 else -1.0)
    assert sel.choose("XAUUSD", "TRENDING_UP").style == "normal"
    # Unseen regime falls back to the symbol's own evidence.
    assert sel.choose("EURUSD", "VOLATILE").scope == "symbol"
    assert any(r["key"] == "All" for r in sel.table())
