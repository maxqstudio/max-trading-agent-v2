"""Paper trades, shadow arena and MT5 position tracking survive a restart."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from backend.bridge.state_machine import IdempotentOrderStateMachine
from backend.core.config import EnvSettings, load_trading_config
from backend.core.models import LivePosition, MarketRegime, Side, TickSnapshot, utc_now_ms
from backend.engine.orchestrator import PAPER_VERSION, RAW_BOOKS, TradingOrchestrator
from backend.llm.router import LLMRouter
from backend.storage.database import Database


def make(tmp_path: Path) -> tuple[TradingOrchestrator, Database]:
    cfg = load_trading_config(local_path=None)
    cfg.app.database_path = str(tmp_path / "p.db")
    cfg.app.snapshots_dir = str(tmp_path / "snaps")
    cfg.app.kill_switch_path = str(tmp_path / "kill.json")
    env = EnvSettings(bridge_auth_token="b" * 32, control_api_token="c" * 32, _env_file=None)
    db = Database(cfg.app.database_path)
    return TradingOrchestrator(cfg, env, db, llm_router=LLMRouter(resolved=[])), db


def tick(sym: str, bid: float, ask: float) -> TickSnapshot:
    return TickSnapshot(symbol=sym, bid=bid, ask=ask, spread_points=(ask - bid) / 0.01, timestamp_ms=utc_now_ms())


def test_paper_position_survives_restart_and_closes_into_journal(tmp_path: Path) -> None:
    async def first() -> None:
        orch, db = make(tmp_path)
        await db.connect()
        try:
            await orch.initialize_from_storage()
            orch.ticks["XAUUSD"] = tick("XAUUSD", 2650.00, 2650.20)
            sig = SimpleNamespace(side=Side.BUY, strategy_id="micro_breakout", atr=1.0)
            risk = SimpleNamespace(volume=0.10, adjusted_sl=2645.0, adjusted_tp=2655.0)
            assess = SimpleNamespace(regime=MarketRegime.TRENDING_BULL)
            await orch._open_paper_position("XAUUSD", sig, risk, assess, orch.specs["XAUUSD"], 0.6, 0.55, 20.0)
            # The "No Veto" books (Normal + Hit & Run) open before the supervisor; here supervisor rejected the setup.
            await orch._open_paper_position(
                "XAUUSD", sig, risk, assess, orch.specs["XAUUSD"], 0.6, 0.55, 20.0, books=RAW_BOOKS, veto="REJECT"
            )
            assert all("XAUUSD" in b.positions for b in orch.paper_books.values())
            orch.shadow_equity = 10123.45
            await orch._save_arena()
        finally:
            await db.close()

    async def second() -> tuple[float, list]:
        orch, db = make(tmp_path)
        await db.connect()
        try:
            await orch.initialize_from_storage()
            assert all("XAUUSD" in b.positions for b in orch.paper_books.values())  # restored
            assert orch.shadow_equity == 10123.45
            await orch._update_paper_positions_on_tick(tick("XAUUSD", 2655.50, 2655.70), int(utc_now_ms() // 1000))
            assert all("XAUUSD" not in b.positions for b in orch.paper_books.values())
            # The rejected setup still taught the bandit (market outcome only) and is on the scoreboard.
            assert [r["verdict"] for r in orch.veto_scoreboard()] == ["REJECT"]
            assert orch.veto_scoreboard()[0]["avg_r"] > 0
            assert orch.bandit.to_dict()
            trades = await db.list_trades(version_id=PAPER_VERSION)
            mt5_only = await db.list_trades(is_shadow=False, exclude_version_prefix="PAPER")
            return orch.paper_books["normal"].equity, [trades, mt5_only]
        finally:
            await db.close()

    asyncio.run(first())
    eq, (paper_trades, mt5_only) = asyncio.run(second())
    assert len(paper_trades) == 1 and paper_trades[0].exit_reason == "TP" and paper_trades[0].pnl_money > 0
    assert mt5_only == []
    assert eq > 10000.0

    async def third() -> float:
        orch, db = make(tmp_path)
        await db.connect()
        try:
            await orch.initialize_from_storage()
            return orch.paper_books["normal"].equity
        finally:
            await db.close()

    assert asyncio.run(third()) == eq  # paper equity persisted too


def test_mt5_position_closed_while_offline_is_recorded() -> None:
    sm = IdempotentOrderStateMachine()
    pos = LivePosition(ticket=777, symbol="XAUUSD", side=Side.BUY, volume=0.1, open_price=2650.0,
                       current_price=2652.0, sl=2645.0, tp=2660.0, profit=20.0, open_time=1000)
    sm.reconcile_positions([pos], {}, "v1.0", 1100)
    saved = sm.export_tracking()

    after_restart = IdempotentOrderStateMachine()
    assert after_restart.import_tracking(saved) == 1
    _, closed = after_restart.reconcile_positions([], {}, "v1.0", 5000)  # gone from MT5 on first sync
    assert len(closed) == 1 and closed[0].ticket == 777 and closed[0].pnl_money == 20.0
