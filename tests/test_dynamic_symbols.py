"""Symbols come from MT5: a symbol the EA reports that is not in the config watchlist is tracked."""

from __future__ import annotations

import asyncio
from pathlib import Path

from backend.bridge.simulator import SimulatedMT5Bridge
from backend.core.config import EnvSettings, load_trading_config
from backend.engine.orchestrator import TradingOrchestrator
from backend.llm.router import LLMRouter
from backend.storage.database import Database


def test_unknown_symbol_from_ea_is_registered(tmp_path: Path) -> None:
    cfg = load_trading_config()
    cfg.app.database_path = str(tmp_path / "dyn.db")
    cfg.app.snapshots_dir = str(tmp_path / "snaps")
    cfg.app.kill_switch_path = str(tmp_path / "kill.json")
    cfg.instruments.symbols = ["XAUUSD"]
    env = EnvSettings(bridge_auth_token="x" * 32, control_api_token="y" * 32, _env_file=None)
    db = Database(cfg.app.database_path)
    orch = TradingOrchestrator(cfg, env, db, llm_router=LLMRouter(resolved=[]))
    sim = SimulatedMT5Bridge(symbols=["XAUUSD", "EURUSD"], seed=3)

    async def run() -> None:
        await db.connect()
        try:
            await orch.initialize_from_storage()
            for _ in range(3):
                await orch.handle_bridge_sync(sim.step_and_build_sync(close_bar_every=1))
        finally:
            await db.close()

    asyncio.run(run())
    assert "EURUSD" in orch.config.instruments.symbols
    assert orch.bars["EURUSD"]
    assert "EURUSD" in orch.ticks
    radar_syms = {r["symbol"] for r in orch.get_live_state_snapshot()["radar"]}
    assert {"XAUUSD", "EURUSD"} <= radar_syms


def test_zero_prices_for_missing_broker_symbol_are_ignored(tmp_path: Path) -> None:
    cfg = load_trading_config()
    cfg.app.database_path = str(tmp_path / "z.db")
    cfg.app.snapshots_dir = str(tmp_path / "snaps")
    cfg.app.kill_switch_path = str(tmp_path / "kill.json")
    env = EnvSettings(bridge_auth_token="x" * 32, control_api_token="y" * 32, _env_file=None)
    db = Database(cfg.app.database_path)
    orch = TradingOrchestrator(cfg, env, db, llm_router=LLMRouter(resolved=[]))
    sim = SimulatedMT5Bridge(symbols=["XAUUSD"], seed=5)

    async def run() -> None:
        await db.connect()
        try:
            await orch.initialize_from_storage()
            req = sim.step_and_build_sync(close_bar_every=1)
            for t in req.ticks:
                t.bid = t.ask = 0.0
            for sp in req.symbol_specs:
                sp.point = 0.0
            for b in req.closed_bars:
                b.open = b.high = b.low = b.close = 0.0
            await orch.handle_bridge_sync(req)
        finally:
            await db.close()

    asyncio.run(run())
    assert orch.last_bridge_sync_ms > 0  # heartbeat still counts
    assert "XAUUSD" not in orch.ticks
    assert orch.specs["XAUUSD"].point > 0
