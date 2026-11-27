from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from backend.api.routes import create_app
from backend.bridge.simulator import SimulatedMT5Bridge
from backend.core.config import EnvSettings, load_trading_config
from backend.core.models import TradingMode
from backend.engine.orchestrator import TradingOrchestrator
from backend.storage.database import Database


def test_e2e_authenticated_bridge_simulation_evolution_and_kill_switch(tmp_path: Path) -> None:
    cfg = load_trading_config()
    cfg.app.database_path = str(tmp_path / "e2e.db")
    cfg.app.snapshots_dir = str(tmp_path / "snaps")
    cfg.evolution.wfo_trials = 4
    cfg.evolution.monte_carlo_iterations = 200
    cfg.ai.dl_epochs = 4
    cfg.app.kill_switch_path = str(tmp_path / "kill.json")
    cfg.app.trading_mode = TradingMode.MT5_DEMO  # default is paper; this test exercises the demo bridge

    env = EnvSettings(
        bridge_auth_token="test_bridge_secret_token",
        control_api_token="test_control_secret_token_xyz",
        llm_provider="deterministic",
        _env_file=None,
    )
    db = Database(cfg.app.database_path)
    orch = TradingOrchestrator(cfg, env, db)
    app = create_app(orch)

    sim = SimulatedMT5Bridge(symbols=cfg.instruments.symbols, seed=99)

    async def _boot():
        await db.connect()
        await orch.initialize_from_storage()
        boot = sim.generate_bootstrap_bars(count=180)
        for sym, blist in boot.items():
            await db.save_bars(blist)
            orch.bars[sym].extend(blist)

    import asyncio

    asyncio.run(_boot())

    try:
        with TestClient(app, headers={"X-Control-Token": "test_control_secret_token_xyz"}) as client:
            # 1. Health endpoint
            assert client.get("/api/health").status_code == 200

            # 2. Bridge auth enforced (401 without valid X-Bridge-Token)
            sync_req = sim.step_and_build_sync(close_bar_every=1)
            r_unauth = client.post("/api/bridge/sync", json=sync_req.model_dump(mode="json"))
            assert r_unauth.status_code == 401

            headers = {"X-Bridge-Token": "test_bridge_secret_token"}
            for _ in range(10):
                req = sim.step_and_build_sync(close_bar_every=2)
                r_ok = client.post("/api/bridge/sync", headers=headers, json=req.model_dump(mode="json"))
                assert r_ok.status_code == 200
                cmds = r_ok.json().get("commands", [])
                if cmds:
                    from backend.core.models import OrderCommand

                    sim.execute_commands([OrderCommand.model_validate(c) for c in cmds])

            # 3. Demo signal + live execution sync round-trip
            r_demo = client.post("/api/demo/signal?symbol=XAUUSD")
            assert r_demo.status_code == 200 and r_demo.json()["ok"] is True
            req2 = sim.step_and_build_sync(close_bar_every=4)
            r_sync2 = client.post("/api/bridge/sync", headers=headers, json=req2.model_dump(mode="json"))
            assert len(r_sync2.json()["commands"]) >= 1

            # 4. State, Brain, Journal, Config endpoints
            st = client.get("/api/state").json()
            assert st["bridge_connected"] is True
            assert len(st["radar"]) == 4

            brain = client.get("/api/brain?symbol=XAUUSD").json()
            assert len(brain["bandit_matrix"]) == 5

            # 5. Trigger Full Self-Improvement Cycle (Reflection + Purged WFO + DL/ML + Monte Carlo)
            evo = client.post("/api/evolution/run").json()
            assert evo["status"] == "completed"
            assert "challenger" in evo and "reflection" in evo

            # 6. Hard-clamped risk config update
            r_risk = client.put("/api/config/risk", json={"risk_per_trade_pct": 99.0}).json()
            assert r_risk["risk"]["risk_per_trade_pct"] <= 1.50

            # 7. Two-step Kill-Switch requires exact KILL confirmation
            assert client.post("/api/control/kill", json={"mode": "FLATTEN_KILL", "confirm_code": "NOPE"}).status_code == 400
            assert client.post("/api/control/kill", json={"mode": "FLATTEN_KILL", "confirm_code": "KILL"}).status_code == 200
            assert client.get("/api/state").json()["killed"] is True
    finally:
        asyncio.run(db.close())
