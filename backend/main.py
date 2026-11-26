from __future__ import annotations

import argparse
import asyncio
import contextlib
from contextlib import asynccontextmanager

import uvicorn

from backend.api.routes import create_app
from backend.bridge.simulator import SimulatedMT5Bridge
from backend.core.config import EnvSettings, load_trading_config
from backend.core.models import TradingMode
from backend.engine.orchestrator import TradingOrchestrator
from backend.notifications.telegram_bot import TelegramControlBot
from backend.storage.database import Database


def build_runtime(sim_mode: bool = False, config_path: str = "config/default.yaml"):
    config = load_trading_config(config_path)
    env = EnvSettings()
    if sim_mode:
        config.app.trading_mode = TradingMode.SIMULATOR
    db = Database(config.app.database_path)
    orchestrator = TradingOrchestrator(config=config, env=env, db=db)
    tg = TelegramControlBot(env.telegram_bot_token, env.telegram_chat_id, orchestrator)
    if tg.configured:
        orchestrator.alert_callback = tg.send_message

    sim_bridge = SimulatedMT5Bridge(symbols=config.instruments.symbols) if sim_mode else None

    @asynccontextmanager
    async def lifespan(_app):
        await db.connect()
        await orchestrator.initialize_from_storage()
        tg.start()

        bg_tasks: list[asyncio.Task] = []

        if sim_bridge is not None:
            # Bootstrap historical M1 bars if DB is fresh so DL/ML/WFO work out-of-the-box
            if len(orchestrator.bars[config.instruments.symbols[0]]) < 150:
                boot = sim_bridge.generate_bootstrap_bars(count=260)
                for sym, blist in boot.items():
                    await db.save_bars(blist)
                    orchestrator.bars[sym].extend(blist)

            async def _sim_loop() -> None:
                while True:
                    req = sim_bridge.step_and_build_sync(close_bar_every=6)
                    resp = await orchestrator.handle_bridge_sync(req)
                    if resp.commands:
                        sim_bridge.execute_commands(resp.commands)
                    await asyncio.sleep(0.45)

            bg_tasks.append(asyncio.create_task(_sim_loop()))

        # Scheduled self-improvement loop
        async def _evolution_scheduler() -> None:
            interval_sec = max(300, config.evolution.schedule_interval_minutes * 60)
            while True:
                await asyncio.sleep(interval_sec)
                with contextlib.suppress(Exception):
                    await orchestrator.run_evolution_cycle()

        bg_tasks.append(asyncio.create_task(_evolution_scheduler()))

        try:
            yield
        finally:
            for t in bg_tasks:
                t.cancel()
            await tg.stop()
            await db.close()

    app = create_app(orchestrator)
    app.router.lifespan_context = lifespan
    return app, orchestrator, env


def main() -> None:
    parser = argparse.ArgumentParser(description="Max Trading Agent v2 — Autonomous Self-Improving MT5 Scalper")
    parser.add_argument("--sim", action="store_true", help="Run with built-in multi-regime MT5 Simulator")
    parser.add_argument("--config", default="config/default.yaml", help="Path to YAML config file")
    parser.add_argument("--host", default=None, help="Bind host override")
    parser.add_argument("--port", type=int, default=None, help="Bind port override")
    args = parser.parse_args()

    app, _, env = build_runtime(sim_mode=args.sim, config_path=args.config)
    host = args.host or env.api_host
    port = args.port or env.api_port
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
