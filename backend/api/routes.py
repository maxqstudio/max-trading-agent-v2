from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.core.models import BridgeSyncRequest, BridgeSyncResponse, ParameterPatch, TradingMode
from backend.core.security import apply_safe_patches, verify_bridge_token
from backend.engine.orchestrator import TradingOrchestrator


class RiskUpdateRequest(BaseModel):
    risk_per_trade_pct: float | None = None
    max_daily_drawdown_pct: float | None = None
    max_spread_atr_ratio: float | None = None
    max_concurrent_positions: int | None = None
    max_usd_correlated_positions: int | None = None
    consecutive_loss_cooldown_min: int | None = None
    ml_min_probability: float | None = None
    rollover_blackout_enabled: bool | None = None
    news_blackout_enabled: bool | None = None
    friday_flatten_enabled: bool | None = None
    auto_promote_challenger: bool | None = None


class ModeUpdateRequest(BaseModel):
    trading_mode: TradingMode


class RollbackRequest(BaseModel):
    version_id: str


class KillRequest(BaseModel):
    confirm_code: str = ""
    mode: str = "SOFT_PAUSE"  # SOFT_PAUSE | FLATTEN_KILL


def create_app(orchestrator: TradingOrchestrator) -> FastAPI:
    app = FastAPI(title="Max Trading Agent v2 API", version="2.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    api = APIRouter(prefix="/api")

    def require_bridge_token(x_bridge_token: str | None = Header(default=None)) -> None:
        if not verify_bridge_token(x_bridge_token, orchestrator.env.bridge_auth_token):
            raise HTTPException(status_code=401, detail="Invalid or missing X-Bridge-Token")

    @api.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": orchestrator.gate.active_snapshot.version_id,
            "mode": orchestrator.config.app.trading_mode.value,
        }

    # --- Authenticated MT5 Zero-DLL Bridge ---
    @api.post("/bridge/sync", response_model=BridgeSyncResponse, dependencies=[Depends(require_bridge_token)])
    async def bridge_sync(payload: BridgeSyncRequest) -> BridgeSyncResponse:
        return await orchestrator.handle_bridge_sync(payload)

    # --- Live Cockpit State ---
    @api.get("/state")
    async def get_state() -> dict[str, Any]:
        snap = orchestrator.get_live_state_snapshot()
        snap["equity_curve"] = await orchestrator.db.list_equity_curve(limit=120)
        return snap

    # --- AI Brain Deep Diagnostics ---
    @api.get("/brain")
    async def get_brain(symbol: str = "XAUUSD") -> dict[str, Any]:
        sym = symbol if symbol in orchestrator.specs else orchestrator.config.instruments.symbols[0]
        return {
            "symbol": sym,
            "bandit_matrix": orchestrator.bandit.get_matrix(sym),
            "dl_prediction": orchestrator.dl_preds.get(sym).model_dump() if sym in orchestrator.dl_preds else {},
            "dl_trained": orchestrator.dl_net.trained,
            "dl_val_acc": round(orchestrator.dl_net.last_val_accuracy, 4),
            "dl_val_loss_history": orchestrator.dl_net.val_loss_history[-25:],
            "ml_diagnostics": orchestrator.ml_labeler.diagnostics(),
            "ml_min_probability": orchestrator.config.ai.ml_min_probability,
            "rl_exit_grid": orchestrator.rl_exit.get_policy_grid(),
            "rl_mfe_capture_pct": orchestrator.rl_exit.avg_mfe_capture_pct,
            "bars": [b.model_dump() for b in list(orchestrator.bars.get(sym, ()))[-32:]],
        }

    # --- Evolution & Shadow Arena ---
    @api.get("/evolution")
    async def get_evolution() -> dict[str, Any]:
        reflections = await orchestrator.db.list_reflections(limit=15)
        equity_curve = await orchestrator.db.list_equity_curve(limit=120)
        return {
            "running": orchestrator.evolution_running,
            "auto_promote": orchestrator.config.app.auto_promote_challenger,
            "active_snapshot": orchestrator.gate.active_snapshot.model_dump(mode="json"),
            "active_challenger": (
                orchestrator.gate.active_challenger.model_dump(mode="json")
                if orchestrator.gate.active_challenger
                else None
            ),
            "snapshots": [s.model_dump(mode="json") for s in orchestrator.gate.snapshots],
            "challengers": [c.model_dump(mode="json") for c in orchestrator.gate.challenger_history[:12]],
            "reflections": [r.model_dump(mode="json") for r in reflections],
            "equity_curve": equity_curve,
        }

    @api.post("/evolution/run")
    async def run_evolution() -> dict[str, Any]:
        return await orchestrator.run_evolution_cycle()

    @api.post("/evolution/approve")
    async def approve_challenger() -> dict[str, Any]:
        snap = orchestrator.gate.promote_active_challenger(
            note="Approved from Quant Cockpit Dashboard",
            current_equity=orchestrator.account.equity,
        )
        if snap is None:
            raise HTTPException(status_code=400, detail="No active Challenger to promote")
        orchestrator.apply_active_config(orchestrator.gate.active_config)
        for s in orchestrator.gate.snapshots:
            await orchestrator.db.save_snapshot(s)
        return {"promoted": snap.model_dump(mode="json")}

    @api.post("/evolution/reject")
    async def reject_challenger() -> dict[str, Any]:
        ok = orchestrator.gate.reject_active_challenger("Rejected from Quant Cockpit Dashboard")
        if not ok:
            raise HTTPException(status_code=400, detail="No active Challenger to reject")
        return {"rejected": True}

    @api.post("/evolution/rollback")
    async def rollback_champion(req: RollbackRequest) -> dict[str, Any]:
        restored = orchestrator.gate.rollback_to(req.version_id, auto=False)
        if restored is None:
            raise HTTPException(status_code=404, detail=f"Snapshot {req.version_id} not found")
        orchestrator.apply_active_config(orchestrator.gate.active_config)
        for s in orchestrator.gate.snapshots:
            await orchestrator.db.save_snapshot(s)
        return {"restored": restored.model_dump(mode="json")}

    # --- Trade Forensics & Analytics ---
    @api.get("/journal")
    async def get_journal(
        limit: int = 100,
        mode: str = "ALL",
        symbol: str | None = None,
    ) -> dict[str, Any]:
        is_shadow = True if mode == "SHADOW" else (False if mode == "LIVE" else None)
        sym = None if (not symbol or symbol == "ALL") else symbol
        trades = await orchestrator.db.list_trades(limit=limit, is_shadow=is_shadow, symbol=sym)

        # Hourly UTC expectancy breakdown
        hourly_r: dict[int, list[float]] = {h: [] for h in range(24)}
        for t in trades:
            hr = int((t.open_time % 86400) // 3600)
            hourly_r[hr].append(t.pnl_r)
        session_heatmap = [
            {
                "hour_utc": h,
                "trades": len(vals),
                "total_r": round(sum(vals), 3),
                "is_blackout": h in (22,),
            }
            for h, vals in hourly_r.items()
        ]
        return {
            "trades": [t.model_dump(mode="json") for t in trades],
            "session_heatmap": session_heatmap,
        }

    # --- Risk, Shields & Broker Config ---
    @api.get("/config")
    async def get_config() -> dict[str, Any]:
        return {
            "risk": orchestrator.config.risk.model_dump(),
            "shields": orchestrator.config.shields.model_dump(),
            "ai": orchestrator.config.ai.model_dump(),
            "auto_promote_challenger": orchestrator.config.app.auto_promote_challenger,
            "hard_guard": orchestrator.hard_guard.model_dump(),
            "symbol_specs": [s.model_dump() for s in orchestrator.specs.values()],
            "llm_provider": orchestrator.env.llm_provider,
            "telegram_configured": bool(orchestrator.env.telegram_bot_token and orchestrator.env.telegram_chat_id),
            "bridge_token_masked": orchestrator.env.bridge_auth_token[:8] + "..." + orchestrator.env.bridge_auth_token[-4:],
        }

    @api.put("/config/risk")
    async def update_risk(req: RiskUpdateRequest) -> dict[str, Any]:
        patches: list[ParameterPatch] = []
        mapping = {
            "risk_per_trade_pct": "risk.risk_per_trade_pct",
            "max_daily_drawdown_pct": "risk.max_daily_drawdown_pct",
            "max_spread_atr_ratio": "risk.max_spread_atr_ratio",
            "max_concurrent_positions": "risk.max_concurrent_positions",
            "max_usd_correlated_positions": "risk.max_usd_correlated_positions",
            "consecutive_loss_cooldown_min": "risk.consecutive_loss_cooldown_min",
            "ml_min_probability": "ai.ml_min_probability",
        }
        for field_name, cfg_path in mapping.items():
            val = getattr(req, field_name)
            if val is not None:
                patches.append(ParameterPatch(path=cfg_path, old_value=0.0, new_value=float(val), rationale="Operator update"))

        updated_cfg, applied = apply_safe_patches(orchestrator.config, patches)
        if req.rollover_blackout_enabled is not None:
            updated_cfg.shields.rollover_blackout_enabled = req.rollover_blackout_enabled
        if req.news_blackout_enabled is not None:
            updated_cfg.shields.news_blackout_enabled = req.news_blackout_enabled
        if req.friday_flatten_enabled is not None:
            updated_cfg.shields.friday_flatten_enabled = req.friday_flatten_enabled
        if req.auto_promote_challenger is not None:
            updated_cfg.app.auto_promote_challenger = req.auto_promote_challenger

        orchestrator.apply_active_config(updated_cfg)
        orchestrator.gate.active_config = updated_cfg
        return {"applied_patches": [p.model_dump() for p in applied], "risk": updated_cfg.risk.model_dump()}

    # --- Operator Controls (Mode, Pause, Shield Toggle, Manual Close, Two-Step Kill-Switch) ---
    @api.post("/control/mode")
    async def set_mode(req: ModeUpdateRequest) -> dict[str, Any]:
        orchestrator.config.app.trading_mode = req.trading_mode
        return {"trading_mode": orchestrator.config.app.trading_mode.value}

    @api.post("/control/pause")
    async def toggle_pause() -> dict[str, Any]:
        orchestrator.risk.paused = not orchestrator.risk.paused
        if not orchestrator.risk.paused:
            orchestrator.risk.killed = False
        return {"paused": orchestrator.risk.paused, "killed": orchestrator.risk.killed}

    @api.post("/control/shield-toggle")
    async def toggle_shield_override() -> dict[str, Any]:
        orchestrator.shields.manual_shock_override = not orchestrator.shields.manual_shock_override
        return {"manual_shield_override": orchestrator.shields.manual_shock_override}

    @api.post("/control/kill")
    async def emergency_kill(req: KillRequest) -> dict[str, Any]:
        if req.mode == "SOFT_PAUSE":
            orchestrator.risk.paused = True
            return {"status": "paused"}
        if req.confirm_code.strip().upper() != "KILL":
            raise HTTPException(status_code=400, detail="Must supply confirm_code='KILL' for FLATTEN_KILL")
        await orchestrator.trigger_flatten_all("DASHBOARD_KILL_SWITCH")
        return {"status": "killed"}

    @api.post("/control/close/{ticket}")
    async def manual_close(ticket: int) -> dict[str, Any]:
        ok = await orchestrator.trigger_manual_close(ticket)
        if not ok:
            raise HTTPException(status_code=404, detail="Position ticket not found")
        return {"closing_ticket": ticket}

    @api.post("/demo/signal")
    async def demo_signal(symbol: str = "XAUUSD") -> dict[str, Any]:
        return await orchestrator.force_demo_signal(symbol)

    app.include_router(api)

    # --- WebSocket Real-Time Telemetry Stream ---
    @app.websocket("/ws/live")
    async def ws_live(websocket: WebSocket) -> None:
        await websocket.accept()
        orchestrator.ws_clients.add(websocket)
        try:
            await websocket.send_json(orchestrator.get_live_state_snapshot())
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            orchestrator.ws_clients.discard(websocket)
        except Exception:
            orchestrator.ws_clients.discard(websocket)

    # Serve built frontend bundle on same port if available
    dist_dir = Path("frontend/dist")
    if dist_dir.exists() and (dist_dir / "index.html").exists():
        app.mount("/", StaticFiles(directory=str(dist_dir), html=True), name="frontend")

    return app
