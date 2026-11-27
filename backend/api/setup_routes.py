"""Dashboard Setup page: AI providers, Claude supervisor, MT5 info. All behind X-Control-Token.

- API keys go only to the local `.env` and are only ever returned masked (last 4 characters).
- Supervisor settings go to `config/local.yaml` (git-ignored).
- Live trading cannot be unlocked here: ALLOW_LIVE_TRADING stays a manual `.env` edit.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.core.config import save_local_config
from backend.core.env_file import EnvWriteError, update_env_file
from backend.engine.orchestrator import TradingOrchestrator
from dataclasses import replace

from backend.llm.base import LLMRequest
from backend.llm.models_catalog import ModelListError, list_models
from backend.llm.presets import DEFAULT_CHAIN, PRESETS
from backend.llm.settings import _merged_env, load_chain, parse_chain, resolve_provider


class ProviderUpdate(BaseModel):
    api_key: str | None = Field(default=None, max_length=500)  # None/"" = keep the current key
    clear_key: bool = False
    model: str | None = Field(default=None, max_length=200)  # "" = back to the preset default
    fallback_models: list[str] | None = Field(default=None, max_length=10)
    fast_model: str | None = Field(default=None, max_length=200)
    base_url: str | None = Field(default=None, max_length=300)


class LLMSetupRequest(BaseModel):
    chain: list[str] = Field(default_factory=list, max_length=12)
    providers: dict[str, ProviderUpdate] = Field(default_factory=dict)


class LLMTestRequest(BaseModel):
    provider: str | None = None


class ModelListRequest(BaseModel):
    provider: str
    api_key: str | None = Field(default=None, max_length=500)  # optional: try a key before saving it
    base_url: str | None = Field(default=None, max_length=300)


class SupervisorSetupRequest(BaseModel):
    enabled: bool | None = None
    timeout_s: float | None = Field(default=None, ge=1.0, le=60.0)
    use_fast_model: bool | None = None
    effort: Literal["low", "medium", "high"] | None = None
    on_unavailable: Literal["block", "allow"] | None = None
    max_calls_per_hour: int | None = Field(default=None, ge=1, le=3600)
    daily_budget_usd: float | None = Field(default=None, ge=0.0, le=500.0)
    max_price_drift_atr: float | None = Field(default=None, ge=0.0, le=2.0)
    review_in_shadow: bool | None = None


class ExitSetupRequest(BaseModel):
    be_at_r: float | None = Field(default=None, ge=0.1, le=3.0)
    trail_start_r: float | None = Field(default=None, ge=0.2, le=5.0)
    trail_dist_r: float | None = Field(default=None, ge=0.05, le=3.0)
    hit_min_net_profit: float | None = Field(default=None, ge=0.0, le=100.0)
    adaptive: bool | None = None


class CostSetupRequest(BaseModel):
    # null = automatic (learned from closed MT5 trades, symbol default until then)
    commission_per_lot: float | None = Field(default=None, ge=0.0, le=200.0)
    slippage_points: float | None = Field(default=None, ge=0.0, le=500.0)


def _mask(key: str) -> str:
    return f"...{key[-4:]}" if len(key) >= 8 else ("(set)" if key else "")


def register_setup_routes(api: APIRouter, orch: TradingOrchestrator, env_path: str | Path = ".env") -> None:
    env_path = Path(env_path)

    def llm_view() -> dict[str, Any]:
        env = _merged_env(env_path)
        chain = parse_chain(env)
        live_stats = {c["provider"]: c["stats"] for c in orch.llm.status()["chain"]}
        providers = []
        for name, preset in PRESETS.items():
            rp = resolve_provider(name, env)
            assert rp is not None
            up = name.upper()
            providers.append({
                "name": name,
                "label": preset.label,
                "note": preset.note,
                "key_url": preset.key_url,
                "needs_key": preset.needs_key,
                "in_chain": name in chain,
                "order": chain.index(name) if name in chain else None,
                "configured": rp.configured,
                "api_key_masked": _mask(rp.api_key),
                "model": rp.model,
                "fast_model": rp.fast_model,
                "base_url": rp.base_url,
                "fallback_models": list(rp.fallback_models),
                "default_model": preset.default_model,
                "default_fast_model": preset.fast_model or preset.default_model,
                "default_base_url": preset.base_url,
                "stats": live_stats.get(name, {}),
                # Values set as real environment variables win over .env, so the UI can't change them.
                "locked_by_environment": [
                    f for f in ("API_KEY", "MODEL", "FAST_MODEL", "BASE_URL", "FALLBACK_MODELS") if f"{up}_{f}" in os.environ
                ],
            })
        return {
            "chain": chain,
            "default_chain": list(DEFAULT_CHAIN),
            "chain_locked_by_environment": "LLM_PROVIDER_CHAIN" in os.environ,
            "providers": providers,
            "active": orch.llm_brief(),
        }

    def supervisor_view() -> dict[str, Any]:
        return {**orch.config.supervisor.model_dump(), "status": orch.supervisor.status()}

    def mt5_view() -> dict[str, Any]:
        return {
            "bridge_connected": bool(orch.last_bridge_sync_ms) and orch.get_live_state_snapshot()["bridge_connected"],
            "bridge_url": "http://127.0.0.1:8765/api/bridge/sync",
            "bridge_token_masked": _mask(orch.env.bridge_auth_token),
            "symbols": list(orch.config.instruments.symbols),
            "account_is_demo": orch.account.is_demo,
            "account_currency": orch.account.currency,
        }

    def costs_view() -> dict[str, Any]:
        p = orch.config.paper
        return {
            "manual": {"commission_per_lot": p.commission_per_lot, "slippage_points": p.slippage_points},
            "symbols": [orch.paper_costs(s) for s in orch.config.instruments.symbols],
        }

    def safety_view() -> dict[str, Any]:
        return {
            "trading_mode": orch.config.app.trading_mode.value,
            "live_trading_enabled": orch.env.allow_live_trading,
            "entry_block_reason": orch.entry_block_reason(),
            "how_to_enable_live": "Set ALLOW_LIVE_TRADING=true in .env by hand and restart. It cannot be changed here.",
        }

    @api.get("/setup")
    async def get_setup() -> dict[str, Any]:
        return {"llm": llm_view(), "supervisor": supervisor_view(), "mt5": mt5_view(), "safety": safety_view(), "costs": costs_view()}

    @api.put("/setup/costs")
    async def put_costs(req: CostSetupRequest) -> dict[str, Any]:
        orch.config.paper = orch.config.paper.model_copy(
            update={"commission_per_lot": req.commission_per_lot, "slippage_points": req.slippage_points}
        )
        save_local_config("paper", orch.config.paper.model_dump())
        return costs_view()

    @api.put("/setup/llm")
    async def put_llm(req: LLMSetupRequest) -> dict[str, Any]:
        unknown = [n for n in [*req.chain, *req.providers] if n not in PRESETS]
        if unknown:
            raise HTTPException(status_code=400, detail=f"Unknown provider(s): {', '.join(unknown)}")
        chain = list(dict.fromkeys(req.chain))
        updates: dict[str, str | None] = {"LLM_PROVIDER_CHAIN": ",".join(chain) if chain else "off"}
        for name, pu in req.providers.items():
            up = name.upper()
            if pu.clear_key:
                updates[f"{up}_API_KEY"] = None
            elif pu.api_key and pu.api_key.strip():
                updates[f"{up}_API_KEY"] = pu.api_key.strip()
            for field, suffix in (("model", "MODEL"), ("fast_model", "FAST_MODEL"), ("base_url", "BASE_URL")):
                val = getattr(pu, field)
                if val is not None:
                    updates[f"{up}_{suffix}"] = val.strip() or None  # empty = back to preset default
            if pu.fallback_models is not None:
                fb = [m.strip() for m in pu.fallback_models if m.strip()]
                updates[f"{up}_FALLBACK_MODELS"] = ",".join(dict.fromkeys(fb)) or None
        try:
            update_env_file(env_path, updates)
        except EnvWriteError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        orch.llm.reload(load_chain(env_path))
        return {"saved": sorted(k for k in updates if not k.endswith("_API_KEY")) + (
            ["API key(s)"] if any(k.endswith("_API_KEY") for k in updates) else []), "llm": llm_view()}

    @api.post("/setup/llm/models")
    async def models(req: ModelListRequest) -> dict[str, Any]:
        rp = resolve_provider(req.provider, _merged_env(env_path)) if req.provider in PRESETS else None
        if rp is None:
            raise HTTPException(status_code=400, detail="Unknown provider")
        if req.api_key and req.api_key.strip():
            rp = replace(rp, api_key=req.api_key.strip())
        if req.base_url and req.base_url.strip():
            rp = replace(rp, base_url=req.base_url.strip().rstrip("/"))
        if rp.preset.needs_key and not rp.api_key:
            raise HTTPException(status_code=400, detail="Enter an API key first")
        try:
            found = await list_models(rp)
        except ModelListError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from None
        return {"provider": rp.name, "models": found}

    @api.post("/setup/llm/test")
    async def test_llm(req: LLMTestRequest) -> dict[str, Any]:
        targets = [r for r in load_chain(env_path) if r.configured]
        if req.provider:
            rp = resolve_provider(req.provider, _merged_env(env_path)) if req.provider in PRESETS else None
            if rp is None:
                raise HTTPException(status_code=400, detail="Unknown provider")
            targets = [rp]
        results = []
        for rp in targets:
            if not rp.configured:
                results.append({"provider": rp.name, "ok": False, "error": "API key not set"})
                continue
            single = type(orch.llm)(resolved=[rp], factory=orch.llm.factory)
            ping = LLMRequest(
                task="ping",
                system="You are a connectivity check. Answer with the JSON requested.",
                user='Reply with {"ok": true}.',
                schema={"type": "object", "additionalProperties": False, "required": ["ok"], "properties": {"ok": {"type": "boolean"}}},
                timeout_s=20.0,
                fast=True,
                max_tokens=1000,
                effort="low",
            )
            try:
                res, _ = await single.complete(ping)
                results.append({"provider": rp.name, "ok": True, "model": res.model, "latency_ms": round(res.latency_ms)})
            except Exception as exc:  # noqa: BLE001
                results.append({"provider": rp.name, "ok": False, "error": str(exc)[:300]})
        return {"results": results}

    @api.put("/setup/exits")
    async def put_exits(req: ExitSetupRequest) -> dict[str, Any]:
        ex = orch.config.exits.model_copy(update={
            k: v for k, v in {"be_at_r": req.be_at_r, "trail_start_r": req.trail_start_r, "trail_dist_r": req.trail_dist_r, "adaptive": req.adaptive}.items()
            if v is not None
        })
        hr = orch.config.paper.hit_run.model_copy(update={
            k: v for k, v in {"min_net_profit": req.hit_min_net_profit}.items()
            if v is not None
        })
        orch.config.exits = ex
        orch.config.paper = orch.config.paper.model_copy(update={"hit_run": hr})
        save_local_config("exits", ex.model_dump())
        save_local_config("paper", orch.config.paper.model_dump())
        return {"exits": ex.model_dump(), "hit_run": hr.model_dump()}

    @api.put("/setup/supervisor")
    async def put_supervisor(req: SupervisorSetupRequest) -> dict[str, Any]:
        changes = req.model_dump(exclude_none=True)
        new = orch.config.supervisor.model_copy(update=changes)
        orch.config.supervisor = new
        orch.apply_supervisor_config()
        save_local_config("supervisor", new.model_dump())
        return {"supervisor": supervisor_view()}
