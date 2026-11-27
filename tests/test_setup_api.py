"""Dashboard Setup page API: token-protected, keys masked, allow-listed .env writes, fallbacks."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from backend.api.routes import create_app
from backend.core.config import EnvSettings, load_trading_config
from backend.core.env_file import EnvWriteError, update_env_file
from backend.engine.orchestrator import TradingOrchestrator
from backend.llm.base import LLMResult
from backend.llm.models_catalog import list_models
from backend.llm.router import LLMRouter
from backend.llm.settings import load_chain, resolve_provider
from backend.storage.database import Database

CTRL = "c" * 32
SECRET = "sk-ant-api03-SECRETSECRETSECRET-1234"


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    for k in list(__import__("os").environ):
        if k.endswith(("_API_KEY", "_MODEL", "_FALLBACK_MODELS", "_BASE_URL")) or k == "LLM_PROVIDER_CHAIN":
            monkeypatch.delenv(k, raising=False)
    monkeypatch.chdir(tmp_path)
    env_path = tmp_path / ".env"
    env_path.write_text(f"# my comment\nBRIDGE_AUTH_TOKEN={'b' * 32}\nCONTROL_API_TOKEN={CTRL}\nALLOW_LIVE_TRADING=false\n")
    cfg = load_trading_config(Path(__file__).resolve().parent.parent / "config" / "default.yaml", local_path=None)
    cfg.app.database_path = str(tmp_path / "s.db")
    cfg.app.snapshots_dir = str(tmp_path / "snaps")
    cfg.app.kill_switch_path = str(tmp_path / "kill.json")
    env = EnvSettings(bridge_auth_token="b" * 32, control_api_token=CTRL, _env_file=None)
    orch = TradingOrchestrator(cfg, env, Database(cfg.app.database_path), llm_router=LLMRouter(resolved=[]))
    app = create_app(orch)
    with TestClient(app, headers={"X-Control-Token": CTRL}) as c:
        yield c, env_path, orch


def test_setup_requires_control_token(client):
    c, _, _ = client
    assert c.get("/api/setup", headers={"X-Control-Token": "wrong" * 6}).status_code == 401


def test_save_key_is_masked_and_written_only_to_env(client):
    c, env_path, orch = client
    r = c.put("/api/setup/llm", json={"chain": ["anthropic", "gemini"], "providers": {
        "anthropic": {"api_key": SECRET, "model": "claude-opus-5-5", "fallback_models": ["claude-sonnet-5-5"]}}})
    assert r.status_code == 200, r.text
    assert SECRET not in r.text
    text = env_path.read_text()
    assert f"ANTHROPIC_API_KEY={SECRET}" in text
    assert "LLM_PROVIDER_CHAIN=anthropic,gemini" in text
    assert "ANTHROPIC_FALLBACK_MODELS=claude-sonnet-5-5" in text
    assert "# my comment" in text and "ALLOW_LIVE_TRADING=false" in text
    got = c.get("/api/setup").json()
    assert SECRET not in str(got)
    anth = next(p for p in got["llm"]["providers"] if p["name"] == "anthropic")
    assert anth["api_key_masked"] == "..." + SECRET[-4:]
    assert anth["fallback_models"] == ["claude-sonnet-5-5"]
    assert got["llm"]["chain"] == ["anthropic", "gemini"]
    # router reloaded in place
    assert [r.name for r in orch.llm.active] == ["anthropic"]


def test_blank_key_keeps_existing_and_clear_removes(client):
    c, env_path, _ = client
    c.put("/api/setup/llm", json={"chain": ["anthropic"], "providers": {"anthropic": {"api_key": SECRET}}})
    c.put("/api/setup/llm", json={"chain": ["anthropic"], "providers": {"anthropic": {"api_key": ""}}})
    assert SECRET in env_path.read_text()
    c.put("/api/setup/llm", json={"chain": ["anthropic"], "providers": {"anthropic": {"clear_key": True}}})
    assert "ANTHROPIC_API_KEY" not in env_path.read_text()


def test_live_trading_and_tokens_cannot_be_written(tmp_path: Path):
    p = tmp_path / ".env"
    p.write_text("ALLOW_LIVE_TRADING=false\n")
    for key in ("ALLOW_LIVE_TRADING", "CONTROL_API_TOKEN", "BRIDGE_AUTH_TOKEN", "SOMETHING_ELSE"):
        with pytest.raises(EnvWriteError):
            update_env_file(p, {key: "true"})
    with pytest.raises(EnvWriteError):
        update_env_file(p, {"ANTHROPIC_MODEL": "x\nALLOW_LIVE_TRADING=true"})
    assert p.read_text() == "ALLOW_LIVE_TRADING=false\n"


def test_unknown_provider_rejected(client):
    c, _, _ = client
    assert c.put("/api/setup/llm", json={"chain": ["evil"]}).status_code == 400


def test_supervisor_settings_saved_to_local_yaml(client, tmp_path: Path):
    c, _, orch = client
    r = c.put("/api/setup/supervisor", json={"timeout_s": 4, "daily_budget_usd": 2.5, "on_unavailable": "allow"})
    assert r.status_code == 200, r.text
    assert orch.supervisor.settings.timeout_s == 4
    assert orch.supervisor.settings.on_unavailable == "allow"
    local = (tmp_path / "config" / "local.yaml").read_text()
    assert "daily_budget_usd: 2.5" in local
    assert c.put("/api/setup/supervisor", json={"timeout_s": 999}).status_code == 422


def test_local_yaml_cannot_set_trading_mode(tmp_path: Path):
    local = tmp_path / "local.yaml"
    local.write_text("app:\n  trading_mode: MT5_LIVE\nsupervisor:\n  timeout_s: 3\n")
    cfg = load_trading_config(Path(__file__).resolve().parent.parent / "config" / "default.yaml", local_path=local)
    assert cfg.app.trading_mode.value == "SHADOW_ONLY"
    assert cfg.supervisor.timeout_s == 3


def _rp(name: str, **env: str):
    return resolve_provider(name, {k: v for k, v in env.items()})


async def test_list_models_openai_compat_and_anthropic():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/v1/models"):
            assert request.headers["x-api-key"] == "k" * 20
            return httpx.Response(200, json={"data": [{"id": "claude-haiku-5-5", "display_name": "Haiku"}, {"id": "claude-opus-5-5"}]})
        assert request.headers["authorization"] == "Bearer " + "g" * 20
        return httpx.Response(200, json={"data": [{"id": "models/gemini-2.5-flash"}, {"id": "gemini-2.5-pro"}]})

    t = httpx.MockTransport(handler)
    a = await list_models(_rp("anthropic", ANTHROPIC_API_KEY="k" * 20), transport=t)
    assert [m["id"] for m in a] == ["claude-haiku-5-5", "claude-opus-5-5"]
    g = await list_models(_rp("gemini", GEMINI_API_KEY="g" * 20), transport=t)
    assert [m["id"] for m in g] == ["gemini-2.5-flash", "gemini-2.5-pro"]


async def test_list_models_bad_key_message_has_no_key():
    from backend.llm.models_catalog import ModelListError

    t = httpx.MockTransport(lambda r: httpx.Response(401, json={"error": "bad key " + "z" * 20}))
    with pytest.raises(ModelListError) as ei:
        await list_models(_rp("openai", OPENAI_API_KEY="z" * 20), transport=t)
    assert "z" * 20 not in str(ei.value)


async def test_router_tries_fallback_model_before_next_provider():
    from backend.llm.base import LLMRequest

    calls: list[str] = []

    class Fake:
        def __init__(self, rp):
            self.rp = rp

        async def complete(self, req):
            calls.append(f"{self.rp.name}:{self.rp.model}")
            if self.rp.model == "main-model":
                raise RuntimeError("overloaded")
            return LLMResult(data={"ok": True}, provider=self.rp.name, model=self.rp.model, latency_ms=1.0)

    resolved = load_chain(None, overrides={
        "LLM_PROVIDER_CHAIN": "anthropic,gemini",
        "ANTHROPIC_API_KEY": "k" * 20, "ANTHROPIC_MODEL": "main-model", "ANTHROPIC_FALLBACK_MODELS": "backup-model",
        "GEMINI_API_KEY": "g" * 20,
    })
    router = LLMRouter(resolved=resolved, factory=Fake)
    res, _ = await router.complete(LLMRequest(task="t", system="s", user="u", schema={}))
    assert res.model == "backup-model"
    assert calls == ["anthropic:main-model", "anthropic:backup-model"]


def test_paper_compare_and_exit_settings(client):
    c, _, orch = client
    c.portal.call(orch.db.connect)  # the fixture does not start the engine
    r = c.get("/api/paper/compare").json()
    assert [row["book"] for row in r["rows"]] == ["normal", "hitrun", "mt5"]
    assert r["learning"]["adaptive"] is True and r["learning"]["min_samples"] == 20
    r = c.put("/api/setup/exits", json={"adaptive": False, "hit_min_net_profit": 2.5, "be_at_r": 0.4}).json()
    assert r["exits"]["adaptive"] is False and r["exits"]["be_at_r"] == 0.4 and r["hit_run"]["min_net_profit"] == 2.5
    assert c.get("/api/paper/compare").json()["learning"]["adaptive"] is False
    assert c.put("/api/setup/exits", json={"be_at_r": 9}).status_code == 422
