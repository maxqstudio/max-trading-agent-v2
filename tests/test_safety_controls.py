"""Safety rails: paper-by-default, live gate, control auth, CORS, persistent kill switch."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.api.routes import create_app
from backend.bridge.simulator import SimulatedMT5Bridge
from backend.core.config import AppConfig, EnvSettings, TradingConfig, load_trading_config
from backend.core.models import OrderAction, OrderCommand, Side, TradingMode
from backend.core.safety import KillSwitchStore, entry_block_reason, is_strong_token
from backend.engine.orchestrator import TradingOrchestrator
from backend.llm.router import LLMRouter
from backend.storage.database import Database

BRIDGE = "test_bridge_secret_token_abc"
CONTROL = "test_control_secret_token_xyz"


def _env(**kw) -> EnvSettings:
    base = dict(bridge_auth_token=BRIDGE, control_api_token=CONTROL, llm_provider="deterministic", _env_file=None)
    base.update(kw)
    return EnvSettings(**base)


_OPEN_DBS: list[Database] = []


@pytest.fixture(autouse=True)
def _close_dbs():
    yield
    while _OPEN_DBS:
        asyncio.run(_OPEN_DBS.pop().close())


def _orch(tmp_path: Path, mode: TradingMode | None = None, **env_kw) -> TradingOrchestrator:
    cfg = load_trading_config()
    cfg.app.database_path = str(tmp_path / "t.db")
    cfg.app.snapshots_dir = str(tmp_path / "snaps")
    cfg.app.kill_switch_path = str(tmp_path / "kill.json")
    if mode is not None:
        cfg.app.trading_mode = mode
    db = Database(cfg.app.database_path)
    asyncio.run(db.connect())
    _OPEN_DBS.append(db)
    return TradingOrchestrator(cfg, _env(**env_kw), db, llm_router=LLMRouter(resolved=[]))


# ---------- pure rules ----------

def test_defaults_are_paper_and_live_disabled() -> None:
    assert AppConfig().trading_mode == TradingMode.SHADOW_ONLY
    assert load_trading_config().app.trading_mode == TradingMode.SHADOW_ONLY
    env = EnvSettings(_env_file=None)
    assert env.allow_live_trading is False
    assert env.bridge_auth_token == "" and env.control_api_token == ""


@pytest.mark.parametrize(
    "token,ok",
    [
        ("", False),
        ("short", False),
        ("mta2_dev_change_me_to_random_secret_32chars", False),
        ("x" * 23, False),
        ("mta2_" + "a1" * 20, True),
    ],
)
def test_token_strength(token: str, ok: bool) -> None:
    assert is_strong_token(token) is ok


@pytest.mark.parametrize(
    "mode,is_demo,allow,killed,expected",
    [
        (TradingMode.MT5_DEMO, True, False, False, None),
        (TradingMode.SIMULATOR, True, False, False, None),
        (TradingMode.SHADOW_ONLY, True, True, False, "SHADOW_ONLY_PAPER_MODE"),
        (TradingMode.MT5_DEMO, True, True, True, "KILL_SWITCH_ACTIVE"),
        (TradingMode.MT5_LIVE, False, False, False, "LIVE_TRADING_NOT_ENABLED"),
        (TradingMode.MT5_DEMO, False, False, False, "REAL_ACCOUNT_NOT_AUTHORIZED"),
        (TradingMode.MT5_DEMO, False, True, False, "REAL_ACCOUNT_NOT_AUTHORIZED"),
        (TradingMode.SIMULATOR, False, True, False, "REAL_ACCOUNT_NOT_AUTHORIZED"),
        (TradingMode.MT5_LIVE, False, True, False, None),
    ],
)
def test_entry_block_matrix(mode, is_demo, allow, killed, expected) -> None:
    assert entry_block_reason(mode=mode, account_is_demo=is_demo, allow_live_trading=allow, killed=killed) == expected


def test_kill_store_persists_and_fails_closed(tmp_path: Path) -> None:
    store = KillSwitchStore(tmp_path / "k.json")
    assert store.load().engaged is False
    store.engage("TEST")
    st = KillSwitchStore(tmp_path / "k.json").load()
    assert st.engaged and st.reason == "TEST"
    store.release()
    assert store.load().engaged is False
    (tmp_path / "k.json").write_text("{not json", encoding="utf-8")
    assert store.load().engaged is True


# ---------- engine ----------

def test_kill_switch_survives_restart(tmp_path: Path) -> None:
    orch = _orch(tmp_path, TradingMode.MT5_DEMO)
    asyncio.run(orch.trigger_flatten_all("TEST_KILL"))
    reborn = _orch(tmp_path, TradingMode.MT5_DEMO)
    assert reborn.risk.killed and reborn.risk.paused
    assert reborn.entry_block_reason() == "KILL_SWITCH_ACTIVE"
    reborn.release_kill_switch()
    assert not reborn.risk.killed and reborn.risk.paused  # release never auto-resumes


def test_real_account_entries_withheld_but_exits_pass(tmp_path: Path) -> None:
    orch = _orch(tmp_path, TradingMode.MT5_DEMO)
    sim = SimulatedMT5Bridge(symbols=orch.config.instruments.symbols, seed=7)
    req = sim.step_and_build_sync(close_bar_every=100)
    req.account.is_demo = False  # EA reports a REAL account
    open_cmd = OrderCommand(action=OrderAction.OPEN, symbol="XAUUSD", side=Side.BUY, volume=0.01, sl=1.0, tp=2.0,
                            expires_at_ms=10**15)
    close_cmd = OrderCommand(action=OrderAction.CLOSE, symbol="XAUUSD", ticket=123, expires_at_ms=10**15)
    orch.order_sm.enqueue(open_cmd)
    orch.order_sm.enqueue(close_cmd)
    resp = asyncio.run(orch.handle_bridge_sync(req))
    actions = [c.action for c in resp.commands]
    assert OrderAction.OPEN not in actions
    assert OrderAction.CLOSE in actions
    assert any(e.tag == "SAFETY RAIL" for e in orch.recent_decisions)


def test_demo_signal_refused_on_real_account(tmp_path: Path) -> None:
    orch = _orch(tmp_path, TradingMode.MT5_DEMO)
    orch.account.is_demo = False
    res = asyncio.run(orch.force_demo_signal("XAUUSD"))
    assert res["ok"] is False


# ---------- API ----------

def _client(orch: TradingOrchestrator, token: str | None = CONTROL) -> TestClient:
    headers = {"X-Control-Token": token} if token else {}
    return TestClient(create_app(orch), headers=headers)


def test_control_endpoints_require_token(tmp_path: Path) -> None:
    orch = _orch(tmp_path)
    anon = _client(orch, token=None)
    assert anon.get("/api/health").status_code == 200
    for method, url, body in [
        ("get", "/api/state", None),
        ("post", "/api/control/mode", {"trading_mode": "MT5_DEMO"}),
        ("put", "/api/config/risk", {"risk_per_trade_pct": 1.0}),
        ("post", "/api/control/pause", None),
        ("post", "/api/evolution/approve", None),
    ]:
        r = getattr(anon, method)(url, json=body) if body is not None else getattr(anon, method)(url)
        assert r.status_code == 401, url
    assert _client(orch, token="wrong_token_wrong_token_wrong").get("/api/state").status_code == 401
    assert _client(orch).get("/api/state").status_code == 200


def test_control_refused_when_server_token_missing(tmp_path: Path) -> None:
    orch = _orch(tmp_path, control_api_token="")
    assert _client(orch, token="anything_anything_anything").get("/api/state").status_code == 503


def test_placeholder_bridge_token_rejects_ea(tmp_path: Path) -> None:
    placeholder = "mta2_dev_change_me_to_random_secret_32chars"
    orch = _orch(tmp_path, bridge_auth_token=placeholder)
    sim = SimulatedMT5Bridge(symbols=orch.config.instruments.symbols, seed=3)
    r = _client(orch).post(
        "/api/bridge/sync", headers={"X-Bridge-Token": placeholder}, json=sim.step_and_build_sync().model_dump(mode="json")
    )
    assert r.status_code == 401


def test_live_mode_needs_env_flag_and_confirmation(tmp_path: Path) -> None:
    c = _client(_orch(tmp_path))
    assert c.post("/api/control/mode", json={"trading_mode": "MT5_LIVE", "confirm_code": "LIVE"}).status_code == 403
    c2 = _client(_orch(tmp_path / "b", allow_live_trading=True))
    assert c2.post("/api/control/mode", json={"trading_mode": "MT5_LIVE"}).status_code == 400
    r = c2.post("/api/control/mode", json={"trading_mode": "MT5_LIVE", "confirm_code": "LIVE"})
    assert r.status_code == 200 and r.json()["trading_mode"] == "MT5_LIVE"


def test_pause_cannot_clear_kill(tmp_path: Path) -> None:
    orch = _orch(tmp_path, TradingMode.MT5_DEMO)
    c = _client(orch)
    assert c.post("/api/control/kill", json={"mode": "FLATTEN_KILL", "confirm_code": "KILL"}).status_code == 200
    assert c.post("/api/control/pause").status_code == 409
    assert orch.risk.killed
    assert c.post("/api/control/kill/release", json={"confirm_code": "nope"}).status_code == 400
    assert c.post("/api/control/kill/release", json={"confirm_code": "RELEASE"}).status_code == 200
    assert not orch.risk.killed and orch.risk.paused


def test_cors_rejects_foreign_origin(tmp_path: Path) -> None:
    c = _client(_orch(tmp_path))
    pre = {"Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "x-control-token,content-type"}
    evil = c.options("/api/control/mode", headers={"Origin": "https://evil.example", **pre})
    assert "access-control-allow-origin" not in {k.lower() for k in evil.headers}
    good = c.options("/api/control/mode", headers={"Origin": "http://127.0.0.1:8765", **pre})
    assert good.headers.get("access-control-allow-origin") == "http://127.0.0.1:8765"


def test_websocket_requires_token(tmp_path: Path) -> None:
    from starlette.websockets import WebSocketDisconnect

    c = _client(_orch(tmp_path))
    with pytest.raises(WebSocketDisconnect):
        with c.websocket_connect("/ws/live") as ws:
            ws.receive_json()
    with c.websocket_connect(f"/ws/live?token={CONTROL}") as ws:
        assert "trading_mode" in ws.receive_json()
