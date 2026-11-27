"""Multi-provider LLM layer: presets, chain, Claude provider, OpenAI-compatible provider, reflection safety.

No real API is called: Claude is a fake SDK client, the others use httpx.MockTransport.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from backend.core.config import EnvSettings, load_trading_config
from backend.core.models import MarketRegime, Side, TradeRecord
from backend.core.security import restrict_llm_patch
from backend.core.models import ParameterPatch
from backend.evolution.llm_reflector import REFLECTION_SYSTEM, LLMReflectionAnalyst, _REFLECTION_SCHEMA
from backend.llm.anthropic_provider import AnthropicProvider
from backend.llm.base import LLMError, LLMRequest, LLMResult, ToolSpec
from backend.llm.openai_compat_provider import OpenAICompatProvider
from backend.llm.router import LLMRouter
from backend.llm.settings import load_chain, parse_chain, resolve_provider

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["ok"], "properties": {"ok": {"type": "boolean"}}}


def _req(**kw) -> LLMRequest:
    base = dict(task="ping", system="sys", user="u", schema=SCHEMA, timeout_s=5.0)
    base.update(kw)
    return LLMRequest(**base)


# ---------- settings / presets ----------

def test_default_chain_is_claude_gemini_openai() -> None:
    assert parse_chain({}) == ["anthropic", "gemini", "openai"]


@pytest.mark.parametrize(
    "env,expected",
    [
        ({"LLM_PROVIDER_CHAIN": "off"}, []),
        ({"LLM_PROVIDER_CHAIN": "groq, ollama ,bogus,groq"}, ["groq", "ollama"]),
        ({"LLM_PROVIDER": "deterministic"}, []),
        ({"LLM_PROVIDER": "gemini"}, ["gemini"]),
        ({"LLM_PROVIDER": "deterministic", "LLM_PROVIDER_CHAIN": "anthropic"}, ["anthropic"]),
    ],
)
def test_chain_parsing(env: dict[str, str], expected: list[str]) -> None:
    assert parse_chain(env) == expected


def test_overrides_and_configured_rules() -> None:
    env = {"GEMINI_API_KEY": "g-key-123456", "GEMINI_MODEL": "my-gemini-model", "OLLAMA_MODEL": "qwen3"}
    g = resolve_provider("gemini", env)
    assert g and g.configured and g.model == "my-gemini-model" and g.masked_key() == "...3456"
    assert resolve_provider("anthropic", env).configured is False  # no key
    assert resolve_provider("ollama", env).configured is True  # local, no key needed
    assert resolve_provider("custom", env).configured is False  # needs base url + model
    chain = load_chain(None, overrides={"LLM_PROVIDER_CHAIN": "anthropic,gemini", **env})
    assert [r.name for r in chain] == ["anthropic", "gemini"]
    assert [r.name for r in LLMRouter(resolved=chain).active] == ["gemini"]


def test_status_never_leaks_keys() -> None:
    secret = "sk-ant-super-secret-value-9999"
    chain = load_chain(None, overrides={"LLM_PROVIDER_CHAIN": "anthropic", "ANTHROPIC_API_KEY": secret})
    dumped = json.dumps(LLMRouter(resolved=chain).status())
    assert secret not in dumped and "...9999" in dumped


# ---------- router ----------

class _FakeProvider:
    def __init__(self, name: str, result: dict[str, Any] | Exception) -> None:
        self.name, self.result, self.calls = name, result, 0

    async def complete(self, req: LLMRequest) -> LLMResult:
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return LLMResult(data=self.result, provider=self.name, model="m", latency_ms=1.0, cost_usd=0.001)


def _router(providers: dict[str, _FakeProvider]) -> LLMRouter:
    chain = load_chain(
        None,
        overrides={"LLM_PROVIDER_CHAIN": ",".join(providers), **{f"{n.upper()}_API_KEY": "k" * 12 for n in providers}},
    )
    return LLMRouter(resolved=chain, factory=lambda rp: providers[rp.name])


def test_router_falls_through_on_error_and_bad_schema() -> None:
    fakes = {
        "anthropic": _FakeProvider("anthropic", LLMError("down")),
        "gemini": _FakeProvider("gemini", {"wrong": 1}),
        "openai": _FakeProvider("openai", {"ok": True}),
    }
    r = _router(fakes)

    def validate(d: dict[str, Any]) -> bool:
        if "ok" not in d:
            raise ValueError("schema")
        return d["ok"]

    res, parsed = asyncio.run(r.complete(_req(), validate=validate))
    assert res.provider == "openai" and parsed is True
    st = r.status()["chain"]
    assert st[0]["stats"]["failures"] == 1 and "down" in st[0]["stats"]["last_error"]
    assert st[2]["stats"]["total_cost_usd"] == pytest.approx(0.001)


def test_router_raises_when_all_fail() -> None:
    r = _router({"anthropic": _FakeProvider("anthropic", LLMError("x"))})
    with pytest.raises(LLMError):
        asyncio.run(r.complete(_req()))


# ---------- Claude provider (fake SDK) ----------

class _FakeMessages:
    def __init__(self, responses: list[Any]) -> None:
        self.responses, self.calls = list(responses), []

    async def create(self, **kwargs):
        self.calls.append(json.loads(json.dumps(kwargs, default=lambda o: getattr(o, "__dict__", str(o)))))
        return self.responses.pop(0)


def _msg(stop: str, blocks: list[Any], model: str = "claude-opus-5-5"):
    usage = SimpleNamespace(input_tokens=1000, output_tokens=200, cache_read_input_tokens=3000, cache_creation_input_tokens=0)
    return SimpleNamespace(stop_reason=stop, content=blocks, usage=usage, model=model)


def _text(t: str):
    return SimpleNamespace(type="text", text=t)


def _claude(responses: list[Any], model_env: dict[str, str] | None = None) -> tuple[AnthropicProvider, _FakeMessages]:
    rp = resolve_provider("anthropic", {"ANTHROPIC_API_KEY": "k" * 20, **(model_env or {})})
    fm = _FakeMessages(responses)
    client = SimpleNamespace(beta=SimpleNamespace(messages=fm))
    return AnthropicProvider(rp, client=client), fm


def test_claude_structured_output_caching_fallback_and_cost() -> None:
    prov, fm = _claude([_msg("end_turn", [_text('{"ok": true}')])])
    res = asyncio.run(prov.complete(_req(effort="low")))
    assert res.data == {"ok": True} and res.model == "claude-opus-5-5"
    call = fm.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert call["output_config"] == {"format": {"type": "json_schema", "schema": SCHEMA}, "effort": "low"}
    assert call["fallbacks"] == "default" and call["betas"] == ["server-side-fallback-2026-07-01"]
    assert "temperature" not in call and "thinking" not in call
    # 1000*4 + 3000*0.2 + 200*20 per 1M
    assert res.cost_usd == pytest.approx((4000 + 600 + 4000) / 1e6)


def test_claude_fast_model_has_no_server_fallback() -> None:
    prov, fm = _claude([_msg("end_turn", [_text('{"ok": false}')], model="claude-haiku-5-5")])
    res = asyncio.run(prov.complete(_req(fast=True)))
    assert fm.calls[0]["model"] == "claude-haiku-5-5" and "fallbacks" not in fm.calls[0]
    assert res.data == {"ok": False}


@pytest.mark.parametrize("stop", ["refusal", "max_tokens"])
def test_claude_bad_stop_reasons_raise(stop: str) -> None:
    prov, _ = _claude([_msg(stop, [_text("")])])
    with pytest.raises(LLMError):
        asyncio.run(prov.complete(_req()))


def test_claude_tool_loop_runs_tools_and_returns_answer() -> None:
    seen: list[dict[str, Any]] = []

    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        seen.append(args)
        return {"n": 12}

    tool_use = SimpleNamespace(type="tool_use", id="tu_1", name="get_breakdown", input={"by": "strategy"})
    prov, fm = _claude([_msg("tool_use", [tool_use]), _msg("end_turn", [_text('{"ok": true}')])])
    spec = ToolSpec("get_breakdown", "d", {"type": "object", "additionalProperties": False, "required": ["by"],
                                           "properties": {"by": {"type": "string"}}}, handler)
    res = asyncio.run(prov.complete(_req(tools=[spec])))
    assert res.data == {"ok": True} and res.tool_calls == 1 and seen == [{"by": "strategy"}]
    assert fm.calls[0]["tools"][0]["strict"] is True
    second = fm.calls[1]["messages"]
    assert second[-1]["content"][0]["type"] == "tool_result" and second[-1]["content"][0]["tool_use_id"] == "tu_1"


def test_claude_timeout_is_an_llm_error() -> None:
    class Slow:
        async def create(self, **kw):
            await asyncio.sleep(1.0)

    rp = resolve_provider("anthropic", {"ANTHROPIC_API_KEY": "k" * 20})
    prov = AnthropicProvider(rp, client=SimpleNamespace(beta=SimpleNamespace(messages=Slow())))
    with pytest.raises(LLMError, match="timeout"):
        asyncio.run(prov.complete(_req(timeout_s=0.05)))


# ---------- OpenAI-compatible provider ----------

def _compat(name: str, handler, env: dict[str, str] | None = None) -> OpenAICompatProvider:
    rp = resolve_provider(name, {f"{name.upper()}_API_KEY": "sk-test-key-0000", **(env or {})})
    return OpenAICompatProvider(rp, transport=httpx.MockTransport(handler))


def test_openai_uses_json_schema_and_bearer_key() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"model": "gpt-x", "choices": [{"message": {"content": '{"ok": true}'}, "finish_reason": "stop"}],
                                         "usage": {"prompt_tokens": 10, "completion_tokens": 3}})

    res = asyncio.run(_compat("openai", handler).complete(_req()))
    assert captured["url"] == "https://api.openai.com/v1/chat/completions"
    assert captured["auth"] == "Bearer sk-test-key-0000"
    assert captured["body"]["response_format"]["type"] == "json_schema"
    assert res.data == {"ok": True} and res.input_tokens == 10


def test_gemini_key_goes_in_header_not_url() -> None:
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok": true}'}}]})

    asyncio.run(_compat("gemini", handler).complete(_req()))
    assert "key=" not in captured["url"] and "sk-test" not in captured["url"]
    assert captured["url"].startswith("https://generativelanguage.googleapis.com/v1beta/openai/")


def test_local_llamacpp_json_object_mode_and_fenced_output() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        captured["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"choices": [{"message": {"content": 'Sure!\n```json\n{"ok": true}\n```'}}]})

    rp = resolve_provider("llamacpp", {})
    res = asyncio.run(OpenAICompatProvider(rp, transport=httpx.MockTransport(handler)).complete(_req()))
    assert res.data == {"ok": True}
    assert captured["body"]["response_format"] == {"type": "json_object"}
    assert "JSON schema" in captured["body"]["messages"][0]["content"]
    assert captured["auth"] is None


def test_http_error_message_does_not_contain_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "invalid api key"})

    with pytest.raises(LLMError) as ei:
        asyncio.run(_compat("groq", handler).complete(_req()))
    assert "401" in str(ei.value) and "sk-test-key-0000" not in str(ei.value)


# ---------- reflection safety ----------

def test_llm_patches_cannot_raise_risk_and_steps_are_capped() -> None:
    cfg = load_trading_config().model_dump(mode="json")
    up = restrict_llm_patch(cfg, ParameterPatch(path="risk.risk_per_trade_pct", old_value=0, new_value=1.5))
    assert up is None
    down = restrict_llm_patch(cfg, ParameterPatch(path="risk.risk_per_trade_pct", old_value=0, new_value=0.1))
    assert down is not None and down.new_value == pytest.approx(0.40)  # 0.50 - 20%
    looser_ml = restrict_llm_patch(cfg, ParameterPatch(path="ai.ml_min_probability", old_value=0, new_value=0.50))
    assert looser_ml is None
    neutral = restrict_llm_patch(cfg, ParameterPatch(path="strategies.liquidity_sweep.sl_atr_mult", old_value=0, new_value=9.0))
    assert neutral is not None and neutral.new_value == pytest.approx(1.74)  # 1.45 + 20%
    assert restrict_llm_patch(cfg, ParameterPatch(path="app.trading_mode", old_value=0, new_value=1)) is None


def _trades(n: int) -> list[TradeRecord]:
    out = []
    for i in range(n):
        win = i % 3 == 0
        out.append(TradeRecord(symbol="XAUUSD", strategy_id="micro_breakout", regime=MarketRegime.TRENDING_BULL, side=Side.BUY,
                               volume=0.1, entry_price=2600, exit_price=2602 if win else 2599, sl_price=2599, tp_price=2602,
                               open_time=3600 * (i % 24), close_time=3600 * (i % 24) + 300, pnl_money=20 if win else -10,
                               pnl_r=1.6 if win else -1.0, mae_r=-0.3 if win else -1.0, mfe_r=1.7 if win else 0.7,
                               spread_at_entry=14 if win else 26))
    return out


def test_reflector_uses_router_and_filters_patches() -> None:
    answer = {
        "finding": "Losers entered at wider spreads.",
        "evidence": "n=30, loser spread 26 vs 14.",
        "patches": [
            {"path": "risk.max_spread_atr_ratio", "new_value": 0.25, "rationale": "tighter spread gate"},
            {"path": "risk.risk_per_trade_pct", "new_value": 1.2, "rationale": "bigger size"},  # must be dropped
        ],
    }
    fake = _FakeProvider("anthropic", answer)
    reflector = LLMReflectionAnalyst(EnvSettings(_env_file=None), router=_router({"anthropic": fake}))
    report = asyncio.run(reflector.reflect(_trades(30), load_trading_config()))
    assert report.provider.startswith("anthropic:")
    assert [p.path for p in report.patches] == ["risk.max_spread_atr_ratio"]
    assert report.patches[0].new_value == pytest.approx(0.25)


def test_reflector_falls_back_to_deterministic_when_all_providers_fail() -> None:
    reflector = LLMReflectionAnalyst(EnvSettings(_env_file=None), router=_router({"anthropic": _FakeProvider("anthropic", LLMError("x"))}))
    report = asyncio.run(reflector.reflect(_trades(30), load_trading_config()))
    assert report.provider == "deterministic_fallback" and reflector.last_error


def test_reflection_schema_is_strict_compatible() -> None:
    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node.get("additionalProperties") is False
                assert set(node.get("required", [])) == set(node.get("properties", {}))
            for k in ("minimum", "maximum", "maxItems", "minLength"):
                assert k not in node  # keep schemas inside the structured-output subset
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(_REFLECTION_SCHEMA)
    assert "{" not in REFLECTION_SYSTEM.split("Tunable parameters")[0].replace("{path", "")  # no timestamps / live data in cached prompt
