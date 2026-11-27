"""Claude via the official Anthropic SDK.

- Structured outputs (`output_config.format`) so the answer is always schema-valid JSON.
- Optional read-only tools (strict schemas) for drill-down during reflection.
- Prompt caching on the stable system prompt (+ tools).
- Server-side refusal fallback on models that support it.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from anthropic import AsyncAnthropic

from backend.llm.base import LLMError, LLMRequest, LLMResult
from backend.llm.presets import PRICING_PER_MTOK
from backend.llm.settings import ResolvedProvider

# Models that accept `fallbacks: "default"` (beta) — see Anthropic docs on refusal fallbacks.
_SERVER_FALLBACK_MODELS = {"claude-opus-5-5", "claude-sonnet-5-5", "claude-opus-5", "claude-fable-5-2", "claude-fable-5-1"}
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


def estimate_cost(model: str, inp: int, out: int, cache_read: int, cache_write: int) -> float | None:
    price = PRICING_PER_MTOK.get(model)
    if price is None:
        return None
    p_in, p_out, p_cr = price
    return round((inp * p_in + cache_write * p_in * 1.25 + cache_read * p_cr + out * p_out) / 1_000_000, 6)


class AnthropicProvider:
    def __init__(self, rp: ResolvedProvider, client: Any | None = None) -> None:
        self.rp = rp
        self.name = rp.name
        # Retries are handled by the router's provider chain; a scalping supervisor cannot wait for backoff.
        self.client = client or AsyncAnthropic(api_key=rp.api_key, base_url=rp.base_url, max_retries=0)

    async def complete(self, req: LLMRequest) -> LLMResult:
        model = self.rp.fast_model if req.fast else self.rp.model
        t0 = time.perf_counter()
        deadline = t0 + req.timeout_s

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": req.max_tokens,
            "system": [{"type": "text", "text": req.system, "cache_control": {"type": "ephemeral"}}],
            "output_config": {"format": {"type": "json_schema", "schema": req.schema}, "effort": req.effort},
        }
        tool_map = {t.name: t for t in req.tools}
        if req.tools:
            kwargs["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.input_schema, "strict": True}
                for t in req.tools
            ]
        if model in _SERVER_FALLBACK_MODELS:
            kwargs["betas"] = [_FALLBACK_BETA]
            kwargs["fallbacks"] = "default"

        messages: list[dict[str, Any]] = [{"role": "user", "content": req.user}]
        tot_in = tot_out = tot_cr = tot_cw = 0
        tool_calls = 0
        served_model = model

        for turn in range(req.max_tool_turns + 1):
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise LLMError("timeout before answer")
            try:
                resp = await asyncio.wait_for(
                    self.client.beta.messages.create(messages=messages, **kwargs), timeout=remaining
                )
            except asyncio.TimeoutError as exc:
                raise LLMError(f"timeout after {req.timeout_s:.1f}s") from exc
            except Exception as exc:  # SDK errors: auth, rate limit, 5xx, connection
                raise LLMError(f"{type(exc).__name__}: {exc}") from exc

            served_model = getattr(resp, "model", model) or model
            u = resp.usage
            tot_in += int(getattr(u, "input_tokens", 0) or 0)
            tot_out += int(getattr(u, "output_tokens", 0) or 0)
            tot_cr += int(getattr(u, "cache_read_input_tokens", 0) or 0)
            tot_cw += int(getattr(u, "cache_creation_input_tokens", 0) or 0)

            if resp.stop_reason == "refusal":
                raise LLMError("model refused")
            if resp.stop_reason == "max_tokens":
                raise LLMError("hit max_tokens before finishing")

            if resp.stop_reason == "tool_use" and tool_map and turn < req.max_tool_turns:
                results = []
                for block in resp.content:
                    if getattr(block, "type", "") != "tool_use":
                        continue
                    tool_calls += 1
                    spec = tool_map.get(block.name)
                    try:
                        if spec is None:
                            raise ValueError(f"unknown tool {block.name}")
                        out = await spec.handler(dict(block.input or {}))
                        results.append(
                            {"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(out, default=str)}
                        )
                    except Exception as exc:
                        results.append(
                            {"type": "tool_result", "tool_use_id": block.id, "content": f"Error: {exc}", "is_error": True}
                        )
                messages.append({"role": "assistant", "content": resp.content})
                messages.append({"role": "user", "content": results})
                continue

            text = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text")
            try:
                data = json.loads(text)
            except json.JSONDecodeError as exc:
                raise LLMError(f"non-JSON answer: {text[:120]!r}") from exc
            return LLMResult(
                data=data,
                provider=self.name,
                model=served_model,
                latency_ms=round((time.perf_counter() - t0) * 1000, 1),
                input_tokens=tot_in,
                output_tokens=tot_out,
                cache_read_tokens=tot_cr,
                cost_usd=estimate_cost(served_model, tot_in, tot_out, tot_cr, tot_cw),
                tool_calls=tool_calls,
            )
        raise LLMError("tool loop did not converge")
