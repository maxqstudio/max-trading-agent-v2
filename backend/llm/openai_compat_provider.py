"""Any OpenAI-compatible chat endpoint: OpenAI, Gemini, OpenRouter, Groq, DeepSeek, Mistral,
xAI, Together, Ollama, llama.cpp server, or a custom URL."""

from __future__ import annotations

import json
import re
import time
from typing import Any

import httpx

from backend.llm.base import LLMError, LLMRequest, LLMResult
from backend.llm.settings import ResolvedProvider

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = _FENCE.sub("", text.strip()).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise LLMError(f"non-JSON answer: {text[:120]!r}") from None
        try:
            data = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMError(f"non-JSON answer: {text[:120]!r}") from exc
    if not isinstance(data, dict):
        raise LLMError("answer is not a JSON object")
    return data


class OpenAICompatProvider:
    def __init__(self, rp: ResolvedProvider, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.rp = rp
        self.name = rp.name
        self._transport = transport

    async def complete(self, req: LLMRequest) -> LLMResult:
        model = self.rp.fast_model if req.fast else self.rp.model
        system = req.system
        body: dict[str, Any] = {"model": model, "max_tokens": req.max_tokens}
        mode = self.rp.preset.json_mode
        if mode == "json_schema":
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": req.task, "schema": req.schema, "strict": True},
            }
        else:
            system = f"{system}\n\nReply with ONLY one JSON object matching this JSON schema:\n{json.dumps(req.schema)}"
            if mode == "json_object":
                body["response_format"] = {"type": "json_object"}
        body["messages"] = [{"role": "system", "content": system}, {"role": "user", "content": req.user}]

        headers = {"Content-Type": "application/json"}
        if self.rp.api_key:
            headers["Authorization"] = f"Bearer {self.rp.api_key}"

        t0 = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=req.timeout_s, transport=self._transport) as client:
                resp = await client.post(f"{self.rp.base_url}/chat/completions", headers=headers, json=body)
        except httpx.TimeoutException as exc:
            raise LLMError(f"timeout after {req.timeout_s:.1f}s") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc
        if resp.status_code >= 400:
            # Never echo request headers (they carry the key); body text is the provider's error message.
            raise LLMError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        try:
            payload = resp.json()
            choice = payload["choices"][0]
            text = choice["message"].get("content") or ""
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError("unexpected response shape") from exc
        if choice.get("finish_reason") == "length":
            raise LLMError("hit max_tokens before finishing")
        usage = payload.get("usage") or {}
        return LLMResult(
            data=_extract_json(text),
            provider=self.name,
            model=str(payload.get("model") or model),
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
        )
