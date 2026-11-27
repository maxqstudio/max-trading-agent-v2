"""List the models a provider offers, so the Setup page can show a dropdown instead of a text box.

Anthropic: GET /v1/models (x-api-key). Everything else (Gemini, OpenAI, OpenRouter, Groq, Ollama,
llama.cpp, ...): the OpenAI-compatible GET {base_url}/models. Errors never include the key.
"""

from __future__ import annotations

from typing import Any

import httpx

from backend.llm.settings import ResolvedProvider


class ModelListError(RuntimeError):
    pass


def _clean(err: str, key: str) -> str:
    return err.replace(key, "***") if key else err


async def list_models(rp: ResolvedProvider, transport: httpx.AsyncBaseTransport | None = None, timeout_s: float = 15.0) -> list[dict[str, Any]]:
    """Return [{id, name}] sorted by id. Raises ModelListError with a key-free message."""
    key = rp.api_key
    if rp.preset.kind == "anthropic":
        url = rp.base_url.rstrip("/") + "/v1/models?limit=1000"
        headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
    else:
        url = rp.base_url.rstrip("/") + "/models"
        headers = {"Authorization": f"Bearer {key}"} if key else {}
    try:
        async with httpx.AsyncClient(timeout=timeout_s, transport=transport) as client:
            r = await client.get(url, headers=headers)
    except httpx.HTTPError as exc:
        raise ModelListError(_clean(f"Could not reach {rp.base_url}: {type(exc).__name__}", key)) from None
    if r.status_code in (401, 403):
        raise ModelListError("The provider rejected the API key (401/403). Check the key.")
    if r.status_code >= 400:
        raise ModelListError(_clean(f"HTTP {r.status_code}: {r.text[:200]}", key))
    try:
        body = r.json()
    except ValueError:
        raise ModelListError("Jawaban provider bukan JSON") from None
    items = body.get("data") if isinstance(body, dict) else body
    if not isinstance(items, list):
        items = body.get("models", []) if isinstance(body, dict) else []
    out: dict[str, dict[str, Any]] = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        mid = str(it.get("id") or it.get("name") or it.get("model") or "").strip()
        if mid.startswith("models/"):  # Gemini style ids
            mid = mid[len("models/"):]
        if mid:
            out[mid] = {"id": mid, "name": str(it.get("display_name") or it.get("name") or mid)}
    return sorted(out.values(), key=lambda m: m["id"])
