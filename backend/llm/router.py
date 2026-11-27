"""Try providers in order (default: Claude -> Gemini -> OpenAI) and keep per-provider stats."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from backend.llm.base import LLMError, LLMRequest, LLMResult, Provider
from backend.llm.settings import ResolvedProvider, load_chain

log = logging.getLogger("max_agent.llm")


@dataclass(slots=True)
class ProviderStats:
    calls: int = 0
    failures: int = 0
    last_error: str = ""
    last_latency_ms: float = 0.0
    total_cost_usd: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read_tokens: int = 0


def build_provider(rp: ResolvedProvider) -> Provider:
    if rp.preset.kind == "anthropic":
        from backend.llm.anthropic_provider import AnthropicProvider

        return AnthropicProvider(rp)
    from backend.llm.openai_compat_provider import OpenAICompatProvider

    return OpenAICompatProvider(rp)


@dataclass
class LLMRouter:
    resolved: list[ResolvedProvider]
    factory: Callable[[ResolvedProvider], Provider] = build_provider
    stats: dict[str, ProviderStats] = field(default_factory=dict)
    _cache: dict[str, Provider] = field(default_factory=dict)

    @classmethod
    def from_env(cls, env_file: str | Path | None = ".env") -> "LLMRouter":
        return cls(resolved=load_chain(env_file))

    def reload(self, resolved: list[ResolvedProvider]) -> None:
        """Swap the provider chain in place (after a settings change); stats are kept."""
        self.resolved = resolved
        self._cache.clear()

    @property
    def active(self) -> list[ResolvedProvider]:
        return [r for r in self.resolved if r.configured]

    @property
    def enabled(self) -> bool:
        return bool(self.active)

    def _provider(self, rp: ResolvedProvider) -> Provider:
        key = f"{rp.name}|{rp.model}|{rp.fast_model}"
        if key not in self._cache:
            self._cache[key] = self.factory(rp)
        return self._cache[key]

    @staticmethod
    def _attempts(rp: ResolvedProvider) -> list[ResolvedProvider]:
        """The provider with its own model first, then each fallback model of the same provider."""
        out = [rp]
        for m in rp.fallback_models:
            if m not in (rp.model, rp.fast_model):
                out.append(replace(rp, model=m, fast_model=m, fallback_models=()))
        return out

    async def complete(self, req: LLMRequest, validate: Callable[[dict[str, Any]], Any] | None = None) -> tuple[LLMResult, Any]:
        """Return (result, validated) from the first provider whose answer passes ``validate``."""
        errors: list[str] = []
        for attempt in (a for rp in self.active for a in self._attempts(rp)):
            rp = attempt
            st = self.stats.setdefault(rp.name, ProviderStats())
            st.calls += 1
            try:
                res = await self._provider(rp).complete(req)
                parsed = validate(res.data) if validate else res.data
            except Exception as exc:  # LLMError, schema validation errors, anything provider-side
                st.failures += 1
                st.last_error = str(exc)[:300]
                label = f"{rp.name}:{rp.fast_model if req.fast else rp.model}"
                errors.append(f"{label}: {st.last_error}")
                log.warning("LLM %s failed for %s: %s", label, req.task, st.last_error)
                continue
            st.last_error = ""
            st.last_latency_ms = res.latency_ms
            st.total_input_tokens += res.input_tokens
            st.total_output_tokens += res.output_tokens
            st.total_cache_read_tokens += res.cache_read_tokens
            if res.cost_usd:
                st.total_cost_usd = round(st.total_cost_usd + res.cost_usd, 6)
            return res, parsed
        raise LLMError("; ".join(errors) or "no LLM provider configured")

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "chain": [
                {
                    "provider": r.name,
                    "label": r.preset.label,
                    "configured": r.configured,
                    "model": r.model,
                    "fast_model": r.fast_model,
                    "fallback_models": list(r.fallback_models),
                    "base_url": r.base_url,
                    "api_key": r.masked_key(),
                    "stats": vars_or_empty(self.stats.get(r.name)),
                }
                for r in self.resolved
            ],
        }


def vars_or_empty(st: ProviderStats | None) -> dict[str, Any]:
    if st is None:
        return {}
    return {k: getattr(st, k) for k in st.__slots__}  # type: ignore[attr-defined]
