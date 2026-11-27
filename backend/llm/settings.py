"""Resolve which providers are usable from `.env` + environment (env vars win)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

from backend.llm.presets import DEFAULT_CHAIN, PRESETS, ProviderPreset

_DISABLED = {"", "off", "none", "deterministic", "disabled"}


@dataclass(frozen=True, slots=True)
class ResolvedProvider:
    preset: ProviderPreset
    api_key: str
    base_url: str
    model: str
    fast_model: str
    fallback_models: tuple[str, ...] = ()  # same provider, tried in order when the main model fails

    @property
    def name(self) -> str:
        return self.preset.name

    @property
    def configured(self) -> bool:
        if not self.base_url or not self.model:
            return False
        return bool(self.api_key) or not self.preset.needs_key

    def masked_key(self) -> str:
        return f"...{self.api_key[-4:]}" if len(self.api_key) >= 8 else ("(set)" if self.api_key else "(none)")


def _merged_env(env_file: str | Path | None) -> dict[str, str]:
    values: dict[str, str] = {}
    if env_file and Path(env_file).exists():
        values.update({k: v for k, v in dotenv_values(env_file).items() if v is not None})
    values.update(os.environ)
    return values


def resolve_provider(name: str, env: dict[str, str]) -> ResolvedProvider | None:
    preset = PRESETS.get(name)
    if preset is None:
        return None
    up = name.upper()
    model = env.get(f"{up}_MODEL", "").strip() or preset.default_model
    return ResolvedProvider(
        preset=preset,
        api_key=env.get(f"{up}_API_KEY", "").strip(),
        base_url=(env.get(f"{up}_BASE_URL", "").strip() or preset.base_url).rstrip("/"),
        model=model,
        fast_model=env.get(f"{up}_FAST_MODEL", "").strip() or preset.fast_model or model,
        fallback_models=tuple(
            m.strip() for m in env.get(f"{up}_FALLBACK_MODELS", "").split(",") if m.strip()
        ),
    )


def parse_chain(env: dict[str, str]) -> list[str]:
    """LLM_PROVIDER_CHAIN=anthropic,gemini,openai (default). Legacy LLM_PROVIDER also accepted.

    Set it to ``off`` to disable LLM calls entirely (deterministic quant only).
    """
    raw = env.get("LLM_PROVIDER_CHAIN")
    if raw is None:
        legacy = env.get("LLM_PROVIDER")
        if legacy is not None and legacy.strip().lower() not in _DISABLED:
            raw = legacy
        elif legacy is not None:
            raw = ""  # legacy explicit 'deterministic' -> off
    if raw is None:
        return list(DEFAULT_CHAIN)
    if raw.strip().lower() in _DISABLED:
        return []
    out: list[str] = []
    for part in raw.split(","):
        n = part.strip().lower()
        if n in PRESETS and n not in out:
            out.append(n)
    return out


def load_chain(env_file: str | Path | None = ".env", overrides: dict[str, str] | None = None) -> list[ResolvedProvider]:
    env = _merged_env(env_file)
    if overrides:
        env.update(overrides)
    resolved = [resolve_provider(n, env) for n in parse_chain(env)]
    return [r for r in resolved if r is not None]
