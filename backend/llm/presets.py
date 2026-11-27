"""Provider presets: pick a name, paste a key, done.

Every value can be overridden in `.env` with ``<NAME>_MODEL``, ``<NAME>_FAST_MODEL``,
``<NAME>_BASE_URL`` and ``<NAME>_API_KEY`` (e.g. ``GEMINI_MODEL=...``). Keys only ever
live in `.env` / environment variables, never in this file or in git.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ProviderKind = Literal["anthropic", "openai_compat"]
JsonMode = Literal["json_schema", "json_object", "none"]


@dataclass(frozen=True, slots=True)
class ProviderPreset:
    name: str
    label: str
    kind: ProviderKind
    base_url: str
    default_model: str
    fast_model: str
    needs_key: bool = True
    json_mode: JsonMode = "json_schema"
    key_url: str = ""
    note: str = ""


PRESETS: dict[str, ProviderPreset] = {
    p.name: p
    for p in [
        ProviderPreset(
            name="anthropic",
            label="Claude (Anthropic API)",
            kind="anthropic",
            base_url="https://api.anthropic.com",
            default_model="claude-opus-5-5",
            fast_model="claude-haiku-5-5",
            key_url="https://console.anthropic.com/settings/keys",
            note="Recommended. Supervisor + reflection with structured outputs, tool use and prompt caching.",
        ),
        ProviderPreset(
            name="gemini",
            label="Google Gemini (AI Studio key)",
            kind="openai_compat",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
            default_model="gemini-2.5-pro",
            fast_model="gemini-2.5-flash",
            key_url="https://aistudio.google.com/apikey",
            note="Set GEMINI_MODEL to the exact model id you want (check AI Studio for the newest).",
        ),
        ProviderPreset(
            name="openai",
            label="OpenAI (ChatGPT API)",
            kind="openai_compat",
            base_url="https://api.openai.com/v1",
            default_model="gpt-4.1",
            fast_model="gpt-4.1-mini",
            key_url="https://platform.openai.com/api-keys",
        ),
        ProviderPreset(
            name="openrouter",
            label="OpenRouter (hundreds of models, one key)",
            kind="openai_compat",
            base_url="https://openrouter.ai/api/v1",
            default_model="anthropic/claude-opus-5-5",
            fast_model="anthropic/claude-haiku-5-5",
            json_mode="json_object",
            key_url="https://openrouter.ai/keys",
        ),
        ProviderPreset(
            name="groq",
            label="Groq (very low latency)",
            kind="openai_compat",
            base_url="https://api.groq.com/openai/v1",
            default_model="llama-3.3-70b-versatile",
            fast_model="llama-3.1-8b-instant",
            json_mode="json_object",
            key_url="https://console.groq.com/keys",
        ),
        ProviderPreset(
            name="deepseek",
            label="DeepSeek",
            kind="openai_compat",
            base_url="https://api.deepseek.com/v1",
            default_model="deepseek-chat",
            fast_model="deepseek-chat",
            json_mode="json_object",
            key_url="https://platform.deepseek.com/api_keys",
        ),
        ProviderPreset(
            name="mistral",
            label="Mistral",
            kind="openai_compat",
            base_url="https://api.mistral.ai/v1",
            default_model="mistral-large-latest",
            fast_model="mistral-small-latest",
            json_mode="json_object",
            key_url="https://console.mistral.ai/api-keys",
        ),
        ProviderPreset(
            name="xai",
            label="xAI (Grok)",
            kind="openai_compat",
            base_url="https://api.x.ai/v1",
            default_model="grok-4",
            fast_model="grok-4-fast",
            json_mode="json_object",
            key_url="https://console.x.ai",
        ),
        ProviderPreset(
            name="together",
            label="Together AI",
            kind="openai_compat",
            base_url="https://api.together.xyz/v1",
            default_model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            fast_model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            json_mode="json_object",
            key_url="https://api.together.ai/settings/api-keys",
        ),
        ProviderPreset(
            name="ollama",
            label="Ollama (local, free)",
            kind="openai_compat",
            base_url="http://127.0.0.1:11434/v1",
            default_model="llama3.1",
            fast_model="llama3.1",
            needs_key=False,
            json_mode="json_object",
            note="Run `ollama serve` and `ollama pull <model>` first.",
        ),
        ProviderPreset(
            name="llamacpp",
            label="llama.cpp server (local, free)",
            kind="openai_compat",
            base_url="http://127.0.0.1:8080/v1",
            default_model="local",
            fast_model="local",
            needs_key=False,
            json_mode="json_object",
            note="Start with `llama-server -m model.gguf --port 8080 --jinja`.",
        ),
        ProviderPreset(
            name="custom",
            label="Any other OpenAI-compatible endpoint",
            kind="openai_compat",
            base_url="",
            default_model="",
            fast_model="",
            needs_key=False,
            json_mode="json_object",
            note="Set CUSTOM_BASE_URL, CUSTOM_MODEL and (if needed) CUSTOM_API_KEY.",
        ),
    ]
}

# Jackson's order: Claude first, then Gemini, OpenAI last.
DEFAULT_CHAIN: tuple[str, ...] = ("anthropic", "gemini", "openai")

# USD per 1M tokens (input, output, cache_read). Only providers whose prices we know.
PRICING_PER_MTOK: dict[str, tuple[float, float, float]] = {
    "claude-opus-5-5": (4.00, 20.00, 0.20),
    "claude-sonnet-5-5": (2.00, 10.00, 0.10),
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
}
