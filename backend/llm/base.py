from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol


class LLMError(Exception):
    """A provider failed in a way the router should fall through on."""


@dataclass(slots=True)
class ToolSpec:
    """Read-only tool the model may call while thinking (Claude only; others get data upfront)."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], Awaitable[Any]]


@dataclass(slots=True)
class LLMRequest:
    task: str  # "supervisor" | "reflection" | "ping"
    system: str  # STABLE text (cached). Never put timestamps / live data here.
    user: str  # volatile context for this call
    schema: dict[str, Any]  # JSON schema of the answer (additionalProperties: false everywhere)
    timeout_s: float = 20.0
    fast: bool = False  # use the provider's fast model (latency-critical paths)
    max_tokens: int = 2000
    effort: str = "medium"  # Claude effort: low | medium | high
    tools: list[ToolSpec] = field(default_factory=list)
    max_tool_turns: int = 4


@dataclass(slots=True)
class LLMResult:
    data: dict[str, Any]
    provider: str
    model: str
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cost_usd: float | None = None
    tool_calls: int = 0


class Provider(Protocol):
    name: str

    async def complete(self, req: LLMRequest) -> LLMResult: ...
