"""Safe edits to the local `.env` from the dashboard Setup page.

Only an allow-list of keys can be written (AI provider settings). Tokens, the bridge secret and
ALLOW_LIVE_TRADING can never be changed from the browser. Writes are atomic and keep comments.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from backend.llm.presets import PRESETS

_PROVIDER_FIELDS = ("API_KEY", "MODEL", "FAST_MODEL", "BASE_URL", "FALLBACK_MODELS")
WRITABLE_KEYS: frozenset[str] = frozenset(
    {"LLM_PROVIDER_CHAIN"} | {f"{n.upper()}_{f}" for n in PRESETS for f in _PROVIDER_FIELDS}
)
NEVER_WRITABLE: frozenset[str] = frozenset(
    {"ALLOW_LIVE_TRADING", "BRIDGE_AUTH_TOKEN", "CONTROL_API_TOKEN", "TELEGRAM_BOT_TOKEN"}
)
_BAD_VALUE = re.compile(r"[\r\n\x00\"'`$\\]")


class EnvWriteError(ValueError):
    pass


def validate_updates(updates: dict[str, str | None]) -> None:
    for key, value in updates.items():
        if key in NEVER_WRITABLE or key not in WRITABLE_KEYS:
            raise EnvWriteError(f"{key} cannot be changed from the dashboard")
        if value is not None and (_BAD_VALUE.search(value) or len(value) > 500):
            raise EnvWriteError(f"{key}: invalid characters or too long")


def update_env_file(path: str | Path, updates: dict[str, str | None]) -> list[str]:
    """Set (or remove, when the value is None) keys in `.env`. Returns the changed key names."""
    validate_updates(updates)
    p = Path(path)
    lines = p.read_text(encoding="utf-8").splitlines() if p.exists() else []
    pending = dict(updates)
    out: list[str] = []
    for line in lines:
        m = re.match(r"^\s*([A-Z0-9_]+)\s*=", line)
        if m and m.group(1) in pending:
            val = pending.pop(m.group(1))
            if val is not None:
                out.append(f"{m.group(1)}={val}")
            continue
        out.append(line)
    for key, val in pending.items():
        if val is not None:
            out.append(f"{key}={val}")
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text("\n".join(out) + "\n", encoding="utf-8")
    os.replace(tmp, p)
    return list(updates)
