"""Hard safety rails that sit underneath every strategy, AI pillar and operator action.

Nothing in here is tunable by the optimizer or by an LLM. These checks answer one
question: "is the engine allowed to send a NEW entry order to MT5 right now?"
Closing, modifying and flattening positions are never blocked by these rails.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.core.models import TradingMode

# Placeholder values that shipped in earlier versions. They are public (this repo is
# public), so a bridge or control token equal to one of these is treated as missing.
KNOWN_PLACEHOLDER_TOKENS: frozenset[str] = frozenset(
    {
        "mta2_dev_change_me_to_random_secret_32chars",
        "change_me",
        "changeme",
    }
)
MIN_TOKEN_LENGTH = 24


def is_strong_token(token: str | None) -> bool:
    """A usable shared secret: long enough and not a published placeholder."""
    if not token:
        return False
    t = token.strip()
    if len(t) < MIN_TOKEN_LENGTH:
        return False
    low = t.lower()
    return low not in KNOWN_PLACEHOLDER_TOKENS and "change_me" not in low


def entry_block_reason(
    *,
    mode: TradingMode,
    account_is_demo: bool,
    allow_live_trading: bool,
    killed: bool,
) -> str | None:
    """Return why a new entry must NOT be dispatched, or None if it may go out.

    Rules (all must pass):
    - Kill switch engaged -> nothing opens.
    - SHADOW_ONLY is paper trading -> nothing opens on MT5.
    - MT5_LIVE requires ALLOW_LIVE_TRADING=true in .env (an operator decision that
      cannot be made from the dashboard, Telegram, or the LLM).
    - A real-money account (as reported by the EA) only accepts entries in MT5_LIVE
      with ALLOW_LIVE_TRADING=true. SIMULATOR / MT5_DEMO on a real account is refused,
      so a mis-attached EA can never trade real money by accident.
    """
    if killed:
        return "KILL_SWITCH_ACTIVE"
    if mode == TradingMode.SHADOW_ONLY:
        return "SHADOW_ONLY_PAPER_MODE"
    if mode == TradingMode.MT5_LIVE and not allow_live_trading:
        return "LIVE_TRADING_NOT_ENABLED"
    if not account_is_demo and not (mode == TradingMode.MT5_LIVE and allow_live_trading):
        return "REAL_ACCOUNT_NOT_AUTHORIZED"
    return None


@dataclass(slots=True)
class KillSwitchState:
    engaged: bool = False
    reason: str = ""
    engaged_at: float = 0.0


class KillSwitchStore:
    """File-backed kill switch so a crash or restart never silently re-arms trading.

    The file is written atomically. If it cannot be read (corrupt / permission error)
    the switch is reported as ENGAGED: failing closed is the safe default.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> KillSwitchState:
        if not self.path.exists():
            return KillSwitchState()
        try:
            data: dict[str, Any] = json.loads(self.path.read_text(encoding="utf-8"))
            return KillSwitchState(
                engaged=bool(data.get("engaged", True)),
                reason=str(data.get("reason", "")),
                engaged_at=float(data.get("engaged_at", 0.0)),
            )
        except Exception:
            return KillSwitchState(engaged=True, reason="KILL_FILE_UNREADABLE", engaged_at=time.time())

    def _write(self, state: KillSwitchState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(
            json.dumps({"engaged": state.engaged, "reason": state.reason, "engaged_at": state.engaged_at}),
            encoding="utf-8",
        )
        os.replace(tmp, self.path)

    def engage(self, reason: str) -> KillSwitchState:
        state = KillSwitchState(engaged=True, reason=reason, engaged_at=time.time())
        self._write(state)
        return state

    def release(self) -> KillSwitchState:
        state = KillSwitchState()
        self._write(state)
        return state
