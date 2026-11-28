from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from backend.engine.orchestrator import TradingOrchestrator


class TelegramControlBot:
    """Lightweight async Telegram alert & command bot via httpx long-polling (zero heavy deps)."""

    def __init__(self, token: str, chat_id: str, orchestrator: TradingOrchestrator) -> None:
        self.token = token.strip()
        self.chat_id = str(chat_id).strip()
        self.orch = orchestrator
        self._running = False
        self._offset = 0
        self._task: asyncio.Task | None = None

    @property
    def configured(self) -> bool:
        return bool(self.token and self.chat_id)

    @property
    def base_url(self) -> str:
        return f"https://api.telegram.org/bot{self.token}"

    async def send_message(self, text: str) -> bool:
        if not self.configured:
            return False
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.post(
                    f"{self.base_url}/sendMessage",
                    json={"chat_id": self.chat_id, "text": text[:3900]},
                )
                return r.status_code == 200
        except Exception:
            return False

    def start(self) -> None:
        if not self.configured or self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _poll_loop(self) -> None:
        async with httpx.AsyncClient(timeout=35.0) as client:
            while self._running:
                try:
                    resp = await client.get(
                        f"{self.base_url}/getUpdates",
                        params={"offset": self._offset, "timeout": 25},
                    )
                    if resp.status_code == 200:
                        for upd in resp.json().get("result", []):
                            self._offset = upd["update_id"] + 1
                            msg = upd.get("message", {})
                            chat = str(msg.get("chat", {}).get("id", ""))
                            if chat != self.chat_id:
                                continue
                            text = (msg.get("text") or "").strip()
                            if text.startswith("/"):
                                reply = await self.handle_command(text)
                                await self.send_message(reply)
                except asyncio.CancelledError:
                    break
                except Exception:
                    await asyncio.sleep(3.0)

    async def handle_command(self, raw_cmd: str) -> str:
        parts = raw_cmd.split()
        cmd = parts[0].lower()

        if cmd == "/status":
            a = self.orch.account
            return (
                f"MAX AGENT {self.orch.gate.active_snapshot.version_id} ({self.orch.config.app.trading_mode.value})\n"
                f"Equity: ${a.equity:,.2f} | Day PnL: {a.daily_pnl:+.2f}$ | DD: {a.daily_dd_pct:.2f}%\n"
                f"Open: {len(self.orch.open_positions)} | Paused: {self.orch.risk.paused} | Killed: {self.orch.risk.killed}"
            )
        if cmd == "/positions":
            if not self.orch.open_positions:
                return "No open live MT5 positions."
            lines = [
                f"#{p.ticket} {p.symbol} {p.side.value} {p.volume} | {p.profit:+.2f}$ | RL: {p.rl_exit_action.value}"
                for p in self.orch.open_positions
            ]
            return "Open Positions:\n" + "\n".join(lines)
        if cmd == "/regime":
            lines = [
                f"{s}: {self.orch.regimes[s].value} | DL Up:{self.orch.dl_preds[s].prob_up:.0%}"
                for s in self.orch.config.instruments.symbols
            ]
            return "Market Regimes:\n" + "\n".join(lines)
        if cmd == "/pause":
            self.orch.risk.paused = True
            return "Engine PAUSED (new entries blocked; SL/TP & RL Exit still active)."
        if cmd == "/resume":
            if self.orch.risk.killed:
                return "Kill switch is engaged. Send '/unkill confirm' first, then /resume."
            self.orch.risk.paused = False
            return "Engine RESUMED."
        if cmd == "/unkill":
            if len(parts) >= 2 and parts[1].lower() == "confirm":
                self.orch.release_kill_switch()
                return "Kill switch released. Engine is still PAUSED — send /resume to allow new entries."
            return "Safety Check: Send '/unkill confirm' to release the kill switch (engine stays paused)."
        if cmd == "/kill":
            if len(parts) >= 2 and parts[1].lower() == "confirm":
                await self.orch.trigger_flatten_all("TELEGRAM_KILL")
                return "EMERGENCY KILL-SWITCH EXECUTED — Flattening all positions!"
            return "Safety Check: Send '/kill confirm' to flatten all MT5 positions & lock engine."
        if cmd == "/evolve":
            res = await self.orch.run_evolution_cycle()
            return f"Evolution Cycle Result: {res.get('status')}"
        if cmd == "/approve":
            cand = self.orch.gate.active_challenger
            if cand is None:
                return "No active Challenger awaiting promotion."
            if cand.status != "PENDING_APPROVAL":
                min_req = self.orch.config.evolution.shadow_min_trades
                return f"Cannot promote {cand.version_id}: still {cand.status} ({cand.shadow_trades}/{min_req} shadow trades)."
            snap = await self.orch.promote_active_challenger(note="Approved via Telegram")
            if snap:
                return f"Promoted {snap.version_id} (config + DL/ML) to Active Champion!"
            return "No active Challenger awaiting promotion."
        if cmd == "/reject":
            ok = self.orch.gate.reject_active_challenger("Rejected via Telegram")
            if ok:
                self.orch.shadow_dl_net = None
                self.orch.shadow_ml_labeler = None
            return "Challenger rejected." if ok else "No active Challenger to reject."
        if cmd == "/rollback":
            snaps = self.orch.gate.snapshots
            prev = next((s for s in snaps if s.status == "ARCHIVED"), None)
            if prev:
                restored = await self.orch.rollback_champion(prev.version_id, auto=False)
                if restored:
                    return f"Rolled back instantly to Champion {restored.version_id} (config + DL/ML)."
            return "No previous archived snapshot available for rollback."
        return "Commands: /status /positions /regime /evolve /approve /reject /rollback /pause /resume /kill /unkill"
