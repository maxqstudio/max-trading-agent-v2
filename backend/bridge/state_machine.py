from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field

from backend.core.models import (
    LivePosition,
    MarketRegime,
    OrderAck,
    OrderAction,
    OrderCommand,
    OrderStatus,
    RLExitAction,
    Side,
    SymbolSpec,
    TradeRecord,
    utc_now_ms,
)


@dataclass
class PendingOrderRecord:
    command: OrderCommand
    status: OrderStatus = OrderStatus.PENDING_DISPATCH
    strategy_id: str = ""
    regime: MarketRegime = MarketRegime.RANGING
    ml_confidence: float = 0.5
    dl_confidence: float = 0.5
    bandit_weight: float = 0.25
    spread_at_entry: float = 0.0
    dispatched_at_ms: int = 0
    exit_style: str = ""  # "normal" or "hitrun", decided at entry
    time_stop_bars: int = 0  # Hit & Run time stop (M1 bars, 0 = off)
    tp_stages: list[float] = field(default_factory=list)  # staged targets before the TP


@dataclass
class TrackedPositionMeta:
    ticket: int
    client_order_id: str
    symbol: str
    side: Side
    volume: float
    entry_price: float
    sl: float
    tp: float
    initial_risk_price: float
    open_time: int
    strategy_id: str
    regime: MarketRegime
    ml_confidence: float
    dl_confidence: float
    bandit_weight: float
    spread_at_entry: float
    slippage_points: float = 0.0
    commission: float = 0.0
    mae_r: float = 0.0
    mfe_r: float = 0.0
    bars_held: int = 0
    already_at_be: bool = False
    already_partial: bool = False
    last_sl_request: float = 0.0  # last SL we asked MT5 for (avoids re-sending every sync)
    last_rl_action: RLExitAction = RLExitAction.HOLD
    rl_trace: list[str] = field(default_factory=lambda: ["HOLD"])
    exit_style: str = ""  # "" = opened before exit styles were fixed at entry
    time_stop_bars: int = 0
    tp_stages: list[float] = field(default_factory=list)


class IdempotentOrderStateMachine:
    """Prevents duplicate/stale orders on HTTP bridge & reconciles authoritative MT5 live positions."""

    def __init__(self, order_ttl_ms: int = 1500) -> None:
        self.order_ttl_ms = order_ttl_ms
        self._queue: deque[OrderCommand] = deque()
        self._orders: dict[str, PendingOrderRecord] = {}
        self._seen_ids: set[str] = set()
        self.active_positions: dict[int, TrackedPositionMeta] = {}
        self._last_mt5_positions: dict[int, LivePosition] = {}
        self.last_blocked: list[tuple[OrderCommand, str]] = []

    def export_tracking(self) -> dict:
        """JSON-safe copy of what we know about open MT5 positions (survives a restart)."""
        return {
            "active": [{**asdict(m), "side": m.side.value, "regime": m.regime.value, "last_rl_action": m.last_rl_action.value}
                       for m in self.active_positions.values()],
            "last": [p.model_dump(mode="json") for p in self._last_mt5_positions.values()],
        }

    def import_tracking(self, data: dict | None) -> int:
        """Restore tracking saved before a restart. Positions that closed while we were down are
        then reported as closed trades on the first MT5 sync instead of being lost."""
        if not data:
            return 0
        for d in data.get("active", []):
            try:
                d = dict(d)
                d["side"] = Side(d["side"])
                d["regime"] = MarketRegime(d["regime"])
                d["last_rl_action"] = RLExitAction(d.get("last_rl_action", "HOLD"))
                meta = TrackedPositionMeta(**d)
            except (KeyError, TypeError, ValueError):
                continue
            self.active_positions[meta.ticket] = meta
        for d in data.get("last", []):
            try:
                pos = LivePosition.model_validate(d)
            except ValueError:
                continue
            self._last_mt5_positions[pos.ticket] = pos
        return len(self.active_positions)

    def has_inflight_open(self, symbol: str) -> bool:
        now = utc_now_ms()
        for rec in self._orders.values():
            if rec.command.action != OrderAction.OPEN or rec.command.symbol != symbol:
                continue
            if rec.status in (OrderStatus.PENDING_DISPATCH, OrderStatus.IN_FLIGHT):
                if now <= rec.command.expires_at_ms + 3000:
                    return True
        return False

    def enqueue(
        self,
        cmd: OrderCommand,
        strategy_id: str = "",
        regime: MarketRegime = MarketRegime.RANGING,
        ml_confidence: float = 0.5,
        dl_confidence: float = 0.5,
        bandit_weight: float = 0.25,
        spread_at_entry: float = 0.0,
        exit_style: str = "",
        time_stop_bars: int = 0,
        tp_stages: list[float] | None = None,
    ) -> bool:
        """Idempotently enqueue an order command. Returns False if duplicate ID or inflight open exists."""
        if cmd.client_order_id in self._seen_ids:
            return False
        if cmd.action == OrderAction.OPEN and self.has_inflight_open(cmd.symbol):
            return False

        self._seen_ids.add(cmd.client_order_id)
        if len(self._seen_ids) > 2000:
            # Trim oldest completed records safely
            for k in list(self._orders.keys())[:500]:
                if self._orders[k].status in (OrderStatus.FILLED, OrderStatus.REJECTED, OrderStatus.EXPIRED, OrderStatus.CLOSED):
                    self._orders.pop(k, None)
                    self._seen_ids.discard(k)

        self._orders[cmd.client_order_id] = PendingOrderRecord(
            command=cmd,
            status=OrderStatus.PENDING_DISPATCH,
            strategy_id=strategy_id or cmd.strategy_id,
            regime=regime,
            ml_confidence=ml_confidence,
            dl_confidence=dl_confidence,
            bandit_weight=bandit_weight,
            spread_at_entry=spread_at_entry,
            exit_style=exit_style,
            time_stop_bars=time_stop_bars,
            tp_stages=list(tp_stages or []),
        )
        self._queue.append(cmd)
        return True

    def pop_commands_for_dispatch(
        self,
        now_ms: int | None = None,
        block_entries_reason: str | None = None,
    ) -> list[OrderCommand]:
        """Pop queued commands for the EA.

        If ``block_entries_reason`` is set, every OPEN command is marked REJECTED and
        withheld (recorded in ``last_blocked``); exits/modifies/flatten still go out.
        """
        now = now_ms if now_ms is not None else utc_now_ms()
        out: list[OrderCommand] = []
        self.last_blocked: list[tuple[OrderCommand, str]] = []
        while self._queue:
            cmd = self._queue.popleft()
            rec = self._orders.get(cmd.client_order_id)
            if rec is None:
                continue
            if cmd.action == OrderAction.OPEN and block_entries_reason:
                rec.status = OrderStatus.REJECTED
                self.last_blocked.append((cmd, block_entries_reason))
                continue
            if cmd.action == OrderAction.OPEN and now > cmd.expires_at_ms:
                rec.status = OrderStatus.EXPIRED
                continue
            rec.status = OrderStatus.IN_FLIGHT
            rec.dispatched_at_ms = now
            out.append(cmd)
        return out

    def process_acks(self, acks: list[OrderAck]) -> list[OrderAck]:
        processed: list[OrderAck] = []
        for ack in acks:
            rec = self._orders.get(ack.client_order_id)
            if rec is None:
                continue
            rec.status = ack.status
            if ack.status == OrderStatus.FILLED and rec.command.action == OrderAction.OPEN and ack.ticket > 0:
                entry = ack.fill_price or rec.command.expected_price
                risk_dist = abs(entry - rec.command.sl) if rec.command.sl > 0 else max(entry * 0.001, 1e-5)
                self.active_positions[ack.ticket] = TrackedPositionMeta(
                    ticket=ack.ticket,
                    client_order_id=ack.client_order_id,
                    symbol=rec.command.symbol,
                    side=rec.command.side or Side.BUY,
                    volume=rec.command.volume,
                    entry_price=entry,
                    sl=rec.command.sl,
                    tp=rec.command.tp,
                    initial_risk_price=max(risk_dist, 1e-9),
                    open_time=int(ack.timestamp_ms // 1000),
                    strategy_id=rec.strategy_id or "unknown",
                    regime=rec.regime,
                    ml_confidence=rec.ml_confidence,
                    dl_confidence=rec.dl_confidence,
                    bandit_weight=rec.bandit_weight,
                    spread_at_entry=rec.spread_at_entry,
                    slippage_points=ack.slippage_points,
                    commission=ack.commission,
                    exit_style=rec.exit_style,
                    time_stop_bars=rec.time_stop_bars,
                    tp_stages=list(rec.tp_stages),
                )
            processed.append(ack)
        return processed

    def reconcile_positions(
        self,
        mt5_positions: list[LivePosition],
        specs: dict[str, SymbolSpec],
        active_version: str,
        now_sec: int,
    ) -> tuple[list[LivePosition], list[TradeRecord]]:
        """Reconcile MT5 authoritative open positions; return enriched LivePositions & newly closed TradeRecords."""
        live_by_ticket: dict[int, LivePosition] = {p.ticket: p for p in mt5_positions}
        enriched: list[LivePosition] = []
        closed_trades: list[TradeRecord] = []

        # 1. Detect positions that disappeared from MT5 (closed via SL/TP/Manual)
        for tk in list(self.active_positions.keys()):
            if tk not in live_by_ticket:
                meta = self.active_positions.pop(tk)
                prev_pos = self._last_mt5_positions.pop(tk, None)
                exit_px = prev_pos.current_price if prev_pos else meta.entry_price
                pnl_money = prev_pos.profit if prev_pos else 0.0
                if meta.side == Side.BUY:
                    pnl_r = (exit_px - meta.entry_price) / max(meta.initial_risk_price, 1e-9)
                else:
                    pnl_r = (meta.entry_price - exit_px) / max(meta.initial_risk_price, 1e-9)

                exit_reason = "TP" if pnl_r > 0.75 else ("TRAIL_BE" if meta.already_at_be and pnl_r >= -0.05 else "SL")
                closed_trades.append(
                    TradeRecord(
                        ticket=tk,
                        is_shadow=False,
                        version_id=active_version,
                        symbol=meta.symbol,
                        strategy_id=meta.strategy_id,
                        regime=meta.regime,
                        side=meta.side,
                        volume=meta.volume,
                        entry_price=meta.entry_price,
                        exit_price=exit_px,
                        sl_price=meta.sl,
                        tp_price=meta.tp,
                        open_time=meta.open_time,
                        close_time=now_sec,
                        pnl_money=round(pnl_money, 2),
                        pnl_r=round(pnl_r, 4),
                        mae_r=round(meta.mae_r, 4),
                        mfe_r=round(meta.mfe_r, 4),
                        spread_at_entry=meta.spread_at_entry,
                        slippage_points=meta.slippage_points,
                        commission=meta.commission,
                        ml_confidence=meta.ml_confidence,
                        dl_confidence=meta.dl_confidence,
                        bandit_weight=meta.bandit_weight,
                        exit_reason=exit_reason,
                        rl_exit_trace=" -> ".join(meta.rl_trace),
                    )
                )

        # 2. Enrich & track MAE/MFE on all currently open MT5 positions (including crash-recovered positions!)
        for tk, pos in live_by_ticket.items():
            meta = self.active_positions.get(tk)
            if meta is None:
                # Crash recovery: reconstruct metadata from MT5 live state + matching order comment
                rec = self._orders.get(pos.client_order_id)
                risk_dist = abs(pos.open_price - pos.sl) if pos.sl > 0 else max(pos.open_price * 0.001, 1e-5)
                meta = TrackedPositionMeta(
                    ticket=tk,
                    client_order_id=pos.client_order_id,
                    symbol=pos.symbol,
                    side=pos.side,
                    volume=pos.volume,
                    entry_price=pos.open_price,
                    sl=pos.sl,
                    tp=pos.tp,
                    initial_risk_price=max(risk_dist, 1e-9),
                    open_time=pos.open_time or now_sec,
                    strategy_id=rec.strategy_id if rec else pos.strategy_id,
                    regime=rec.regime if rec else pos.regime,
                    ml_confidence=rec.ml_confidence if rec else pos.ml_confidence,
                    dl_confidence=rec.dl_confidence if rec else pos.dl_confidence,
                    bandit_weight=rec.bandit_weight if rec else 0.25,
                    spread_at_entry=rec.spread_at_entry if rec else 0.0,
                )
                self.active_positions[tk] = meta

            cur_r = (
                (pos.current_price - meta.entry_price) / meta.initial_risk_price
                if meta.side == Side.BUY
                else (meta.entry_price - pos.current_price) / meta.initial_risk_price
            )
            meta.mae_r = min(meta.mae_r, cur_r)
            meta.mfe_r = max(meta.mfe_r, cur_r)
            meta.sl = pos.sl
            meta.tp = pos.tp
            meta.volume = pos.volume
            meta.bars_held = max(0, (now_sec - meta.open_time) // 60)

            pos.strategy_id = meta.strategy_id
            pos.regime = meta.regime
            pos.ml_confidence = meta.ml_confidence
            pos.dl_confidence = meta.dl_confidence
            pos.rl_exit_action = meta.last_rl_action
            pos.mae_r = round(meta.mae_r, 3)
            pos.mfe_r = round(meta.mfe_r, 3)
            pos.initial_risk_price = meta.initial_risk_price
            pos.bars_held = meta.bars_held
            self._last_mt5_positions[tk] = pos
            enriched.append(pos)

        return enriched, closed_trades
