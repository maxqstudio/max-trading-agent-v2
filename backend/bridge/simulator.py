from __future__ import annotations

import math
import time

import numpy as np

from backend.core.models import (
    AccountState,
    Bar,
    BridgeSyncRequest,
    LivePosition,
    MQLHardGuardTelemetry,
    OrderAck,
    OrderAction,
    OrderCommand,
    OrderStatus,
    Side,
    SymbolSpec,
    TickSnapshot,
    utc_now_ms,
)
from backend.risk.manager import DEFAULT_SPECS


class SimulatedMT5Bridge:
    """Realistic multi-regime MT5 Broker Simulator for offline testing, demos, and verification.

    Generates correlated XAUUSD & Major Forex ticks/M1 bars with regime switching,
    executes OrderCommands with realistic Bid/Ask spread, slippage & broker commissions,
    and simulates SL/TP intrabar triggers identically to the real MQL5 EA.
    """

    def __init__(
        self,
        symbols: list[str] | None = None,
        initial_balance: float = 10000.0,
        seed: int = 42,
    ) -> None:
        self.symbols = symbols or ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY"]
        self.rng = np.random.default_rng(seed)
        self.balance = initial_balance
        self.day_start_equity = initial_balance
        self.specs: dict[str, SymbolSpec] = {
            s: DEFAULT_SPECS.get(s, DEFAULT_SPECS["XAUUSD"]).model_copy(deep=True)
            for s in self.symbols
        }
        self._prices: dict[str, float] = {
            "XAUUSD": 2648.00,
            "EURUSD": 1.08420,
            "GBPUSD": 1.26900,
            "USDJPY": 151.400,
        }
        self._spreads: dict[str, float] = {
            "XAUUSD": 20.0,
            "EURUSD": 8.0,
            "GBPUSD": 12.0,
            "USDJPY": 10.0,
        }
        self._regime_drift: dict[str, float] = {s: 0.0 for s in self.symbols}
        self._regime_vol: dict[str, float] = {s: 1.0 for s in self.symbols}
        self._bar_open: dict[str, float] = {}
        self._bar_high: dict[str, float] = {}
        self._bar_low: dict[str, float] = {}
        self._bar_vol: dict[str, float] = {}
        self._tick_step: int = 0
        self._sim_time_sec: int = int(time.time()) - 320 * 60
        self._next_ticket: int = 884000
        self._positions: dict[int, LivePosition] = {}
        self._pending_acks: list[OrderAck] = []
        self._seen_order_ids: set[str] = set()
        self._specs_sent: bool = False

    def _base_step(self, sym: str) -> float:
        return {"XAUUSD": 0.45, "EURUSD": 0.00018, "GBPUSD": 0.00022, "USDJPY": 0.022}.get(sym, 0.0002)

    def _maybe_switch_regime(self, sym: str) -> None:
        if self.rng.random() < 0.035:
            mode = int(self.rng.integers(0, 4))
            if mode == 0:  # Trending bull
                self._regime_drift[sym] = 0.24
                self._regime_vol[sym] = 1.15
            elif mode == 1:  # Trending bear
                self._regime_drift[sym] = -0.24
                self._regime_vol[sym] = 1.15
            elif mode == 2:  # Ranging
                self._regime_drift[sym] = 0.0
                self._regime_vol[sym] = 0.85
            else:  # High-vol impulse
                self._regime_drift[sym] = float(self.rng.choice([-0.32, 0.32]))
                self._regime_vol[sym] = 1.75

    def generate_bootstrap_bars(self, count: int = 280) -> dict[str, list[Bar]]:
        """Generate realistic multi-regime historical M1 bars for instant AI warmup & WFO."""
        out: dict[str, list[Bar]] = {s: [] for s in self.symbols}
        for _ in range(count):
            self._sim_time_sec += 60
            for sym in self.symbols:
                self._maybe_switch_regime(sym)
                spec = self.specs[sym]
                step = self._base_step(sym) * self._regime_vol[sym]
                o = self._prices[sym]
                # Simulate 4 sub-steps inside each M1 candle to produce realistic wicks & sweeps
                sub = [o]
                for _k in range(4):
                    nxt = sub[-1] + (float(self.rng.normal(self._regime_drift[sym], 1.0)) * step * 0.55)
                    sub.append(nxt)
                c = round(sub[-1], spec.digits)
                h = round(max(sub) + abs(float(self.rng.normal(0, 0.25))) * step, spec.digits)
                l = round(min(sub) - abs(float(self.rng.normal(0, 0.25))) * step, spec.digits)
                self._prices[sym] = c
                sp = max(4.0, round(self._spreads[sym] + float(self.rng.normal(0, 2.0)), 1))
                self._spreads[sym] = sp
                out[sym].append(
                    Bar(
                        symbol=sym,
                        timeframe="M1",
                        timestamp=self._sim_time_sec,
                        open=round(o, spec.digits),
                        high=h,
                        low=l,
                        close=c,
                        volume=float(int(self.rng.integers(35, 220))),
                        spread_points=sp,
                    )
                )
        for sym in self.symbols:
            p = self._prices[sym]
            self._bar_open[sym] = p
            self._bar_high[sym] = p
            self._bar_low[sym] = p
            self._bar_vol[sym] = 10.0
        self._sim_time_sec = int(time.time())
        return out

    def step_and_build_sync(self, close_bar_every: int = 8) -> BridgeSyncRequest:
        """Advance live ticks one step, evaluate open position SL/TP, and emit BridgeSyncRequest."""
        self._tick_step += 1
        now_ms = utc_now_ms()
        now_sec = int(now_ms // 1000)
        ticks: list[TickSnapshot] = []
        closed_bars: list[Bar] = []

        for sym in self.symbols:
            self._maybe_switch_regime(sym)
            spec = self.specs[sym]
            step = self._base_step(sym) * self._regime_vol[sym] * 0.45
            delta = float(self.rng.normal(self._regime_drift[sym] * 0.35, 1.0)) * step
            bid = round(max(spec.point * 100, self._prices[sym] + delta), spec.digits)
            self._prices[sym] = bid
            sp = max(5.0, min(65.0, round(self._spreads[sym] + float(self.rng.normal(0, 1.4)), 1)))
            self._spreads[sym] = sp
            ask = round(bid + sp * spec.point, spec.digits)

            ticks.append(TickSnapshot(symbol=sym, bid=bid, ask=ask, spread_points=sp, timestamp_ms=now_ms))

            if sym not in self._bar_open:
                self._bar_open[sym] = bid
                self._bar_high[sym] = bid
                self._bar_low[sym] = bid
                self._bar_vol[sym] = 1.0
            else:
                self._bar_high[sym] = max(self._bar_high[sym], bid)
                self._bar_low[sym] = min(self._bar_low[sym], bid)
                self._bar_vol[sym] += float(self.rng.integers(5, 25))

            if self._tick_step % close_bar_every == 0:
                closed_bars.append(
                    Bar(
                        symbol=sym,
                        timeframe="M1",
                        timestamp=now_sec,
                        open=self._bar_open[sym],
                        high=self._bar_high[sym],
                        low=self._bar_low[sym],
                        close=bid,
                        volume=self._bar_vol[sym],
                        spread_points=sp,
                    )
                )
                self._bar_open[sym] = bid
                self._bar_high[sym] = bid
                self._bar_low[sym] = bid
                self._bar_vol[sym] = 1.0

        # Update floating P&L and check SL/TP hits on open simulated positions
        floating_total = 0.0
        for tk in list(self._positions.keys()):
            pos = self._positions[tk]
            spec = self.specs[pos.symbol]
            bid = self._prices[pos.symbol]
            ask = round(bid + self._spreads[pos.symbol] * spec.point, spec.digits)
            cur_px = bid if pos.side == Side.BUY else ask
            pos.current_price = cur_px

            diff = (cur_px - pos.open_price) if pos.side == Side.BUY else (pos.open_price - cur_px)
            ticks_pnl = diff / spec.tick_size
            pos.profit = round(ticks_pnl * spec.tick_value * pos.volume - pos.commission, 2)

            hit_sl = (pos.sl > 0) and ((pos.side == Side.BUY and cur_px <= pos.sl) or (pos.side == Side.SELL and cur_px >= pos.sl))
            hit_tp = (pos.tp > 0) and ((pos.side == Side.BUY and cur_px >= pos.tp) or (pos.side == Side.SELL and cur_px <= pos.tp))
            if hit_sl or hit_tp:
                self.balance = round(self.balance + pos.profit, 2)
                self._positions.pop(tk, None)
            else:
                floating_total += pos.profit

        equity = round(self.balance + floating_total, 2)
        dd_pct = max(0.0, ((self.day_start_equity - equity) / max(self.day_start_equity, 1.0)) * 100.0)

        acks = list(self._pending_acks)
        self._pending_acks.clear()
        send_specs = list(self.specs.values()) if not self._specs_sent else []
        self._specs_sent = True

        return BridgeSyncRequest(
            account=AccountState(
                balance=self.balance,
                equity=equity,
                free_margin=max(0.0, equity - len(self._positions) * 200.0),
                day_start_equity=self.day_start_equity,
                daily_pnl=round(equity - self.day_start_equity, 2),
                daily_dd_pct=round(dd_pct, 4),
                is_demo=True,
            ),
            ticks=ticks,
            closed_bars=closed_bars,
            positions=[p.model_copy(deep=True) for p in self._positions.values()],
            symbol_specs=send_specs,
            acks=acks,
            hard_guard=MQLHardGuardTelemetry(synced=True, ea_version="2.0.0-SIM"),
            timestamp_ms=now_ms,
        )

    def execute_commands(self, commands: list[OrderCommand]) -> None:
        now_ms = utc_now_ms()
        for cmd in commands:
            if cmd.client_order_id in self._seen_order_ids:
                self._pending_acks.append(
                    OrderAck(client_order_id=cmd.client_order_id, status=OrderStatus.DUPLICATE, message="Duplicate ID")
                )
                continue
            self._seen_order_ids.add(cmd.client_order_id)

            if cmd.action == OrderAction.FLATTEN_ALL:
                for tk in list(self._positions.keys()):
                    pos = self._positions.pop(tk)
                    self.balance = round(self.balance + pos.profit, 2)
                self._pending_acks.append(
                    OrderAck(client_order_id=cmd.client_order_id, status=OrderStatus.FILLED, message="Flattened all")
                )
                continue

            if cmd.action == OrderAction.CLOSE:
                pos = self._positions.pop(cmd.ticket, None)
                if pos:
                    self.balance = round(self.balance + pos.profit, 2)
                    self._pending_acks.append(
                        OrderAck(client_order_id=cmd.client_order_id, status=OrderStatus.FILLED, ticket=cmd.ticket, fill_price=pos.current_price)
                    )
                else:
                    self._pending_acks.append(
                        OrderAck(client_order_id=cmd.client_order_id, status=OrderStatus.REJECTED, ticket=cmd.ticket, message="Not found")
                    )
                continue

            if cmd.action == OrderAction.PARTIAL_CLOSE:
                pos = self._positions.get(cmd.ticket)
                if pos and pos.volume >= 0.02:
                    half = round(math.floor((pos.volume * 0.5) / 0.01) * 0.01, 2)
                    if half >= 0.01:
                        realized = round(pos.profit * (half / pos.volume), 2)
                        self.balance = round(self.balance + realized, 2)
                        pos.volume = round(pos.volume - half, 2)
                        self._pending_acks.append(
                            OrderAck(client_order_id=cmd.client_order_id, status=OrderStatus.FILLED, ticket=cmd.ticket, fill_price=pos.current_price)
                        )
                continue

            if cmd.action == OrderAction.MODIFY:
                pos = self._positions.get(cmd.ticket)
                if pos:
                    if cmd.sl > 0:
                        pos.sl = cmd.sl
                    if cmd.tp > 0:
                        pos.tp = cmd.tp
                    self._pending_acks.append(
                        OrderAck(client_order_id=cmd.client_order_id, status=OrderStatus.FILLED, ticket=cmd.ticket)
                    )
                continue

            if cmd.action == OrderAction.OPEN:
                if now_ms > cmd.expires_at_ms:
                    self._pending_acks.append(
                        OrderAck(client_order_id=cmd.client_order_id, status=OrderStatus.EXPIRED, message="Expired TTL")
                    )
                    continue
                spec = self.specs.get(cmd.symbol, DEFAULT_SPECS["XAUUSD"])
                bid = self._prices.get(cmd.symbol, cmd.expected_price)
                ask = round(bid + self._spreads.get(cmd.symbol, 12.0) * spec.point, spec.digits)
                slip_pts = round(float(self.rng.uniform(0.5, 2.5)), 2)
                fill_px = (
                    round(ask + slip_pts * spec.point, spec.digits)
                    if cmd.side == Side.BUY
                    else round(bid - slip_pts * spec.point, spec.digits)
                )
                comm = round(spec.commission_per_lot * cmd.volume, 2)
                self._next_ticket += 1
                tk = self._next_ticket
                self._positions[tk] = LivePosition(
                    ticket=tk,
                    client_order_id=cmd.client_order_id,
                    symbol=cmd.symbol,
                    side=cmd.side or Side.BUY,
                    volume=cmd.volume,
                    open_price=fill_px,
                    current_price=fill_px,
                    sl=cmd.sl,
                    tp=cmd.tp,
                    commission=comm,
                    open_time=int(now_ms // 1000),
                    strategy_id=cmd.strategy_id,
                )
                self._pending_acks.append(
                    OrderAck(
                        client_order_id=cmd.client_order_id,
                        status=OrderStatus.FILLED,
                        ticket=tk,
                        fill_price=fill_px,
                        slippage_points=slip_pts,
                        commission=comm,
                        message="Simulated MT5 fill",
                    )
                )
