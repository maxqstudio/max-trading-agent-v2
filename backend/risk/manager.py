from __future__ import annotations

import math
from dataclasses import dataclass

from backend.core.config import RiskConfig
from backend.core.models import AccountState, LivePosition, Side, SymbolSpec, TradeSignal


@dataclass(slots=True)
class RiskDecision:
    approved: bool
    volume: float = 0.0
    adjusted_sl: float = 0.0
    adjusted_tp: float = 0.0
    risk_pct_used: float = 0.0
    reason_code: str = "APPROVED"
    detail: str = ""


DEFAULT_SPECS: dict[str, SymbolSpec] = {
    "XAUUSD": SymbolSpec(
        symbol="XAUUSD",
        broker_symbol="XAUUSD",
        digits=2,
        point=0.01,
        tick_size=0.01,
        tick_value=1.0,
        contract_size=100.0,
        volume_min=0.01,
        volume_max=50.0,
        volume_step=0.01,
        stops_level=15,
        filling_mode="IOC",
        commission_per_lot=7.0,
    ),
    "EURUSD": SymbolSpec(
        symbol="EURUSD",
        broker_symbol="EURUSD",
        digits=5,
        point=0.00001,
        tick_size=0.00001,
        tick_value=1.0,
        contract_size=100000.0,
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
        stops_level=5,
        filling_mode="IOC",
        commission_per_lot=7.0,
    ),
    "GBPUSD": SymbolSpec(
        symbol="GBPUSD",
        broker_symbol="GBPUSD",
        digits=5,
        point=0.00001,
        tick_size=0.00001,
        tick_value=1.0,
        contract_size=100000.0,
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
        stops_level=8,
        filling_mode="IOC",
        commission_per_lot=7.0,
    ),
    "USDJPY": SymbolSpec(
        symbol="USDJPY",
        broker_symbol="USDJPY",
        digits=3,
        point=0.001,
        tick_size=0.001,
        tick_value=0.66,
        contract_size=100000.0,
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
        stops_level=8,
        filling_mode="IOC",
        commission_per_lot=7.0,
    ),
}


class PortfolioRiskManager:
    """Institutional broker-spec position sizer and multi-layer portfolio circuit breaker."""

    def __init__(self, config: RiskConfig) -> None:
        self.config = config
        self.paused: bool = False
        self.killed: bool = False
        self.consecutive_losses: int = 0
        self.cooldown_until_ts: int = 0
        self._rolling_win_rate: float = 0.54
        self._rolling_payoff: float = 1.45

    def update_performance_stats(self, win_rate: float, payoff_ratio: float) -> None:
        self._rolling_win_rate = max(0.25, min(0.80, float(win_rate)))
        self._rolling_payoff = max(0.60, min(3.50, float(payoff_ratio)))

    def record_trade_outcome(self, pnl_money: float, now_ts: int) -> None:
        if pnl_money < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.config.consecutive_loss_limit:
                self.cooldown_until_ts = now_ts + self.config.consecutive_loss_cooldown_min * 60
        else:
            self.consecutive_losses = 0

    def compute_dynamic_risk_pct(self, ml_probability: float, bandit_weight: float) -> float:
        base = self.config.risk_per_trade_pct
        if self.config.half_kelly_enabled:
            p = 0.5 * self._rolling_win_rate + 0.5 * max(0.40, min(0.85, ml_probability))
            b = self._rolling_payoff
            kelly = (p * b - (1.0 - p)) / max(b, 0.25)
            half_kelly_pct = max(0.0, 0.5 * kelly * 100.0)
            # Blend base risk with half-Kelly gently scaled by bandit weight
            weight_scale = 0.75 + 0.5 * min(1.0, max(0.0, bandit_weight * 2.0))
            target = min(base * weight_scale, max(self.config.min_risk_pct, half_kelly_pct * 0.15))
        else:
            target = base
        return round(max(self.config.min_risk_pct, min(self.config.max_risk_pct, target)), 4)

    def calculate_lot_from_spec(
        self,
        equity: float,
        risk_pct: float,
        sl_distance_price: float,
        spec: SymbolSpec,
        commission_per_lot: float | None = None,
    ) -> float:
        """Compute exact broker-compliant lot size strictly from live tick_value and tick_size."""
        if equity <= 0 or sl_distance_price <= 0 or spec.tick_size <= 0 or spec.tick_value <= 0:
            return 0.0
        risk_money = equity * (risk_pct / 100.0)
        ticks_at_risk = sl_distance_price / spec.tick_size
        comm = spec.commission_per_lot if commission_per_lot is None else max(0.0, float(commission_per_lot))
        loss_per_lot = ticks_at_risk * spec.tick_value + comm
        if loss_per_lot <= 0:
            return 0.0
        raw_lots = risk_money / loss_per_lot
        step = max(spec.volume_step, 0.001)
        floored = math.floor(raw_lots / step) * step
        if floored < spec.volume_min:
            # Strict cap: never allow minimum lot to exceed configured max_risk_pct
            min_lot_risk_pct = ((spec.volume_min * loss_per_lot) / equity) * 100.0
            if min_lot_risk_pct <= self.config.max_risk_pct:
                return round(spec.volume_min, 4)
            return 0.0
        return round(min(floored, spec.volume_max), 4)

    def evaluate_entry(
        self,
        signal: TradeSignal,
        account: AccountState,
        open_positions: list[LivePosition],
        spec: SymbolSpec,
        spread_points: float,
        ml_probability: float,
        bandit_weight: float,
        now_ts: int,
        commission_per_lot: float | None = None,
    ) -> RiskDecision:
        if self.killed:
            return RiskDecision(False, reason_code="KILL_SWITCH_ACTIVE", detail="Global Kill-Switch locked")
        if self.paused:
            return RiskDecision(False, reason_code="ENGINE_PAUSED", detail="New entries paused by operator")

        # Daily drawdown guard
        dd_pct = account.daily_dd_pct
        if dd_pct <= 0 and account.day_start_equity > 0:
            dd_pct = max(0.0, ((account.day_start_equity - account.equity) / account.day_start_equity) * 100.0)
        if dd_pct >= self.config.max_daily_drawdown_pct:
            return RiskDecision(
                False,
                reason_code="DAILY_DD_LOCKOUT",
                detail=f"Daily DD {dd_pct:.2f}% >= limit {self.config.max_daily_drawdown_pct:.2f}%",
            )

        # Consecutive loss cooldown
        if now_ts < self.cooldown_until_ts:
            rem = self.cooldown_until_ts - now_ts
            return RiskDecision(False, reason_code="LOSS_COOLDOWN", detail=f"Consecutive loss cooldown ({rem}s left)")

        # Max concurrent positions
        if len(open_positions) >= self.config.max_concurrent_positions:
            return RiskDecision(
                False,
                reason_code="MAX_POSITIONS",
                detail=f"Open slots full ({len(open_positions)}/{self.config.max_concurrent_positions})",
            )

        # Prevent duplicate symbol position stacking
        if any(p.symbol == signal.symbol for p in open_positions):
            return RiskDecision(
                False,
                reason_code="SYMBOL_ALREADY_OPEN",
                detail=f"Existing open position on {signal.symbol}",
            )

        # USD currency correlation cap (XAUUSD, EURUSD, GBPUSD, USDJPY all expose USD)
        if "USD" in signal.symbol:
            usd_open = sum(1 for p in open_positions if "USD" in p.symbol)
            if usd_open >= self.config.max_usd_correlated_positions:
                return RiskDecision(
                    False,
                    reason_code="USD_CORRELATION_CAP",
                    detail=f"USD correlated positions ({usd_open}/{self.config.max_usd_correlated_positions})",
                )

        # Spread-to-ATR gate (critical for M1-M5 scalping!)
        spread_price = spread_points * spec.point
        spread_atr = spread_price / max(signal.atr, 1e-9)
        if spread_atr > self.config.max_spread_atr_ratio:
            return RiskDecision(
                False,
                reason_code="SPREAD_ATR_GATE",
                detail=f"Spread/ATR {spread_atr:.2f} > max {self.config.max_spread_atr_ratio:.2f}",
            )

        # Enforce broker stops_level + spread buffer on SL & TP
        min_stop_dist = max(spec.stops_level * spec.point + spread_price * 1.2, spec.point * 10)
        entry = signal.entry_price
        if signal.side == Side.BUY:
            sl_dist = max(entry - signal.sl_price, min_stop_dist)
            tp_dist = max(signal.tp_price - entry, sl_dist * self.config.min_rr_ratio)
            adj_sl = round(entry - sl_dist, spec.digits)
            adj_tp = round(entry + tp_dist, spec.digits)
        else:
            sl_dist = max(signal.sl_price - entry, min_stop_dist)
            tp_dist = max(entry - signal.tp_price, sl_dist * self.config.min_rr_ratio)
            adj_sl = round(entry + sl_dist, spec.digits)
            adj_tp = round(entry - tp_dist, spec.digits)

        # Total Cost-to-Risk Gate: prevents tight M1 stops where broker commission + spread eat 30-50%+ of risk!
        eff_comm = spec.commission_per_lot if commission_per_lot is None else max(0.0, float(commission_per_lot))
        money_per_px = (spec.tick_value / spec.tick_size) if spec.tick_size > 0 else 0.0
        comm_px = (eff_comm / money_per_px) if money_per_px > 0 else 0.0
        cost_to_risk = (spread_price + comm_px) / max(sl_dist, 1e-12)
        if self.config.max_cost_to_risk_ratio > 0 and cost_to_risk > self.config.max_cost_to_risk_ratio:
            return RiskDecision(
                False,
                reason_code="COST_TO_RISK_GATE",
                detail=f"Spread+commission is {cost_to_risk:.0%} of SL risk > max {self.config.max_cost_to_risk_ratio:.0%} (SL too tight for broker costs)",
            )

        risk_pct = self.compute_dynamic_risk_pct(ml_probability, bandit_weight)
        lots = self.calculate_lot_from_spec(account.equity, risk_pct, sl_dist, spec, commission_per_lot=eff_comm)
        if lots <= 0:
            return RiskDecision(
                False,
                reason_code="LOT_TOO_SMALL",
                detail="Calculated lot below broker minimum within safe risk budget",
            )

        return RiskDecision(
            approved=True,
            volume=lots,
            adjusted_sl=adj_sl,
            adjusted_tp=adj_tp,
            risk_pct_used=risk_pct,
            reason_code="APPROVED",
            detail=f"Lot {lots:.2f} | Risk {risk_pct:.2f}% | Spread/ATR {spread_atr:.2f} | Cost/SL {cost_to_risk:.0%}",
        )
