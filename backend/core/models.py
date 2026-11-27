from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_ms() -> int:
    return int(utc_now().timestamp() * 1000)


class TradingMode(str, Enum):
    SIMULATOR = "SIMULATOR"
    SHADOW_ONLY = "SHADOW_ONLY"
    MT5_DEMO = "MT5_DEMO"
    MT5_LIVE = "MT5_LIVE"


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class MarketRegime(str, Enum):
    TRENDING_BULL = "TRENDING_BULL"
    TRENDING_BEAR = "TRENDING_BEAR"
    RANGING = "RANGING"
    HIGH_VOL_SPIKE = "HIGH_VOL_SPIKE"
    LOW_VOL_CHOP = "LOW_VOL_CHOP"


class OrderAction(str, Enum):
    OPEN = "OPEN"
    MODIFY = "MODIFY"
    CLOSE = "CLOSE"
    PARTIAL_CLOSE = "PARTIAL_CLOSE"
    FLATTEN_ALL = "FLATTEN_ALL"


class OrderStatus(str, Enum):
    PENDING_DISPATCH = "PENDING_DISPATCH"
    IN_FLIGHT = "IN_FLIGHT"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    DUPLICATE = "DUPLICATE"
    CLOSED = "CLOSED"


class RLExitAction(str, Enum):
    HOLD = "HOLD"
    MOVE_BE = "MOVE_BE"
    TIGHT_TRAIL = "TIGHT_TRAIL"
    PARTIAL_50 = "PARTIAL_50"
    EXIT_NOW = "EXIT_NOW"


class Bar(BaseModel):
    symbol: str
    timeframe: str = "M1"
    timestamp: int  # Unix epoch seconds (bar open time)
    open: float
    high: float
    low: float
    close: float
    volume: float = 1.0
    spread_points: float = 10.0


class SymbolSpec(BaseModel):
    symbol: str  # Logical symbol e.g. XAUUSD
    broker_symbol: str  # Broker symbol e.g. XAUUSDm
    digits: int = 5
    point: float = 0.00001
    tick_size: float = 0.00001
    tick_value: float = 1.0
    contract_size: float = 100000.0
    volume_min: float = 0.01
    volume_max: float = 100.0
    volume_step: float = 0.01
    stops_level: int = 10
    freeze_level: int = 0
    filling_mode: Literal["IOC", "FOK", "RETURN"] = "IOC"
    commission_per_lot: float = 7.0


class TickSnapshot(BaseModel):
    symbol: str
    bid: float
    ask: float
    spread_points: float
    timestamp_ms: int = Field(default_factory=utc_now_ms)


class LivePosition(BaseModel):
    ticket: int
    client_order_id: str = ""
    symbol: str
    side: Side
    volume: float
    open_price: float
    current_price: float
    sl: float = 0.0
    tp: float = 0.0
    profit: float = 0.0
    swap: float = 0.0
    commission: float = 0.0
    open_time: int = 0
    strategy_id: str = "unknown"
    regime: MarketRegime = MarketRegime.RANGING
    ml_confidence: float = 0.5
    dl_confidence: float = 0.5
    rl_exit_action: RLExitAction = RLExitAction.HOLD
    mae_r: float = 0.0
    mfe_r: float = 0.0
    initial_risk_price: float = 0.0
    bars_held: int = 0


class AccountState(BaseModel):
    balance: float = 10000.0
    equity: float = 10000.0
    margin: float = 0.0
    free_margin: float = 10000.0
    day_start_equity: float = 10000.0
    daily_pnl: float = 0.0
    daily_dd_pct: float = 0.0
    currency: str = "USD"
    is_demo: bool = True


class MQLHardGuardTelemetry(BaseModel):
    hard_max_daily_dd_pct: float = 3.5
    hard_max_lot: float = 0.50
    hard_max_positions: int = 4
    heartbeat_timeout_sec: int = 15
    order_ttl_ms: int = 1500
    ea_version: str = "2.0.0"
    synced: bool = False


class OrderCommand(BaseModel):
    client_order_id: str = Field(default_factory=lambda: f"mta_{uuid4().hex[:16]}")
    action: OrderAction
    symbol: str
    broker_symbol: str = ""
    side: Side | None = None
    volume: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    ticket: int = 0
    magic: int = 202602
    comment: str = ""
    strategy_id: str = ""
    expected_price: float = 0.0
    created_at_ms: int = Field(default_factory=utc_now_ms)
    expires_at_ms: int = Field(default_factory=lambda: utc_now_ms() + 1500)


class OrderAck(BaseModel):
    client_order_id: str
    status: OrderStatus
    ticket: int = 0
    fill_price: float = 0.0
    slippage_points: float = 0.0
    commission: float = 0.0
    error_code: int = 0
    message: str = ""
    timestamp_ms: int = Field(default_factory=utc_now_ms)


class CalendarEvent(BaseModel):
    """High-impact economic calendar event from MT5's built-in calendar (time in UTC)."""

    title: str
    currency: str
    time_utc: int  # unix seconds
    impact: str = "HIGH"


class BridgeSyncRequest(BaseModel):
    account: AccountState
    ticks: list[TickSnapshot] = Field(default_factory=list)
    closed_bars: list[Bar] = Field(default_factory=list)
    history_bars: list[Bar] = Field(default_factory=list)  # Startup backfill: stored & buffered, never traded live
    positions: list[LivePosition] = Field(default_factory=list)
    symbol_specs: list[SymbolSpec] = Field(default_factory=list)
    acks: list[OrderAck] = Field(default_factory=list)
    hard_guard: MQLHardGuardTelemetry | None = None
    timestamp_ms: int = Field(default_factory=utc_now_ms)
    # Sent by EA 2.1+: broker server time minus UTC (bar timestamps are server time), and the
    # next day of high-impact calendar events (None = not refreshed on this sync).
    server_utc_offset_sec: int | None = None
    calendar: list[CalendarEvent] | None = None


class BridgeSyncResponse(BaseModel):
    commands: list[OrderCommand] = Field(default_factory=list)
    server_time_ms: int = Field(default_factory=utc_now_ms)
    paused: bool = False
    killed: bool = False


class DLPrediction(BaseModel):
    prob_down: float = 0.33
    prob_flat: float = 0.34
    prob_up: float = 0.33
    vol_expansion: float = 0.50
    trained: bool = False


class TradeSignal(BaseModel):
    symbol: str
    strategy_id: str
    side: Side
    entry_price: float
    sl_price: float
    tp_price: float
    atr: float
    confidence: float  # Raw strategy setup quality [0, 1]
    regime: MarketRegime
    reason: str = ""
    features: dict[str, float] = Field(default_factory=dict)
    # Staged targets before tp_price (nearest first). Passing one steps the stop up.
    tp_stages: list[float] = Field(default_factory=list)
    timestamp: int = 0


class DecisionEvent(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    timestamp_ms: int = Field(default_factory=utc_now_ms)
    symbol: str
    category: Literal["EXEC", "AI", "BLOCK", "EVOLUTION", "RISK"] = "AI"
    tag: str
    strategy_id: str = ""
    side: str = ""
    message: str
    latency_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class TradeRecord(BaseModel):
    trade_id: str = Field(default_factory=lambda: f"TR-{uuid4().hex[:8].upper()}")
    ticket: int = 0
    is_shadow: bool = False
    version_id: str = "v1.0"
    symbol: str
    strategy_id: str
    regime: MarketRegime
    side: Side
    volume: float
    entry_price: float
    exit_price: float
    sl_price: float
    tp_price: float
    open_time: int
    close_time: int
    pnl_money: float
    pnl_r: float
    mae_r: float = 0.0
    mfe_r: float = 0.0
    spread_at_entry: float = 0.0
    slippage_points: float = 0.0
    commission: float = 0.0
    ml_confidence: float = 0.5
    dl_confidence: float = 0.5
    bandit_weight: float = 0.25
    exit_reason: str = "TP"
    rl_exit_trace: str = "HOLD"


class ParameterPatch(BaseModel):
    """Strictly bounded parameter patch proposed by WFO or LLM Reflection."""

    path: str  # e.g. "risk.max_spread_atr_ratio" or "strategies.liquidity_sweep.sl_atr_mult"
    old_value: float
    new_value: float
    rationale: str = ""


class ReflectionReport(BaseModel):
    report_id: str = Field(default_factory=lambda: f"REF-{uuid4().hex[:6].upper()}")
    created_at_ms: int = Field(default_factory=utc_now_ms)
    provider: str = "deterministic"
    trades_analyzed: int = 0
    finding: str
    evidence: str
    patches: list[ParameterPatch] = Field(default_factory=list)
    gate_status: Literal["PROPOSED", "PASSED_OOS", "IN_SHADOW", "PROMOTED", "REJECTED"] = "PROPOSED"


class BacktestMetrics(BaseModel):
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    expectancy_r: float = 0.0
    max_drawdown_pct: float = 0.0
    net_return_pct: float = 0.0
    avg_mae_r: float = 0.0
    avg_mfe_r: float = 0.0
    mc_p95_dd_pct: float = 0.0
    mc_ruin_prob: float = 0.0


class ChallengerCandidate(BaseModel):
    version_id: str
    parent_version_id: str
    source: Literal["WFO", "LLM_REFLECTION", "HYBRID"] = "HYBRID"
    created_at_ms: int = Field(default_factory=utc_now_ms)
    status: Literal["CANDIDATE", "REJECTED", "SHADOWING", "PENDING_APPROVAL", "PROMOTED", "ROLLED_BACK"] = "CANDIDATE"
    patches: list[ParameterPatch] = Field(default_factory=list)
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    oos_metrics: BacktestMetrics = Field(default_factory=BacktestMetrics)
    champion_oos_metrics: BacktestMetrics = Field(default_factory=BacktestMetrics)
    shadow_trades: int = 0
    shadow_pnl_r: float = 0.0
    shadow_pf: float = 0.0
    shadow_max_dd_pct: float = 0.0
    rejection_reason: str = ""
    reflection_id: str | None = None


class ChampionSnapshot(BaseModel):
    version_id: str
    created_at_ms: int = Field(default_factory=utc_now_ms)
    status: Literal["ACTIVE", "ARCHIVED", "AUTO_ROLLED_BACK"] = "ACTIVE"
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    oos_metrics: BacktestMetrics = Field(default_factory=BacktestMetrics)
    note: str = ""
    live_trades_since_promo: int = 0
    live_peak_equity: float = 10000.0
    live_max_dd_pct: float = 0.0
