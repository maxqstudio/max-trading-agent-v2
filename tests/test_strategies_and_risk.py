from __future__ import annotations

from datetime import datetime, timezone

from backend.core.config import RiskConfig, ShieldsConfig, load_trading_config
from backend.core.models import (
    AccountState,
    Bar,
    LivePosition,
    MarketRegime,
    ParameterPatch,
    Side,
    TradeSignal,
)
from backend.core.security import apply_safe_patches
from backend.risk.manager import DEFAULT_SPECS, PortfolioRiskManager
from backend.risk.shields import MarketShields
from backend.strategies.registry import StrategySuite


def test_broker_spec_position_sizing_across_gold_and_fx() -> None:
    rm = PortfolioRiskManager(RiskConfig(risk_per_trade_pct=0.50, max_risk_pct=1.0))

    # XAUUSD: equity $10,000, 0.5% risk = $50, SL distance $2.50 = 250 ticks * $1/tick + $7 comm = $257/lot -> ~0.19 lots
    xau_lot = rm.calculate_lot_from_spec(10000.0, 0.50, 2.50, DEFAULT_SPECS["XAUUSD"])
    assert 0.15 <= xau_lot <= 0.21

    # USDJPY: tick_value=$0.66, tick_size=0.001, SL=0.25 JPY = 250 ticks * $0.66 + $7 = $172/lot -> ~0.29 lots
    jpy_lot = rm.calculate_lot_from_spec(10000.0, 0.50, 0.25, DEFAULT_SPECS["USDJPY"])
    assert 0.25 <= jpy_lot <= 0.32

    # EURUSD: tick_size=0.00001, tick_value=$1, SL=0.0015 (15 pips) = 150 ticks * $1 + $7 = $157/lot -> ~0.31 lots
    eur_lot = rm.calculate_lot_from_spec(10000.0, 0.50, 0.0015, DEFAULT_SPECS["EURUSD"])
    assert 0.28 <= eur_lot <= 0.34


def test_risk_manager_blocks_dd_correlation_spread_and_kill_switch() -> None:
    rm = PortfolioRiskManager(
        RiskConfig(
            max_daily_drawdown_pct=3.0,
            max_usd_correlated_positions=2,
            max_spread_atr_ratio=0.30,
        )
    )
    sig = TradeSignal(
        symbol="GBPUSD",
        strategy_id="liquidity_sweep",
        side=Side.BUY,
        entry_price=1.27000,
        sl_price=1.26850,
        tp_price=1.27300,
        atr=0.00100,
        confidence=0.72,
        regime=MarketRegime.TRENDING_BULL,
    )
    spec = DEFAULT_SPECS["GBPUSD"]

    # 1. Daily DD lockout
    acc_dd = AccountState(balance=9650, equity=9650, day_start_equity=10000, daily_dd_pct=3.5)
    d1 = rm.evaluate_entry(sig, acc_dd, [], spec, 10.0, 0.68, 0.30, 1000)
    assert not d1.approved and d1.reason_code == "DAILY_DD_LOCKOUT"

    # 2. USD correlation cap (already holding XAUUSD + EURUSD)
    acc_ok = AccountState(balance=10000, equity=10000, day_start_equity=10000, daily_dd_pct=0.2)
    open_pos = [
        LivePosition(ticket=1, symbol="XAUUSD", side=Side.BUY, volume=0.1, open_price=2640, current_price=2641),
        LivePosition(ticket=2, symbol="EURUSD", side=Side.SELL, volume=0.2, open_price=1.08, current_price=1.08),
    ]
    d2 = rm.evaluate_entry(sig, acc_ok, open_pos, spec, 10.0, 0.68, 0.30, 1000)
    assert not d2.approved and d2.reason_code == "USD_CORRELATION_CAP"

    # 3. Spread-to-ATR gate (spread 45 pts = 0.00045 > 0.30 * 0.00100)
    d3 = rm.evaluate_entry(sig, acc_ok, [], spec, 45.0, 0.68, 0.30, 1000)
    assert not d3.approved and d3.reason_code == "SPREAD_ATR_GATE"

    # 4. Normal approval
    d4 = rm.evaluate_entry(sig, acc_ok, [], spec, 10.0, 0.68, 0.30, 1000)
    assert d4.approved and d4.volume >= spec.volume_min

    # 5. Kill switch hard block
    rm.killed = True
    d5 = rm.evaluate_entry(sig, acc_ok, [], spec, 10.0, 0.68, 0.30, 1000)
    assert not d5.approved and d5.reason_code == "KILL_SWITCH_ACTIVE"


def test_market_shields_rollover_spread_spike_shock_and_friday_flatten() -> None:
    shields = MarketShields(
        ShieldsConfig(
            rollover_blackout_enabled=True,
            rollover_start_utc="21:50",
            rollover_end_utc="23:10",
            spread_anomaly_multiplier=2.5,
            shock_bar_atr_multiplier=3.0,
            friday_flatten_enabled=True,
            friday_flatten_utc="20:45",
        )
    )
    for _ in range(25):
        shields.record_spread("XAUUSD", 20.0)

    # Normal London session
    t_normal = datetime(2026, 11, 25, 13, 0, tzinfo=timezone.utc)  # Wednesday 13:00 UTC
    assert shields.evaluate("XAUUSD", 22.0, now_utc=t_normal).allowed is True

    # Spread spike (60pt > 2.5 * 20pt)
    v_sp = shields.evaluate("XAUUSD", 60.0, now_utc=t_normal)
    assert not v_sp.allowed and v_sp.reason_code == "SPREAD_SPIKE"

    # NY Rollover blackout (22:15 UTC)
    t_roll = datetime(2026, 11, 25, 22, 15, tzinfo=timezone.utc)
    v_roll = shields.evaluate("XAUUSD", 20.0, now_utc=t_roll)
    assert not v_roll.allowed and v_roll.reason_code == "ROLLOVER_BLACKOUT"

    # Friday pre-close auto-flatten (Friday 20:50 UTC)
    t_fri = datetime(2026, 11, 27, 20, 50, tzinfo=timezone.utc)
    v_fri = shields.evaluate("XAUUSD", 20.0, now_utc=t_fri)
    assert not v_fri.allowed and v_fri.should_flatten is True

    # Shock bar cooldown
    shock_bar = Bar(symbol="XAUUSD", timestamp=int(t_normal.timestamp()), open=2640, high=2652, low=2640, close=2651)
    assert shields.record_bar_for_shock("XAUUSD", shock_bar, atr_val=3.0, now_ts=int(t_normal.timestamp())) is True
    v_shock = shields.evaluate("XAUUSD", 20.0, now_utc=t_normal)
    assert not v_shock.allowed and v_shock.reason_code == "SHOCK_COOLDOWN"


def test_hard_safety_bounds_clamp_extreme_llm_or_optimizer_patches() -> None:
    cfg = load_trading_config()
    extreme_patches = [
        ParameterPatch(path="risk.risk_per_trade_pct", old_value=0.5, new_value=50.0, rationale="Malicious/hallucinated"),
        ParameterPatch(path="app.trading_mode", old_value=0.0, new_value=99.0, rationale="Not whitelisted"),
    ]
    new_cfg, applied = apply_safe_patches(cfg, extreme_patches)
    assert new_cfg.risk.risk_per_trade_pct <= 1.50
    assert all(p.path != "app.trading_mode" for p in applied)


def test_strategy_suite_loads_all_four_scalping_strategies() -> None:
    cfg = load_trading_config()
    suite = StrategySuite.from_config(cfg)
    assert set(suite.strategies.keys()) == {
        "liquidity_sweep",
        "micro_breakout",
        "ema_vwap_pullback",
        "tick_imbalance",
    }
