from __future__ import annotations

from backend.bridge.state_machine import IdempotentOrderStateMachine
from backend.core.models import (
    LivePosition,
    MarketRegime,
    OrderAck,
    OrderAction,
    OrderCommand,
    OrderStatus,
    Side,
    utc_now_ms,
)
from backend.core.security import verify_bridge_token
from backend.risk.manager import DEFAULT_SPECS


def test_idempotent_state_machine_blocks_duplicates_and_expired_orders() -> None:
    sm = IdempotentOrderStateMachine(order_ttl_ms=1500)
    now = utc_now_ms()

    cmd1 = OrderCommand(
        client_order_id="ord_unique_001",
        action=OrderAction.OPEN,
        symbol="XAUUSD",
        side=Side.BUY,
        volume=0.10,
        sl=2640.0,
        tp=2650.0,
        expected_price=2643.0,
        created_at_ms=now,
        expires_at_ms=now + 1500,
    )
    assert sm.enqueue(cmd1, strategy_id="liquidity_sweep", regime=MarketRegime.TRENDING_BULL) is True
    # Duplicate client_order_id MUST be rejected
    assert sm.enqueue(cmd1) is False

    # Second concurrent OPEN on same symbol while first is inflight MUST be rejected
    cmd_dup_sym = OrderCommand(
        client_order_id="ord_unique_002",
        action=OrderAction.OPEN,
        symbol="XAUUSD",
        side=Side.BUY,
        volume=0.10,
        expires_at_ms=now + 1500,
    )
    assert sm.enqueue(cmd_dup_sym) is False

    # Stale expired order on another symbol MUST be dropped before dispatch
    cmd_expired = OrderCommand(
        client_order_id="ord_stale_003",
        action=OrderAction.OPEN,
        symbol="EURUSD",
        side=Side.SELL,
        volume=0.10,
        expires_at_ms=now - 500,
    )
    assert sm.enqueue(cmd_expired) is True

    dispatched = sm.pop_commands_for_dispatch(now_ms=now + 100)
    assert len(dispatched) == 1
    assert dispatched[0].client_order_id == "ord_unique_001"


def test_reconciliation_tracks_mae_mfe_and_detects_closed_trades_and_crash_recovery() -> None:
    sm = IdempotentOrderStateMachine()
    now = utc_now_ms()
    cmd = OrderCommand(
        client_order_id="ord_live_100",
        action=OrderAction.OPEN,
        symbol="XAUUSD",
        side=Side.BUY,
        volume=0.10,
        sl=2640.0,
        tp=2650.0,
        expected_price=2644.0,
        expires_at_ms=now + 3000,
    )
    sm.enqueue(cmd, strategy_id="micro_breakout", regime=MarketRegime.TRENDING_BULL, ml_confidence=0.71)
    sm.pop_commands_for_dispatch(now)
    sm.process_acks([OrderAck(client_order_id="ord_live_100", status=OrderStatus.FILLED, ticket=9001, fill_price=2644.0)])

    # Position moves favorably then closes externally on MT5
    pos_open = LivePosition(
        ticket=9001,
        client_order_id="ord_live_100",
        symbol="XAUUSD",
        side=Side.BUY,
        volume=0.10,
        open_price=2644.0,
        current_price=2648.0,
        sl=2640.0,
        tp=2650.0,
        profit=40.0,
    )
    enriched, closed1 = sm.reconcile_positions([pos_open], DEFAULT_SPECS, "v1.0", int(now // 1000) + 120)
    assert len(enriched) == 1
    assert len(closed1) == 0
    assert enriched[0].mfe_r >= 0.99

    # Crash recovery: simulate unknown ticket already open on MT5 after restart
    orphan_pos = LivePosition(
        ticket=7777,
        symbol="EURUSD",
        side=Side.SELL,
        volume=0.20,
        open_price=1.08500,
        current_price=1.08420,
        sl=1.08600,
        tp=1.08300,
        profit=16.0,
    )
    enriched2, closed2 = sm.reconcile_positions([orphan_pos], DEFAULT_SPECS, "v1.0", int(now // 1000) + 180)
    assert len(enriched2) == 1 and enriched2[0].ticket == 7777
    assert len(closed2) == 1 and closed2[0].ticket == 9001
    assert closed2[0].pnl_r > 0.9


def test_bridge_token_verification() -> None:
    assert verify_bridge_token("secret123", "secret123") is True
    assert verify_bridge_token("wrong", "secret123") is False
    assert verify_bridge_token(None, "secret123") is False
