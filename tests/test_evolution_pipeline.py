from __future__ import annotations

import numpy as np

from backend.bridge.simulator import SimulatedMT5Bridge
from backend.core.config import EnvSettings, load_trading_config
from backend.core.models import BacktestMetrics, MarketRegime, ParameterPatch, RLExitAction, Side, TradeRecord
from backend.evolution.backtester import CausalBacktester
from backend.evolution.bandit import ContextualBanditRL
from backend.evolution.champion_gate import ChampionChallengerGate
from backend.evolution.llm_reflector import LLMReflectionAnalyst
from backend.llm.router import LLMRouter
from backend.evolution.meta_labeler import CalibratedMetaLabeler
from backend.evolution.monte_carlo import run_monte_carlo_stress
from backend.evolution.rl_exit_agent import RLExitAgent


def test_contextual_bandit_converges_to_winning_strategy() -> None:
    bandit = ContextualBanditRL(forgetting_factor=0.99)
    for _ in range(25):
        bandit.update("XAUUSD", MarketRegime.RANGING, "liquidity_sweep", +1.6)
        bandit.update("XAUUSD", MarketRegime.RANGING, "micro_breakout", -1.0)
    weights = bandit.get_weights("XAUUSD", MarketRegime.RANGING, explore=False)
    assert weights["liquidity_sweep"] > weights["micro_breakout"] * 1.8


def test_rl_exit_agent_respects_trigger_masks_and_learns() -> None:
    agent = RLExitAgent(be_trigger_r=0.80, trail_trigger_r=1.25)
    # Below BE trigger: cannot choose MOVE_BE or TIGHT_TRAIL
    act_low = agent.select_action(0.30, bars_held=2, regime=MarketRegime.TRENDING_BULL, dl_aligned=True)
    assert act_low == RLExitAction.HOLD
    # Above BE trigger: prefers MOVE_BE
    act_be = agent.select_action(0.95, bars_held=3, regime=MarketRegime.TRENDING_BULL, dl_aligned=True)
    assert act_be == RLExitAction.MOVE_BE
    # Above trail trigger & already at BE: prefers TIGHT_TRAIL or PARTIAL_50
    act_tr = agent.select_action(1.50, bars_held=5, regime=MarketRegime.TRENDING_BULL, dl_aligned=True, already_at_be=True)
    assert act_tr in (RLExitAction.TIGHT_TRAIL, RLExitAction.PARTIAL_50)

    agent.update_from_closed_trade(1.6, 1.3, 5, MarketRegime.TRENDING_BULL, True, RLExitAction.TIGHT_TRAIL)
    assert agent.avg_mfe_capture_pct > 50.0


def test_backtester_monte_carlo_and_calibrated_ml() -> None:
    cfg = load_trading_config()
    sim = SimulatedMT5Bridge(symbols=["XAUUSD", "EURUSD"], seed=55)
    bars_map = sim.generate_bootstrap_bars(count=260)

    bt = CausalBacktester()
    metrics, trades, (mx, my) = bt.run(bars_map, cfg, collect_meta_samples=True)
    assert metrics.total_trades >= 1
    assert len(mx) == len(trades)

    mc = run_monte_carlo_stress([t.pnl_r for t in trades] + [1.2, -1.0, 1.5, -0.8, 1.1], iterations=500)
    assert mc.iterations == 500
    assert mc.p95_max_dd_pct >= mc.p50_max_dd_pct

    # Synthetic 80-sample check for calibrated ML Meta-Labeler
    rng = np.random.default_rng(7)
    x_syn = rng.normal(0, 1, size=(80, 24))
    y_syn = (x_syn[:, 20] - 0.5 * x_syn[:, 17] > 0).astype(int)
    ml = CalibratedMetaLabeler()
    res = ml.fit_purged_walk_forward(x_syn, y_syn)
    assert res["trained"] == 1.0
    prob = ml.predict_win_probability(x_syn[0])
    assert 0.0 <= prob <= 1.0


def test_deterministic_reflector_and_champion_gate_promote_and_auto_rollback() -> None:
    cfg = load_trading_config()
    cfg.evolution.shadow_min_trades = 4
    cfg.evolution.rollback_live_dd_pct = 1.50

    # 1. Reflector detects wide-spread losses & high MFE reversals
    trades = [
        TradeRecord(
            symbol="XAUUSD",
            strategy_id="liquidity_sweep",
            regime=MarketRegime.RANGING,
            side=Side.BUY,
            volume=0.1,
            entry_price=2640,
            exit_price=2638,
            sl_price=2638,
            tp_price=2644,
            open_time=1000,
            close_time=1200,
            pnl_money=-40,
            pnl_r=-1.0,
            mae_r=-1.0,
            mfe_r=0.85,
            spread_at_entry=34.0,
        )
        for _ in range(8)
    ] + [
        TradeRecord(
            symbol="XAUUSD",
            strategy_id="liquidity_sweep",
            regime=MarketRegime.RANGING,
            side=Side.BUY,
            volume=0.1,
            entry_price=2640,
            exit_price=2644,
            sl_price=2638,
            tp_price=2644,
            open_time=1300,
            close_time=1500,
            pnl_money=60,
            pnl_r=1.5,
            mae_r=-0.25,
            mfe_r=1.6,
            spread_at_entry=16.0,
        )
        for _ in range(4)
    ]

    reflector = LLMReflectionAnalyst(EnvSettings(_env_file=None), router=LLMRouter(resolved=[]))
    report = reflector.deterministic_reflect(trades, cfg)
    assert len(report.patches) >= 1

    # A healthy journal must NOT force a parameter change (tuning on noise = overfitting)
    healthy = [t for t in trades if t.pnl_r > 0] * 3
    calm = reflector.deterministic_reflect(healthy, cfg)
    assert calm.patches == []
    assert "no parameter change" in calm.finding

    # 2. Champion Gate: reject weak candidate, accept strong candidate -> Shadow Arena -> Promote -> Auto-Rollback
    gate = ChampionChallengerGate(cfg, initial_equity=10000.0)
    weak = BacktestMetrics(total_trades=10, profit_factor=0.9, sharpe_ratio=-0.5, expectancy_r=-0.1, max_drawdown_pct=5.0, mc_p95_dd_pct=8.0)
    strong = BacktestMetrics(total_trades=35, profit_factor=1.65, sharpe_ratio=2.1, expectancy_r=0.35, max_drawdown_pct=1.4, mc_p95_dd_pct=2.5, mc_ruin_prob=0.0)
    base = BacktestMetrics(total_trades=35, profit_factor=1.30, sharpe_ratio=1.2, expectancy_r=0.15, max_drawdown_pct=2.0, mc_p95_dd_pct=3.5)

    rej = gate.submit_candidate(cfg, [ParameterPatch(path="ai.ml_min_probability", old_value=0.56, new_value=0.60)], weak, base)
    assert rej.status == "REJECTED"

    cand = gate.submit_candidate(cfg, [ParameterPatch(path="ai.ml_min_probability", old_value=0.56, new_value=0.60)], strong, base)
    assert cand.status == "SHADOWING"

    # Feed 4 winning shadow trades -> PENDING_APPROVAL
    transition = None
    for idx in range(4):
        transition = gate.record_shadow_trade(
            TradeRecord(
                is_shadow=True,
                symbol="XAUUSD",
                strategy_id="liquidity_sweep",
                regime=MarketRegime.RANGING,
                side=Side.BUY,
                volume=0.1,
                entry_price=2640,
                exit_price=2644,
                sl_price=2638,
                tp_price=2644,
                open_time=2000 + idx * 60,
                close_time=2050 + idx * 60,
                pnl_money=50,
                pnl_r=1.2,
            )
        )
    assert transition == "PENDING_APPROVAL"

    promoted = gate.promote_active_challenger(note="Test promo", current_equity=10200.0)
    assert promoted is not None and gate.active_snapshot.version_id == "v2.3"

    # Trigger Stage-4 Auto-Rollback on >1.5% post-promotion drawdown (10200 -> 10020 = 1.76% DD)
    rolled = gate.check_live_rollback_guard(current_equity=10020.0, trade_closed=True)
    assert rolled is not None
    assert gate.active_snapshot.version_id == "v2.2"
