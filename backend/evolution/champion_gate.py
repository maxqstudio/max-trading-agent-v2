from __future__ import annotations

from dataclasses import dataclass

from backend.core.config import TradingConfig
from backend.core.models import (
    BacktestMetrics,
    ChallengerCandidate,
    ChampionSnapshot,
    ParameterPatch,
    TradeRecord,
    utc_now_ms,
)


@dataclass(slots=True)
class GateEvaluation:
    passed: bool
    reason: str


class ChampionChallengerGate:
    """4-Stage Anti-Overfit Governance & Instant Auto-Rollback Controller.

    Stage 1: Candidate generation (WFO + LLM bounded patches)
    Stage 2: Purged OOS Backtest + 1,000x Monte Carlo P95 DD gate
    Stage 3: Live Shadow Arena forward validation on real MT5 ticks
    Stage 4: Promotion (auto or human approval) + Live Post-Promotion Auto-Rollback Guard
    """

    def __init__(self, initial_config: TradingConfig, initial_equity: float = 10000.0) -> None:
        self.active_config: TradingConfig = initial_config
        self.snapshots: list[ChampionSnapshot] = [
            ChampionSnapshot(
                version_id=initial_config.app.version,
                status="ACTIVE",
                config_snapshot=initial_config.model_dump(mode="json"),
                note="Initial Champion baseline",
                live_peak_equity=initial_equity,
            )
        ]
        self.active_challenger: ChallengerCandidate | None = None
        self.challenger_history: list[ChallengerCandidate] = []
        self._shadow_trades_for_challenger: list[TradeRecord] = []

    @property
    def active_snapshot(self) -> ChampionSnapshot:
        for s in self.snapshots:
            if s.status == "ACTIVE":
                return s
        return self.snapshots[0]

    def _next_version_id(self) -> str:
        base = self.active_snapshot.version_id.lstrip("v").split("-")[0]
        try:
            major, minor = base.split(".")
            return f"v{major}.{int(minor) + 1}-cand"
        except Exception:
            return f"v1.{len(self.snapshots) + 1}-cand"

    def evaluate_oos_and_monte_carlo(
        self,
        challenger_oos: BacktestMetrics,
        champion_oos: BacktestMetrics,
    ) -> GateEvaluation:
        evo = self.active_config.evolution
        if challenger_oos.total_trades < evo.oos_min_trades:
            return GateEvaluation(
                False,
                f"Insufficient OOS trades ({challenger_oos.total_trades} < {evo.oos_min_trades})",
            )
        if challenger_oos.profit_factor < evo.oos_min_profit_factor:
            return GateEvaluation(
                False,
                f"OOS Profit Factor {challenger_oos.profit_factor:.2f} < min {evo.oos_min_profit_factor:.2f}",
            )
        if challenger_oos.expectancy_r <= 0:
            return GateEvaluation(False, f"Non-positive OOS expectancy ({challenger_oos.expectancy_r:.3f}R)")
        if challenger_oos.max_drawdown_pct > evo.oos_max_drawdown_pct:
            return GateEvaluation(
                False,
                f"OOS Max DD {challenger_oos.max_drawdown_pct:.2f}% > cap {evo.oos_max_drawdown_pct:.2f}%",
            )
        if challenger_oos.mc_p95_dd_pct > evo.monte_carlo_p95_max_dd_pct:
            return GateEvaluation(
                False,
                f"Monte Carlo P95 DD {challenger_oos.mc_p95_dd_pct:.2f}% > cap {evo.monte_carlo_p95_max_dd_pct:.2f}%",
            )
        if challenger_oos.mc_ruin_prob > 0.01:
            return GateEvaluation(False, f"Monte Carlo ruin risk {challenger_oos.mc_ruin_prob:.2%} > 1%")

        # Require demonstrable edge over Champion (or allow if Champion had negative/zero Sharpe)
        if champion_oos.total_trades >= evo.oos_min_trades and challenger_oos.sharpe_ratio < champion_oos.sharpe_ratio:
            return GateEvaluation(
                False,
                f"OOS Sharpe {challenger_oos.sharpe_ratio:.2f} did not beat Champion {champion_oos.sharpe_ratio:.2f}",
            )
        return GateEvaluation(True, "Passed Purged OOS Backtest & 1,000x Monte Carlo Stress Test")

    def submit_candidate(
        self,
        candidate_config: TradingConfig,
        patches: list[ParameterPatch],
        challenger_oos: BacktestMetrics,
        champion_oos: BacktestMetrics,
        source: str = "HYBRID",
        reflection_id: str | None = None,
        force_shadow_if_close: bool = False,
    ) -> ChallengerCandidate:
        ver_id = self._next_version_id()
        gate = self.evaluate_oos_and_monte_carlo(challenger_oos, champion_oos)

        if not gate.passed and not force_shadow_if_close:
            cand = ChallengerCandidate(
                version_id=ver_id,
                parent_version_id=self.active_snapshot.version_id,
                source=source,  # type: ignore[arg-type]
                status="REJECTED",
                patches=patches,
                config_snapshot=candidate_config.model_dump(mode="json"),
                oos_metrics=challenger_oos,
                champion_oos_metrics=champion_oos,
                rejection_reason=gate.reason,
                reflection_id=reflection_id,
            )
            self.challenger_history.insert(0, cand)
            return cand

        cand = ChallengerCandidate(
            version_id=ver_id,
            parent_version_id=self.active_snapshot.version_id,
            source=source,  # type: ignore[arg-type]
            status="SHADOWING",
            patches=patches,
            config_snapshot=candidate_config.model_dump(mode="json"),
            oos_metrics=challenger_oos,
            champion_oos_metrics=champion_oos,
            reflection_id=reflection_id,
        )
        self.active_challenger = cand
        self._shadow_trades_for_challenger = []
        self.challenger_history.insert(0, cand)
        return cand

    def record_shadow_trade(self, trade: TradeRecord) -> str | None:
        """Record a live Shadow Arena trade for the active Challenger; returns status transition if ready."""
        if self.active_challenger is None or self.active_challenger.status != "SHADOWING":
            return None
        self._shadow_trades_for_challenger.append(trade)
        rs = [t.pnl_r for t in self._shadow_trades_for_challenger]
        wins = sum(r for r in rs if r > 0)
        losses = sum(-r for r in rs if r <= 0)
        pf = wins / losses if losses > 1e-9 else (3.0 if wins > 0 else 0.0)

        # Compute shadow peak-to-trough DD in %
        eq = 100.0
        peak = 100.0
        max_dd = 0.0
        risk_pct = self.active_config.risk.risk_per_trade_pct
        for r in rs:
            eq *= 1.0 + (r * risk_pct / 100.0)
            peak = max(peak, eq)
            max_dd = max(max_dd, ((peak - eq) / peak) * 100.0)

        c = self.active_challenger
        c.shadow_trades = len(rs)
        c.shadow_pnl_r = round(sum(rs), 3)
        c.shadow_pf = round(min(pf, 9.99), 3)
        c.shadow_max_dd_pct = round(max_dd, 3)

        min_req = self.active_config.evolution.shadow_min_trades
        if c.shadow_trades >= min_req:
            if c.shadow_pf >= 1.15 and c.shadow_pnl_r > 0 and c.shadow_max_dd_pct <= self.active_config.evolution.oos_max_drawdown_pct:
                c.status = "PENDING_APPROVAL"
                if self.active_config.app.auto_promote_challenger:
                    self.promote_active_challenger(note="Auto-promoted after passing Live Shadow Arena")
                    return "AUTO_PROMOTED"
                return "PENDING_APPROVAL"
            else:
                c.status = "REJECTED"
                c.rejection_reason = f"Failed Live Shadow Arena (PF={c.shadow_pf:.2f}, R={c.shadow_pnl_r:+.2f}R)"
                self.active_challenger = None
                return "SHADOW_REJECTED"
        return None

    def promote_active_challenger(self, note: str = "Approved by operator", current_equity: float = 10000.0) -> ChampionSnapshot | None:
        if self.active_challenger is None:
            return None
        cand = self.active_challenger
        promoted_ver = cand.version_id.replace("-cand", "")
        new_cfg = TradingConfig.model_validate(cand.config_snapshot)
        new_cfg.app.version = promoted_ver

        for snap in self.snapshots:
            if snap.status == "ACTIVE":
                snap.status = "ARCHIVED"

        new_snap = ChampionSnapshot(
            version_id=promoted_ver,
            created_at_ms=utc_now_ms(),
            status="ACTIVE",
            config_snapshot=new_cfg.model_dump(mode="json"),
            oos_metrics=cand.oos_metrics,
            note=note,
            live_trades_since_promo=0,
            live_peak_equity=current_equity,
            live_max_dd_pct=0.0,
        )
        self.snapshots.insert(0, new_snap)
        cand.status = "PROMOTED"
        self.active_config = new_cfg
        self.active_challenger = None
        return new_snap

    def reject_active_challenger(self, reason: str = "Rejected by operator") -> bool:
        if self.active_challenger is None:
            return False
        self.active_challenger.status = "REJECTED"
        self.active_challenger.rejection_reason = reason
        self.active_challenger = None
        return True

    def rollback_to(self, version_id: str, auto: bool = False) -> ChampionSnapshot | None:
        target = next((s for s in self.snapshots if s.version_id == version_id), None)
        if target is None:
            return None
        for snap in self.snapshots:
            if snap.status == "ACTIVE":
                snap.status = "AUTO_ROLLED_BACK" if auto else "ARCHIVED"
        target.status = "ACTIVE"
        self.active_config = TradingConfig.model_validate(target.config_snapshot)
        return target

    def check_live_rollback_guard(self, current_equity: float, trade_closed: bool = False) -> ChampionSnapshot | None:
        """Stage 4: Auto-rollback to previous Champion if post-promotion drawdown exceeds safety limit."""
        if len(self.snapshots) < 2:
            return None
        active = self.active_snapshot
        if trade_closed:
            active.live_trades_since_promo += 1
        active.live_peak_equity = max(active.live_peak_equity, current_equity)
        if active.live_peak_equity > 0:
            dd = ((active.live_peak_equity - current_equity) / active.live_peak_equity) * 100.0
            active.live_max_dd_pct = max(active.live_max_dd_pct, round(dd, 3))

        evo = self.active_config.evolution
        if active.live_max_dd_pct >= evo.rollback_live_dd_pct:
            # Find most recent archived previous snapshot
            prev = next((s for s in self.snapshots if s.version_id != active.version_id and s.status == "ARCHIVED"), None)
            if prev is not None:
                return self.rollback_to(prev.version_id, auto=True)
        return None
