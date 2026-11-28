from __future__ import annotations

import asyncio
import time
from collections import deque
from datetime import datetime, timezone
from dataclasses import asdict, dataclass, field
from typing import Any

from backend.bridge.state_machine import IdempotentOrderStateMachine
from backend.core.safety import KillSwitchStore, entry_block_reason
from backend.core.config import EnvSettings, TradingConfig
from backend.core.models import (
    AccountState,
    Bar,
    BridgeSyncRequest,
    BridgeSyncResponse,
    DecisionEvent,
    DLPrediction,
    LivePosition,
    MarketRegime,
    MQLHardGuardTelemetry,
    OrderAction,
    OrderCommand,
    RLExitAction,
    Side,
    SymbolSpec,
    TickSnapshot,
    TradeRecord,
    TradingMode,
    utc_now_ms,
)
from backend.evolution.bandit import ContextualBanditRL
from backend.evolution.champion_gate import ChampionChallengerGate
from backend.evolution.llm_reflector import LLMReflectionAnalyst
from backend.llm.router import LLMRouter
from backend.engine.exit_policy import NetTake, ProfitLock, evaluate, net_take, stage_stop
from backend.engine.exit_selector import ExitStyleSelector
from backend.engine.filters import HTF_BUCKETS, ICHI_BUCKETS, SESSIONS, FilterLearner, decide
from backend.engine.hitrun_stop import Ghost, HitRunArm, HitRunPlan, HitRunTuner, plan_stop, time_stop_due
from backend.engine.supervisor import SupervisorSettings, TradeSupervisor, build_context, price_drifted, scale_volume
from backend.evolution.meta_labeler import CalibratedMetaLabeler, effective_ml_cutoff
from backend.evolution.optimizer import PurgedWalkForwardOptimizer
from backend.evolution.rl_exit_agent import RLExitAgent
from backend.evolution.sequence_net import SequenceConvNet
from backend.features.dataset import build_sequence_tensor, extract_meta_features
from backend.features.htf import alignment, htf_bias, session_of
from backend.features import ichimoku
from backend.features.daylevels import prev_day_range
from backend.features.structure import structure_context
from backend.features.regime import classify_regime
from backend.risk.manager import DEFAULT_SPECS, PortfolioRiskManager
from backend.risk.shields import MarketShields
from backend.storage.database import Database
from backend.strategies.registry import StrategySuite


@dataclass
class ShadowPosition:
    symbol: str
    strategy_id: str
    regime: MarketRegime
    side: Side
    entry_price: float
    sl: float
    tp: float
    risk_dist: float
    open_time: int
    ml_conf: float
    dl_conf: float
    spread_pts: float
    mae_r: float = 0.0
    mfe_r: float = 0.0
    version_id: str = "shadow"
    volume: float = 0.10
    veto: str = ""  # No Veto book: what the supervisor said about this setup (evaluation only)
    time_stop_bars: int = 0  # Hit & Run book: close if not in net profit after this many M1 bars
    htf_m15: int = 0  # entry context: +1 with the M15 direction, -1 against, 0 none
    htf_h1: int = 0
    session: str = ""
    ichi_m15: int = 0  # M15 Ichimoku checklist score signed with the trade (-5..+5)
    tp_stages: list[float] = field(default_factory=list)  # staged targets before tp (Normal and No Veto books)


PAPER_VERSION = "PAPER"  # Normal book (kept for journal compatibility)
HITRUN_VERSION = "PAPER-HITRUN"
RAW_VERSION = "PAPER-RAW"
RAW_HITRUN_VERSION = "PAPER-RAW-HITRUN"
PAPER_BOOKS: dict[str, str] = {
    "normal": PAPER_VERSION,
    "hitrun": HITRUN_VERSION,
    "raw": RAW_VERSION,
    "raw_hitrun": RAW_HITRUN_VERSION,
}
PAPER_LABELS: dict[str, str] = {
    "normal": "Normal",
    "hitrun": "Hit & Run",
    "raw": "No Veto (Normal)",
    "raw_hitrun": "No Veto (Hit & Run)",
}
# Normal and Hit & Run open only what the supervisor + filters allowed.
# Raw books ("raw" and "raw_hitrun") open every risk-approved setup (2x2 matrix).
VETOED_BOOKS = ("normal", "hitrun")
RAW_BOOKS = ("raw", "raw_hitrun")
HITRUN_BOOK_NAMES = frozenset({"hitrun", "raw_hitrun"})
VETO_KEYS = ("APPROVE", "REDUCE", "REJECT", "NONE")
PAPER_START_EQUITY = 10000.0
BAR_BUFFER = 3000  # M1 bars kept per symbol: about 50 hours (H1 direction, M15 Ichimoku, yesterday's range)
SIGNAL_BARS = 600  # what the strategies, regime and models look at


@dataclass
class PaperBook:
    equity: float = PAPER_START_EQUITY
    positions: dict[str, ShadowPosition] = field(default_factory=dict)


def _pos_to_dict(p: ShadowPosition) -> dict[str, Any]:
    return {**asdict(p), "side": p.side.value, "regime": p.regime.value}


def _pos_from_dict(d: dict[str, Any]) -> ShadowPosition | None:
    try:
        known = {k: v for k, v in d.items() if k in ShadowPosition.__dataclass_fields__}
        return ShadowPosition(**{**known, "side": Side(d["side"]), "regime": MarketRegime(d["regime"])})
    except (KeyError, TypeError, ValueError):
        return None


def _supervisor_settings(cfg: TradingConfig) -> SupervisorSettings:
    sc = cfg.supervisor
    return SupervisorSettings(
        enabled=sc.enabled,
        mode=sc.mode,
        brief_interval_min=sc.brief_interval_min,
        timeout_s=sc.timeout_s,
        use_fast_model=sc.use_fast_model,
        effort=sc.effort,
        on_unavailable="allow" if sc.on_unavailable == "allow" else "block",
        max_calls_per_hour=sc.max_calls_per_hour,
        daily_budget_usd=sc.daily_budget_usd,
        max_price_drift_atr=sc.max_price_drift_atr,
    )


class TradingOrchestrator:
    """Central real-time brain uniting MT5 Bridge, 4-Pillar Hybrid AI, Shadow Arena & Evolution Loop."""

    def __init__(
        self,
        config: TradingConfig,
        env: EnvSettings,
        db: Database,
        llm_router: LLMRouter | None = None,
    ) -> None:
        self.config = config
        self.llm = llm_router if llm_router is not None else LLMRouter.from_env(".env")
        self.env = env
        self.db = db

        # Core state & specs
        self.specs: dict[str, SymbolSpec] = {
            s: DEFAULT_SPECS.get(s, DEFAULT_SPECS["XAUUSD"]).model_copy(deep=True)
            for s in config.instruments.symbols
        }
        self.account = AccountState()
        self.hard_guard = MQLHardGuardTelemetry()
        self.last_bridge_sync_ms: int = 0
        self.bridge_latency_ms: float = 0.0
        self.ticks_per_sec: float = 0.0
        self._tick_timestamps: deque[int] = deque(maxlen=60)
        self.scans_count: int = 0
        self.active_thought: str = "Initializing 4-pillar hybrid AI brain & awaiting live MT5 ticks..."

        # Market buffers & live per-symbol cockpit telemetry
        self.bars: dict[str, deque[Bar]] = {s: deque(maxlen=BAR_BUFFER) for s in config.instruments.symbols}
        self.ticks: dict[str, TickSnapshot] = {}
        self.regimes: dict[str, MarketRegime] = {s: MarketRegime.RANGING for s in config.instruments.symbols}
        self.dl_preds: dict[str, DLPrediction] = {s: DLPrediction() for s in config.instruments.symbols}
        self.readiness: dict[str, int] = {s: 35 for s in config.instruments.symbols}
        self.buy_pressure: dict[str, int] = {s: 50 for s in config.instruments.symbols}
        self.shield_status: dict[str, str] = {s: "CLEAR" for s in config.instruments.symbols}
        self.ichi_scores: dict[str, int] = {s: 0 for s in config.instruments.symbols}
        self.open_positions: list[LivePosition] = []
        self.recent_decisions: deque[DecisionEvent] = deque(maxlen=120)

        # Subsystems
        self.shields = MarketShields(config.shields)
        self.risk = PortfolioRiskManager(config.risk)
        # Persistent kill switch: survives restarts and crashes.
        self.kill_store = KillSwitchStore(config.app.kill_switch_path)
        self.kill_state = self.kill_store.load()
        if self.kill_state.engaged:
            self.risk.killed = True
            self.risk.paused = True
        self.order_sm = IdempotentOrderStateMachine(order_ttl_ms=config.instruments.order_ttl_ms)
        self.suite = StrategySuite.from_config(config)

        # 4-Pillar Self-Improving AI
        self.dl_net = SequenceConvNet()
        self.ml_labeler = CalibratedMetaLabeler()
        self.shadow_dl_net: SequenceConvNet | None = None
        self.shadow_ml_labeler: CalibratedMetaLabeler | None = None
        self.bandit = ContextualBanditRL(forgetting_factor=config.ai.bandit_forgetting_factor)
        self.rl_exit = RLExitAgent(
            be_trigger_r=config.ai.rl_exit_be_trigger_r,
            trail_trigger_r=config.ai.rl_exit_trail_trigger_r,
        )
        self.optimizer = PurgedWalkForwardOptimizer()
        self.reflector = LLMReflectionAnalyst(env, router=self.llm)
        self.supervisor = TradeSupervisor(self.llm, _supervisor_settings(config))
        self._supervising: set[str] = set()
        self._supervisor_tasks: set[asyncio.Task] = set()
        self._recent_closed: deque[TradeRecord] = deque(maxlen=60)
        self.gate = ChampionChallengerGate(config, initial_equity=self.account.equity)

        # Parallel Live Shadow Arena (1 = with AI Supervisor veto & filters, 1 = pure raw Challenger)
        self.shadow_positions: dict[str, ShadowPosition] = {}
        self.shadow_equity: float = 10000.0
        self.shadow_raw_positions: dict[str, ShadowPosition] = {}
        self.shadow_raw_equity: float = 10000.0
        # Paper trading (SHADOW_ONLY mode): the champion's approved trades, filled virtually.
        # Paper books run side by side on the same approved setups, in every mode:
        # "normal" = existing exits + profit lock, "hitrun" = take profit fast.
        self.exit_selector = ExitStyleSelector(min_samples=config.exits.adaptive_min_samples)
        self.hitrun_tuner = HitRunTuner(min_samples=config.paper.hit_run.adaptive_min_samples)
        self.filter_learner = FilterLearner(min_samples=config.filters.min_samples)
        self.server_offset_sec = 0  # broker server time minus UTC (bar timestamps are server time)
        self.veto_stats: dict[str, dict[str, float]] = {}
        self.learned_costs: dict[str, dict[str, float]] = {}  # per symbol, from closed MT5 trades
        self.paper_books: dict[str, PaperBook] = {
            k: PaperBook(equity=config.paper.start_equity) for k in PAPER_BOOKS
        }
        self._tracking_sig: tuple[int, ...] = ()
        self.shadow_suite: StrategySuite = StrategySuite.from_config(config)

        # Async callbacks / subscribers (Telegram + WebSocket)
        self.ws_clients: set[Any] = set()
        self.alert_callback: Any | None = None
        self.evolution_running: bool = False
        self.research_running: bool = False
        self.last_research_audit: dict[str, Any] | None = None
        self._warmup_triggered: bool = False
        self._brief_refreshing: bool = False
        self._bg_tasks: set[asyncio.Task] = set()
        self._last_eq_record_ms: int = 0

    async def initialize_from_storage(self) -> None:
        """Crash recovery & state restoration on startup."""
        for sym in self.config.instruments.symbols:
            saved_bars = await self.db.load_recent_bars(sym, "M1", limit=BAR_BUFFER)
            if saved_bars:
                self.bars[sym].extend(saved_bars)

        bandit_kv = await self.db.get_kv("bandit_state")
        if bandit_kv:
            self.bandit.load_dict(bandit_kv)

        rl_kv = await self.db.get_kv("rl_exit_state")
        if rl_kv:
            self.rl_exit.load_dict(rl_kv)

        snaps = await self.db.list_snapshots()
        if snaps:
            self.gate.snapshots = snaps
            active_snap = self.gate.active_snapshot
            # Only restore evolved champion parameters if this snapshot actually came from an evolution promotion;
            # never let a legacy baseline snapshot clobber user Setup/local.yaml settings.
            if active_snap.config_snapshot and active_snap.note != "Initial Champion baseline":
                self.apply_champion_config(TradingConfig.model_validate(active_snap.config_snapshot))
            else:
                self.gate.active_config = self.config
        else:
            await self.db.save_snapshot(self.gate.active_snapshot)

        challs = await self.db.list_challengers(limit=10)
        if challs:
            self.gate.challenger_history = challs
            for c in challs:
                if c.status in ("SHADOWING", "PENDING_APPROVAL"):
                    self.gate.active_challenger = c
                    self.shadow_suite = StrategySuite.from_config(TradingConfig.model_validate(c.config_snapshot))
                    break

        arena = await self.db.get_kv("shadow_arena")
        if arena:
            self.shadow_equity = float(arena.get("equity", self.shadow_equity))
            for d in arena.get("positions", []):
                p = _pos_from_dict(d)
                if p:
                    self.shadow_positions[p.symbol] = p
            self.shadow_raw_equity = float(arena.get("raw_equity", self.shadow_raw_equity))
            for d in arena.get("raw_positions", []):
                p = _pos_from_dict(d)
                if p:
                    self.shadow_raw_positions[p.symbol] = p
        books = await self.db.get_kv("paper_books")
        if not books:
            legacy = await self.db.get_kv("paper_book")  # single book from the previous version
            books = {"normal": legacy} if legacy else {}
        for name, data in (books or {}).items():
            if name not in self.paper_books or not isinstance(data, dict):
                continue
            book = self.paper_books[name]
            book.equity = float(data.get("equity", book.equity))
            for d in data.get("positions", []):
                p = _pos_from_dict(d)
                if p:
                    book.positions[p.symbol] = p
        await self._rebuild_exit_selector()
        self.veto_stats = await self.db.get_kv("veto_stats") or {}
        self.hitrun_tuner.load(await self.db.get_kv("hitrun_tuner"))
        self.filter_learner.load(await self.db.get_kv("filter_stats"))
        self.server_offset_sec = int(await self.db.get_kv("server_offset_sec") or 0)
        await self._learn_broker_costs()
        self.order_sm.import_tracking(await self.db.get_kv("mt5_tracking"))
        self._tracking_sig = tuple(sorted(self.order_sm.active_positions))

        saved_decs = await self.db.list_decisions(limit=120)
        for ev in reversed(saved_decs):
            self.recent_decisions.appendleft(ev)

        active_ver = self.gate.active_snapshot.version_id
        dl_ver_path, ml_ver_path = self._version_model_paths(active_ver)
        if not self.dl_net.load(dl_ver_path):
            self.dl_net.load(f"{self.config.app.snapshots_dir}/dl_champion.npz")
            if self.dl_net.trained:
                self.dl_net.save(dl_ver_path)
        if not self.ml_labeler.load(ml_ver_path):
            self.ml_labeler.load(f"{self.config.app.snapshots_dir}/ml_champion.pkl")
            if self.ml_labeler.trained:
                self.ml_labeler.save(ml_ver_path)
        if self.gate.active_challenger is not None:
            self._load_challenger_models(self.gate.active_challenger.version_id)
        self.last_research_audit = await self.db.get_kv("last_research_audit")

    def apply_active_config(self, cfg: TradingConfig) -> None:
        self.config = cfg
        self.supervisor.settings = _supervisor_settings(cfg)
        self.risk.config = cfg.risk
        self.shields.config = cfg.shields
        self.suite = StrategySuite.from_config(cfg)
        self.rl_exit.be_trigger_r = cfg.ai.rl_exit_be_trigger_r
        self.rl_exit.trail_trigger_r = cfg.ai.rl_exit_trail_trigger_r

    def apply_champion_config(self, champion_cfg: TradingConfig) -> TradingConfig:
        """Apply evolved strategy/AI/risk params while preserving user Setup settings (local.yaml)."""
        ops = self.config
        merged = champion_cfg.model_copy(
            update={
                "app": ops.app.model_copy(update={"version": champion_cfg.app.version}),
                "supervisor": ops.supervisor,
                "exits": ops.exits,
                "paper": ops.paper,
                "filters": ops.filters,
                "shields": ops.shields,
                "risk": champion_cfg.risk.model_copy(
                    update={"max_cost_to_risk_ratio": ops.risk.max_cost_to_risk_ratio}
                ),
            }
        )
        self.apply_active_config(merged)
        self.gate.active_config = merged
        return merged

    async def _log_decision(self, event: DecisionEvent) -> None:
        self.recent_decisions.appendleft(event)
        await self.db.save_decision(event)

    async def _notify(self, text: str) -> None:
        if self.alert_callback is not None:
            try:
                await self.alert_callback(text)
            except Exception:
                pass

    def llm_brief(self) -> dict[str, Any]:
        """Which AI answers right now (from active fast/main chain) and today's spend."""
        fast_list = self.llm.active_fast
        main_list = self.llm.active_main
        first_fast = fast_list[0] if fast_list else (main_list[0] if main_list else None)
        first_main = main_list[0] if main_list else first_fast
        fallbacks = [f"{r.name}:{r.fast_model}" for r in fast_list[1:]]
        return {
            "enabled": self.llm.enabled,
            "provider": first_fast.name if first_fast else "",
            "label": first_fast.preset.label if first_fast else "Deterministic (no AI key)",
            "model": first_main.model if first_main else "",
            "fast_model": first_fast.fast_model if first_fast else "",
            "fallbacks": fallbacks,
            "total_cost_usd": round(sum(st.total_cost_usd for st in self.llm.stats.values()), 4),
        }

    def apply_supervisor_config(self) -> None:
        self.supervisor.settings = _supervisor_settings(self.config)

    def ensure_symbol(self, sym: str) -> None:
        """Start tracking a symbol the EA reports that is not in the config watchlist.

        The EA maps broker names (XAUUSD.m, EURUSDm, ...) to plain names and can add the chart's
        own symbol, so the set of symbols comes from MT5 instead of a fixed list.
        """
        if not sym or sym in self.config.instruments.symbols:
            return
        self.config.instruments.symbols.append(sym)
        self.specs.setdefault(sym, DEFAULT_SPECS.get(sym, DEFAULT_SPECS["XAUUSD"]).model_copy(deep=True))
        self.bars.setdefault(sym, deque(maxlen=BAR_BUFFER))
        self.regimes.setdefault(sym, MarketRegime.RANGING)
        self.dl_preds.setdefault(sym, DLPrediction())
        self.readiness.setdefault(sym, 35)
        self.buy_pressure.setdefault(sym, 50)
        self.shield_status.setdefault(sym, "CLEAR")

    async def handle_bridge_sync(self, req: BridgeSyncRequest) -> BridgeSyncResponse:
        """Primary real-time execution heartbeat called by MQL5 EA or Simulator."""
        t0 = time.perf_counter()
        now_ms = utc_now_ms()
        now_sec = int(now_ms // 1000)

        self.last_bridge_sync_ms = now_ms
        self._tick_timestamps.append(now_ms)
        if len(self._tick_timestamps) >= 2:
            span = max((self._tick_timestamps[-1] - self._tick_timestamps[0]) / 1000.0, 0.25)
            self.ticks_per_sec = round(len(self._tick_timestamps) / span, 1)

        # 0. Drop data for symbols the broker doesn't offer (an EA asking for "XAUUSD" on an
        #    "XAUUSD.m" broker gets zeros back); zeros would poison the models and sizing.
        req.symbol_specs = [sp for sp in req.symbol_specs if sp.point > 0]
        req.ticks = [t for t in req.ticks if t.bid > 0 and t.ask > 0]
        req.closed_bars = [b for b in req.closed_bars if b.close > 0 and b.high >= b.low > 0]
        req.history_bars = [b for b in req.history_bars if b.close > 0 and b.high >= b.low > 0]

        # 1. Update specs, account, hard guard
        for sym in (
            {sp.symbol for sp in req.symbol_specs}
            | {t.symbol for t in req.ticks}
            | {b.symbol for b in req.closed_bars}
            | {b.symbol for b in req.history_bars}
        ):
            self.ensure_symbol(sym)
        for sp in req.symbol_specs:
            self.specs[sp.symbol] = sp
        self.account = req.account
        if req.hard_guard is not None:
            self.hard_guard = req.hard_guard
        if req.server_utc_offset_sec is not None and req.server_utc_offset_sec != self.server_offset_sec:
            self.server_offset_sec = int(req.server_utc_offset_sec)
            await self.db.set_kv("server_offset_sec", self.server_offset_sec, now_ms)
        if req.calendar is not None:
            self.shields.set_live_news(
                [
                    {
                        "title": f"{ev.currency} {ev.title}",
                        "currencies": ["XAU", "USD"] if ev.currency == "USD" else [ev.currency],
                        "timestamp_utc": datetime.fromtimestamp(ev.time_utc, timezone.utc).isoformat(),
                        "impact": ev.impact,
                    }
                    for ev in req.calendar
                ],
                now_sec,
            )

        # 2. Process order ACKs & reconcile MT5 positions (Single Source of Truth)
        acks = self.order_sm.process_acks(req.acks)
        for ack in acks:
            await self._log_decision(
                DecisionEvent(
                    symbol="MT5",
                    category="EXEC" if ack.status.value == "FILLED" else "BLOCK",
                    tag=f"ACK {ack.status.value}",
                    message=f"Order {ack.client_order_id} -> {ack.status.value} (tk={ack.ticket}, px={ack.fill_price}, slip={ack.slippage_points}pt) {ack.message}",
                )
            )

        self.open_positions, closed_trades = self.order_sm.reconcile_positions(
            req.positions,
            self.specs,
            self.gate.active_snapshot.version_id,
            now_sec,
        )
        # Persist what we know about open MT5 positions whenever the set changes, so a position
        # that closes while the bot is off is still recorded on the next start.
        sig = tuple(sorted(self.order_sm.active_positions))
        if sig != self._tracking_sig:
            self._tracking_sig = sig
            await self.db.set_kv("mt5_tracking", self.order_sm.export_tracking(), now_ms)

        # 3. Online learning on every newly closed live trade (RL-1 Bandit + RL-2 Exit + Rollback Guard)
        if closed_trades:
            await self._learn_broker_costs(closed_trades)
        for tr in closed_trades:
            await self.db.save_trade(tr)
            self._recent_closed.append(tr)
            self.risk.record_trade_outcome(tr.pnl_money, now_sec)
            # With paper on, the bandit and RL exit learn from the "No Veto" book instead, which
            # sees every setup (MT5 only sees what the supervisor approved, and would double count).
            if not self.config.paper.enabled:
                self.bandit.update(tr.symbol, tr.regime, tr.strategy_id, tr.pnl_r)
                self.rl_exit.update_from_closed_trade(
                    peak_r=tr.mfe_r,
                    realized_r=tr.pnl_r,
                    bars_held=max(1, (tr.close_time - tr.open_time) // 60),
                    regime=tr.regime,
                    dl_aligned=tr.dl_confidence >= 0.5,
                    last_action=RLExitAction.MOVE_BE if "MOVE_BE" in tr.rl_exit_trace else RLExitAction.HOLD,
                )
                await self.db.set_kv("bandit_state", self.bandit.to_dict(), now_ms)
                await self.db.set_kv("rl_exit_state", self.rl_exit.to_dict(), now_ms)
            await self._log_decision(
                DecisionEvent(
                    symbol=tr.symbol,
                    category="EXEC",
                    tag=f"CLOSED {tr.exit_reason}",
                    strategy_id=tr.strategy_id,
                    side=tr.side.value,
                    message=f"#{tr.ticket} {tr.side.value} {tr.pnl_money:+.2f}$ ({tr.pnl_r:+.2f}R) | MAE {tr.mae_r:.2f}R / MFE +{tr.mfe_r:.2f}R",
                    metadata={"pnl": round(tr.pnl_money, 2), "pnl_r": round(tr.pnl_r, 3)},
                )
            )
            await self._notify(
                f"Trade Closed: {tr.symbol} {tr.side.value} ({tr.strategy_id})\n"
                f"PnL: {tr.pnl_money:+.2f}$ ({tr.pnl_r:+.2f}R) | Reason: {tr.exit_reason}"
            )

            rolled = self.gate.check_live_rollback_guard(self.account.equity, trade_closed=True)
            if rolled is not None:
                self._restore_version_models(rolled.version_id)
                self.apply_champion_config(self.gate.active_config)
                for sn in self.gate.snapshots:
                    await self.db.save_snapshot(sn)
                await self._log_decision(
                    DecisionEvent(
                        symbol="SYSTEM",
                        category="EVOLUTION",
                        tag="AUTO ROLLBACK",
                        message=f"Post-promotion drawdown triggered automatic rollback to Champion {rolled.version_id} (config + DL/ML weights)",
                    )
                )
                await self._notify(f"AUTO-ROLLBACK TRIGGERED! Restored Champion {rolled.version_id} (config + DL/ML)")

        # 4. Ingest live ticks & update Shadow Arena positions
        for tk in req.ticks:
            prev = self.ticks.get(tk.symbol)
            self.ticks[tk.symbol] = tk
            self.shields.record_spread(tk.symbol, tk.spread_points)
            if prev is not None:
                cur_p = self.buy_pressure.get(tk.symbol, 50)
                delta = 3 if tk.bid > prev.bid else (-3 if tk.bid < prev.bid else 0)
                self.buy_pressure[tk.symbol] = max(18, min(84, cur_p + delta))
            await self._update_shadow_positions_on_tick(tk, now_sec)
            await self._update_paper_positions_on_tick(tk, now_sec)

        # 4b. Ingest startup historical backfill bars (buffer & DB only — never fire live trades on old bars!)
        if req.history_bars:
            await self.db.save_bars(req.history_bars)
            by_sym: dict[str, list[Bar]] = {}
            for hb in req.history_bars:
                by_sym.setdefault(hb.symbol, []).append(hb)
            for sym, new_hist in by_sym.items():
                buf = self.bars.setdefault(sym, deque(maxlen=BAR_BUFFER))
                merged = {b.timestamp: b for b in buf}
                for hb in new_hist:
                    merged[hb.timestamp] = hb
                ordered = [merged[k] for k in sorted(merged.keys())][-BAR_BUFFER:]
                buf.clear()
                buf.extend(ordered)
            self._last_backfill_ts = time.monotonic()

        # 5. Ingest newly closed bars & run full 4-Pillar AI scan on closed bars
        if req.closed_bars:
            await self.db.save_bars(req.closed_bars)
            for bar in req.closed_bars:
                buf = self.bars.setdefault(bar.symbol, deque(maxlen=BAR_BUFFER))
                if not buf or bar.timestamp > buf[-1].timestamp:
                    buf.append(bar)
                await self._evaluate_symbol_on_bar_close(bar, now_sec)

        # 5b. Auto-train DL/ML once startup backfill finishes streaming (or enough live bars accumulate)
        self._maybe_trigger_auto_warmup()

        # 6. Run RL-2 Dynamic Exit management on all open live positions
        if self.config.ai.rl_exit_enabled and not self.risk.killed:
            await self._manage_open_positions_with_rl_exit(now_ms)

        # 7. Periodically record equity curve
        if now_ms - self._last_eq_record_ms >= 5000:
            self._last_eq_record_ms = now_ms
            await self.db.record_equity_point(
                now_ms,
                self.account.equity,
                self.shadow_equity,
                self.account.daily_dd_pct,
            )

        commands = self.order_sm.pop_commands_for_dispatch(
            now_ms, block_entries_reason=self.entry_block_reason()
        )
        for blocked_cmd, why in self.order_sm.last_blocked:
            await self._log_decision(
                DecisionEvent(
                    symbol=blocked_cmd.symbol,
                    category="BLOCK",
                    tag="SAFETY RAIL",
                    strategy_id=blocked_cmd.strategy_id,
                    side=blocked_cmd.side.value if blocked_cmd.side else "",
                    message=f"Entry {blocked_cmd.client_order_id} withheld: {why}",
                )
            )
        self.bridge_latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        # Broadcast live frame to WebSocket dashboards
        if self.ws_clients:
            asyncio.create_task(self.broadcast_state())

        return BridgeSyncResponse(
            commands=commands,
            server_time_ms=now_ms,
            paused=self.risk.paused,
            killed=self.risk.killed,
        )

    async def _evaluate_symbol_on_bar_close(self, bar: Bar, now_sec: int) -> None:
        t0 = time.perf_counter()
        sym = bar.symbol
        self.scans_count += 1
        all_bars = list(self.bars.get(sym, ()))
        bar_list = all_bars[-SIGNAL_BARS:]
        if len(bar_list) < 40:
            return

        assess = classify_regime(bar_list)
        if assess is None:
            return
        self.regimes[sym] = assess.regime
        spec = self.specs.get(sym, DEFAULT_SPECS["XAUUSD"])

        # Shock bar detection
        self.shields.record_bar_for_shock(sym, bar, assess.indicators.atr14, now_sec)

        # Pillar 1 (DL): 1D-CNN Sequence prediction
        seq = build_sequence_tensor(bar_list, seq_len=self.config.ai.dl_seq_len, point=spec.point)
        dl_pred = self.dl_net.predict(seq)
        self.dl_preds[sym] = dl_pred

        # Market Shields evaluation
        tick = self.ticks.get(sym)
        spread_pts = tick.spread_points if tick else bar.spread_points
        verdict = self.shields.evaluate(sym, spread_pts)
        self.shield_status[sym] = verdict.reason_code
        if verdict.should_flatten and self.open_positions:
            await self.trigger_flatten_all("FRIDAY_CLOSE_FLATTEN")
            return
        if not verdict.allowed:
            self.readiness[sym] = 15
            self.active_thought = f"{sym} shielded ({verdict.reason_code}): {verdict.detail}"
            return

        # Market facts from the full buffer: M15/H1 direction, M15 Ichimoku, yesterday's range
        htf = (htf_bias(all_bars, 15), htf_bias(all_bars, 60))
        ichi_m15 = ichimoku.htf_score(all_bars, 15)  # M15 Ichimoku checklist, needs ~26 h of M1 bars
        self.ichi_scores[sym] = ichi_m15
        prev_day = prev_day_range(all_bars)
        mkt = {"prev_day": prev_day, "htf_m15": htf[0], "htf_h1": htf[1]}

        signals = self.suite.scan_all(sym, bar_list, assess.indicators, assess.regime, context=mkt)

        # Also evaluate Challenger in parallel Live Shadow Arena!
        await self._scan_shadow_challenger(
            sym, bar_list, assess.indicators, assess.regime, dl_pred, spread_pts, spec, now_sec, htf, ichi_m15, prev_day,
        )

        if not signals:
            # Compute live hunting readiness from indicator alignment so Cockpit always shows active hunt progress
            adx_score = min(45, int(assess.indicators.adx14 * 1.4))
            dl_dir = int(max(dl_pred.prob_up, dl_pred.prob_down) * 45)
            self.readiness[sym] = max(18, min(79, adx_score + dl_dir // 2))
            self.active_thought = (
                f"{sym} [{assess.regime.value}] hunting M1 setups — ADX {assess.indicators.adx14:.1f} • "
                f"DL Up {dl_pred.prob_up:.0%}/Dn {dl_pred.prob_down:.0%} • Readiness {self.readiness[sym]}%"
            )
            return

        # Pillar 3a (RL-1): Contextual Thompson Bandit selects highest-weighted signal
        weights = self.bandit.get_weights(sym, assess.regime, explore=True)
        best_sig = max(signals, key=lambda s: s.confidence * (0.5 + weights.get(s.strategy_id, 0.25)))
        b_weight = weights.get(best_sig.strategy_id, 0.25)

        # Pillar 2 (ML): Calibrated Meta-Labeler predicts P(win)
        spread_price = spread_pts * spec.point
        feat_vec = extract_meta_features(
            best_sig, assess.indicators, spread_price, dl_pred, bar.timestamp, htf=htf, server_offset_sec=self.server_offset_sec,
            structure=structure_context(bar_list, ichi_m15, prev_day),
        )
        side_sign = 1.0 if best_sig.side == Side.BUY else -1.0
        entry_ctx = {
            "htf_m15": alignment(side_sign, htf[0]),
            "htf_h1": alignment(side_sign, htf[1]),
            "session": session_of(bar.timestamp, self.server_offset_sec),
            "ichi_m15": int(ichi_m15 * side_sign),
        }
        ml_prob = self.ml_labeler.predict_win_probability(feat_vec)
        dl_align = dl_pred.prob_up if best_sig.side == Side.BUY else dl_pred.prob_down
        lat_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        sig_sl_dist = max(abs(best_sig.entry_price - best_sig.sl_price), spec.point)
        sig_rr = abs(best_sig.tp_price - best_sig.entry_price) / sig_sl_dist
        ml_cutoff = effective_ml_cutoff(self.config.ai.ml_min_probability, sig_rr)
        self.readiness[sym] = min(98, int(ml_prob * 100))

        if ml_prob < ml_cutoff:
            self.active_thought = f"{sym} {best_sig.strategy_id} {best_sig.side.value} filtered by ML Meta-Labeler (P={ml_prob:.2f} < {ml_cutoff:.2f}, RR {sig_rr:.1f})"
            await self._log_decision(
                DecisionEvent(
                    symbol=sym,
                    category="BLOCK",
                    tag="ML FILTER",
                    strategy_id=best_sig.strategy_id,
                    side=best_sig.side.value,
                    message=f"Rejected: Calibrated P(win) {ml_prob:.2f} < {ml_cutoff:.2f} (RR {sig_rr:.1f}) | DL align {dl_align:.2f}",
                    latency_ms=lat_ms,
                )
            )
            return

        # Portfolio Risk Guard & Broker-Spec Position Sizing (uses real learned broker commission!)
        risk_dec = self.risk.evaluate_entry(
            signal=best_sig,
            account=self.account,
            open_positions=self.open_positions,
            spec=spec,
            spread_points=spread_pts,
            ml_probability=ml_prob,
            bandit_weight=b_weight,
            now_ts=now_sec,
            commission_per_lot=self.paper_costs(sym)["commission_per_lot"],
        )
        if not risk_dec.approved:
            self.active_thought = f"{sym} {best_sig.strategy_id} blocked by Risk Guard ({risk_dec.reason_code})"
            await self._log_decision(
                DecisionEvent(
                    symbol=sym,
                    category="BLOCK",
                    tag=risk_dec.reason_code,
                    strategy_id=best_sig.strategy_id,
                    side=best_sig.side.value,
                    message=risk_dec.detail,
                    latency_ms=lat_ms,
                )
            )
            return

        reviewed = self.supervisor.active and (
            self.config.app.trading_mode != TradingMode.SHADOW_ONLY or self.config.supervisor.review_in_shadow
        )
        per_trade_llm = reviewed and self.supervisor.uses_per_trade_review(best_sig.strategy_id)
        if self.config.paper.enabled and not (per_trade_llm and sym in self._supervising):
            # Every risk-approved setup, before (and regardless of) the supervisor & entry filters (both Normal & Hit & Run).
            await self._open_paper_position(
                sym, best_sig, risk_dec, assess, spec, ml_prob, dl_align, spread_pts,
                books=RAW_BOOKS, veto="PENDING" if reviewed else "NONE", context=entry_ctx,
            )
        # Direction, regime and session filters: after the No Veto books, so they keep seeing every setup.
        block = self._entry_filter(entry_ctx, strategy_id=best_sig.strategy_id, regime=assess.regime.value)
        if block.code:
            raw = self.paper_books["raw"].positions.get(sym)
            if raw is not None and raw.veto == "PENDING":
                if reviewed and not per_trade_llm:
                    bv = self.supervisor.evaluate_from_brief(sym, best_sig.side, best_sig.strategy_id)
                    raw.veto = bv.decision if not bv.error else "NONE"
                else:
                    raw.veto = "NONE"
                await self._save_paper()
            self.active_thought = f"{sym} {best_sig.strategy_id} {best_sig.side.value} filtered: {block.detail}"
            await self._log_decision(
                DecisionEvent(
                    symbol=sym, category="BLOCK", tag=block.code.replace("_", " "), strategy_id=best_sig.strategy_id,
                    side=best_sig.side.value, message=block.detail, latency_ms=lat_ms,
                )
            )
            return
        if reviewed and not per_trade_llm:
            # Periodic / Hybrid fast path: instant (<0.1ms) evaluation against latest DeskBrief — ZERO price drift!
            if self.supervisor.brief_is_due() and not self._brief_refreshing:
                t = asyncio.create_task(self.refresh_supervisor_brief(force=False))
                self._bg_tasks.add(t)
                t.add_done_callback(self._bg_tasks.discard)
            verdict = self.supervisor.evaluate_from_brief(sym, best_sig.side, best_sig.strategy_id)
            if self.supervisor.last:
                self.supervisor.last["symbol"] = sym
            raw = self.paper_books["raw"].positions.get(sym)
            if raw is not None and raw.veto == "PENDING":
                raw.veto = verdict.decision if not verdict.error else "NONE"
                await self._save_paper()
            note = f" | AI Supervisor {verdict.decision} (brief, 0ms): {verdict.reason}"
            if not verdict.allows_entry:
                await self._log_decision(
                    DecisionEvent(
                        symbol=sym,
                        category="BLOCK",
                        tag="SUPERVISOR VETO" if not verdict.error else "SUPERVISOR DOWN",
                        strategy_id=best_sig.strategy_id,
                        side=best_sig.side.value,
                        message=f"{best_sig.side.value} {risk_dec.volume} lot skipped{note}",
                        latency_ms=lat_ms,
                    )
                )
                self.active_thought = f"{sym} vetoed by AI Supervisor brief: {verdict.reason}"
                return
            if verdict.size_multiplier < 1.0:
                new_vol = scale_volume(risk_dec.volume, verdict.size_multiplier, spec)
                if new_vol <= 0:
                    await self._log_decision(
                        DecisionEvent(symbol=sym, category="BLOCK", tag="SUPERVISOR REDUCE", strategy_id=best_sig.strategy_id,
                                      side=best_sig.side.value, message=f"Reduced size below broker minimum — skipped{note}")
                    )
                    return
                risk_dec.volume = new_vol
            await self._finalize_entry(sym, best_sig, risk_dec, assess, spec, ml_prob, dl_align, b_weight, spread_pts, lat_ms, note)
            return
        if per_trade_llm:
            if sym in self._supervising:
                return
            self._supervising.add(sym)
            ctx = build_context(
                signal=best_sig,
                spec=spec,
                volume=risk_dec.volume,
                sl=risk_dec.adjusted_sl,
                tp=risk_dec.adjusted_tp,
                spread_points=spread_pts,
                bars=bar_list,
                ind=assess.indicators,
                dl_up=dl_pred.prob_up,
                dl_down=dl_pred.prob_down,
                ml_prob=ml_prob,
                bandit_weight=b_weight,
                open_positions=self.open_positions,
                recent_trades=list(self._recent_closed),
                daily_dd_pct=self.account.daily_dd_pct,
                max_daily_dd_pct=self.config.risk.max_daily_drawdown_pct,
                consecutive_losses=self.risk.consecutive_losses,
                hour_utc=int((now_sec % 86400) // 3600),
            )
            self.active_thought = f"{sym} {best_sig.side.value} via {best_sig.strategy_id} — waiting for supervisor"
            task = asyncio.create_task(
                self._supervise_then_finalize(ctx, sym, best_sig, risk_dec, assess, spec, ml_prob, dl_align, b_weight, spread_pts, lat_ms)
            )
            self._supervisor_tasks.add(task)
            task.add_done_callback(self._supervisor_tasks.discard)
            return

        await self._finalize_entry(sym, best_sig, risk_dec, assess, spec, ml_prob, dl_align, b_weight, spread_pts, lat_ms)

    async def _supervise_then_finalize(
        self, ctx: str, sym: str, best_sig: Any, risk_dec: Any, assess: Any, spec: SymbolSpec,
        ml_prob: float, dl_align: float, b_weight: float, spread_pts: float, lat_ms: float,
    ) -> None:
        try:
            verdict = await self.supervisor.review(ctx)
            if self.supervisor.last:
                self.supervisor.last["symbol"] = sym
            raw = self.paper_books["raw"].positions.get(sym)
            if raw is not None and raw.veto == "PENDING":
                raw.veto = verdict.decision if not verdict.error else "NONE"
                await self._save_paper()
            note = f" | Supervisor {verdict.decision} ({verdict.provider or 'policy'}, {verdict.latency_ms:.0f}ms): {verdict.reason}"
            if not verdict.allows_entry:
                await self._log_decision(
                    DecisionEvent(
                        symbol=sym,
                        category="BLOCK",
                        tag="SUPERVISOR VETO" if not verdict.error else "SUPERVISOR DOWN",
                        strategy_id=best_sig.strategy_id,
                        side=best_sig.side.value,
                        message=f"{best_sig.side.value} {risk_dec.volume} lot skipped{note}",
                        latency_ms=verdict.latency_ms,
                    )
                )
                self.active_thought = f"{sym} vetoed by supervisor: {verdict.reason}"
                return
            if verdict.size_multiplier < 1.0:
                new_vol = scale_volume(risk_dec.volume, verdict.size_multiplier, spec)
                if new_vol <= 0:
                    await self._log_decision(
                        DecisionEvent(symbol=sym, category="BLOCK", tag="SUPERVISOR REDUCE", strategy_id=best_sig.strategy_id,
                                      side=best_sig.side.value, message=f"Reduced size below broker minimum — skipped{note}")
                    )
                    return
                risk_dec.volume = new_vol
            # The world moved while we waited: re-check the hard gates and the price.
            if self.risk.killed or self.risk.paused:
                return
            if any(p.symbol == sym for p in self.open_positions):
                return
            tick = self.ticks.get(sym)
            if tick is not None:
                px = tick.ask if best_sig.side == Side.BUY else tick.bid
                if price_drifted(best_sig.side, best_sig.entry_price, px, best_sig.atr, self.config.supervisor.max_price_drift_atr):
                    await self._log_decision(
                        DecisionEvent(symbol=sym, category="BLOCK", tag="STALE AFTER REVIEW", strategy_id=best_sig.strategy_id,
                                      side=best_sig.side.value, message=f"Price moved >{self.config.supervisor.max_price_drift_atr} ATR against entry while reviewing{note}")
                    )
                    return
            await self._finalize_entry(sym, best_sig, risk_dec, assess, spec, ml_prob, dl_align, b_weight, spread_pts, lat_ms, note)
        except Exception as exc:  # never let a background review crash the engine
            await self._log_decision(
                DecisionEvent(symbol=sym, category="BLOCK", tag="SUPERVISOR ERROR", message=str(exc)[:200])
            )
        finally:
            self._supervising.discard(sym)

    async def _finalize_entry(
        self, sym: str, best_sig: Any, risk_dec: Any, assess: Any, spec: SymbolSpec,
        ml_prob: float, dl_align: float, b_weight: float, spread_pts: float, lat_ms: float, note: str = "",
    ) -> None:
        tick = self.ticks.get(sym)
        if self.config.paper.enabled:
            await self._open_paper_position(sym, best_sig, risk_dec, assess, spec, ml_prob, dl_align, spread_pts, books=VETOED_BOOKS)
        # In SHADOW_ONLY mode, log & skip real MT5 dispatch
        if self.config.app.trading_mode == TradingMode.SHADOW_ONLY:
            await self._log_decision(
                DecisionEvent(
                    symbol=sym,
                    category="AI",
                    tag="SHADOW ONLY",
                    strategy_id=best_sig.strategy_id,
                    side=best_sig.side.value,
                    message=f"Paper trade {best_sig.side.value} {risk_dec.volume} lot (ML {ml_prob:.0%}, DL {dl_align:.0%}) — no MT5 order in SHADOW_ONLY{note}",
                    latency_ms=lat_ms,
                )
            )
            return

        # Exit style is fixed at entry: a Hit & Run position gets its own SL and lot from the start.
        volume, sl, ts_bars = risk_dec.volume, risk_dec.adjusted_sl, 0
        style = self._mt5_exit_style(sym, assess.regime)
        if style == "hitrun" and self.config.paper.hit_run.own_stop:
            px = tick.ask if (tick and best_sig.side == Side.BUY) else (tick.bid if tick else best_sig.entry_price)
            spread_px = max(tick.ask - tick.bid, 0.0) if tick else spread_pts * spec.point
            plan = self._hitrun_plan(sym, best_sig.side, px, best_sig.atr, spread_px, spec, sl, volume,
                                     self._hitrun_arm(sym, assess.regime))
            if plan is None:
                style = "normal"
                note += " | Hit & Run cap not reachable with the minimum lot: Normal exit"
            else:
                volume, sl, ts_bars = plan.volume, plan.sl, plan.arm.time_stop_bars
                note += f" | Hit & Run: SL {plan.sl}, max loss ${plan.max_loss:.2f}, {plan.arm.label()}"

        # Dispatch idempotent OrderCommand to MT5 EA
        now_ms = utc_now_ms()
        cmd = OrderCommand(
            action=OrderAction.OPEN,
            symbol=sym,
            broker_symbol=spec.broker_symbol,
            side=best_sig.side,
            volume=volume,
            sl=sl,
            tp=risk_dec.adjusted_tp,
            magic=self.config.instruments.magic_number,
            strategy_id=best_sig.strategy_id,
            expected_price=tick.ask if (tick and best_sig.side == Side.BUY) else (tick.bid if tick else best_sig.entry_price),
            created_at_ms=now_ms,
            expires_at_ms=now_ms + self.config.instruments.order_ttl_ms,
        )
        queued = self.order_sm.enqueue(
            cmd,
            strategy_id=best_sig.strategy_id,
            regime=assess.regime,
            ml_confidence=ml_prob,
            dl_confidence=dl_align,
            bandit_weight=b_weight,
            spread_at_entry=spread_pts,
            exit_style=style,
            time_stop_bars=ts_bars,
            tp_stages=self._stages_inside(best_sig, best_sig.entry_price, risk_dec.adjusted_tp) if style != "hitrun" else [],
        )
        if queued:
            self.active_thought = (
                f"FIRING {sym} {best_sig.side.value} {volume} lot via {best_sig.strategy_id} "
                f"(ML {ml_prob:.0%}, DL {dl_align:.0%}, Bandit {b_weight:.0%})"
            )
            await self._log_decision(
                DecisionEvent(
                    symbol=sym,
                    category="EXEC",
                    tag="ORDER SENT",
                    strategy_id=best_sig.strategy_id,
                    side=best_sig.side.value,
                    message=f"{best_sig.side.value} {volume} lot | ML {ml_prob:.0%} • DL {dl_align:.0%} • {best_sig.reason}{note}",
                    latency_ms=lat_ms,
                )
            )

    async def _save_arena(self) -> None:
        await self.db.set_kv(
            "shadow_arena",
            {
                "equity": self.shadow_equity,
                "positions": [_pos_to_dict(p) for p in self.shadow_positions.values()],
                "raw_equity": self.shadow_raw_equity,
                "raw_positions": [_pos_to_dict(p) for p in self.shadow_raw_positions.values()],
            },
            utc_now_ms(),
        )

    async def _save_paper(self) -> None:
        await self.db.set_kv(
            "paper_books",
            {
                name: {"equity": b.equity, "positions": [_pos_to_dict(p) for p in b.positions.values()]}
                for name, b in self.paper_books.items()
            },
            utc_now_ms(),
        )

    async def _open_paper_position(
        self, sym: str, sig: Any, risk_dec: Any, assess: Any, spec: SymbolSpec,
        ml_prob: float, dl_align: float, spread_pts: float,
        books: tuple[str, ...] = VETOED_BOOKS, veto: str = "", context: dict[str, Any] | None = None,
    ) -> None:
        """Fill a trade virtually at bid/ask in the given paper books (one position per symbol)."""
        ctx = context or {}
        tick = self.ticks.get(sym)
        if tick is None or risk_dec.volume <= 0:
            return
        slip = self.paper_costs(sym)["slippage_points"] * spec.point
        entry = (tick.ask + slip) if sig.side == Side.BUY else (tick.bid - slip)
        sl, tp = risk_dec.adjusted_sl, risk_dec.adjusted_tp
        spread_px = max(tick.ask - tick.bid, 0.0)
        h = self.config.paper.hit_run
        opened = False
        for name in books:
            book = self.paper_books[name]
            if sym in book.positions:
                continue
            p_sl, p_vol, p_risk, p_ts = sl, risk_dec.volume, max(abs(entry - sl), spec.point), 0
            if name in HITRUN_BOOK_NAMES and h.own_stop:
                plan = self._hitrun_plan(sym, sig.side, entry, sig.atr, spread_px, spec, sl, risk_dec.volume,
                                         self._hitrun_arm(sym, assess.regime))
                if plan is None:
                    continue  # even the minimum lot would lose more than the Hit & Run cap
                p_sl, p_vol, p_risk, p_ts = plan.sl, plan.volume, plan.risk_dist, plan.arm.time_stop_bars
            book.positions[sym] = ShadowPosition(
                symbol=sym,
                strategy_id=sig.strategy_id,
                regime=assess.regime,
                side=sig.side,
                entry_price=round(entry, spec.digits),
                sl=p_sl,
                tp=tp,
                risk_dist=p_risk,
                open_time=int(utc_now_ms() // 1000),
                ml_conf=ml_prob,
                dl_conf=dl_align,
                spread_pts=spread_pts,
                version_id=PAPER_BOOKS[name],
                volume=p_vol,
                veto=veto if name == "raw" else "",
                time_stop_bars=p_ts,
                htf_m15=int(ctx.get("htf_m15", 0)),
                htf_h1=int(ctx.get("htf_h1", 0)),
                session=str(ctx.get("session", "")),
                ichi_m15=int(ctx.get("ichi_m15", 0)),
                tp_stages=self._stages_inside(sig, entry, tp) if name not in HITRUN_BOOK_NAMES else [],
            )
            opened = True
        if "raw" in books and h.own_stop and h.adaptive:
            # Ghost trades for every Hit & Run stop setting, on every risk-approved setup.
            plans = [
                pl for arm in self.hitrun_tuner.arms(self._hitrun_default_arm())
                if (pl := self._hitrun_plan(sym, sig.side, entry, sig.atr, spread_px, spec, sl, risk_dec.volume, arm))
            ]
            self.hitrun_tuner.spawn(sym, assess.regime.value, sig.side, entry, int(utc_now_ms() // 1000), plans)
        if opened:
            await self._save_paper()

    @staticmethod
    def _stages_inside(sig: Any, entry: float, tp: float) -> list[float]:
        """The signal's staged targets that lie between the entry and the final TP."""
        stages = list(getattr(sig, "tp_stages", None) or [])
        lo, hi = min(entry, tp), max(entry, tp)
        return [float(x) for x in stages if lo < x < hi]

    def _paper_money(self, p: ShadowPosition, exit_px: float) -> float:
        spec = self.specs.get(p.symbol)
        move = (exit_px - p.entry_price) if p.side == Side.BUY else (p.entry_price - exit_px)
        if spec is None or spec.tick_size <= 0:
            return 0.0
        return round(move / spec.tick_size * spec.tick_value * p.volume, 2)

    async def _rebuild_exit_selector(self) -> None:
        """Re-learn the exit-style comparison from every closed paper trade in the journal."""
        self.exit_selector = ExitStyleSelector(min_samples=self.config.exits.adaptive_min_samples)
        for name, version in PAPER_BOOKS.items():
            trades = await self.db.list_trades(limit=20000, version_id=version)
            for t in sorted(trades, key=lambda t: t.close_time):
                self.exit_selector.record(name, t.symbol, t.regime.value, t.pnl_r)

    def _mt5_exit_style(self, symbol: str, regime: MarketRegime) -> str:
        if not self.config.exits.adaptive:
            return "normal"
        return self.exit_selector.choose(symbol, regime.value).style

    async def _learn_broker_costs(self, new_trades: list[TradeRecord] | None = None) -> None:
        """Commission per lot and slippage per symbol, from the last closed MT5 trades.

        MT5 reports the entry deal's commission; the exit side usually costs the same, so the
        round trip is taken as twice the entry commission."""
        trades = await self.db.list_trades(limit=2000, is_shadow=False, exclude_version_prefix="PAPER")
        if new_trades:
            seen = {t.trade_id for t in trades}
            trades += [t for t in new_trades if t.trade_id not in seen]
        by_sym: dict[str, list[TradeRecord]] = {}
        for t in sorted(trades, key=lambda t: t.close_time):
            if t.volume > 0:
                by_sym.setdefault(t.symbol, []).append(t)
        out: dict[str, dict[str, float]] = {}
        for sym, ts in by_sym.items():
            last = ts[-30:]
            comm = sorted(abs(t.commission) * 2 / t.volume for t in last)
            slip = sorted(abs(t.slippage_points) for t in last)
            out[sym] = {
                "commission_per_lot": round(comm[len(comm) // 2], 2),
                "slippage_points": round(slip[len(slip) // 2], 1),
                "trades": float(len(last)),
            }
        self.learned_costs = out

    def paper_costs(self, symbol: str) -> dict[str, Any]:
        """Costs used for paper trades on ``symbol``, and where each number comes from."""
        p = self.config.paper
        learned = self.learned_costs.get(symbol)
        spec = self.specs.get(symbol)
        if p.commission_per_lot is not None:
            comm, comm_src = p.commission_per_lot, "manual"
        elif learned:
            comm, comm_src = learned["commission_per_lot"], "mt5"
        else:
            comm, comm_src = (spec.commission_per_lot if spec else 0.0), "default"
        if p.slippage_points is not None:
            slip, slip_src = p.slippage_points, "manual"
        elif learned:
            slip, slip_src = learned["slippage_points"], "mt5"
        else:
            slip, slip_src = 0.0, "default"
        return {
            "symbol": symbol,
            "commission_per_lot": float(comm),
            "commission_source": comm_src,
            "slippage_points": float(slip),
            "slippage_source": slip_src,
            "mt5_trades": int(learned["trades"]) if learned else 0,
        }

    def _paper_exit_px(self, p: ShadowPosition, px: float) -> float:
        spec = self.specs.get(p.symbol)
        slip = self.paper_costs(p.symbol)["slippage_points"] * (spec.point if spec else 0.0)
        return px - slip if p.side == Side.BUY else px + slip

    def _hitrun_default_arm(self) -> HitRunArm:
        h = self.config.paper.hit_run
        return HitRunArm(h.sl_atr_mult, h.time_stop_bars)

    def _hitrun_arm(self, symbol: str, regime: MarketRegime) -> HitRunArm:
        h = self.config.paper.hit_run
        default = self._hitrun_default_arm()
        if not h.adaptive:
            return default
        self.hitrun_tuner.min_samples = h.adaptive_min_samples
        return self.hitrun_tuner.choose(symbol, regime.value, default).arm

    def _hitrun_plan(
        self, sym: str, side: Side, entry: float, atr: float, spread_px: float, spec: SymbolSpec,
        normal_sl: float, normal_volume: float, arm: HitRunArm,
    ) -> HitRunPlan | None:
        h = self.config.paper.hit_run
        return plan_stop(
            side=side, entry=entry, atr=atr, spread_px=spread_px, spec=spec, normal_sl=normal_sl,
            normal_volume=normal_volume, commission_per_lot=self.paper_costs(sym)["commission_per_lot"],
            min_net_profit=h.min_net_profit, max_loss_mult=h.max_loss_mult,
            sl_min_spread_mult=h.sl_min_spread_mult, arm=arm,
        )

    def _ghost_net(self, g: Ghost, px: float) -> float:
        spec = self.specs.get(g.symbol)
        if spec is None or spec.tick_size <= 0:
            return 0.0
        costs = self.paper_costs(g.symbol)
        slip = costs["slippage_points"] * spec.point
        exit_px = px - slip if g.side == Side.BUY else px + slip
        move = (exit_px - g.entry) if g.side == Side.BUY else (g.entry - exit_px)
        return round(move / spec.tick_size * spec.tick_value * g.volume - costs["commission_per_lot"] * g.volume, 2)

    def hitrun_learning(self) -> dict[str, Any]:
        h = self.config.paper.hit_run
        default = self._hitrun_default_arm()
        return {
            "adaptive": h.adaptive,
            "min_samples": h.adaptive_min_samples,
            "default": default.label(),
            "groups": self.hitrun_tuner.table(default),
        }

    def _net_take(self, symbol: str) -> NetTake:
        h = self.config.paper.hit_run
        return NetTake(
            min_net_profit=h.min_net_profit,
            commission_per_lot=self.paper_costs(symbol)["commission_per_lot"],
            min_cost_multiple=h.min_cost_multiple,
        )

    def _commission_price(
        self,
        symbol: str,
        spec: SymbolSpec | None,
        live_comm: float | None = None,
        live_vol: float = 0.0,
    ) -> float:
        """Convert round-trip broker commission into exact price distance so Breakeven truly breaks even."""
        if spec is None or spec.tick_size <= 0 or spec.tick_value <= 0:
            return 0.0
        money_per_px = spec.tick_value / spec.tick_size
        if live_comm is not None and live_comm != 0 and live_vol > 0:
            comm_per_lot = abs(live_comm) * 2.0 / live_vol
        else:
            comm_per_lot = self.paper_costs(symbol)["commission_per_lot"]
        return max(0.0, comm_per_lot / money_per_px)

    def _exit_policy(self) -> ProfitLock:
        e = self.config.exits
        return ProfitLock(be_at_r=e.be_at_r, be_offset_r=e.be_offset_r, trail_start_r=e.trail_start_r, trail_dist_r=e.trail_dist_r)

    async def _update_paper_positions_on_tick(self, tk: TickSnapshot, now_sec: int) -> None:
        changed = False
        if self.hitrun_tuner.ghosts.get(tk.symbol):
            spec_g = self.specs.get(tk.symbol)
            sp_px_g = (tk.ask - tk.bid) if tk.ask > tk.bid else (tk.spread_points * (spec_g.point if spec_g else 0.0))
            comm_per_lot_g = self.paper_costs(tk.symbol)["commission_per_lot"]

            def _ghost_friction(g: Ghost) -> float:
                sp_m = (sp_px_g / spec_g.tick_size * spec_g.tick_value * g.volume) if (spec_g and spec_g.tick_size > 0) else 0.0
                return comm_per_lot_g * g.volume + sp_m

            if self.hitrun_tuner.on_tick(
                tk.symbol,
                tk.bid,
                tk.ask,
                now_sec,
                self.config.paper.hit_run.min_net_profit,
                self._ghost_net,
                min_cost_multiple=self.config.paper.hit_run.min_cost_multiple,
                friction_at=_ghost_friction,
            ):
                await self.db.set_kv("hitrun_tuner", self.hitrun_tuner.to_dict(), utc_now_ms())

        for name, book in self.paper_books.items():
            p = book.positions.get(tk.symbol)
            if p is None:
                continue
            px = tk.bid if p.side == Side.BUY else tk.ask
            cur_r = ((px - p.entry_price) if p.side == Side.BUY else (p.entry_price - px)) / max(p.risk_dist, 1e-12)
            p.mae_r = min(p.mae_r, cur_r)
            p.mfe_r = max(p.mfe_r, cur_r)
            spec = self.specs.get(p.symbol)
            spread_px = (tk.ask - tk.bid) if tk.ask > tk.bid else (tk.spread_points * (spec.point if spec else 0.0))
            comm_px = self._commission_price(p.symbol, spec)
            spread_money = (spread_px / spec.tick_size * spec.tick_value * p.volume) if (spec and spec.tick_size > 0) else 0.0
            close_reason = ""
            if name in HITRUN_BOOK_NAMES:
                # Hit & Run (both filtered and No Veto): take net profit above min_net_profit and cost-multiple floor.
                take, net = net_take(
                    self._net_take(p.symbol),
                    gross_money=self._paper_money(p, self._paper_exit_px(p, px)),
                    volume=p.volume,
                    spread_money=spread_money,
                )
                action, new_sl = ("CLOSE" if take else "HOLD"), p.sl
                if not take and time_stop_due(p.open_time, now_sec, p.time_stop_bars, net):
                    action, close_reason = "CLOSE", "TIME_STOP"
            else:
                action, new_sl = evaluate(
                    self._exit_policy(), side=p.side, entry=p.entry_price, risk=p.risk_dist,
                    price=px, peak_r=p.mfe_r, sl=p.sl, spread_px=spread_px, commission_px=comm_px,
                )
            if name not in HITRUN_BOOK_NAMES and p.tp_stages:
                peak_px = p.entry_price + p.mfe_r * p.risk_dist if p.side == Side.BUY else p.entry_price - p.mfe_r * p.risk_dist
                staged = stage_stop(side=p.side, entry=p.entry_price, stages=p.tp_stages, peak_price=peak_px,
                                    sl=new_sl if action == "MOVE_SL" else p.sl, cost_px=spread_px + comm_px)
                if staged is not None and action != "CLOSE":
                    action, new_sl = "MOVE_SL", staged
            if action == "MOVE_SL":
                p.sl = round(new_sl, spec.digits if spec else 5)
                changed = True
            hit_sl = p.sl > 0 and ((p.side == Side.BUY and px <= p.sl) or (p.side == Side.SELL and px >= p.sl))
            hit_tp = p.tp > 0 and ((p.side == Side.BUY and px >= p.tp) or (p.side == Side.SELL and px <= p.tp))
            timed_out = now_sec - p.open_time >= self.config.paper.max_hold_min * 60
            if not (action == "CLOSE" or hit_sl or hit_tp or timed_out):
                continue
            book.positions.pop(tk.symbol, None)
            changed = True
            # All paper books pay the same round-trip commission so every comparison is fair (including Net R).
            exit_px = self._paper_exit_px(p, px)
            pnl = round(self._paper_money(p, exit_px) - self._net_take(p.symbol).commission_per_lot * p.volume, 2)
            exit_r = ((exit_px - p.entry_price) if p.side == Side.BUY else (p.entry_price - exit_px)) / max(p.risk_dist, 1e-12)
            net_r = round(exit_r - (comm_px / max(p.risk_dist, 1e-12)), 4)
            book.equity = round(book.equity + pnl, 2)
            self.exit_selector.record(name, p.symbol, p.regime.value, net_r)
            if name == "raw":
                await self._learn_from_unvetoed(p, net_r, pnl, now_sec)
            in_profit_stop = hit_sl and cur_r > 0
            reason = close_reason or (
                "TP" if hit_tp else "TAKE" if action == "CLOSE" else "LOCK" if in_profit_stop
                else "SL" if hit_sl else "TIME_EXPIRY"
            )
            await self.db.save_trade(
                TradeRecord(
                    is_shadow=False,
                    version_id=PAPER_BOOKS[name],
                    symbol=p.symbol,
                    strategy_id=p.strategy_id,
                    regime=p.regime,
                    side=p.side,
                    volume=p.volume,
                    entry_price=p.entry_price,
                    exit_price=round(exit_px, spec.digits if spec else 5),
                    sl_price=p.sl,
                    tp_price=p.tp,
                    open_time=p.open_time,
                    close_time=now_sec,
                    pnl_money=pnl,
                    commission=-round(self._net_take(p.symbol).commission_per_lot * p.volume, 2),
                    slippage_points=self.paper_costs(p.symbol)["slippage_points"],
                    pnl_r=net_r,
                    mae_r=round(p.mae_r, 4),
                    mfe_r=round(p.mfe_r, 4),
                    spread_at_entry=p.spread_pts,
                    ml_confidence=p.ml_conf,
                    dl_confidence=p.dl_conf,
                    exit_reason=reason,
                )
            )
            await self._log_decision(
                DecisionEvent(
                    symbol=p.symbol,
                    category="EXEC",
                    tag=f"{PAPER_LABELS[name].upper()} {reason}",
                    strategy_id=p.strategy_id,
                    side=p.side.value,
                    message=f"Paper {PAPER_LABELS[name]} {p.side.value} {p.volume} lot closed @ {px} ({net_r:+.2f}R, ${pnl:+.2f}). Equity ${book.equity:,.2f}",
                    metadata={"pnl": pnl, "pnl_r": round(net_r, 3)},
                )
            )
        if changed:
            await self._save_paper()

    async def _learn_from_unvetoed(self, p: ShadowPosition, r: float, pnl: float, now_sec: int) -> None:
        """Online learning from a closed "No Veto" trade. The labels are market results only;
        the supervisor's verdict is never a training target (it is only counted for the scoreboard)."""
        self.bandit.update(p.symbol, p.regime, p.strategy_id, r)
        moved_to_be = (p.sl >= p.entry_price) if p.side == Side.BUY else (0 < p.sl <= p.entry_price)
        self.rl_exit.update_from_closed_trade(
            peak_r=p.mfe_r,
            realized_r=r,
            bars_held=max(1, (now_sec - p.open_time) // 60),
            regime=p.regime,
            dl_aligned=p.dl_conf >= 0.5,
            last_action=RLExitAction.MOVE_BE if moved_to_be else RLExitAction.HOLD,
        )
        if p.session:
            self.filter_learner.record(p.htf_m15, p.htf_h1, p.session, r, ichi=p.ichi_m15)
            await self.db.set_kv("filter_stats", self.filter_learner.to_dict(), utc_now_ms())
        key = p.veto if p.veto in VETO_KEYS else "NONE"
        st = self.veto_stats.setdefault(key, {"n": 0, "wins": 0, "sum_r": 0.0, "sum_money": 0.0})
        st["n"] += 1
        st["wins"] += int(pnl > 0)
        st["sum_r"] = round(st["sum_r"] + r, 4)
        st["sum_money"] = round(st["sum_money"] + pnl, 2)
        now_ms = utc_now_ms()
        await self.db.set_kv("bandit_state", self.bandit.to_dict(), now_ms)
        await self.db.set_kv("rl_exit_state", self.rl_exit.to_dict(), now_ms)
        await self.db.set_kv("veto_stats", self.veto_stats, now_ms)

    def _entry_filter(self, ctx: dict[str, Any], *, strategy_id: str = "", regime: str = ""):
        f = self.config.filters
        self.filter_learner.min_samples = f.min_samples
        session_mode = "on" if self.config.shields.preferred_sessions_only and f.session_mode == "off" else f.session_mode
        return decide(
            self.filter_learner, htf_mode=f.htf_mode, htf_timeframe=f.htf_timeframe, session_mode=session_mode,
            blocked_sessions=f.blocked_sessions, m15=ctx["htf_m15"], h1=ctx["htf_h1"], session=ctx["session"],
            ichimoku_mode=f.ichimoku_mode, ichi=int(ctx.get("ichi_m15", 0)), ichi_threshold=f.ichimoku_threshold,
            strategy_id=strategy_id, regime=regime,
            trend_strategies_require_htf=f.trend_strategies_require_htf,
            block_breakout_in_ranging=f.block_breakout_in_ranging,
        )

    def _maybe_trigger_auto_warmup(self) -> None:
        """Auto-run the first DL/ML training cycle once historical backfill settles or enough bars exist."""
        if (
            self._warmup_triggered
            or self.evolution_running
            or not self.config.evolution.auto_warmup_on_backfill
            or (self.dl_net.trained and self.ml_labeler.trained)
        ):
            return
        last_bf = getattr(self, "_last_backfill_ts", 0.0)
        if last_bf > 0.0:
            # Wait 2.5s after the last history_bars chunk so multi-symbol chunked backfill completes first.
            if (time.monotonic() - last_bf) < 2.5:
                return
            min_req = max(120, self.config.evolution.min_bars_for_wfo)
        else:
            # Without explicit MT5 history backfill, require a deeper live bar cushion (600+ M1 bars).
            min_req = max(600, self.config.evolution.min_bars_for_wfo)
        ready = sum(1 for buf in self.bars.values() if len(buf) >= min_req)
        if ready == 0:
            return
        self._warmup_triggered = True
        try:
            task = asyncio.create_task(self._run_auto_warmup())
            self._bg_tasks.add(task)
            task.add_done_callback(self._bg_tasks.discard)
        except RuntimeError:
            self._warmup_triggered = False

    async def _run_auto_warmup(self) -> None:
        try:
            await self.run_evolution_cycle()
        except Exception:
            self._warmup_triggered = False

    def build_desk_brief_digest(self) -> str:
        """Build compact multi-symbol market digest for periodic AI Supervisor Desk Brief."""
        import json

        rows = []
        for sym in self.config.instruments.symbols:
            bars = list(self.bars.get(sym, ()))
            tk = self.ticks.get(sym)
            spec = self.specs.get(sym, DEFAULT_SPECS["XAUUSD"])
            if len(bars) < 40:
                continue
            assess = classify_regime(bars[-SIGNAL_BARS:])
            if assess is None:
                continue
            m15_b = htf_bias(bars, 15)
            h1_b = htf_bias(bars, 60)
            ichi_s = ichimoku.htf_score(bars, 15)
            sp_pts = tk.spread_points if tk else bars[-1].spread_points
            sp_atr = round((sp_pts * spec.point) / max(assess.indicators.atr14, 1e-9), 3)
            rows.append({
                "symbol": sym,
                "regime": assess.regime.value,
                "vol_ratio": assess.volatility_ratio,
                "adx14": round(assess.indicators.adx14, 1),
                "m15_bias": m15_b,
                "h1_bias": h1_b,
                "ichi_m15": ichi_s,
                "spread_atr": sp_atr,
                "shield": self.shield_status.get(sym, "CLEAR"),
            })
        payload = {
            "utc_hour": int((utc_now_ms() // 1000 % 86400) // 3600),
            "daily_dd_pct": round(self.account.daily_dd_pct, 2),
            "daily_dd_limit_pct": self.config.risk.max_daily_drawdown_pct,
            "consecutive_losses": self.risk.consecutive_losses,
            "recent_trades_r": [round(t.pnl_r, 2) for t in list(self._recent_closed)[-8:]],
            "symbols": rows,
        }
        return "Multi-symbol desk digest:\n" + json.dumps(payload, separators=(",", ":"))

    async def refresh_supervisor_brief(self, force: bool = False) -> dict[str, Any]:
        if not self.supervisor.active or self._brief_refreshing:
            return self.supervisor.status()
        if not force and not self.supervisor.brief_is_due():
            return self.supervisor.status()
        digest = self.build_desk_brief_digest()
        if '"symbols":[]' in digest:
            return self.supervisor.status()
        self._brief_refreshing = True
        try:
            brief = await self.supervisor.refresh_brief(digest)
            if brief:
                summary = ", ".join(f"{k}:{v.bias}" for k, v in brief.items())
                await self._log_decision(
                    DecisionEvent(
                        symbol="SUPERVISOR",
                        category="AI",
                        tag="DESK BRIEF",
                        message=f"Updated {self.supervisor.brief_provider}: {summary} | {self.supervisor.brief_summary}",
                    )
                )
        finally:
            self._brief_refreshing = False
        return self.supervisor.status()

    async def reset_paper_books(self, reset_learners: bool = True) -> dict[str, Any]:
        """Archive/wipe pre-fix virtual paper & shadow history for a clean forward test marathon."""
        deleted = await self.db.clear_paper_trades()
        start = self.config.paper.start_equity
        self.paper_books = {k: PaperBook(equity=start) for k in PAPER_BOOKS}
        self.shadow_positions.clear()
        self.shadow_equity = start
        self.shadow_raw_positions.clear()
        self.shadow_raw_equity = start
        self.veto_stats = {}
        self.exit_selector = ExitStyleSelector(min_samples=self.config.exits.adaptive_min_samples)
        self.hitrun_tuner = HitRunTuner(min_samples=self.config.paper.hit_run.adaptive_min_samples)
        self.filter_learner = FilterLearner(min_samples=self.config.filters.min_samples)
        now_ms = utc_now_ms()
        if reset_learners:
            self.bandit = ContextualBanditRL(forgetting_factor=self.config.ai.bandit_forgetting_factor)
            self.rl_exit = RLExitAgent(
                be_trigger_r=self.config.ai.rl_exit_be_trigger_r,
                trail_trigger_r=self.config.ai.rl_exit_trail_trigger_r,
            )
            await self.db.set_kv("bandit_state", self.bandit.to_dict(), now_ms)
            await self.db.set_kv("rl_exit_state", self.rl_exit.to_dict(), now_ms)
        await self._save_paper()
        await self._save_arena()
        await self.db.set_kv("veto_stats", self.veto_stats, now_ms)
        await self.db.set_kv("hitrun_tuner", self.hitrun_tuner.to_dict(), now_ms)
        await self.db.set_kv("filter_stats", self.filter_learner.to_dict(), now_ms)
        await self._log_decision(
            DecisionEvent(
                symbol="PAPER",
                category="EVOLUTION",
                tag="PAPER RESET",
                message=f"Cleared {deleted} old paper/shadow trades and reset paper books to ${start:,.0f} (learners reset={reset_learners})",
            )
        )
        return {"cleared_trades": deleted, "start_equity": start, "reset_learners": reset_learners}

    def filters_view(self) -> dict[str, Any]:
        f = self.config.filters
        self.filter_learner.min_samples = f.min_samples
        now_ts = int(utc_now_ms() // 1000)
        upcoming = []
        for ev in self.shields.news_events():
            try:
                ts = int(datetime.fromisoformat(str(ev["timestamp_utc"]).replace("Z", "+00:00")).timestamp())
            except (KeyError, ValueError):
                continue
            if -3600 <= ts - now_ts <= 36 * 3600:
                upcoming.append({"title": ev.get("title", ""), "time_utc": ts})
        upcoming.sort(key=lambda e: e["time_utc"])
        return {
            "settings": f.model_dump(),
            "news": {
                "enabled": self.config.shields.news_blackout_enabled,
                "buffer_minutes": self.config.shields.news_buffer_minutes,
                "from_mt5": bool(self.shields.live_news_updated_ts),
                "mt5_updated_ts": self.shields.live_news_updated_ts,
                "upcoming": upcoming[:12],
            },
            "htf": {tf: self.filter_learner.table(f"htf:{tf}", HTF_BUCKETS) for tf in ("M15", "H1", "BOTH")},
            "session": self.filter_learner.table("session", SESSIONS),
            "ichimoku": self.filter_learner.table(f"ichi:{f.ichimoku_threshold}", ICHI_BUCKETS),
            "server_offset_sec": self.server_offset_sec,
        }

    def veto_scoreboard(self) -> list[dict[str, Any]]:
        rows = []
        for k in VETO_KEYS:
            st = self.veto_stats.get(k)
            if not st or not st["n"]:
                continue
            rows.append({
                "verdict": k, "trades": st["n"], "win_rate": round(st["wins"] / st["n"] * 100, 1),
                "avg_r": round(st["sum_r"] / st["n"], 3), "net_pnl": st["sum_money"],
            })
        return rows

    def paper_snapshot(self) -> dict[str, Any]:
        books = {}
        for name, book in self.paper_books.items():
            out = []
            for p in book.positions.values():
                tk = self.ticks.get(p.symbol)
                px = (tk.bid if p.side == Side.BUY else tk.ask) if tk else p.entry_price
                out.append({
                    "book": name, "label": PAPER_LABELS[name],
                    "symbol": p.symbol, "side": p.side.value, "volume": p.volume, "entry_price": p.entry_price,
                    "current_price": px, "sl": p.sl, "tp": p.tp, "strategy_id": p.strategy_id,
                    "profit": self._paper_money(p, px), "open_time": p.open_time,
                })
            books[name] = {"label": PAPER_LABELS[name], "equity": book.equity, "positions": out}
        normal = books["normal"]
        # "equity"/"positions" keep the earlier shape (Normal book) for older dashboard code.
        return {"equity": normal["equity"], "positions": [p for b in books.values() for p in b["positions"]], "books": books}

    async def _manage_open_positions_with_rl_exit(self, now_ms: int) -> None:
        for pos in self.open_positions:
            meta = self.order_sm.active_positions.get(pos.ticket)
            if meta is None:
                continue
            spec = self.specs.get(pos.symbol, DEFAULT_SPECS["XAUUSD"])
            cur_r = (
                (pos.current_price - meta.entry_price) / meta.initial_risk_price
                if meta.side == Side.BUY
                else (meta.entry_price - pos.current_price) / meta.initial_risk_price
            )
            # Profit lock first: BE at +be_at_r, trail from +trail_start_r. Independent of the RL
            # agent, which only moved to BE at +0.8R and let +0.5R trades fall back to a full loss.
            tk = self.ticks.get(pos.symbol)
            spread_px = (tk.ask - tk.bid) if tk and tk.ask > tk.bid else 0.0
            comm_px = self._commission_price(pos.symbol, spec, live_comm=meta.commission, live_vol=pos.volume)
            spread_money = (spread_px / spec.tick_size * spec.tick_value * pos.volume) if spec.tick_size > 0 else 0.0
            style = meta.exit_style or self._mt5_exit_style(pos.symbol, meta.regime)
            close_tag = "HIT & RUN TAKE"
            if style == "hitrun":
                # MT5 profit is valued at the closing price (spread included); add swap and the
                # round-trip commission (MT5 reports the entry side, the exit side costs the same).
                comm = abs(meta.commission) * 2 if meta.commission else None
                take, net = net_take(
                    self._net_take(pos.symbol), gross_money=pos.profit, volume=pos.volume,
                    swap=pos.swap, commission=comm, spread_money=spread_money,
                )
                lock_action, lock_sl = ("CLOSE" if take else "HOLD"), pos.sl
                if not take and meta.time_stop_bars and meta.bars_held >= meta.time_stop_bars and net <= 0:
                    lock_action, close_tag = "CLOSE", "HIT & RUN TIME STOP"
            else:
                lock_action, lock_sl = evaluate(
                    self._exit_policy(), side=meta.side, entry=meta.entry_price, risk=meta.initial_risk_price,
                    price=pos.current_price, peak_r=meta.mfe_r, sl=pos.sl, spread_px=spread_px, commission_px=comm_px,
                )
                if meta.tp_stages:
                    risk = meta.initial_risk_price
                    peak_px = meta.entry_price + meta.mfe_r * risk if meta.side == Side.BUY else meta.entry_price - meta.mfe_r * risk
                    staged = stage_stop(side=meta.side, entry=meta.entry_price, stages=meta.tp_stages, peak_price=peak_px,
                                        sl=lock_sl if lock_action == "MOVE_SL" else pos.sl, cost_px=spread_px + comm_px)
                    if staged is not None:
                        lock_action, lock_sl = "MOVE_SL", staged
            if lock_action == "CLOSE":
                trace = close_tag.replace(" & ", "_").replace(" ", "_")
                if meta.rl_trace[-1] != trace:
                    meta.rl_trace.append(trace)
                    self.order_sm.enqueue(
                        OrderCommand(
                            action=OrderAction.CLOSE,
                            symbol=pos.symbol,
                            broker_symbol=spec.broker_symbol,
                            ticket=pos.ticket,
                            expires_at_ms=now_ms + 3000,
                        )
                    )
                    await self._log_decision(
                        DecisionEvent(
                            symbol=pos.symbol, category="EXEC", tag=close_tag, strategy_id=meta.strategy_id,
                            message=(
                                f"#{pos.ticket} closed at net profit ${net:.2f} ({cur_r:+.2f}R), Hit & Run on {pos.symbol} {meta.regime.value}"
                                if close_tag == "HIT & RUN TAKE" else
                                f"#{pos.ticket} closed: not in net profit after {meta.bars_held} bars (${net:.2f}, {cur_r:+.2f}R)"
                            ),
                        )
                    )
                continue
            if lock_action == "MOVE_SL":
                lock_sl = round(lock_sl, spec.digits)
                ref = meta.last_sl_request or pos.sl
                moved_enough = ref <= 0 or abs(lock_sl - ref) >= spec.point * 5
                better_than_ref = ref <= 0 or ((lock_sl > ref) if meta.side == Side.BUY else (lock_sl < ref))
                if moved_enough and better_than_ref:
                    first = not meta.already_at_be
                    meta.already_at_be = True
                    meta.last_sl_request = lock_sl
                    if meta.rl_trace[-1] != "PROFIT_LOCK":
                        meta.rl_trace.append("PROFIT_LOCK")
                    self.order_sm.enqueue(
                        OrderCommand(
                            action=OrderAction.MODIFY,
                            symbol=pos.symbol,
                            broker_symbol=spec.broker_symbol,
                            ticket=pos.ticket,
                            sl=lock_sl,
                            tp=pos.tp,
                            expires_at_ms=now_ms + 3000,
                        )
                    )
                    if first:
                        await self._log_decision(
                            DecisionEvent(
                                symbol=pos.symbol,
                                category="EXEC",
                                tag="PROFIT LOCK",
                                strategy_id=meta.strategy_id,
                                message=f"#{pos.ticket} reached +{meta.mfe_r:.2f}R: SL moved to {lock_sl} (breakeven + costs)",
                            )
                        )
                    continue

            dl = self.dl_preds.get(pos.symbol, DLPrediction())
            dl_aligned = (dl.prob_up >= dl.prob_down) if meta.side == Side.BUY else (dl.prob_down >= dl.prob_up)

            action = self.rl_exit.select_action(
                unrealized_r=cur_r,
                bars_held=meta.bars_held,
                regime=meta.regime,
                dl_aligned=dl_aligned,
                already_at_be=meta.already_at_be,
                already_partial=meta.already_partial,
            )
            meta.last_rl_action = action
            pos.rl_exit_action = action

            if action == RLExitAction.MOVE_BE and not meta.already_at_be:
                be_buf = 0.06 * meta.initial_risk_price + spread_px + comm_px
                be_sl = (
                    round(meta.entry_price + be_buf, spec.digits)
                    if meta.side == Side.BUY
                    else round(meta.entry_price - be_buf, spec.digits)
                )
                meta.already_at_be = True
                meta.rl_trace.append("MOVE_BE")
                self.order_sm.enqueue(
                    OrderCommand(
                        action=OrderAction.MODIFY,
                        symbol=pos.symbol,
                        broker_symbol=spec.broker_symbol,
                        ticket=pos.ticket,
                        sl=be_sl,
                        tp=pos.tp,
                        expires_at_ms=now_ms + 3000,
                    )
                )
                await self._log_decision(
                    DecisionEvent(
                        symbol=pos.symbol,
                        category="EXEC",
                        tag="RL-2 MOVE_BE",
                        strategy_id=meta.strategy_id,
                        message=f"Moved #{pos.ticket} SL to Breakeven ({be_sl}) at +{cur_r:.2f}R",
                    )
                )
            elif action == RLExitAction.TIGHT_TRAIL:
                trail_sl = (
                    round(pos.current_price - 0.55 * meta.initial_risk_price, spec.digits)
                    if meta.side == Side.BUY
                    else round(pos.current_price + 0.55 * meta.initial_risk_price, spec.digits)
                )
                improves = (meta.side == Side.BUY and trail_sl > pos.sl + spec.point * 5) or (
                    meta.side == Side.SELL and (pos.sl == 0 or trail_sl < pos.sl - spec.point * 5)
                )
                if improves:
                    meta.already_at_be = True
                    if meta.rl_trace[-1] != "TIGHT_TRAIL":
                        meta.rl_trace.append("TIGHT_TRAIL")
                    self.order_sm.enqueue(
                        OrderCommand(
                            action=OrderAction.MODIFY,
                            symbol=pos.symbol,
                            broker_symbol=spec.broker_symbol,
                            ticket=pos.ticket,
                            sl=trail_sl,
                            tp=pos.tp,
                            expires_at_ms=now_ms + 3000,
                        )
                    )
            elif action == RLExitAction.PARTIAL_50 and not meta.already_partial and pos.volume >= spec.volume_min * 2:
                meta.already_partial = True
                meta.rl_trace.append("PARTIAL_50")
                self.order_sm.enqueue(
                    OrderCommand(
                        action=OrderAction.PARTIAL_CLOSE,
                        symbol=pos.symbol,
                        broker_symbol=spec.broker_symbol,
                        ticket=pos.ticket,
                        expires_at_ms=now_ms + 3000,
                    )
                )
            elif action == RLExitAction.EXIT_NOW and cur_r < -0.25 and meta.bars_held >= 9:
                meta.rl_trace.append("EXIT_NOW")
                self.order_sm.enqueue(
                    OrderCommand(
                        action=OrderAction.CLOSE,
                        symbol=pos.symbol,
                        broker_symbol=spec.broker_symbol,
                        ticket=pos.ticket,
                        expires_at_ms=now_ms + 3000,
                    )
                )

    async def _scan_shadow_challenger(
        self,
        sym: str,
        bars: list[Bar],
        snap: Any,
        regime: MarketRegime,
        dl_pred: DLPrediction,
        spread_pts: float,
        spec: SymbolSpec,
        now_sec: int,
        htf: tuple[int, int] = (0, 0),
        ichi_m15: int = 0,
        prev_day: tuple[float, float] | None = None,
    ) -> None:
        if sym in self.shadow_positions and sym in self.shadow_raw_positions:
            return
        cand = self.gate.active_challenger
        shadow_cfg = TradingConfig.model_validate(cand.config_snapshot) if cand else self.config
        sigs = self.shadow_suite.scan_all(sym, bars, snap, regime, context={"prev_day": prev_day, "htf_m15": htf[0], "htf_h1": htf[1]})
        if not sigs:
            return
        best = max(sigs, key=lambda s: s.confidence)
        spread_px = spread_pts * spec.point
        if (spread_px / max(snap.atr14, 1e-9)) > shadow_cfg.risk.max_spread_atr_ratio:
            return

        # Mirror Champion's Cost-to-Risk Gate using real broker commission
        entry = best.entry_price + (spread_px if best.side == Side.BUY else 0.0)
        risk_d = max(abs(entry - best.sl_price), snap.atr14 * 0.5, spec.point)
        comm_px = self._commission_price(sym, spec)
        max_cost_ratio = self.config.risk.max_cost_to_risk_ratio
        if max_cost_ratio > 0 and (spread_px + comm_px) / risk_d > max_cost_ratio:
            return

        # Mirror Champion's RR-aware ML Meta-Labeler gate using Challenger's staged DL/ML bundle when present
        sh_dl = self.shadow_dl_net if (self.shadow_dl_net is not None and self.shadow_dl_net.trained) else self.dl_net
        sh_ml = self.shadow_ml_labeler if (self.shadow_ml_labeler is not None and self.shadow_ml_labeler.trained) else self.ml_labeler
        if sh_dl is not self.dl_net and sh_dl.trained:
            seq = build_sequence_tensor(bars, seq_len=shadow_cfg.ai.dl_seq_len, point=spec.point)
            sh_dl_pred = sh_dl.predict(seq)
        else:
            sh_dl_pred = dl_pred
        feat = extract_meta_features(
            best, snap, spread_px, sh_dl_pred, bars[-1].timestamp, htf=htf, server_offset_sec=self.server_offset_sec,
            structure=structure_context(bars, ichi_m15, prev_day),
        )
        ml_p = sh_ml.predict_win_probability(feat)
        sig_rr = abs(best.tp_price - best.entry_price) / max(abs(best.entry_price - best.sl_price), spec.point)
        if ml_p < effective_ml_cutoff(shadow_cfg.ai.ml_min_probability, sig_rr):
            return

        base_ver = cand.version_id if cand else f"{self.config.app.version}-shadow"
        stages = self._stages_inside(best, entry, best.tp_price)
        changed = False

        # 1) Pure Challenger (Raw): opens before filters & AI Supervisor veto for direct A/B comparison
        if sym not in self.shadow_raw_positions:
            self.shadow_raw_positions[sym] = ShadowPosition(
                symbol=sym,
                strategy_id=best.strategy_id,
                regime=regime,
                side=best.side,
                entry_price=round(entry, spec.digits),
                sl=round(best.sl_price, spec.digits),
                tp=round(best.tp_price, spec.digits),
                risk_dist=risk_d,
                open_time=now_sec,
                ml_conf=ml_p,
                dl_conf=dl_pred.prob_up if best.side == Side.BUY else dl_pred.prob_down,
                spread_pts=spread_pts,
                volume=1.0,
                version_id=f"{base_ver}-raw",
                tp_stages=list(stages),
            )
            changed = True

        # 2) Supervised Challenger (AI Veto): passes P0 Entry Filters + shared 0ms AI Supervisor DeskBrief
        if sym not in self.shadow_positions:
            side_sign = 1.0 if best.side == Side.BUY else -1.0
            entry_ctx = {
                "htf_m15": alignment(side_sign, htf[0]),
                "htf_h1": alignment(side_sign, htf[1]),
                "session": session_of(bars[-1].timestamp, self.server_offset_sec),
                "ichi_m15": int(ichi_m15 * side_sign),
            }
            if not self._entry_filter(entry_ctx, strategy_id=best.strategy_id, regime=regime.value).code:
                size_mult = 1.0
                allowed = True
                reviewed = self.supervisor.active and (
                    self.config.app.trading_mode != TradingMode.SHADOW_ONLY or self.config.supervisor.review_in_shadow
                )
                if reviewed:
                    verdict = self.supervisor.evaluate_from_brief(sym, best.side, best.strategy_id, record_stats=False)
                    allowed = verdict.allows_entry
                    size_mult = max(0.1, min(1.0, verdict.size_multiplier))
                if allowed:
                    self.shadow_positions[sym] = ShadowPosition(
                        symbol=sym,
                        strategy_id=best.strategy_id,
                        regime=regime,
                        side=best.side,
                        entry_price=round(entry, spec.digits),
                        sl=round(best.sl_price, spec.digits),
                        tp=round(best.tp_price, spec.digits),
                        risk_dist=risk_d,
                        open_time=now_sec,
                        ml_conf=ml_p,
                        dl_conf=dl_pred.prob_up if best.side == Side.BUY else dl_pred.prob_down,
                        spread_pts=spread_pts,
                        volume=round(size_mult, 2),
                        version_id=base_ver,
                        tp_stages=list(stages),
                    )
                    changed = True

        if changed:
            await self._save_arena()

    async def _update_shadow_positions_on_tick(self, tk: TickSnapshot, now_sec: int) -> None:
        spec = self.specs.get(tk.symbol)
        spread_px = (tk.ask - tk.bid) if tk.ask > tk.bid else (tk.spread_points * (spec.point if spec else 0.0))
        comm_px = self._commission_price(tk.symbol, spec)
        arena_changed = False

        for is_raw, pos_map in ((False, self.shadow_positions), (True, self.shadow_raw_positions)):
            spos = pos_map.get(tk.symbol)
            if spos is None:
                continue
            px = tk.bid if spos.side == Side.BUY else tk.ask
            cur_r = (
                (px - spos.entry_price) / spos.risk_dist
                if spos.side == Side.BUY
                else (spos.entry_price - px) / spos.risk_dist
            )
            spos.mae_r = min(spos.mae_r, cur_r)
            spos.mfe_r = max(spos.mfe_r, cur_r)

            action, new_sl = evaluate(
                self._exit_policy(), side=spos.side, entry=spos.entry_price, risk=spos.risk_dist,
                price=px, peak_r=spos.mfe_r, sl=spos.sl, spread_px=spread_px, commission_px=comm_px,
            )
            if spos.tp_stages:
                peak_px = spos.entry_price + spos.mfe_r * spos.risk_dist if spos.side == Side.BUY else spos.entry_price - spos.mfe_r * spos.risk_dist
                staged = stage_stop(
                    side=spos.side, entry=spos.entry_price, stages=spos.tp_stages, peak_price=peak_px,
                    sl=new_sl if action == "MOVE_SL" else spos.sl, cost_px=spread_px + comm_px,
                )
                if staged is not None:
                    action, new_sl = "MOVE_SL", staged
            if action == "MOVE_SL":
                spos.sl = round(new_sl, spec.digits if spec else 5)
                arena_changed = True

            hit_sl = (spos.side == Side.BUY and px <= spos.sl) or (spos.side == Side.SELL and px >= spos.sl)
            hit_tp = (spos.side == Side.BUY and px >= spos.tp) or (spos.side == Side.SELL and px <= spos.tp)
            timed_out = (now_sec - spos.open_time) >= self.config.paper.max_hold_min * 60

            if not (hit_sl or hit_tp or timed_out):
                continue

            pos_map.pop(tk.symbol, None)
            arena_changed = True
            comm_r = (comm_px / max(spos.risk_dist, 1e-12)) if comm_px > 0 else 0.04
            net_r = cur_r - comm_r
            size_scale = spos.volume if 0.05 <= spos.volume <= 1.0 else 1.0
            base_eq = self.shadow_raw_equity if is_raw else self.shadow_equity
            risk_money = base_eq * (self.config.risk.risk_per_trade_pct / 100.0) * size_scale
            pnl = round(risk_money * net_r, 2)
            if is_raw:
                self.shadow_raw_equity = round(self.shadow_raw_equity + pnl, 2)
            else:
                self.shadow_equity = round(self.shadow_equity + pnl, 2)
            in_profit_stop = hit_sl and cur_r > 0
            tr = TradeRecord(
                is_shadow=True,
                version_id=spos.version_id,
                symbol=spos.symbol,
                strategy_id=spos.strategy_id,
                regime=spos.regime,
                side=spos.side,
                volume=round(0.10 * size_scale, 2),
                entry_price=spos.entry_price,
                exit_price=px,
                sl_price=spos.sl,
                tp_price=spos.tp,
                open_time=spos.open_time,
                close_time=now_sec,
                pnl_money=pnl,
                pnl_r=round(net_r, 4),
                mae_r=round(spos.mae_r, 4),
                mfe_r=round(spos.mfe_r, 4),
                spread_at_entry=spos.spread_pts,
                ml_confidence=spos.ml_conf,
                dl_confidence=spos.dl_conf,
                exit_reason="TP" if hit_tp else ("LOCK" if in_profit_stop else ("SL" if hit_sl else "TIME_EXPIRY")),
            )
            await self.db.save_trade(tr)
            if not is_raw:
                transition = self.gate.record_shadow_trade(tr)
                if transition and self.gate.active_challenger:
                    await self.db.save_challenger(self.gate.active_challenger)
                if transition == "PENDING_APPROVAL":
                    await self._notify("Challenger passed Live Shadow Arena! Ready for /approve on Dashboard or Telegram.")
                elif transition == "AUTO_PROMOTED":
                    self._activate_challenger_models(self.gate.active_snapshot.version_id)
                    self.apply_champion_config(self.gate.active_config)
                    for sn in self.gate.snapshots:
                        await self.db.save_snapshot(sn)
                    await self._notify(f"Challenger AUTO-PROMOTED to Champion {self.gate.active_snapshot.version_id} (config + DL/ML)!")

        if arena_changed:
            await self._save_arena()

    async def _training_bars(self) -> dict[str, list[Bar]]:
        """Several days of M1 bars per symbol from the database (falls back to the live buffer)."""
        limit = max(600, self.config.evolution.train_window_bars)
        out: dict[str, list[Bar]] = {}
        for sym, buf in self.bars.items():
            stored = await self.db.load_recent_bars(sym, "M1", limit=limit)
            live = list(buf)
            if live and stored and live[-1].timestamp > stored[-1].timestamp:
                stored += [b for b in live if b.timestamp > stored[-1].timestamp]
            bars = stored if len(stored) >= len(live) else live
            if len(bars) >= 120:
                out[sym] = bars[-limit:]
        return out

    async def run_evolution_cycle(self) -> dict[str, Any]:
        self.optimizer.backtester.server_offset_sec = self.server_offset_sec
        """Full 4-Stage Self-Improvement Cycle: Reflection -> Purged WFO + DL/ML Retrain -> Monte Carlo -> Shadow Gate."""
        if self.evolution_running:
            return {"status": "already_running"}
        self.evolution_running = True
        try:
            bars_map = await self._training_bars()
            if not bars_map:
                return {"status": "insufficient_bars", "message": "Need at least 120 closed M1 bars per symbol"}

            recent_trades = await self.db.list_trades(limit=60)
            reflection = await self.reflector.reflect(recent_trades, self.config)
            await self.db.save_reflection(reflection)

            # Run heavy WFO + DL/ML retraining off the event loop so tick handling never stalls
            wfo = await asyncio.to_thread(
                self.optimizer.run,
                bars_map,
                self.config,
                reflection.patches,
                self.specs,
                self.dl_net,
                self.ml_labeler,
                self.config.evolution.param_window_bars,
            )

            # Bootstrap cold-start models once if Champion has no trained DL/ML yet.
            # Otherwise, new DL/ML weights NEVER bypass Shadow Arena: they are staged with the
            # Challenger bundle and only promoted (or rolled back) atomically alongside config.
            active_ver = self.gate.active_snapshot.version_id
            dl_active_path, ml_active_path = self._version_model_paths(active_ver)
            if not self.dl_net.trained and wfo.dl_decision.get("adopted"):
                self.dl_net = wfo.trained_dl
                self.dl_net.save(f"{self.config.app.snapshots_dir}/dl_champion.npz")
                self.dl_net.save(dl_active_path)
            if not self.ml_labeler.trained and wfo.ml_decision.get("adopted"):
                self.ml_labeler = wfo.trained_ml
                self.ml_labeler.save(f"{self.config.app.snapshots_dir}/ml_champion.pkl")
                self.ml_labeler.save(ml_active_path)

            cand = self.gate.submit_candidate(
                candidate_config=wfo.best_config,
                patches=wfo.best_patches,
                challenger_oos=wfo.challenger_oos,
                champion_oos=wfo.champion_oos,
                source="HYBRID" if reflection.patches else "WFO",
                reflection_id=reflection.report_id,
            )
            reflection.gate_status = "IN_SHADOW" if cand.status == "SHADOWING" else "REJECTED"
            await self.db.save_reflection(reflection)
            await self.db.save_challenger(cand)

            if cand.status == "SHADOWING":
                self.shadow_suite = StrategySuite.from_config(wfo.best_config)
                self.shadow_dl_net = wfo.trained_dl if wfo.trained_dl.trained else self.dl_net
                self.shadow_ml_labeler = wfo.trained_ml if wfo.trained_ml.trained else self.ml_labeler
                self._save_version_models(cand.version_id, self.shadow_dl_net, self.shadow_ml_labeler)

            n_bars = {s: len(b) for s, b in bars_map.items()}
            for name, dec in (("1D-CNN", wfo.dl_decision), ("Meta-labeler", wfo.ml_decision)):
                staged = dec.get("adopted") and cand.status == "SHADOWING"
                status_lbl = "STAGED IN SHADOW" if staged else ("ADOPTED (INIT)" if dec.get("adopted") else "KEPT")
                await self._log_decision(
                    DecisionEvent(
                        symbol="EVOLUTION",
                        category="EVOLUTION",
                        tag=f"{name} {status_lbl}",
                        message=(
                            f"{name}: {status_lbl.lower()}"
                            f"{' (continued from the champion weights)' if dec.get('warm_start') else ''}. {dec.get('reason', '')}. "
                            f"Data: {max(n_bars.values())} M1 bars per symbol"
                        ),
                    )
                )

            await self._log_decision(
                DecisionEvent(
                    symbol="EVOLUTION",
                    category="EVOLUTION",
                    tag=f"WFO {cand.status}",
                    message=f"{cand.version_id} | OOS PF={cand.oos_metrics.profit_factor:.2f} MC-P95={cand.oos_metrics.mc_p95_dd_pct:.2f}% | {cand.rejection_reason or 'Deployed to Live Shadow Arena'}",
                )
            )
            await self._notify(
                f"Evolution Cycle Complete ({cand.version_id}): {cand.status}\n"
                f"OOS PF: {cand.oos_metrics.profit_factor:.2f} | MC P95 DD: {cand.oos_metrics.mc_p95_dd_pct:.2f}%"
            )
            return {
                "status": "completed",
                "challenger": cand.model_dump(mode="json"),
                "reflection": reflection.model_dump(mode="json"),
                "dl_train": wfo.dl_train_summary,
                "ml_train": wfo.ml_train_summary,
                "dl_decision": wfo.dl_decision,
                "ml_decision": wfo.ml_decision,
                "training_bars": n_bars,
            }
        finally:
            self.evolution_running = False

    def _version_model_paths(self, version_id: str) -> tuple[str, str]:
        safe = "".join(ch if (ch.isalnum() or ch in ".-_") else "_" for ch in version_id)
        base = self.config.app.snapshots_dir
        return f"{base}/dl_{safe}.npz", f"{base}/ml_{safe}.pkl"

    def _save_version_models(
        self,
        version_id: str,
        dl_net: SequenceConvNet | None,
        ml_labeler: CalibratedMetaLabeler | None,
    ) -> None:
        dl_p, ml_p = self._version_model_paths(version_id)
        if dl_net is not None and dl_net.trained:
            dl_net.save(dl_p)
        if ml_labeler is not None and ml_labeler.trained:
            ml_labeler.save(ml_p)

    def _load_challenger_models(self, cand_version_id: str) -> None:
        dl_p, ml_p = self._version_model_paths(cand_version_id)
        cand_dl = SequenceConvNet()
        self.shadow_dl_net = cand_dl if cand_dl.load(dl_p) and cand_dl.trained else None
        cand_ml = CalibratedMetaLabeler()
        self.shadow_ml_labeler = cand_ml if cand_ml.load(ml_p) and cand_ml.trained else None

    def _activate_challenger_models(self, promoted_version_id: str) -> None:
        """Promote staged Challenger DL/ML weights atomically into active Champion & persist versioned copies."""
        if self.shadow_dl_net is not None and self.shadow_dl_net.trained:
            self.dl_net = self.shadow_dl_net
        if self.shadow_ml_labeler is not None and self.shadow_ml_labeler.trained:
            self.ml_labeler = self.shadow_ml_labeler
        if self.dl_net.trained:
            self.dl_net.save(f"{self.config.app.snapshots_dir}/dl_champion.npz")
        if self.ml_labeler.trained:
            self.ml_labeler.save(f"{self.config.app.snapshots_dir}/ml_champion.pkl")
        self._save_version_models(promoted_version_id, self.dl_net, self.ml_labeler)
        self.shadow_dl_net = None
        self.shadow_ml_labeler = None

    def _restore_version_models(self, version_id: str) -> None:
        """Restore DL/ML weights belonging to a previous Champion snapshot during rollback."""
        dl_p, ml_p = self._version_model_paths(version_id)
        restored_dl = SequenceConvNet()
        if restored_dl.load(dl_p) and restored_dl.trained:
            self.dl_net = restored_dl
            self.dl_net.save(f"{self.config.app.snapshots_dir}/dl_champion.npz")
        restored_ml = CalibratedMetaLabeler()
        if restored_ml.load(ml_p) and restored_ml.trained:
            self.ml_labeler = restored_ml
            self.ml_labeler.save(f"{self.config.app.snapshots_dir}/ml_champion.pkl")

    async def promote_active_challenger(self, note: str = "Approved by operator") -> Any:
        """Atomically promote a PENDING_APPROVAL Challenger (config + staged DL/ML weights) to Champion."""
        prev_ver = self.gate.active_snapshot.version_id
        self._save_version_models(prev_ver, self.dl_net, self.ml_labeler)
        snap = self.gate.promote_active_challenger(note=note, current_equity=self.account.equity)
        if snap is None:
            return None
        self._activate_challenger_models(snap.version_id)
        self.apply_champion_config(self.gate.active_config)
        for s in self.gate.snapshots:
            await self.db.save_snapshot(s)
        return snap

    async def rollback_champion(self, version_id: str, auto: bool = False) -> Any:
        """Atomically roll back Champion config AND versioned DL/ML weights."""
        restored = self.gate.rollback_to(version_id, auto=auto)
        if restored is None:
            return None
        self._restore_version_models(restored.version_id)
        self.apply_champion_config(self.gate.active_config)
        for s in self.gate.snapshots:
            await self.db.save_snapshot(s)
        return restored

    async def get_research_status(self) -> dict[str, Any]:
        """Training Ground / Research Lab status: DB M1 bar coverage, active DL/ML champions, and last OOS audit."""
        raw_cov = await self.db.get_bar_coverage("M1")
        target_bars = max(600, self.config.evolution.train_window_bars)
        offset = self.server_offset_sec
        coverage = []
        seen: set[str] = set()
        for row in raw_cov:
            sym = str(row["symbol"])
            seen.add(sym)
            n_bars = int(row["bars"] or 0)
            min_ts = int(row["min_ts"] or 0)
            max_ts = int(row["max_ts"] or 0)
            dur = round(max(0.0, (max_ts - min_ts) / 86400.0), 2) if n_bars > 1 else 0.0
            from_utc = (
                datetime.fromtimestamp(min_ts - offset, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                if min_ts
                else "—"
            )
            to_utc = (
                datetime.fromtimestamp(max_ts - offset, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                if max_ts
                else "—"
            )
            coverage.append(
                {
                    "symbol": sym,
                    "bars": n_bars,
                    "from_utc": from_utc,
                    "to_utc": to_utc,
                    "duration_days": dur,
                    "pct_of_window": round(min(100.0, (n_bars / target_bars) * 100.0), 1),
                }
            )
        for sym in self.config.instruments.symbols:
            if sym not in seen:
                coverage.append(
                    {
                        "symbol": sym,
                        "bars": 0,
                        "from_utc": "—",
                        "to_utc": "—",
                        "duration_days": 0.0,
                        "pct_of_window": 0.0,
                    }
                )
        return {
            "running": self.research_running or self.evolution_running,
            "window_bars_default": target_bars,
            "recency_half_life_days": self.config.evolution.recency_half_life_days,
            "dl_min_oos_gain": self.config.evolution.dl_min_oos_gain,
            "ml_min_oos_gain": self.config.evolution.ml_min_oos_gain,
            "server_offset_sec": offset,
            "coverage": coverage,
            "dl_champion": {
                "trained": self.dl_net.trained,
                "val_acc": round(self.dl_net.last_val_accuracy, 4),
            },
            "ml_champion": self.ml_labeler.diagnostics(),
            "last_audit": self.last_research_audit,
        }

    async def import_research_csv(self, filename: str, content: str) -> dict[str, Any]:
        """Import an MT5 History Center CSV/TSV file into SQLite bars without triggering live entries."""
        from backend.evolution.pretrain import parse_mt5_csv_text

        parsed = await asyncio.to_thread(
            parse_mt5_csv_text, filename, content, self.server_offset_sec
        )
        if not parsed.bars:
            raise ValueError(f"No valid M1 OHLC bars found in '{filename}' (skipped {parsed.skipped_rows} rows)")
        await self.db.save_bars(parsed.bars)
        self.ensure_symbol(parsed.symbol)
        # Refresh in-memory tail buffer cleanly (historical backfill only, zero live entries)
        recent = await self.db.load_recent_bars(parsed.symbol, "M1", limit=BAR_BUFFER)
        buf = self.bars.setdefault(parsed.symbol, deque(maxlen=BAR_BUFFER))
        buf.clear()
        buf.extend(recent)
        await self._log_decision(
            DecisionEvent(
                symbol=parsed.symbol,
                category="EVOLUTION",
                tag="CSV IMPORT",
                message=f"Imported {len(parsed.bars):,} M1 bars for {parsed.symbol} from {filename} (skipped {parsed.skipped_rows})",
            )
        )
        return {
            "ok": True,
            "symbol": parsed.symbol,
            "imported": len(parsed.bars),
            "skipped": parsed.skipped_rows,
        }

    async def run_research_pretrain(
        self,
        *,
        window_bars: int | None = None,
        dry_run: bool = False,
        force: bool = False,
    ) -> dict[str, Any]:
        """Run offline DL (1D-CNN) + ML (Calibrated Meta-Labeler) backtest pre-training & Scientist OOS Audit."""
        from backend.evolution.pretrain import run_pretrain_audit

        if self.research_running or self.evolution_running:
            return {"status": "already_running", "message": "Training or evolution cycle already in progress"}

        self.research_running = True
        try:
            limit = max(300, int(window_bars or self.config.evolution.train_window_bars))
            cov_rows = await self.db.get_bar_coverage("M1")
            symbols = list(dict.fromkeys([*self.config.instruments.symbols, *(str(r["symbol"]) for r in cov_rows)]))
            bars_map: dict[str, list[Bar]] = {}
            for sym in symbols:
                stored = await self.db.load_recent_bars(sym, "M1", limit=limit)
                live = list(self.bars.get(sym, ()))
                if live and stored and live[-1].timestamp > stored[-1].timestamp:
                    stored += [b for b in live if b.timestamp > stored[-1].timestamp]
                bars = stored if len(stored) >= len(live) else live
                if len(bars) >= 120:
                    bars_map[sym] = bars[-limit:]
            if not bars_map:
                return {
                    "status": "insufficient_bars",
                    "message": "Need at least 120 closed M1 bars in at least one symbol",
                }

            res = await asyncio.to_thread(
                run_pretrain_audit,
                bars_by_symbol=bars_map,
                config=self.config,
                specs=self.specs,
                champion_dl=self.dl_net if self.dl_net.trained else None,
                champion_ml=self.ml_labeler if self.ml_labeler.trained else None,
                server_offset_sec=self.server_offset_sec,
                window_bars=limit,
                dry_run=dry_run,
                force=force,
                snapshots_dir=self.config.app.snapshots_dir,
            )

            if not dry_run:
                if res.dl_saved and res.selected_dl is not None:
                    self.dl_net = res.selected_dl
                if res.ml_saved and res.selected_ml is not None:
                    self.ml_labeler = res.selected_ml

            self.last_research_audit = res.audit
            await self.db.set_kv("last_research_audit", res.audit, utc_now_ms())
            await self._log_decision(
                DecisionEvent(
                    symbol="RESEARCH",
                    category="EVOLUTION",
                    tag="TRAINING GROUND",
                    message=(
                        f"DL {res.audit['dl']['status']} ({res.audit['dl']['save_action']}) | "
                        f"ML {res.audit['ml']['status']} ({res.audit['ml']['save_action']}) | "
                        f"OOS PF {res.audit['oos_backtest']['profit_factor']:.2f} ({res.audit['oos_backtest']['total_trades']} trades)"
                    ),
                )
            )
            return {"status": "completed", "audit": res.audit}
        finally:
            self.research_running = False

    def entry_block_reason(self) -> str | None:
        """Final, non-tunable gate checked right before any OPEN command leaves for MT5."""
        return entry_block_reason(
            mode=self.config.app.trading_mode,
            account_is_demo=self.account.is_demo,
            allow_live_trading=self.env.allow_live_trading,
            killed=self.risk.killed,
        )

    def release_kill_switch(self) -> None:
        """Clear the kill lock. The engine stays PAUSED until the operator resumes it."""
        self.kill_state = self.kill_store.release()
        self.risk.killed = False
        self.risk.paused = True

    async def trigger_flatten_all(self, reason: str = "MANUAL_KILL") -> None:
        self.risk.killed = True
        self.risk.paused = True
        self.kill_state = self.kill_store.engage(reason)
        now_ms = utc_now_ms()
        self.order_sm.enqueue(
            OrderCommand(
                action=OrderAction.FLATTEN_ALL,
                symbol="ALL",
                expires_at_ms=now_ms + 10000,
            )
        )
        await self._log_decision(
            DecisionEvent(
                symbol="SYSTEM",
                category="RISK",
                tag="KILL SWITCH",
                message=f"FLATTEN_ALL dispatched ({reason}) — Engine locked.",
            )
        )
        await self._notify(f"EMERGENCY KILL-SWITCH ACTIVATED ({reason})! Flattening all positions.")

    async def trigger_manual_close(self, ticket: int) -> bool:
        pos = next((p for p in self.open_positions if p.ticket == ticket), None)
        if pos is None:
            return False
        spec = self.specs.get(pos.symbol, DEFAULT_SPECS["XAUUSD"])
        self.order_sm.enqueue(
            OrderCommand(
                action=OrderAction.CLOSE,
                symbol=pos.symbol,
                broker_symbol=spec.broker_symbol,
                ticket=ticket,
                expires_at_ms=utc_now_ms() + 5000,
            )
        )
        await self._log_decision(
            DecisionEvent(symbol=pos.symbol, category="EXEC", tag="MANUAL CLOSE", message=f"Dispatched close for #{ticket}")
        )
        return True

    async def force_demo_signal(self, symbol: str) -> dict[str, Any]:
        """Allow operator in Simulator/Demo mode to trigger a live validated signal immediately."""
        # This path bypasses the risk manager, so it is only allowed on simulated/demo money.
        if self.config.app.trading_mode == TradingMode.MT5_LIVE or not self.account.is_demo:
            return {"ok": False, "reason": "Demo signal is disabled on live / real-money accounts"}
        if self.risk.killed:
            return {"ok": False, "reason": "Kill switch active"}
        sym = symbol if symbol in self.specs else self.config.instruments.symbols[0]
        spec = self.specs[sym]
        tick = self.ticks.get(sym)
        if tick is None:
            return {"ok": False, "reason": "No tick data yet"}
        dl = self.dl_preds.get(sym, DLPrediction())
        side = Side.BUY if dl.prob_up >= dl.prob_down else Side.SELL
        bars = list(self.bars.get(sym, ()))
        atr_est = (bars[-1].high - bars[-1].low) * 1.5 if bars else spec.point * 120
        atr_est = max(atr_est, spec.point * 40)
        entry = tick.ask if side == Side.BUY else tick.bid
        sl = round(entry - 1.4 * atr_est if side == Side.BUY else entry + 1.4 * atr_est, spec.digits)
        tp = round(entry + 2.2 * atr_est if side == Side.BUY else entry - 2.2 * atr_est, spec.digits)
        cmd = OrderCommand(
            action=OrderAction.OPEN,
            symbol=sym,
            broker_symbol=spec.broker_symbol,
            side=side,
            volume=max(spec.volume_min, 0.10),
            sl=sl,
            tp=tp,
            magic=self.config.instruments.magic_number,
            strategy_id="liquidity_sweep",
            expected_price=entry,
            expires_at_ms=utc_now_ms() + 4000,
        )
        ok = self.order_sm.enqueue(
            cmd,
            strategy_id="liquidity_sweep",
            regime=self.regimes.get(sym, MarketRegime.RANGING),
            ml_confidence=0.72,
            dl_confidence=max(dl.prob_up, dl.prob_down, 0.65),
            bandit_weight=0.32,
            spread_at_entry=tick.spread_points,
        )
        if ok:
            self.readiness[sym] = 96
            await self._log_decision(
                DecisionEvent(
                    symbol=sym,
                    category="EXEC",
                    tag="DEMO SIGNAL",
                    strategy_id="liquidity_sweep",
                    side=side.value,
                    message=f"Operator triggered {side.value} {cmd.volume} lot @ {entry} (SL {sl} / TP {tp})",
                )
            )
        return {"ok": ok, "client_order_id": cmd.client_order_id}

    def get_live_state_snapshot(self) -> dict[str, Any]:
        now_ms = utc_now_ms()
        stale_sec = round((now_ms - self.last_bridge_sync_ms) / 1000.0, 1) if self.last_bridge_sync_ms else 999.0
        radar = []
        for sym in self.config.instruments.symbols:
            tk = self.ticks.get(sym)
            spec = self.specs.get(sym, DEFAULT_SPECS["XAUUSD"])
            b_list = list(self.bars.get(sym, ()))
            last_bars = [b.model_dump() for b in b_list[-110:]]
            atr_approx = max((b_list[-1].high - b_list[-1].low), spec.point * 25) if b_list else spec.point * 50
            sp_atr_pct = round(((tk.spread_points * spec.point) / atr_approx) * 100.0, 1) if tk else 0.0
            radar.append(
                {
                    "symbol": sym,
                    "digits": spec.digits,
                    "point": spec.point,
                    "bid": tk.bid if tk else 0.0,
                    "ask": tk.ask if tk else 0.0,
                    "spread_points": tk.spread_points if tk else 0.0,
                    "spread_atr_pct": min(99.0, sp_atr_pct),
                    "regime": self.regimes.get(sym, MarketRegime.RANGING).value,
                    "dl": self.dl_preds.get(sym, DLPrediction()).model_dump(),
                    "readiness": self.readiness.get(sym, 30),
                    "buy_pressure": self.buy_pressure.get(sym, 50),
                    "shield": self.shield_status.get(sym, "CLEAR"),
                    "ichi_m15": self.ichi_scores.get(sym, 0),
                    "bars": last_bars,
                }
            )

        return {
            "server_time_ms": now_ms,
            "trading_mode": self.config.app.trading_mode.value,
            "champion_version": self.gate.active_snapshot.version_id,
            "auto_promote": self.config.app.auto_promote_challenger,
            "paused": self.risk.paused,
            "killed": self.risk.killed,
            "kill_reason": self.kill_state.reason if self.risk.killed else "",
            "live_trading_enabled": self.env.allow_live_trading,
            "supervisor": self.supervisor.status(),
            "llm": self.llm_brief(),
            "entry_block_reason": self.entry_block_reason(),
            "evolution_running": self.evolution_running,
            "bridge_connected": stale_sec < 6.0,
            "bridge_stale_sec": stale_sec,
            "bridge_latency_ms": self.bridge_latency_ms,
            "ticks_per_sec": self.ticks_per_sec,
            "scans_count": self.scans_count,
            "active_thought": self.active_thought,
            "account": self.account.model_dump(),
            "shadow_equity": self.shadow_equity,
            "shadow_raw_equity": self.shadow_raw_equity,
            "paper": self.paper_snapshot(),
            "hard_guard": self.hard_guard.model_dump(),
            "radar": radar,
            "positions": [p.model_dump(mode="json") for p in self.open_positions],
            "decisions": [d.model_dump(mode="json") for d in list(self.recent_decisions)[:40]],
            "active_challenger": self.gate.active_challenger.model_dump(mode="json") if self.gate.active_challenger else None,
            "manual_shield_override": self.shields.manual_shock_override,
        }

    async def broadcast_state(self) -> None:
        if not self.ws_clients:
            return
        payload = self.get_live_state_snapshot()
        dead = []
        for ws in list(self.ws_clients):
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.ws_clients.discard(ws)
