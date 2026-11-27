from __future__ import annotations

import asyncio
import time
from collections import deque
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
from backend.engine.exit_policy import NetTake, ProfitLock, evaluate, net_take
from backend.engine.exit_selector import ExitStyleSelector
from backend.engine.supervisor import SupervisorSettings, TradeSupervisor, build_context, price_drifted, scale_volume
from backend.evolution.meta_labeler import CalibratedMetaLabeler
from backend.evolution.optimizer import PurgedWalkForwardOptimizer
from backend.evolution.rl_exit_agent import RLExitAgent
from backend.evolution.sequence_net import SequenceConvNet
from backend.features.dataset import build_sequence_tensor, extract_meta_features
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
    veto: str = ""  # Tanpa Veto book: what the supervisor said about this setup (evaluation only)


PAPER_VERSION = "PAPER"  # Normal book (kept for journal compatibility)
HITRUN_VERSION = "PAPER-HITRUN"
RAW_VERSION = "PAPER-RAW"
PAPER_BOOKS: dict[str, str] = {"normal": PAPER_VERSION, "hitrun": HITRUN_VERSION, "raw": RAW_VERSION}
PAPER_LABELS: dict[str, str] = {"normal": "Normal", "hitrun": "Hit & Run", "raw": "Tanpa Veto"}
# Normal and Hit & Run open only what the supervisor allowed. "raw" opens every setup that
# passed the risk checks, so the bandit and RL exit learn from all setups, not only approved ones.
VETOED_BOOKS = ("normal", "hitrun")
VETO_KEYS = ("APPROVE", "REDUCE", "REJECT", "NONE")
PAPER_START_EQUITY = 10000.0


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
        self.bars: dict[str, deque[Bar]] = {s: deque(maxlen=600) for s in config.instruments.symbols}
        self.ticks: dict[str, TickSnapshot] = {}
        self.regimes: dict[str, MarketRegime] = {s: MarketRegime.RANGING for s in config.instruments.symbols}
        self.dl_preds: dict[str, DLPrediction] = {s: DLPrediction() for s in config.instruments.symbols}
        self.readiness: dict[str, int] = {s: 35 for s in config.instruments.symbols}
        self.buy_pressure: dict[str, int] = {s: 50 for s in config.instruments.symbols}
        self.shield_status: dict[str, str] = {s: "CLEAR" for s in config.instruments.symbols}
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

        # Parallel Live Shadow Arena
        self.shadow_positions: dict[str, ShadowPosition] = {}
        self.shadow_equity: float = 10000.0
        # Paper trading (SHADOW_ONLY mode): the champion's approved trades, filled virtually.
        # Paper books run side by side on the same approved setups, in every mode:
        # "normal" = existing exits + profit lock, "hitrun" = take profit fast.
        self.exit_selector = ExitStyleSelector(min_samples=config.exits.adaptive_min_samples)
        self.veto_stats: dict[str, dict[str, float]] = {}
        self.paper_books: dict[str, PaperBook] = {
            k: PaperBook(equity=config.paper.start_equity) for k in PAPER_BOOKS
        }
        self._tracking_sig: tuple[int, ...] = ()
        self.shadow_suite: StrategySuite = StrategySuite.from_config(config)

        # Async callbacks / subscribers (Telegram + WebSocket)
        self.ws_clients: set[Any] = set()
        self.alert_callback: Any | None = None
        self.evolution_running: bool = False
        self._last_eq_record_ms: int = 0

    async def initialize_from_storage(self) -> None:
        """Crash recovery & state restoration on startup."""
        for sym in self.config.instruments.symbols:
            saved_bars = await self.db.load_recent_bars(sym, "M1", limit=500)
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
            self.apply_active_config(self.gate.active_config)
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
        self.order_sm.import_tracking(await self.db.get_kv("mt5_tracking"))
        self._tracking_sig = tuple(sorted(self.order_sm.active_positions))

        saved_decs = await self.db.list_decisions(limit=120)
        for ev in reversed(saved_decs):
            self.recent_decisions.appendleft(ev)

        dl_path = f"{self.config.app.snapshots_dir}/dl_champion.npz"
        self.dl_net.load(dl_path)
        self.ml_labeler.load(f"{self.config.app.snapshots_dir}/ml_champion.pkl")

    def apply_active_config(self, cfg: TradingConfig) -> None:
        self.config = cfg
        self.supervisor.settings = _supervisor_settings(cfg)
        self.risk.config = cfg.risk
        self.shields.config = cfg.shields
        self.suite = StrategySuite.from_config(cfg)
        self.rl_exit.be_trigger_r = cfg.ai.rl_exit_be_trigger_r
        self.rl_exit.trail_trigger_r = cfg.ai.rl_exit_trail_trigger_r

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
        """Which AI answers right now (first configured provider in the chain) and today's spend."""
        active = self.llm.active
        first = active[0] if active else None
        return {
            "enabled": bool(active),
            "provider": first.name if first else "",
            "label": first.preset.label if first else "Deterministic (no AI key)",
            "model": first.model if first else "",
            "fast_model": first.fast_model if first else "",
            "fallbacks": [r.name for r in active[1:]],
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
        self.bars.setdefault(sym, deque(maxlen=600))
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

        # 1. Update specs, account, hard guard
        for sym in {sp.symbol for sp in req.symbol_specs} | {t.symbol for t in req.ticks} | {b.symbol for b in req.closed_bars}:
            self.ensure_symbol(sym)
        for sp in req.symbol_specs:
            self.specs[sp.symbol] = sp
        self.account = req.account
        if req.hard_guard is not None:
            self.hard_guard = req.hard_guard

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
        for tr in closed_trades:
            await self.db.save_trade(tr)
            self._recent_closed.append(tr)
            self.risk.record_trade_outcome(tr.pnl_money, now_sec)
            # With paper on, the bandit and RL exit learn from the "Tanpa Veto" book instead, which
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
                    message=f"#{tr.ticket} {tr.side.value} {tr.pnl_money:+.2f}$ ({tr.pnl_r:+.2f}R) | MAE {tr.mae_r:.2f}R / MFE +{tr.mfe_r:.2f}R — RL Bandit updated",
                )
            )
            await self._notify(
                f"Trade Closed: {tr.symbol} {tr.side.value} ({tr.strategy_id})\n"
                f"PnL: {tr.pnl_money:+.2f}$ ({tr.pnl_r:+.2f}R) | Reason: {tr.exit_reason}"
            )

            rolled = self.gate.check_live_rollback_guard(self.account.equity, trade_closed=True)
            if rolled is not None:
                self.apply_active_config(self.gate.active_config)
                for sn in self.gate.snapshots:
                    await self.db.save_snapshot(sn)
                await self._log_decision(
                    DecisionEvent(
                        symbol="SYSTEM",
                        category="EVOLUTION",
                        tag="AUTO ROLLBACK",
                        message=f"Post-promotion drawdown triggered automatic rollback to Champion {rolled.version_id}",
                    )
                )
                await self._notify(f"AUTO-ROLLBACK TRIGGERED! Restored Champion {rolled.version_id}")

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

        # 5. Ingest newly closed bars & run full 4-Pillar AI scan on closed bars
        if req.closed_bars:
            await self.db.save_bars(req.closed_bars)
            for bar in req.closed_bars:
                buf = self.bars.setdefault(bar.symbol, deque(maxlen=600))
                if not buf or bar.timestamp > buf[-1].timestamp:
                    buf.append(bar)
                await self._evaluate_symbol_on_bar_close(bar, now_sec)

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
        bar_list = list(self.bars.get(sym, ()))
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

        # Scan 4 Scalping Micro-Strategies
        signals = self.suite.scan_all(sym, bar_list, assess.indicators, assess.regime)

        # Also evaluate Challenger in parallel Live Shadow Arena!
        await self._scan_shadow_challenger(sym, bar_list, assess.indicators, assess.regime, dl_pred, spread_pts, spec, now_sec)

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
        feat_vec = extract_meta_features(best_sig, assess.indicators, spread_price, dl_pred, bar.timestamp)
        ml_prob = self.ml_labeler.predict_win_probability(feat_vec)
        dl_align = dl_pred.prob_up if best_sig.side == Side.BUY else dl_pred.prob_down
        lat_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        self.readiness[sym] = min(98, int(ml_prob * 100))

        if ml_prob < self.config.ai.ml_min_probability:
            self.active_thought = f"{sym} {best_sig.strategy_id} {best_sig.side.value} filtered by ML Meta-Labeler (P={ml_prob:.2f} < {self.config.ai.ml_min_probability:.2f})"
            await self._log_decision(
                DecisionEvent(
                    symbol=sym,
                    category="BLOCK",
                    tag="ML FILTER",
                    strategy_id=best_sig.strategy_id,
                    side=best_sig.side.value,
                    message=f"Rejected: Calibrated P(win) {ml_prob:.2f} < {self.config.ai.ml_min_probability:.2f} | DL align {dl_align:.2f}",
                    latency_ms=lat_ms,
                )
            )
            return

        # Portfolio Risk Guard & Broker-Spec Position Sizing
        risk_dec = self.risk.evaluate_entry(
            signal=best_sig,
            account=self.account,
            open_positions=self.open_positions,
            spec=spec,
            spread_points=spread_pts,
            ml_probability=ml_prob,
            bandit_weight=b_weight,
            now_ts=now_sec,
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
        if self.config.paper.enabled and not (reviewed and sym in self._supervising):
            # Every risk-approved setup, before (and regardless of) the supervisor's verdict.
            await self._open_paper_position(
                sym, best_sig, risk_dec, assess, spec, ml_prob, dl_align, spread_pts,
                books=("raw",), veto="PENDING" if reviewed else "NONE",
            )
        if reviewed:
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

        # Dispatch idempotent OrderCommand to MT5 EA
        now_ms = utc_now_ms()
        cmd = OrderCommand(
            action=OrderAction.OPEN,
            symbol=sym,
            broker_symbol=spec.broker_symbol,
            side=best_sig.side,
            volume=risk_dec.volume,
            sl=risk_dec.adjusted_sl,
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
        )
        if queued:
            self.active_thought = (
                f"FIRING {sym} {best_sig.side.value} {risk_dec.volume} lot via {best_sig.strategy_id} "
                f"(ML {ml_prob:.0%}, DL {dl_align:.0%}, Bandit {b_weight:.0%})"
            )
            await self._log_decision(
                DecisionEvent(
                    symbol=sym,
                    category="EXEC",
                    tag="ORDER SENT",
                    strategy_id=best_sig.strategy_id,
                    side=best_sig.side.value,
                    message=f"{best_sig.side.value} {risk_dec.volume} lot | ML {ml_prob:.0%} • DL {dl_align:.0%} • {best_sig.reason}{note}",
                    latency_ms=lat_ms,
                )
            )

    async def _save_arena(self) -> None:
        await self.db.set_kv(
            "shadow_arena",
            {"equity": self.shadow_equity, "positions": [_pos_to_dict(p) for p in self.shadow_positions.values()]},
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
        books: tuple[str, ...] = VETOED_BOOKS, veto: str = "",
    ) -> None:
        """Fill a trade virtually at bid/ask in the given paper books (one position per symbol)."""
        tick = self.ticks.get(sym)
        if tick is None or risk_dec.volume <= 0:
            return
        entry = tick.ask if sig.side == Side.BUY else tick.bid
        sl, tp = risk_dec.adjusted_sl, risk_dec.adjusted_tp
        opened = False
        for name in books:
            book = self.paper_books[name]
            if sym in book.positions:
                continue
            book.positions[sym] = ShadowPosition(
                symbol=sym,
                strategy_id=sig.strategy_id,
                regime=assess.regime,
                side=sig.side,
                entry_price=round(entry, spec.digits),
                sl=sl,
                tp=tp,
                risk_dist=max(abs(entry - sl), spec.point),
                open_time=int(utc_now_ms() // 1000),
                ml_conf=ml_prob,
                dl_conf=dl_align,
                spread_pts=spread_pts,
                version_id=PAPER_BOOKS[name],
                volume=risk_dec.volume,
                veto=veto if name == "raw" else "",
            )
            opened = True
        if opened:
            await self._save_paper()

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

    def _net_take(self, symbol: str) -> NetTake:
        h = self.config.paper.hit_run
        spec = self.specs.get(symbol)
        comm = h.commission_per_lot if h.commission_per_lot is not None else (spec.commission_per_lot if spec else 0.0)
        return NetTake(min_net_profit=h.min_net_profit, commission_per_lot=comm)

    def _exit_policy(self) -> ProfitLock:
        e = self.config.exits
        return ProfitLock(be_at_r=e.be_at_r, be_offset_r=e.be_offset_r, trail_start_r=e.trail_start_r, trail_dist_r=e.trail_dist_r)

    async def _update_paper_positions_on_tick(self, tk: TickSnapshot, now_sec: int) -> None:
        changed = False

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
            if name == "hitrun":
                # Hit & Run: take any net profit. Valued at bid/ask, so spread is already paid.
                take, _net = net_take(self._net_take(p.symbol), gross_money=self._paper_money(p, px), volume=p.volume)
                action, new_sl = ("CLOSE" if take else "HOLD"), p.sl
            else:
                action, new_sl = evaluate(
                    self._exit_policy(), side=p.side, entry=p.entry_price, risk=p.risk_dist,
                    price=px, peak_r=p.mfe_r, sl=p.sl, spread_px=spread_px,
                )
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
            # Both books pay the same round-trip commission so the comparison is fair.
            pnl = round(self._paper_money(p, px) - self._net_take(p.symbol).commission_per_lot * p.volume, 2)
            book.equity = round(book.equity + pnl, 2)
            self.exit_selector.record(name, p.symbol, p.regime.value, round(cur_r, 4))
            if name == "raw":
                await self._learn_from_unvetoed(p, round(cur_r, 4), pnl, now_sec)
            in_profit_stop = hit_sl and cur_r > 0
            reason = (
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
                    exit_price=px,
                    sl_price=p.sl,
                    tp_price=p.tp,
                    open_time=p.open_time,
                    close_time=now_sec,
                    pnl_money=pnl,
                    pnl_r=round(cur_r, 4),
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
                    message=f"Paper {PAPER_LABELS[name]} {p.side.value} {p.volume} lot closed @ {px} ({cur_r:+.2f}R, ${pnl:+.2f}). Equity ${book.equity:,.2f}",
                )
            )
        if changed:
            await self._save_paper()

    async def _learn_from_unvetoed(self, p: ShadowPosition, r: float, pnl: float, now_sec: int) -> None:
        """Online learning from a closed "Tanpa Veto" trade. The labels are market results only;
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
            style = self._mt5_exit_style(pos.symbol, meta.regime)
            if style == "hitrun":
                # MT5 profit is valued at the closing price (spread included); add swap and the
                # round-trip commission (MT5 reports the entry side, the exit side costs the same).
                comm = abs(meta.commission) * 2 if meta.commission else None
                take, net = net_take(
                    self._net_take(pos.symbol), gross_money=pos.profit, volume=pos.volume, swap=pos.swap, commission=comm
                )
                lock_action, lock_sl = ("CLOSE" if take else "HOLD"), pos.sl
            else:
                lock_action, lock_sl = evaluate(
                    self._exit_policy(), side=meta.side, entry=meta.entry_price, risk=meta.initial_risk_price,
                    price=pos.current_price, peak_r=meta.mfe_r, sl=pos.sl, spread_px=spread_px,
                )
            if lock_action == "CLOSE":
                if meta.rl_trace[-1] != "HIT_RUN_TAKE":
                    meta.rl_trace.append("HIT_RUN_TAKE")
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
                            symbol=pos.symbol, category="EXEC", tag="HIT & RUN TAKE", strategy_id=meta.strategy_id,
                            message=f"#{pos.ticket} ditutup di untung bersih ${net:.2f} (+{cur_r:.2f}R), Hit & Run dipilih untuk {pos.symbol} {meta.regime.value}",
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
                be_sl = (
                    round(meta.entry_price + 0.06 * meta.initial_risk_price, spec.digits)
                    if meta.side == Side.BUY
                    else round(meta.entry_price - 0.06 * meta.initial_risk_price, spec.digits)
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
    ) -> None:
        if sym in self.shadow_positions:
            return
        cand = self.gate.active_challenger
        shadow_cfg = TradingConfig.model_validate(cand.config_snapshot) if cand else self.config
        sigs = self.shadow_suite.scan_all(sym, bars, snap, regime)
        if not sigs:
            return
        best = max(sigs, key=lambda s: s.confidence)
        spread_px = spread_pts * spec.point
        if (spread_px / max(snap.atr14, 1e-9)) > shadow_cfg.risk.max_spread_atr_ratio:
            return
        feat = extract_meta_features(best, snap, spread_px, dl_pred, bars[-1].timestamp)
        ml_p = self.ml_labeler.predict_win_probability(feat)
        if ml_p < shadow_cfg.ai.ml_min_probability:
            return
        entry = best.entry_price + (spread_px if best.side == Side.BUY else 0.0)
        risk_d = max(abs(entry - best.sl_price), snap.atr14 * 0.5)
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
            version_id=cand.version_id if cand else f"{self.config.app.version}-shadow",
        )
        await self._save_arena()

    async def _update_shadow_positions_on_tick(self, tk: TickSnapshot, now_sec: int) -> None:
        spos = self.shadow_positions.get(tk.symbol)
        if spos is None:
            return
        px = tk.bid if spos.side == Side.BUY else tk.ask
        cur_r = (
            (px - spos.entry_price) / spos.risk_dist
            if spos.side == Side.BUY
            else (spos.entry_price - px) / spos.risk_dist
        )
        spos.mae_r = min(spos.mae_r, cur_r)
        spos.mfe_r = max(spos.mfe_r, cur_r)

        hit_sl = (spos.side == Side.BUY and px <= spos.sl) or (spos.side == Side.SELL and px >= spos.sl)
        hit_tp = (spos.side == Side.BUY and px >= spos.tp) or (spos.side == Side.SELL and px <= spos.tp)
        timed_out = (now_sec - spos.open_time) >= 20 * 60

        if hit_sl or hit_tp or timed_out:
            self.shadow_positions.pop(tk.symbol, None)
            net_r = cur_r - 0.04
            risk_money = self.shadow_equity * (self.config.risk.risk_per_trade_pct / 100.0)
            pnl = round(risk_money * net_r, 2)
            self.shadow_equity = round(self.shadow_equity + pnl, 2)
            tr = TradeRecord(
                is_shadow=True,
                version_id=spos.version_id,
                symbol=spos.symbol,
                strategy_id=spos.strategy_id,
                regime=spos.regime,
                side=spos.side,
                volume=0.10,
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
                exit_reason="TP" if hit_tp else ("SL" if hit_sl else "TIME_EXPIRY"),
            )
            await self.db.save_trade(tr)
            await self._save_arena()
            transition = self.gate.record_shadow_trade(tr)
            if transition and self.gate.active_challenger:
                await self.db.save_challenger(self.gate.active_challenger)
            if transition == "PENDING_APPROVAL":
                await self._notify(f"Challenger passed Live Shadow Arena! Ready for /approve on Dashboard or Telegram.")
            elif transition == "AUTO_PROMOTED":
                self.apply_active_config(self.gate.active_config)
                for sn in self.gate.snapshots:
                    await self.db.save_snapshot(sn)
                await self._notify(f"Challenger AUTO-PROMOTED to Champion {self.gate.active_snapshot.version_id}!")

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

            # A new model replaces the champion only when it scored better on unseen data.
            if wfo.dl_decision.get("adopted"):
                self.dl_net = wfo.trained_dl
                self.dl_net.save(f"{self.config.app.snapshots_dir}/dl_champion.npz")
            if wfo.ml_decision.get("adopted"):
                self.ml_labeler = wfo.trained_ml
                self.ml_labeler.save(f"{self.config.app.snapshots_dir}/ml_champion.pkl")
            n_bars = {s: len(b) for s, b in bars_map.items()}
            for name, dec in (("1D-CNN", wfo.dl_decision), ("Meta-labeler", wfo.ml_decision)):
                await self._log_decision(
                    DecisionEvent(
                        symbol="EVOLUTION",
                        category="EVOLUTION",
                        tag=f"{name} {'UPDATED' if dec.get('adopted') else 'KEPT'}",
                        message=(
                            f"{name}: {'model baru dipakai' if dec.get('adopted') else 'champion lama dipertahankan'}"
                            f"{' (lanjut dari weight champion)' if dec.get('warm_start') else ''}. {dec.get('reason', '')}. "
                            f"Data: {max(n_bars.values())} bar M1 per pasar"
                        ),
                    )
                )

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
            last_bars = [b.model_dump() for b in b_list[-60:]]
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
