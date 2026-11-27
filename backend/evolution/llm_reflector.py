from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from backend.core.config import EnvSettings, TradingConfig, get_nested_value
from backend.core.models import ParameterPatch, ReflectionReport, TradeRecord
from backend.core.security import HARD_SAFETY_BOUNDS, RISK_DIRECTION, clamp_and_validate_patch, restrict_llm_patch
from backend.llm.base import LLMRequest, ToolSpec
from backend.llm.router import LLMRouter
from backend.risk.manager import DEFAULT_SPECS

log = logging.getLogger("max_agent.reflection")

# What each tunable parameter does, so the model reasons about mechanics instead of names.
PARAM_DOCS: dict[str, str] = {
    "risk.risk_per_trade_pct": "Base % of equity risked per trade (before half-Kelly scaling).",
    "risk.min_risk_pct": "Floor for dynamic risk % per trade.",
    "risk.max_risk_pct": "Ceiling for dynamic risk % per trade.",
    "risk.max_daily_drawdown_pct": "Daily drawdown % at which new entries lock out for the day.",
    "risk.max_concurrent_positions": "Max simultaneous open positions (integer).",
    "risk.max_usd_correlated_positions": "Max simultaneous USD-exposed positions (integer).",
    "risk.max_spread_atr_ratio": "Entry blocked if spread/ATR exceeds this. Lower = stricter spread filter.",
    "risk.consecutive_loss_limit": "Losses in a row that trigger a cooldown (integer).",
    "risk.consecutive_loss_cooldown_min": "Cooldown length in minutes after the loss streak (integer).",
    "risk.min_rr_ratio": "Minimum TP distance as a multiple of SL distance.",
    "ai.ml_min_probability": "Calibrated meta-labeler P(win) needed to take a signal. Higher = fewer, cleaner trades.",
    "ai.rl_exit_be_trigger_r": "Open profit (in R) at which the exit agent may move SL to breakeven.",
    "ai.rl_exit_trail_trigger_r": "Open profit (in R) at which trailing starts.",
    "ai.rl_exit_partial_trigger_r": "Open profit (in R) at which a partial close is allowed.",
}
for _path in HARD_SAFETY_BOUNDS:
    if _path.startswith("strategies."):
        _, sid, key = _path.split(".")
        PARAM_DOCS[_path] = {
            "sl_atr_mult": f"{sid}: stop-loss distance in ATR multiples.",
            "tp_atr_mult": f"{sid}: take-profit distance in ATR multiples.",
            "min_wick_ratio": f"{sid}: minimum rejection-wick fraction of the bar to qualify a sweep.",
            "min_body_atr": f"{sid}: minimum breakout bar body in ATR multiples.",
            "min_adx": f"{sid}: minimum ADX trend strength to take pullbacks.",
            "min_momentum_atr": f"{sid}: minimum momentum move in ATR multiples.",
        }.get(key, f"{sid}: {key}")


def _param_table() -> str:
    lines = []
    for path, (lo, hi) in HARD_SAFETY_BOUNDS.items():
        d = RISK_DIRECTION.get(path, 0)
        rule = " May only DECREASE." if d > 0 else (" May only INCREASE." if d < 0 else "")
        lines.append(f"- `{path}` [{lo} .. {hi}]: {PARAM_DOCS.get(path, '')}{rule}")
    return "\n".join(lines)


REFLECTION_SYSTEM = f"""You are the post-mortem analyst for an automated MT5 scalping system (M1 entries, M5 confirmation) trading XAUUSD, EURUSD, GBPUSD and USDJPY.

Your job: read the closed-trade journal, find the most likely cause of underperformance (or confirm the system is healthy), and propose at most 3 small parameter changes as hypotheses.

How your output is used:
- Every patch is a HYPOTHESIS. The system re-runs a purged walk-forward backtest and a Monte Carlo stress test with your patches, then shadow-trades them. Only if they beat the current champion out-of-sample does a human approve them. So be specific and testable, not confident.
- Patches are code-enforced: unknown paths are dropped, values are clamped to the bounds below, any single change is capped at 20% of the current value, and risk-increasing changes are discarded. Do not waste a patch on those.
- Returning an empty `patches` list is a correct answer when the evidence is thin (fewer than ~30 trades in a group) or the system is healthy. Changing parameters on noise is the main way scalping systems overfit.

Units: pnl_r / mae_r / mfe_r are in R (multiples of initial risk; -1R = full stop-loss). Spread is in broker points. MAE is how far price went against the trade, MFE how far in favour, before exit.

Tunable parameters [hard bounds]:
{_param_table()}

Method:
1. Look at sample size first. Name it in `evidence`.
2. Compare winners vs losers on spread, MAE, MFE, strategy, regime and hour. Use the tools to drill down when a summary number looks suspicious.
3. Prefer one well-supported change over three weak ones.

Output fields:
- `finding`: 1-3 sentences, the diagnosis in plain language.
- `evidence`: the specific numbers that support it, including sample sizes.
- `patches`: list of {{path, new_value, rationale}}; `rationale` says what you expect to change and how the backtest would show it.
"""

_REFLECTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["finding", "evidence", "patches"],
    "properties": {
        "finding": {"type": "string"},
        "evidence": {"type": "string"},
        "patches": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["path", "new_value", "rationale"],
                "properties": {
                    "path": {"type": "string", "enum": sorted(HARD_SAFETY_BOUNDS)},
                    "new_value": {"type": "number"},
                    "rationale": {"type": "string"},
                },
            },
        },
    },
}


class _LLMPatch(BaseModel):
    model_config = ConfigDict(extra="ignore")
    path: str
    new_value: float
    rationale: str = ""


class _LLMReflection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    finding: str
    evidence: str = ""
    patches: list[_LLMPatch] = []


class LLMReflectionAnalyst:
    """Pillar 4 (LLM): Post-mortem trade journal analyst proposing strictly bounded ParameterPatches.

    Supports Gemini, OpenAI, Anthropic, and Ollama; automatically falls back to the built-in
    Deterministic Quant Reflection Engine if no API key is configured or network fails.
    """

    def __init__(self, env: EnvSettings, router: LLMRouter | None = None) -> None:
        self.env = env
        self.router = router if router is not None else LLMRouter.from_env(".env")
        self.last_error: str = ""

    def _summarize_journal(self, trades: list[TradeRecord], sl_mults: dict[str, float] | None = None) -> dict[str, Any]:
        sl_mults = sl_mults or {}
        if not trades:
            return {"count": 0}
        wins = [t for t in trades if t.pnl_r > 0]
        losses = [t for t in trades if t.pnl_r <= 0]
        win_spread = float(np.mean([t.spread_at_entry for t in wins])) if wins else 0.0
        loss_spread = float(np.mean([t.spread_at_entry for t in losses])) if losses else 0.0
        avg_loss_mfe = float(np.mean([t.mfe_r for t in losses])) if losses else 0.0
        avg_win_mae = float(np.mean([abs(t.mae_r) for t in wins])) if wins else 0.0
        avg_win_mfe = float(np.mean([t.mfe_r for t in wins])) if wins else 0.0
        avg_win_r = float(np.mean([t.pnl_r for t in wins])) if wins else 0.0

        by_strategy: dict[str, list[float]] = {}
        for t in trades:
            by_strategy.setdefault(t.strategy_id, []).append(t.pnl_r)
        strat_stats = {
            sid: {"n": len(rs), "win_rate": round(float(np.mean([r > 0 for r in rs])), 3), "avg_r": round(float(np.mean(rs)), 3)}
            for sid, rs in by_strategy.items()
        }

        def spread_atr_est(t: TradeRecord) -> float | None:
            # spread / ATR, estimated from spread / initial risk * the strategy's SL ATR multiple,
            # so it is directly comparable to risk.max_spread_atr_ratio.
            spec = DEFAULT_SPECS.get(t.symbol)
            risk_px = abs(t.entry_price - t.sl_price)
            if spec is None or risk_px <= 0:
                return None
            sl_mult = float(sl_mults.get(t.strategy_id, 1.35))
            return (t.spread_at_entry * spec.point) / risk_px * sl_mult

        def avg_est(group: list[TradeRecord]) -> float | None:
            vals = [v for t in group if (v := spread_atr_est(t)) is not None]
            return round(float(np.mean(vals)), 3) if vals else None

        def pct(group: list[TradeRecord], cond) -> float:
            return round(float(np.mean([cond(t) for t in group])), 3) if group else 0.0

        return {
            "count": len(trades),
            "n_wins": len(wins),
            "n_losses": len(losses),
            "win_avg_spread_atr_est": avg_est(wins),
            "loss_avg_spread_atr_est": avg_est(losses),
            "losers_reaching_0_5r_pct": pct(losses, lambda t: t.mfe_r >= 0.5),
            "losers_reaching_1r_pct": pct(losses, lambda t: t.mfe_r >= 1.0),
            "exit_reasons": {r: sum(1 for t in trades if t.exit_reason == r) for r in sorted({t.exit_reason for t in trades})},
            "win_rate": round(len(wins) / len(trades), 3),
            "avg_r": round(float(np.mean([t.pnl_r for t in trades])), 3),
            "win_avg_spread": round(win_spread, 2),
            "loss_avg_spread": round(loss_spread, 2),
            "loss_avg_mfe_r": round(avg_loss_mfe, 3),
            "win_avg_mae_r": round(avg_win_mae, 3),
            "win_avg_mfe_r": round(avg_win_mfe, 3),
            "win_avg_realized_r": round(avg_win_r, 3),
            "strategies": strat_stats,
        }

    def deterministic_reflect(self, trades: list[TradeRecord], config: TradingConfig) -> ReflectionReport:
        stats = self._summarize_journal(trades)
        cfg_dict = config.model_dump(mode="json")
        raw_patches: list[ParameterPatch] = []
        findings: list[str] = []

        if stats.get("count", 0) == 0:
            return ReflectionReport(
                provider="deterministic_quant",
                trades_analyzed=0,
                finding="Baseline warmup phase — insufficient closed trades for post-mortem diagnosis.",
                evidence="Awaiting closed live/shadow trades to measure MAE/MFE & spread impact.",
                patches=[],
            )

        # Rule 1: Loss group suffered significantly higher spreads
        if stats["loss_avg_spread"] > max(stats["win_avg_spread"] * 1.18, 12.0):
            cur = config.risk.max_spread_atr_ratio
            raw_patches.append(
                ParameterPatch(
                    path="risk.max_spread_atr_ratio",
                    old_value=cur,
                    new_value=round(cur * 0.88, 3),
                    rationale="Tighten spread-to-ATR gate because losing trades entered at wider spreads",
                )
            )
            findings.append(
                f"Losing trades averaged {stats['loss_avg_spread']:.1f}pt spread vs {stats['win_avg_spread']:.1f}pt on winners."
            )

        # Rule 2: Losers frequently went well into profit before reversing -> tighten BE trigger
        if stats["loss_avg_mfe_r"] >= 0.65:
            cur_be = config.ai.rl_exit_be_trigger_r
            raw_patches.append(
                ParameterPatch(
                    path="ai.rl_exit_be_trigger_r",
                    old_value=cur_be,
                    new_value=round(max(0.55, cur_be - 0.12), 3),
                    rationale="Engage breakeven earlier to prevent profitable scalps turning into full losses",
                )
            )
            findings.append(
                f"Losing trades reached +{stats['loss_avg_mfe_r']:.2f}R average MFE before reversing into SL."
            )

        # Rule 3: Low win rate -> raise ML Meta-Labeler confidence cutoff slightly
        if stats["win_rate"] < 0.52:
            cur_ml = config.ai.ml_min_probability
            raw_patches.append(
                ParameterPatch(
                    path="ai.ml_min_probability",
                    old_value=cur_ml,
                    new_value=round(cur_ml + 0.03, 3),
                    rationale="Raise calibrated ML confidence threshold to filter marginal noise setups",
                )
            )
            findings.append(
                f"Win rate ({stats['win_rate']:.1%}) below 52% target — tightening ML Meta-Labeler filter."
            )

        # Rule 4: Winners taking heavy MAE heat before winning -> give SL slightly more breathing room
        if stats["win_avg_mae_r"] > 0.60:
            cur_sl = float(config.strategies.get("liquidity_sweep", {}).get("sl_atr_mult", 1.45))
            raw_patches.append(
                ParameterPatch(
                    path="strategies.liquidity_sweep.sl_atr_mult",
                    old_value=cur_sl,
                    new_value=round(cur_sl + 0.15, 3),
                    rationale="Slightly widen SL ATR multiplier to avoid stop-out noise before expansion",
                )
            )
            findings.append(
                f"Winning setups endured {stats['win_avg_mae_r']:.2f}R average MAE heat — widening sweep SL buffer."
            )

        if not raw_patches:
            findings.append(
                f"Portfolio healthy ({stats['win_rate']:.1%} WR, {stats['avg_r']:+.2f}R avg) — no parameter change proposed."
            )

        validated = [
            safe
            for p in raw_patches
            if (safe := clamp_and_validate_patch(cfg_dict, p)) is not None
        ]
        evidence = (
            f"WinRate={stats['win_rate']:.1%} | AvgR={stats['avg_r']:+.2f}R | "
            f"LossMFE=+{stats['loss_avg_mfe_r']:.2f}R | WinMAE=-{stats['win_avg_mae_r']:.2f}R | "
            f"Spread(W/L)={stats['win_avg_spread']:.1f}/{stats['loss_avg_spread']:.1f}pt"
        )
        return ReflectionReport(
            provider="deterministic_quant",
            trades_analyzed=stats["count"],
            finding=" ".join(findings),
            evidence=evidence,
            patches=validated,
        )

    def _trade_tools(self, trades: list[TradeRecord]) -> list[ToolSpec]:
        def _row(t: TradeRecord) -> dict[str, Any]:
            return {
                "symbol": t.symbol, "strategy": t.strategy_id, "regime": t.regime.value, "side": t.side.value,
                "hour_utc": int((t.open_time % 86400) // 3600), "pnl_r": round(t.pnl_r, 2),
                "mae_r": round(t.mae_r, 2), "mfe_r": round(t.mfe_r, 2), "spread": round(t.spread_at_entry, 1),
                "ml_p": round(t.ml_confidence, 2), "dl_p": round(t.dl_confidence, 2), "exit": t.exit_reason,
                "bars_held": max(1, (t.close_time - t.open_time) // 60),
            }

        async def get_trades(args: dict[str, Any]) -> list[dict[str, Any]]:
            sym, strat, outcome = args.get("symbol", ""), args.get("strategy_id", ""), args.get("outcome", "all")
            rows = [
                t for t in trades
                if (not sym or t.symbol == sym) and (not strat or t.strategy_id == strat)
                and (outcome == "all" or (outcome == "win") == (t.pnl_r > 0))
            ]
            limit = max(1, min(int(args.get("limit", 40) or 40), 100))
            return [_row(t) for t in rows[-limit:]]

        async def get_breakdown(args: dict[str, Any]) -> dict[str, Any]:
            by = args.get("by", "strategy")
            key = {
                "strategy": lambda t: t.strategy_id, "symbol": lambda t: t.symbol,
                "regime": lambda t: t.regime.value, "hour_utc": lambda t: str(int((t.open_time % 86400) // 3600)),
                "exit_reason": lambda t: t.exit_reason,
            }.get(by, lambda t: t.strategy_id)
            groups: dict[str, list[TradeRecord]] = {}
            for t in trades:
                groups.setdefault(key(t), []).append(t)
            return {
                g: {
                    "n": len(ts),
                    "win_rate": round(float(np.mean([t.pnl_r > 0 for t in ts])), 3),
                    "avg_r": round(float(np.mean([t.pnl_r for t in ts])), 3),
                    "avg_mae_r": round(float(np.mean([t.mae_r for t in ts])), 3),
                    "avg_mfe_r": round(float(np.mean([t.mfe_r for t in ts])), 3),
                    "avg_spread": round(float(np.mean([t.spread_at_entry for t in ts])), 1),
                }
                for g, ts in groups.items()
            }

        return [
            ToolSpec(
                name="get_trades",
                description="List closed trades (most recent last). Empty string = no filter.",
                input_schema={
                    "type": "object", "additionalProperties": False,
                    "required": ["symbol", "strategy_id", "outcome", "limit"],
                    "properties": {
                        "symbol": {"type": "string"}, "strategy_id": {"type": "string"},
                        "outcome": {"type": "string", "enum": ["all", "win", "loss"]},
                        "limit": {"type": "integer"},
                    },
                },
                handler=get_trades,
            ),
            ToolSpec(
                name="get_breakdown",
                description="Win rate, avg R, MAE, MFE and spread grouped by one dimension.",
                input_schema={
                    "type": "object", "additionalProperties": False, "required": ["by"],
                    "properties": {"by": {"type": "string", "enum": ["strategy", "symbol", "regime", "hour_utc", "exit_reason"]}},
                },
                handler=get_breakdown,
            ),
        ]

    def build_user_message(self, stats: dict[str, Any], config: TradingConfig) -> str:
        cfg_dict = config.model_dump(mode="json")
        current = {}
        for path in HARD_SAFETY_BOUNDS:
            try:
                current[path] = get_nested_value(cfg_dict, path)
            except KeyError:
                continue
        return (
            "Closed-trade journal summary (MAE/MFE averages are absolute R; *_spread_atr_est is spread/ATR at entry,"
            " comparable to risk.max_spread_atr_ratio; spreads are broker points):\n"
            f"{json.dumps(stats, sort_keys=True)}\n\n"
            "Current parameter values:\n"
            f"{json.dumps(current, sort_keys=True)}\n\n"
            "Diagnose and answer in the required JSON format."
        )

    async def reflect(self, trades: list[TradeRecord], config: TradingConfig) -> ReflectionReport:
        if len(trades) < 5 or not self.router.enabled:
            return self.deterministic_reflect(trades, config)

        sl_mults = {sid: float(v.get("sl_atr_mult", 1.35)) for sid, v in config.strategies.items()}
        stats = self._summarize_journal(trades, sl_mults)
        req = LLMRequest(
            task="reflection",
            system=REFLECTION_SYSTEM,
            user=self.build_user_message(stats, config),
            schema=_REFLECTION_SCHEMA,
            timeout_s=90.0,
            max_tokens=16000,
            effort="high",
            tools=self._trade_tools(trades),
        )
        try:
            res, parsed = await self.router.complete(req, validate=_LLMReflection.model_validate)
        except Exception as exc:
            self.last_error = str(exc)[:300]
            log.warning("LLM reflection failed on every provider, using deterministic: %s", self.last_error)
            fallback = self.deterministic_reflect(trades, config)
            fallback.provider = "deterministic_fallback"
            return fallback

        cfg_dict = config.model_dump(mode="json")
        validated: list[ParameterPatch] = []
        for raw in parsed.patches[:3]:
            safe = restrict_llm_patch(
                cfg_dict, ParameterPatch(path=raw.path, old_value=0.0, new_value=raw.new_value, rationale=raw.rationale[:300])
            )
            if safe is not None:
                validated.append(safe)
        self.last_error = ""
        cost = f" | ${res.cost_usd:.4f}" if res.cost_usd else ""
        return ReflectionReport(
            provider=f"{res.provider}:{res.model}",
            trades_analyzed=len(trades),
            finding=parsed.finding[:600],
            evidence=(parsed.evidence[:440] + f" [{res.latency_ms:.0f}ms, {res.tool_calls} tool calls{cost}]"),
            patches=validated,
        )
