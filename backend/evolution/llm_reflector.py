from __future__ import annotations

import json
import re
from typing import Any

import httpx
import numpy as np

from backend.core.config import EnvSettings, TradingConfig
from backend.core.models import ParameterPatch, ReflectionReport, TradeRecord
from backend.core.security import HARD_SAFETY_BOUNDS, clamp_and_validate_patch


class LLMReflectionAnalyst:
    """Pillar 4 (LLM): Post-mortem trade journal analyst proposing strictly bounded ParameterPatches.

    Supports Gemini, OpenAI, Anthropic, and Ollama; automatically falls back to the built-in
    Deterministic Quant Reflection Engine if no API key is configured or network fails.
    """

    def __init__(self, env: EnvSettings) -> None:
        self.env = env

    def _summarize_journal(self, trades: list[TradeRecord]) -> dict[str, Any]:
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

        return {
            "count": len(trades),
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

        # Rule 5: Default subtle refinement if metrics look healthy
        if not raw_patches:
            cur_ml = config.ai.ml_min_probability
            raw_patches.append(
                ParameterPatch(
                    path="ai.ml_min_probability",
                    old_value=cur_ml,
                    new_value=round(cur_ml + 0.015, 3),
                    rationale="Incremental precision optimization on calibrated ML probability gate",
                )
            )
            findings.append(
                f"Portfolio healthy ({stats['win_rate']:.1%} WR, {stats['avg_r']:+.2f}R avg) — testing tighter ML selectivity."
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

    def _build_llm_prompt(self, stats: dict[str, Any], config: TradingConfig) -> str:
        allowed_keys = list(HARD_SAFETY_BOUNDS.keys())
        return (
            "You are a senior quantitative scalping researcher auditing an MT5 M1-M5 trading agent.\n"
            f"Trade Forensics Summary: {json.dumps(stats)}\n"
            f"Current Risk & AI Config: {json.dumps({'risk': config.risk.model_dump(), 'ai': config.ai.model_dump()})}\n"
            f"Allowed patch paths (ONLY use these exact keys): {json.dumps(allowed_keys)}\n"
            "Return STRICT JSON only with schema:\n"
            '{"finding": "...", "evidence": "...", "patches": [{"path": "...", "old_value": 0.0, "new_value": 0.0, "rationale": "..."}]}'
        )

    def _parse_llm_json(self, text: str, config: TradingConfig, provider: str, n_trades: int) -> ReflectionReport | None:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        cfg_dict = config.model_dump(mode="json")
        validated: list[ParameterPatch] = []
        for raw in data.get("patches", []):
            try:
                patch = ParameterPatch.model_validate(raw)
                safe = clamp_and_validate_patch(cfg_dict, patch)
                if safe is not None:
                    validated.append(safe)
            except Exception:
                continue
        if not data.get("finding"):
            return None
        return ReflectionReport(
            provider=provider,
            trades_analyzed=n_trades,
            finding=str(data["finding"])[:600],
            evidence=str(data.get("evidence", ""))[:500],
            patches=validated,
        )

    async def reflect(self, trades: list[TradeRecord], config: TradingConfig) -> ReflectionReport:
        provider = (self.env.llm_provider or "deterministic").lower().strip()
        if provider == "deterministic" or len(trades) < 5:
            return self.deterministic_reflect(trades, config)

        stats = self._summarize_journal(trades)
        prompt = self._build_llm_prompt(stats, config)

        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                raw_text = ""
                if provider == "gemini" and self.env.gemini_api_key:
                    url = (
                        f"https://generativelanguage.googleapis.com/v1beta/models/"
                        f"{self.env.llm_model}:generateContent?key={self.env.gemini_api_key}"
                    )
                    resp = await client.post(url, json={"contents": [{"parts": [{"text": prompt}]}]})
                    resp.raise_for_status()
                    raw_text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
                elif provider == "openai" and self.env.openai_api_key:
                    resp = await client.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers={"Authorization": f"Bearer {self.env.openai_api_key}"},
                        json={
                            "model": self.env.llm_model or "gpt-4o-mini",
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": 0.2,
                        },
                    )
                    resp.raise_for_status()
                    raw_text = resp.json()["choices"][0]["message"]["content"]
                elif provider == "anthropic" and self.env.anthropic_api_key:
                    resp = await client.post(
                        "https://api.anthropic.com/v1/messages",
                        headers={
                            "x-api-key": self.env.anthropic_api_key,
                            "anthropic-version": "2023-06-01",
                        },
                        json={
                            "model": self.env.llm_model or "claude-3-5-sonnet-latest",
                            "max_tokens": 800,
                            "messages": [{"role": "user", "content": prompt}],
                        },
                    )
                    resp.raise_for_status()
                    raw_text = resp.json()["content"][0]["text"]
                elif provider == "ollama":
                    resp = await client.post(
                        f"{self.env.ollama_base_url.rstrip('/')}/api/generate",
                        json={"model": self.env.llm_model or "llama3.1", "prompt": prompt, "stream": False},
                    )
                    resp.raise_for_status()
                    raw_text = resp.json().get("response", "")

                if raw_text:
                    parsed = self._parse_llm_json(raw_text, config, provider, len(trades))
                    if parsed is not None:
                        return parsed
        except Exception:
            pass

        # Safe fallback so evolution never stalls on API outages
        fallback = self.deterministic_reflect(trades, config)
        fallback.provider = f"{provider}_fallback_quant"
        return fallback
