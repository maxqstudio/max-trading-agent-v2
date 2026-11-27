from __future__ import annotations

import hmac
from typing import Any

from backend.core.config import TradingConfig, get_nested_value, set_nested_value
from backend.core.models import ParameterPatch
from backend.core.safety import is_strong_token

# Absolute non-negotiable safety bounds that neither Optimizer nor LLM can ever bypass.
HARD_SAFETY_BOUNDS: dict[str, tuple[float, float]] = {
    "risk.risk_per_trade_pct": (0.10, 1.50),
    "risk.min_risk_pct": (0.10, 0.50),
    "risk.max_risk_pct": (0.30, 1.50),
    "risk.max_daily_drawdown_pct": (1.00, 5.00),
    "risk.max_concurrent_positions": (1.0, 5.0),
    "risk.max_usd_correlated_positions": (1.0, 3.0),
    "risk.max_spread_atr_ratio": (0.12, 0.45),
    "risk.consecutive_loss_limit": (2.0, 6.0),
    "risk.consecutive_loss_cooldown_min": (5.0, 120.0),
    "risk.min_rr_ratio": (1.05, 3.50),
    "ai.ml_min_probability": (0.50, 0.78),
    "ai.rl_exit_be_trigger_r": (0.50, 1.50),
    "ai.rl_exit_trail_trigger_r": (0.80, 2.20),
    "ai.rl_exit_partial_trigger_r": (1.00, 3.00),
    "strategies.liquidity_sweep.min_wick_ratio": (0.40, 0.80),
    "strategies.liquidity_sweep.sl_atr_mult": (0.90, 2.50),
    "strategies.liquidity_sweep.tp_atr_mult": (1.40, 4.00),
    "strategies.micro_breakout.min_body_atr": (0.25, 1.20),
    "strategies.micro_breakout.sl_atr_mult": (0.90, 2.50),
    "strategies.micro_breakout.tp_atr_mult": (1.40, 4.50),
    "strategies.ema_vwap_pullback.min_adx": (14.0, 35.0),
    "strategies.ema_vwap_pullback.sl_atr_mult": (0.90, 2.40),
    "strategies.ema_vwap_pullback.tp_atr_mult": (1.40, 4.00),
    "strategies.tick_imbalance.min_momentum_atr": (0.30, 1.40),
    "strategies.tick_imbalance.sl_atr_mult": (0.85, 2.20),
    "strategies.tick_imbalance.tp_atr_mult": (1.30, 3.80),
    "strategies.structure_shift.min_engulf_body_atr": (0.20, 1.20),
    "strategies.structure_shift.sl_buffer_atr": (0.05, 0.80),
    "strategies.structure_shift.retest_min": (0.236, 0.618),
    "strategies.structure_shift.target_stage": (1.0, 3.0),
    "strategies.day_range_reversal.sl_buffer_atr": (0.05, 0.80),
    "strategies.day_range_reversal.tp_rr": (1.0, 3.0),
    "strategies.band_reversion.sl_buffer_atr": (0.05, 0.80),
}

INTEGER_PATHS: set[str] = {
    "risk.max_concurrent_positions",
    "risk.max_usd_correlated_positions",
    "risk.consecutive_loss_limit",
    "risk.consecutive_loss_cooldown_min",
}


def verify_shared_token(provided: str | None, expected: str) -> bool:
    """Constant-time token check that also refuses weak / placeholder expected tokens."""
    if not provided or not is_strong_token(expected):
        return False
    return hmac.compare_digest(provided.strip().encode(), expected.strip().encode())


def verify_bridge_token(provided: str | None, expected: str) -> bool:
    return verify_shared_token(provided, expected)


def clamp_and_validate_patch(
    cfg_dict: dict[str, Any],
    patch: ParameterPatch,
) -> ParameterPatch | None:
    """Validate patch path against whitelist and hard-clamp value within absolute safety bounds."""
    if patch.path not in HARD_SAFETY_BOUNDS:
        return None
    low, high = HARD_SAFETY_BOUNDS[patch.path]
    clamped = max(low, min(high, float(patch.new_value)))
    if patch.path in INTEGER_PATHS:
        clamped = float(int(round(clamped)))
    else:
        clamped = round(clamped, 4)

    try:
        current = float(get_nested_value(cfg_dict, patch.path))
    except (KeyError, TypeError, ValueError):
        return None

    if abs(clamped - current) < 1e-6:
        return None

    return ParameterPatch(
        path=patch.path,
        old_value=round(current, 4),
        new_value=clamped,
        rationale=patch.rationale,
    )


def apply_safe_patches(
    config: TradingConfig,
    patches: list[ParameterPatch],
) -> tuple[TradingConfig, list[ParameterPatch]]:
    """Apply only whitelisted, hard-clamped patches to a copy of TradingConfig."""
    cfg_dict = config.model_dump(mode="json")
    applied: list[ParameterPatch] = []
    for raw_patch in patches:
        safe = clamp_and_validate_patch(cfg_dict, raw_patch)
        if safe is None:
            continue
        val: Any = int(safe.new_value) if safe.path in INTEGER_PATHS else safe.new_value
        cfg_dict = set_nested_value(cfg_dict, safe.path, val)
        applied.append(safe)
    return TradingConfig.model_validate(cfg_dict), applied


# +1: raising the value takes MORE risk / loosens a filter.  -1: lowering it does.
# LLM-proposed patches may only move these in the risk-REDUCING direction.
RISK_DIRECTION: dict[str, int] = {
    "risk.risk_per_trade_pct": +1,
    "risk.min_risk_pct": +1,
    "risk.max_risk_pct": +1,
    "risk.max_daily_drawdown_pct": +1,
    "risk.max_concurrent_positions": +1,
    "risk.max_usd_correlated_positions": +1,
    "risk.max_spread_atr_ratio": +1,
    "risk.consecutive_loss_limit": +1,
    "risk.consecutive_loss_cooldown_min": -1,
    "risk.min_rr_ratio": -1,
    "ai.ml_min_probability": -1,
}
MAX_LLM_STEP_FRACTION = 0.20  # one reflection can move a parameter at most 20% of its current value


def restrict_llm_patch(cfg_dict: dict[str, Any], patch: ParameterPatch) -> ParameterPatch | None:
    """Stricter gate for LLM ideas: whitelist + hard clamp + no risk increase + capped step size."""
    try:
        current = float(get_nested_value(cfg_dict, patch.path))
    except (KeyError, TypeError, ValueError):
        return None
    target = float(patch.new_value)
    direction = RISK_DIRECTION.get(patch.path, 0)
    if direction and (target - current) * direction > 0:
        return None
    max_step = abs(current) * MAX_LLM_STEP_FRACTION
    if max_step > 0:
        target = max(current - max_step, min(current + max_step, target))
    return clamp_and_validate_patch(
        cfg_dict, ParameterPatch(path=patch.path, old_value=current, new_value=target, rationale=patch.rationale)
    )
