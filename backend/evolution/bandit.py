from __future__ import annotations

import math
from typing import Any

import numpy as np

from backend.core.models import MarketRegime

DEFAULT_STRATEGIES = [
    "liquidity_sweep",
    "micro_breakout",
    "ema_vwap_pullback",
    "tick_imbalance",
    "structure_shift",
]


class ContextualBanditRL:
    """Pillar 3a (RL-1): Contextual Thompson Sampling Bandit with exponential forgetting factor.

    Adapts strategy allocation weights per (symbol, regime) after every closed trade.
    """

    def __init__(
        self,
        strategies: list[str] | None = None,
        forgetting_factor: float = 0.985,
        seed: int = 42,
    ) -> None:
        self.strategies = list(strategies or DEFAULT_STRATEGIES)
        self.gamma = float(forgetting_factor)
        self.rng = np.random.default_rng(seed)
        # state[symbol][regime][strategy_id] = {"alpha": float, "beta": float, "n": int, "avg_r": float}
        self._arms: dict[str, dict[str, dict[str, dict[str, float]]]] = {}

    def _ensure_context(self, symbol: str, regime: MarketRegime | str) -> dict[str, dict[str, float]]:
        reg = regime.value if isinstance(regime, MarketRegime) else str(regime)
        sym_map = self._arms.setdefault(symbol, {})
        if reg not in sym_map:
            # Mild domain-informed prior (still balanced so data dominates rapidly)
            sym_map[reg] = {}
            for s in self.strategies:
                prior_a, prior_b = 2.0, 2.0
                if reg == MarketRegime.RANGING.value and s == "liquidity_sweep":
                    prior_a = 2.6
                elif reg in (MarketRegime.TRENDING_BULL.value, MarketRegime.TRENDING_BEAR.value) and s in (
                    "micro_breakout",
                    "ema_vwap_pullback",
                ):
                    prior_a = 2.5
                sym_map[reg][s] = {"alpha": prior_a, "beta": prior_b, "n": 0.0, "avg_r": 0.0}
        ctx = sym_map[reg]
        for s in self.strategies:
            # A strategy added after this state was saved starts with a neutral prior.
            ctx.setdefault(s, {"alpha": 2.0, "beta": 2.0, "n": 0.0, "avg_r": 0.0})
        return ctx

    def update(self, symbol: str, regime: MarketRegime | str, strategy_id: str, pnl_r: float) -> None:
        ctx = self._ensure_context(symbol, regime)
        if strategy_id not in ctx:
            ctx[strategy_id] = {"alpha": 2.0, "beta": 2.0, "n": 0.0, "avg_r": 0.0}

        # Apply gentle forgetting to all arms in this context so bandit tracks non-stationary market shifts
        for arm in ctx.values():
            arm["alpha"] = max(1.0, 1.0 + (arm["alpha"] - 1.0) * self.gamma)
            arm["beta"] = max(1.0, 1.0 + (arm["beta"] - 1.0) * self.gamma)

        arm = ctx[strategy_id]
        # Map risk-adjusted R-multiple into soft success [0, 1] via logistic transform
        clipped_r = max(-2.5, min(3.0, float(pnl_r)))
        soft_success = 1.0 / (1.0 + math.exp(-1.1 * clipped_r))
        arm["alpha"] += soft_success
        arm["beta"] += 1.0 - soft_success
        n_prev = arm["n"]
        arm["n"] = n_prev + 1.0
        arm["avg_r"] = (arm["avg_r"] * n_prev + clipped_r) / arm["n"]

    def get_weights(self, symbol: str, regime: MarketRegime | str, explore: bool = False) -> dict[str, float]:
        ctx = self._ensure_context(symbol, regime)
        raw: dict[str, float] = {}
        for sid, arm in ctx.items():
            if explore:
                score = float(self.rng.beta(max(arm["alpha"], 0.5), max(arm["beta"], 0.5)))
            else:
                score = arm["alpha"] / (arm["alpha"] + arm["beta"])
            raw[sid] = max(0.05, score)
        total = sum(raw.values())
        return {sid: round(val / total, 4) for sid, val in raw.items()}

    def get_matrix(self, symbol: str) -> list[dict[str, Any]]:
        regimes = [
            MarketRegime.TRENDING_BULL,
            MarketRegime.TRENDING_BEAR,
            MarketRegime.RANGING,
            MarketRegime.HIGH_VOL_SPIKE,
            MarketRegime.LOW_VOL_CHOP,
        ]
        rows: list[dict[str, Any]] = []
        for reg in regimes:
            weights = self.get_weights(symbol, reg, explore=False)
            ctx = self._ensure_context(symbol, reg)
            rows.append(
                {
                    "regime": reg.value,
                    "weights": weights,
                    "trades": {s: int(ctx[s]["n"]) for s in self.strategies},
                    "avg_r": {s: round(ctx[s]["avg_r"], 3) for s in self.strategies},
                }
            )
        return rows

    def to_dict(self) -> dict[str, Any]:
        return {"gamma": self.gamma, "arms": self._arms}

    def load_dict(self, payload: dict[str, Any]) -> None:
        if not payload:
            return
        self.gamma = float(payload.get("gamma", self.gamma))
        self._arms = payload.get("arms", {})
