from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class MonteCarloResult:
    iterations: int
    p50_max_dd_pct: float
    p95_max_dd_pct: float
    p99_max_dd_pct: float
    ruin_probability: float
    median_return_pct: float


def run_monte_carlo_stress(
    pnl_r_list: list[float],
    risk_per_trade_pct: float = 0.5,
    iterations: int = 1000,
    ruin_dd_threshold_pct: float = 15.0,
    seed: int = 42,
) -> MonteCarloResult:
    """Vectorized 1,000x trade-order reshuffling + bootstrap stress test to reject lucky curve fits."""
    if len(pnl_r_list) < 5:
        return MonteCarloResult(
            iterations=iterations,
            p50_max_dd_pct=0.0,
            p95_max_dd_pct=0.0,
            p99_max_dd_pct=0.0,
            ruin_probability=0.0,
            median_return_pct=0.0,
        )

    rng = np.random.default_rng(seed)
    rs = np.array(pnl_r_list, dtype=np.float64)
    n = len(rs)
    # Bootstrap resample with replacement across all iterations
    indices = rng.integers(0, n, size=(iterations, n))
    sampled = rs[indices]  # (I, N)
    step_returns = np.clip(sampled * (risk_per_trade_pct / 100.0), -0.5, 0.5)

    # Compounded equity paths starting at 1.0
    equity_paths = np.cumprod(1.0 + step_returns, axis=1)
    equity_paths = np.concatenate([np.ones((iterations, 1), dtype=np.float64), equity_paths], axis=1)
    peaks = np.maximum.accumulate(equity_paths, axis=1)
    drawdowns_pct = ((peaks - equity_paths) / np.maximum(peaks, 1e-9)) * 100.0
    max_dds = np.max(drawdowns_pct, axis=1)
    final_returns_pct = (equity_paths[:, -1] - 1.0) * 100.0

    return MonteCarloResult(
        iterations=iterations,
        p50_max_dd_pct=round(float(np.percentile(max_dds, 50)), 4),
        p95_max_dd_pct=round(float(np.percentile(max_dds, 95)), 4),
        p99_max_dd_pct=round(float(np.percentile(max_dds, 99)), 4),
        ruin_probability=round(float(np.mean(max_dds >= ruin_dd_threshold_pct)), 4),
        median_return_pct=round(float(np.median(final_returns_pct)), 4),
    )
