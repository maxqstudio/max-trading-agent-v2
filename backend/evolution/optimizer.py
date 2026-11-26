from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backend.core.config import TradingConfig, get_nested_value
from backend.core.models import BacktestMetrics, Bar, ParameterPatch, SymbolSpec
from backend.core.security import apply_safe_patches
from backend.evolution.backtester import CausalBacktester
from backend.evolution.meta_labeler import CalibratedMetaLabeler
from backend.evolution.monte_carlo import run_monte_carlo_stress
from backend.evolution.sequence_net import SequenceConvNet
from backend.features.dataset import build_dl_training_dataset
from backend.risk.manager import DEFAULT_SPECS

TUNABLE_SPACE: dict[str, tuple[float, float]] = {
    "risk.max_spread_atr_ratio": (0.18, 0.38),
    "ai.ml_min_probability": (0.52, 0.68),
    "ai.rl_exit_be_trigger_r": (0.65, 1.15),
    "ai.rl_exit_trail_trigger_r": (1.00, 1.70),
    "strategies.liquidity_sweep.min_wick_ratio": (0.48, 0.70),
    "strategies.liquidity_sweep.sl_atr_mult": (1.15, 1.95),
    "strategies.liquidity_sweep.tp_atr_mult": (1.80, 3.00),
    "strategies.micro_breakout.min_body_atr": (0.35, 0.75),
    "strategies.micro_breakout.sl_atr_mult": (1.10, 1.90),
    "strategies.micro_breakout.tp_atr_mult": (1.90, 3.20),
    "strategies.ema_vwap_pullback.min_adx": (16.0, 26.0),
}


@dataclass(slots=True)
class WFOResult:
    best_config: TradingConfig
    best_patches: list[ParameterPatch]
    champion_oos: BacktestMetrics
    challenger_oos: BacktestMetrics
    dl_train_summary: dict[str, float]
    ml_train_summary: dict[str, float]
    trained_dl: SequenceConvNet
    trained_ml: CalibratedMetaLabeler


class PurgedWalkForwardOptimizer:
    """Purged & Embargoed Walk-Forward Retrainer for DL 1D-CNN, ML Meta-Labeler, and Strategy Params."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)
        self.backtester = CausalBacktester()

    def _split_bars_purged(
        self,
        bars_by_symbol: dict[str, list[Bar]],
        train_ratio: float = 0.65,
        embargo_bars: int = 12,
    ) -> tuple[dict[str, list[Bar]], dict[str, list[Bar]]]:
        train_map: dict[str, list[Bar]] = {}
        oos_map: dict[str, list[Bar]] = {}
        for sym, bars in bars_by_symbol.items():
            if len(bars) < 120:
                continue
            cut = int(len(bars) * train_ratio)
            oos_start = min(len(bars) - 65, cut + embargo_bars)
            train_map[sym] = bars[:cut]
            # Prefix 48 warmup bars before oos_start solely for indicator calculation; signals start at idx 48
            oos_map[sym] = bars[max(0, oos_start - 48) :]
        return train_map, oos_map

    def run(
        self,
        bars_by_symbol: dict[str, list[Bar]],
        champion_config: TradingConfig,
        extra_patches: list[ParameterPatch] | None = None,
        specs: dict[str, SymbolSpec] | None = None,
    ) -> WFOResult:
        sym_specs = specs or DEFAULT_SPECS
        train_bars, oos_bars = self._split_bars_purged(bars_by_symbol)

        # 1. Retrain Lightweight DL 1D-CNN strictly on Train split
        dl_net = SequenceConvNet(seed=int(self.rng.integers(1, 10000)))
        all_x, all_yd, all_yv = [], [], []
        for sym, b_list in train_bars.items():
            spec = sym_specs.get(sym, DEFAULT_SPECS["XAUUSD"])
            xs, yds, yvs = build_dl_training_dataset(
                b_list,
                seq_len=champion_config.ai.dl_seq_len,
                horizon=8,
                point=spec.point,
            )
            if len(xs) > 0:
                all_x.append(xs)
                all_yd.append(yds)
                all_yv.append(yvs)

        dl_summary = {"train_loss": 0.0, "val_loss": 0.0, "val_acc": 0.0}
        if all_x:
            x_cat = np.concatenate(all_x, axis=0)
            yd_cat = np.concatenate(all_yd, axis=0)
            yv_cat = np.concatenate(all_yv, axis=0)
            if len(x_cat) >= 40:
                dl_summary = dl_net.fit(
                    x_cat,
                    yd_cat,
                    yv_cat,
                    epochs=champion_config.ai.dl_epochs,
                    lr=champion_config.ai.dl_lr,
                )

        # 2. Collect strictly in-sample meta-labeling features & retrain Calibrated ML Meta-Labeler
        ml_labeler = CalibratedMetaLabeler()
        _, _, (mx, my) = self.backtester.run(
            train_bars,
            champion_config,
            specs=sym_specs,
            dl_net=dl_net,
            ml_labeler=None,
            collect_meta_samples=True,
        )
        ml_summary = ml_labeler.fit_purged_walk_forward(mx, my)

        # 3. Evaluate Champion baseline on Out-of-Sample split + Monte Carlo
        champ_oos, champ_trades, _ = self.backtester.run(
            oos_bars,
            champion_config,
            specs=sym_specs,
            dl_net=dl_net,
            ml_labeler=ml_labeler,
        )
        mc_c = run_monte_carlo_stress(
            [t.pnl_r for t in champ_trades],
            risk_per_trade_pct=champion_config.risk.risk_per_trade_pct,
            iterations=champion_config.evolution.monte_carlo_iterations,
        )
        champ_oos.mc_p95_dd_pct = mc_c.p95_max_dd_pct
        champ_oos.mc_ruin_prob = mc_c.ruin_probability

        # 4. Build candidate pool (LLM patches first + bounded random search proposals)
        cfg_dict = champion_config.model_dump(mode="json")
        candidate_patch_sets: list[list[ParameterPatch]] = []
        if extra_patches:
            candidate_patch_sets.append(extra_patches)

        keys = list(TUNABLE_SPACE.keys())
        trials = max(4, champion_config.evolution.wfo_trials)
        for _ in range(trials):
            chosen = self.rng.choice(keys, size=3, replace=False)
            pset: list[ParameterPatch] = []
            for k in chosen:
                lo, hi = TUNABLE_SPACE[str(k)]
                cur = float(get_nested_value(cfg_dict, str(k)))
                proposed = float(self.rng.uniform(lo, hi))
                pset.append(
                    ParameterPatch(
                        path=str(k),
                        old_value=cur,
                        new_value=round(proposed, 3),
                        rationale="Walk-Forward bounded exploration",
                    )
                )
            if extra_patches:
                pset = list(extra_patches[:2]) + pset[:2]
            candidate_patch_sets.append(pset)

        # Score candidates on Train first, then validate top finalists on OOS (anti-overfit!)
        scored_train: list[tuple[float, TradingConfig, list[ParameterPatch]]] = []
        for pset in candidate_patch_sets:
            cand_cfg, applied = apply_safe_patches(champion_config, pset)
            if not applied:
                continue
            tr_m, _, _ = self.backtester.run(
                train_bars,
                cand_cfg,
                specs=sym_specs,
                dl_net=dl_net,
                ml_labeler=ml_labeler,
            )
            tr_score = tr_m.sharpe_ratio + 0.5 * tr_m.profit_factor - 0.35 * tr_m.max_drawdown_pct
            scored_train.append((tr_score, cand_cfg, applied))

        scored_train.sort(key=lambda item: item[0], reverse=True)
        finalists = scored_train[:3]

        best_cfg = champion_config
        best_patches: list[ParameterPatch] = []
        best_oos = champ_oos
        best_oos_score = champ_oos.sharpe_ratio + 0.4 * champ_oos.profit_factor - 0.25 * champ_oos.mc_p95_dd_pct

        for _, cand_cfg, applied in finalists:
            oos_m, oos_trades, _ = self.backtester.run(
                oos_bars,
                cand_cfg,
                specs=sym_specs,
                dl_net=dl_net,
                ml_labeler=ml_labeler,
            )
            mc = run_monte_carlo_stress(
                [t.pnl_r for t in oos_trades],
                risk_per_trade_pct=cand_cfg.risk.risk_per_trade_pct,
                iterations=cand_cfg.evolution.monte_carlo_iterations,
            )
            oos_m.mc_p95_dd_pct = mc.p95_max_dd_pct
            oos_m.mc_ruin_prob = mc.ruin_probability
            score = oos_m.sharpe_ratio + 0.4 * oos_m.profit_factor - 0.25 * oos_m.mc_p95_dd_pct
            if score > best_oos_score or not best_patches:
                best_oos_score = score
                best_cfg = cand_cfg
                best_oos = oos_m
                best_patches = applied

        return WFOResult(
            best_config=best_cfg,
            best_patches=best_patches,
            champion_oos=champ_oos,
            challenger_oos=best_oos,
            dl_train_summary=dl_summary,
            ml_train_summary=ml_summary,
            trained_dl=dl_net,
            trained_ml=ml_labeler,
        )
