from __future__ import annotations

import copy
from dataclasses import dataclass, field

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
    "ai.ml_min_probability": (0.46, 0.64),
    "ai.rl_exit_be_trigger_r": (0.65, 1.15),
    "ai.rl_exit_trail_trigger_r": (1.00, 1.70),
    "strategies.liquidity_sweep.min_wick_ratio": (0.48, 0.70),
    "strategies.liquidity_sweep.sl_atr_mult": (1.15, 1.95),
    "strategies.liquidity_sweep.tp_atr_mult": (1.80, 3.00),
    "strategies.micro_breakout.min_body_atr": (0.35, 0.75),
    "strategies.micro_breakout.sl_atr_mult": (1.10, 1.90),
    "strategies.micro_breakout.tp_atr_mult": (1.90, 3.20),
    "strategies.ema_vwap_pullback.min_adx": (16.0, 26.0),
    "strategies.structure_shift.min_engulf_body_atr": (0.35, 0.80),
    "strategies.structure_shift.sl_buffer_atr": (0.10, 0.50),
    "strategies.day_range_reversal.tp_rr": (1.2, 2.5),
    "strategies.band_reversion.sl_buffer_atr": (0.15, 0.60),
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
    # What happened to each model this cycle: kept the champion or adopted the new one, and why.
    dl_decision: dict = field(default_factory=dict)
    ml_decision: dict = field(default_factory=dict)


def compute_timestamp_recency_weights(
    timestamps: list[int] | np.ndarray,
    half_life_days: float = 5.0,
    min_weight: float = 0.20,
) -> np.ndarray:
    """Exponential time-decay weights from real Unix timestamps (mean-normalized to 1.0).

    Recent bars/trades receive weight ~1.0 while older regimes decay gently toward ``min_weight``
    so the models adapt to current volatility without forgetting older market regimes.
    """
    ts = np.asarray(timestamps, dtype=np.float64)
    if len(ts) == 0:
        return np.empty((0,), dtype=np.float64)
    max_ts = float(np.max(ts))
    age_days = np.maximum(0.0, (max_ts - ts) / 86400.0)
    hl = max(float(half_life_days), 0.25)
    raw = np.clip(np.power(0.5, age_days / hl), min_weight, 1.0)
    mean_w = float(np.mean(raw))
    return raw / mean_w if mean_w > 1e-12 else np.ones_like(raw)


def _multiscale_indices(n: int, recent_keep: int = 2400, max_total: int = 3600) -> np.ndarray:
    """Keep 100% of recent bars and evenly strided older history so 14-day CPU training stays fast."""
    if n <= max_total:
        return np.arange(n, dtype=np.int64)
    recent_start = max(0, n - recent_keep)
    older_budget = max(200, max_total - (n - recent_start))
    older_idx = np.linspace(0, max(0, recent_start - 1), older_budget, dtype=np.int64)
    older_idx = np.unique(older_idx)
    recent_idx = np.arange(recent_start, n, dtype=np.int64)
    return np.concatenate([older_idx, recent_idx])


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
        champion_dl: SequenceConvNet | None = None,
        champion_ml: CalibratedMetaLabeler | None = None,
        param_window_bars: int | None = None,
    ) -> WFOResult:
        sym_specs = specs or DEFAULT_SPECS
        train_bars, oos_bars = self._split_bars_purged(bars_by_symbol)

        # 1. DL 1D-CNN: continue from the champion's weights (warm start) on the Train split with
        #    per-symbol exponential time-decay weights (14-day multi-regime memory without forgetting),
        #    then adopt only when OOS loss beats champion by at least ``dl_min_oos_gain``.
        hl_days = champion_config.evolution.recency_half_life_days

        def dl_dataset(
            bmap: dict[str, list[Bar]], *, with_weights: bool = False
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None] | None:
            xs_all, yd_all, yv_all, ts_all = [], [], [], []
            seq_len = champion_config.ai.dl_seq_len
            horizon = 8
            for sym, b_list in bmap.items():
                spec = sym_specs.get(sym, DEFAULT_SPECS["XAUUSD"])
                xs, yds, yvs = build_dl_training_dataset(b_list, seq_len=seq_len, horizon=horizon, point=spec.point)
                if len(xs) == 0:
                    continue
                # Timestamps aligned with build_dl_training_dataset's end_idx = range(seq_len+16, len(bars)-horizon)
                start_bar = seq_len + 16
                bar_ts = np.array([b.timestamp for b in b_list[start_bar : start_bar + len(xs)]], dtype=np.int64)
                if with_weights and len(xs) > 3600:
                    keep = _multiscale_indices(len(xs), recent_keep=2400, max_total=3600)
                    xs, yds, yvs, bar_ts = xs[keep], yds[keep], yvs[keep], bar_ts[keep]
                xs_all.append(xs)
                yd_all.append(yds)
                yv_all.append(yvs)
                ts_all.append(bar_ts)
            if not xs_all:
                return None
            x_cat = np.concatenate(xs_all, axis=0)
            yd_cat = np.concatenate(yd_all, axis=0)
            yv_cat = np.concatenate(yv_all, axis=0)
            if not with_weights:
                return x_cat, yd_cat, yv_cat, None
            # Sort globally by timestamp so chronological train/val split inside fit() never leaks across symbols!
            ts_cat = np.concatenate(ts_all, axis=0)
            order = np.argsort(ts_cat, kind="stable")
            sw = compute_timestamp_recency_weights(ts_cat[order], half_life_days=hl_days)
            return x_cat[order], yd_cat[order], yv_cat[order], sw

        warm = champion_dl is not None and champion_dl.trained
        dl_net = copy.deepcopy(champion_dl) if warm else SequenceConvNet(seed=int(self.rng.integers(1, 10000)))
        dl_summary = {"train_loss": 0.0, "val_loss": 0.0, "val_acc": 0.0}
        tr_ds = dl_dataset(train_bars, with_weights=True)
        oos_ds_full = dl_dataset(oos_bars, with_weights=False)
        oos_ds = oos_ds_full[:3] if oos_ds_full is not None else None
        if tr_ds is not None and len(tr_ds[0]) >= 40:
            lr = champion_config.ai.dl_lr * (champion_config.evolution.dl_finetune_lr_mult if warm else 1.0)
            dl_summary = dl_net.fit(tr_ds[0], tr_ds[1], tr_ds[2], epochs=champion_config.ai.dl_epochs, lr=lr, sample_weight=tr_ds[3])
        dl_net, dl_decision = _pick_dl(
            champion_dl, dl_net, oos_ds, warm, min_oos_gain=champion_config.evolution.dl_min_oos_gain
        )

        # 2. ML Meta-Labeler: trained on multi-day setups sorted chronologically with exponential
        #    time-decay weights, adopted only if OOS Brier beats champion by ``ml_min_oos_gain``.
        _, tr_trades, (mx, my) = self.backtester.run(
            train_bars, champion_config, specs=sym_specs, dl_net=dl_net, ml_labeler=None, collect_meta_samples=True
        )
        ml_sw: np.ndarray | None = None
        if len(mx) > 0 and len(tr_trades) == len(mx):
            tr_ts = np.array([t.open_time for t in tr_trades], dtype=np.int64)
            order_ml = np.argsort(tr_ts, kind="stable")
            mx, my = mx[order_ml], my[order_ml]
            ml_sw = compute_timestamp_recency_weights(tr_ts[order_ml], half_life_days=hl_days)
        new_ml = CalibratedMetaLabeler()
        ml_summary = new_ml.fit_purged_walk_forward(mx, my, sample_weight=ml_sw)
        _, _, (ox, oy) = self.backtester.run(
            oos_bars, champion_config, specs=sym_specs, dl_net=dl_net, ml_labeler=None, collect_meta_samples=True
        )
        ml_labeler, ml_decision = _pick_ml(
            champion_ml, new_ml, ox, oy, min_oos_gain=champion_config.evolution.ml_min_oos_gain
        )

        # Strategy parameters are searched on the most recent part of the window only (much cheaper).
        if param_window_bars:
            pw = max(param_window_bars, 200)
            train_bars, oos_bars = self._split_bars_purged({k: v[-pw:] for k, v in bars_by_symbol.items()})

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
            dl_decision=dl_decision,
            ml_decision=ml_decision,
        )


def _pick_dl(
    champion: SequenceConvNet | None,
    new: SequenceConvNet,
    oos: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    warm: bool,
    *,
    min_oos_gain: float = 0.005,
) -> tuple[SequenceConvNet, dict]:
    has_champ = champion is not None and champion.trained
    if not new.trained:
        return (champion if has_champ else new), {"adopted": False, "reason": "training skipped (too little data)", "warm_start": warm}
    if not has_champ:
        return new, {"adopted": True, "reason": "first model", "warm_start": False}
    c = champion.evaluate(*oos) if oos is not None else None
    n = new.evaluate(*oos) if oos is not None else None
    if c is None or n is None:
        return champion, {"adopted": False, "reason": "no out-of-sample data to compare", "warm_start": warm}
    gain = round(c["loss"] - n["loss"], 5)
    better = gain >= min_oos_gain
    return (new if better else champion), {
        "adopted": better,
        "reason": f"OOS loss {n['loss']:.4f} vs champion {c['loss']:.4f} (gain {gain:+.4f}, min {min_oos_gain:.4f})",
        "warm_start": warm,
        "new": n,
        "champion": c,
    }


def _pick_ml(
    champion: CalibratedMetaLabeler | None,
    new: CalibratedMetaLabeler,
    ox: np.ndarray,
    oy: np.ndarray,
    *,
    min_oos_gain: float = 0.005,
) -> tuple[CalibratedMetaLabeler, dict]:
    has_champ = champion is not None and champion.trained
    if not new.trained:
        return (champion if has_champ else new), {"adopted": False, "reason": "training skipped (too few setups)"}
    if not has_champ:
        return new, {"adopted": True, "reason": "first model"}
    c, n = champion.score(ox, oy), new.score(ox, oy)
    if c is None or n is None:
        # Too few unseen setups to compare. Prefer the model that learned from more setups.
        more = new.n_samples >= champion.n_samples
        return (new if more else champion), {
            "adopted": more,
            "reason": f"only {len(oy)} out-of-sample setups; trained on {new.n_samples} vs champion {champion.n_samples} setups",
        }
    gain = round(c - n, 5)
    better = gain >= min_oos_gain
    return (new if better else champion), {
        "adopted": better,
        "reason": f"OOS Brier {n:.4f} vs champion {c:.4f} (gain {gain:+.4f}, min {min_oos_gain:.4f})",
        "new_brier": n,
        "champion_brier": c,
        "oos_setups": int(len(oy)),
    }
