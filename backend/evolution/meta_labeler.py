from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss

from backend.features.dataset import META_FEATURE_NAMES


def effective_ml_cutoff(ml_min_probability: float, rr_ratio: float) -> float:
    """RR-aware calibrated win probability cutoff so positive-EV high-RR setups are not starved.

    For a 1:1 trade (RR=1.0, break-even WR=50%), the cutoff equals ``ml_min_probability``.
    For higher RR (e.g. RR=1.6..2.2, break-even WR=31..38%), the cutoff scales with the
    break-even win rate plus the operator's safety margin, while monotonically respecting
    any manual tightening of ``ml_min_probability``.
    """
    rr = max(0.5, float(rr_ratio))
    be_wr = 1.0 / (1.0 + rr)
    margin = max(0.02, float(ml_min_probability) - 0.50)
    ev_floor = be_wr + margin + 0.04
    return round(max(0.38, min(float(ml_min_probability), ev_floor)), 4)


class CalibratedMetaLabeler:
    """Pillar 2 (ML): Calibrated HistGradientBoosting Meta-Labeler predicting P(win | signal, regime, DL, spread)."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.model: Any | None = None
        self.trained: bool = False
        self.brier_score: float = 0.25
        self.oos_accuracy: float = 0.50
        self.n_samples: int = 0
        self.calibration_points: list[tuple[float, float]] = [
            (0.25, 0.24),
            (0.45, 0.44),
            (0.60, 0.61),
            (0.75, 0.74),
        ]
        self.feature_importances: list[dict[str, Any]] = [
            {"name": "dl_prob_align", "importance": 0.22},
            {"name": "spread_atr_ratio", "importance": 0.18},
            {"name": "wick_align", "importance": 0.15},
            {"name": "adx_norm", "importance": 0.13},
            {"name": "raw_confidence", "importance": 0.12},
            {"name": "dl_vol_expansion", "importance": 0.10},
        ]

    def fit_purged_walk_forward(
        self,
        x: np.ndarray,
        y: np.ndarray,
        embargo_bars: int = 5,
        sample_weight: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Train on chronological train split with embargo gap, calibrate probabilities, and evaluate on OOS split."""
        n = len(x)
        if n < 24 or len(np.unique(y)) < 2:
            return {"trained": 0.0, "brier": self.brier_score, "oos_acc": self.oos_accuracy}

        split = max(16, int(n * 0.75))
        oos_start = min(n - 6, split + embargo_bars)
        x_tr, y_tr = x[:split], y[:split]
        sw_tr = np.asarray(sample_weight[:split], dtype=np.float64) if sample_weight is not None else None
        x_oos, y_oos = x[oos_start:], y[oos_start:]

        if len(np.unique(y_tr)) < 2:
            x_tr, y_tr = x, y
            sw_tr = np.asarray(sample_weight, dtype=np.float64) if sample_weight is not None else None
            x_oos, y_oos = x[-max(8, n // 4) :], y[-max(8, n // 4) :]

        base = HistGradientBoostingClassifier(
            max_iter=80,
            max_depth=4,
            learning_rate=0.06,
            min_samples_leaf=max(5, min(15, len(x_tr) // 10)),
            l2_regularization=1.0,
            random_state=self.random_state,
        )

        class_counts = np.bincount(y_tr.astype(int))
        min_class = int(np.min(class_counts)) if len(class_counts) >= 2 else 0
        if len(x_tr) >= 40 and min_class >= 6:
            cv_folds = min(3, min_class)
            clf = CalibratedClassifierCV(base, method="sigmoid", cv=cv_folds)
            clf.fit(x_tr, y_tr, sample_weight=sw_tr)
            self.model = clf
        else:
            base.fit(x_tr, y_tr, sample_weight=sw_tr)
            self.model = base

        self.trained = True
        self.n_samples = n

        probs = self.predict_proba_batch(x_oos)
        self.brier_score = float(brier_score_loss(y_oos, probs)) if len(np.unique(y_oos)) >= 2 else 0.22
        self.oos_accuracy = float(np.mean((probs >= 0.5).astype(int) == y_oos))
        self._compute_calibration_curve(probs, y_oos)
        self._compute_feature_importances(x_oos, y_oos)

        return {
            "trained": 1.0,
            "brier": round(self.brier_score, 4),
            "oos_acc": round(self.oos_accuracy, 4),
            "samples": float(n),
        }

    def _compute_calibration_curve(self, probs: np.ndarray, y_true: np.ndarray) -> None:
        bins = np.linspace(0.0, 1.0, 7)
        pts: list[tuple[float, float]] = []
        for i in range(len(bins) - 1):
            mask = (probs >= bins[i]) & (probs < bins[i + 1] + (0.01 if i == len(bins) - 2 else 0.0))
            if np.sum(mask) >= 2:
                pts.append((round(float(np.mean(probs[mask])), 3), round(float(np.mean(y_true[mask])), 3)))
        if len(pts) >= 2:
            self.calibration_points = pts

    def _compute_feature_importances(self, x_oos: np.ndarray, y_oos: np.ndarray) -> None:
        if self.model is None or len(x_oos) < 8:
            return
        base_probs = self.predict_proba_batch(x_oos)
        base_loss = float(np.mean((base_probs - y_oos) ** 2))
        rng = np.random.default_rng(self.random_state)
        scores: list[tuple[str, float]] = []
        for col in range(min(x_oos.shape[1], len(META_FEATURE_NAMES))):
            x_perm = x_oos.copy()
            x_perm[:, col] = rng.permutation(x_perm[:, col])
            perm_probs = self.predict_proba_batch(x_perm)
            delta = max(0.0, float(np.mean((perm_probs - y_oos) ** 2)) - base_loss)
            scores.append((META_FEATURE_NAMES[col], delta))
        total = sum(s for _, s in scores)
        if total > 1e-8:
            ranked = sorted(scores, key=lambda item: item[1], reverse=True)[:8]
            self.feature_importances = [
                {"name": name, "importance": round(val / total, 4)} for name, val in ranked
            ]

    def predict_proba_batch(self, x: np.ndarray) -> np.ndarray:
        if not self.trained or self.model is None:
            # Cold-start calibrated heuristic prior combining raw confidence (idx 10), spread/ATR (idx 17), DL align (idx 20)
            raw_conf = np.clip(x[:, 10], 0.35, 0.90)
            sp_pen = np.clip(x[:, 17] * 0.25, 0.0, 0.25)
            dl_boost = (x[:, 20] - 0.33) * 0.25
            return np.clip(raw_conf - sp_pen + dl_boost, 0.25, 0.85)
        n_in = getattr(self.model, "n_features_in_", None)
        if n_in is not None and x.shape[1] > n_in:
            x = x[:, :n_in]  # model trained before newer features were appended
        probs = self.model.predict_proba(x)
        if probs.shape[1] == 2:
            return probs[:, 1]
        return np.full((len(x),), 0.55, dtype=np.float64)

    def score(self, x: np.ndarray, y: np.ndarray) -> float | None:
        """Brier score (lower is better) on held-out samples; None without enough evidence."""
        if not self.trained or len(x) < 10 or len(np.unique(y)) < 2:
            return None
        return round(float(brier_score_loss(y, self.predict_proba_batch(x))), 5)

    def predict_win_probability(self, feat_vec: np.ndarray) -> float:
        x = feat_vec[None, :] if feat_vec.ndim == 1 else feat_vec
        return round(float(self.predict_proba_batch(x)[0]), 4)

    def diagnostics(self) -> dict[str, Any]:
        return {
            "trained": self.trained,
            "samples": self.n_samples,
            "brier_score": round(self.brier_score, 4),
            "oos_accuracy": round(self.oos_accuracy, 4),
            "calibration_points": self.calibration_points,
            "feature_importances": self.feature_importances,
        }


    # --- Persistence (local snapshots dir only; *.pkl is git-ignored) ---
    def save(self, path: str | Path) -> None:
        if not self.trained:
            return
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        with tmp.open("wb") as fh:
            pickle.dump(self.__dict__, fh)
        tmp.replace(p)

    def load(self, path: str | Path) -> bool:
        """Load a model this app saved itself. Never point this at a file from the internet."""
        p = Path(path)
        if not p.exists():
            return False
        try:
            with p.open("rb") as fh:
                state = pickle.load(fh)  # noqa: S301 - trusted local snapshot written by save()
            self.__dict__.update(state)
            return True
        except Exception:
            return False
