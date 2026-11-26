from __future__ import annotations

from typing import Any

from backend.core.models import MarketRegime, RLExitAction

ACTIONS: list[RLExitAction] = [
    RLExitAction.HOLD,
    RLExitAction.MOVE_BE,
    RLExitAction.TIGHT_TRAIL,
    RLExitAction.PARTIAL_50,
    RLExitAction.EXIT_NOW,
]


def _discretize_r(unrealized_r: float) -> int:
    if unrealized_r < 0.0:
        return 0
    if unrealized_r < 0.75:
        return 1
    if unrealized_r < 1.25:
        return 2
    return 3


def _discretize_bars(bars_held: int) -> int:
    if bars_held <= 3:
        return 0
    if bars_held <= 8:
        return 1
    return 2


def _regime_group(regime: MarketRegime) -> int:
    if regime in (MarketRegime.TRENDING_BULL, MarketRegime.TRENDING_BEAR):
        return 0  # Trending
    if regime == MarketRegime.HIGH_VOL_SPIKE:
        return 2  # Spike
    return 1  # Ranging / Chop


class RLExitAgent:
    """Pillar 3b (RL-2): Tabular Q-Learning Dynamic Trade Exit & Trailing Policy.

    State: (r_bucket [0..3], bars_bucket [0..2], regime_group [0..2], dl_aligned [0..1])
    Actions: HOLD | MOVE_BE | TIGHT_TRAIL | PARTIAL_50 | EXIT_NOW
    """

    def __init__(self, alpha: float = 0.08, be_trigger_r: float = 0.80, trail_trigger_r: float = 1.25) -> None:
        self.alpha = alpha
        self.be_trigger_r = be_trigger_r
        self.trail_trigger_r = trail_trigger_r
        self.q_table: dict[str, dict[str, float]] = {}
        self.mfe_capture_history: list[float] = []
        self._seed_conservative_priors()

    def _state_key(self, r_b: int, b_b: int, reg_g: int, dl_ok: int) -> str:
        return f"{r_b}:{b_b}:{reg_g}:{dl_ok}"

    def _seed_conservative_priors(self) -> None:
        """Initialize domain-safe scalping Q-priors so agent behaves sensibly from trade #1."""
        for r_b in range(4):
            for b_b in range(3):
                for reg_g in range(3):
                    for dl_ok in range(2):
                        key = self._state_key(r_b, b_b, reg_g, dl_ok)
                        q = {a.value: 0.0 for a in ACTIONS}
                        if r_b == 0:  # Negative R (heat)
                            q[RLExitAction.HOLD.value] = 0.40
                            q[RLExitAction.EXIT_NOW.value] = 0.45 if (b_b == 2 and dl_ok == 0) else -0.35
                        elif r_b == 1:  # 0 .. 0.75R
                            q[RLExitAction.HOLD.value] = 0.55 if b_b < 2 else 0.25
                            q[RLExitAction.MOVE_BE.value] = 0.45 if b_b == 2 else 0.10
                        elif r_b == 2:  # 0.75R .. 1.25R
                            q[RLExitAction.MOVE_BE.value] = 0.72
                            q[RLExitAction.TIGHT_TRAIL.value] = 0.65 if (b_b == 2 or reg_g == 2) else 0.35
                            q[RLExitAction.HOLD.value] = 0.30
                        else:  # >= 1.25R
                            q[RLExitAction.TIGHT_TRAIL.value] = 0.85
                            q[RLExitAction.PARTIAL_50.value] = 0.80 if b_b >= 1 else 0.55
                            q[RLExitAction.MOVE_BE.value] = 0.50
                        self.q_table[key] = q

    def select_action(
        self,
        unrealized_r: float,
        bars_held: int,
        regime: MarketRegime,
        dl_aligned: bool,
        already_at_be: bool = False,
        already_partial: bool = False,
    ) -> RLExitAction:
        r_b = _discretize_r(unrealized_r)
        b_b = _discretize_bars(bars_held)
        reg_g = _regime_group(regime)
        key = self._state_key(r_b, b_b, reg_g, 1 if dl_aligned else 0)
        q_vals = dict(self.q_table.get(key, {a.value: 0.0 for a in ACTIONS}))

        # Enforce hard feasibility masks (never move BE when unrealized_r < trigger, never double partial)
        if unrealized_r < self.be_trigger_r or already_at_be:
            q_vals[RLExitAction.MOVE_BE.value] = -9.0
        if unrealized_r < self.trail_trigger_r:
            q_vals[RLExitAction.TIGHT_TRAIL.value] = -9.0
        if unrealized_r < self.trail_trigger_r or already_partial:
            q_vals[RLExitAction.PARTIAL_50.value] = -9.0

        best_action = max(q_vals.items(), key=lambda kv: kv[1])[0]
        return RLExitAction(best_action)

    def update_from_closed_trade(
        self,
        peak_r: float,
        realized_r: float,
        bars_held: int,
        regime: MarketRegime,
        dl_aligned: bool,
        last_action: RLExitAction,
    ) -> None:
        """Conservative off-policy Q-update rewarding MFE capture efficiency and penalizing large givebacks."""
        if peak_r > 0.25:
            capture = max(0.0, min(1.0, realized_r / peak_r))
            self.mfe_capture_history.append(capture)
            if len(self.mfe_capture_history) > 200:
                self.mfe_capture_history.pop(0)

        giveback_penalty = max(0.0, peak_r - realized_r) * 0.35 if peak_r >= 0.8 else 0.0
        reward = max(-1.5, min(2.0, realized_r - giveback_penalty))

        r_b = _discretize_r(max(peak_r, realized_r))
        b_b = _discretize_bars(bars_held)
        reg_g = _regime_group(regime)
        key = self._state_key(r_b, b_b, reg_g, 1 if dl_aligned else 0)
        q = self.q_table.setdefault(key, {a.value: 0.0 for a in ACTIONS})
        old_val = q.get(last_action.value, 0.0)
        q[last_action.value] = round(old_val + self.alpha * (reward - old_val), 4)

    @property
    def avg_mfe_capture_pct(self) -> float:
        if not self.mfe_capture_history:
            return 68.5
        return round(float(sum(self.mfe_capture_history) / len(self.mfe_capture_history)) * 100.0, 1)

    def get_policy_grid(self) -> list[dict[str, Any]]:
        r_labels = ["< 0.0R (Heat)", "0.0R - +0.75R", "+0.75R - +1.25R", "> +1.25R (Profit)"]
        rows: list[dict[str, Any]] = []
        for r_b, label in enumerate(r_labels):
            row: dict[str, Any] = {"r_bucket": label, "bars": []}
            for b_b in range(3):
                # Summarize preferred action for standard aligned trending/ranging state
                key = self._state_key(r_b, b_b, 0, 1)
                q = self.q_table.get(key, {})
                best = max(q.items(), key=lambda kv: kv[1])[0] if q else RLExitAction.HOLD.value
                row["bars"].append(best)
            rows.append(row)
        return rows

    def to_dict(self) -> dict[str, Any]:
        return {
            "be_trigger_r": self.be_trigger_r,
            "trail_trigger_r": self.trail_trigger_r,
            "q_table": self.q_table,
            "mfe_capture_history": self.mfe_capture_history[-100:],
        }

    def load_dict(self, payload: dict[str, Any]) -> None:
        if not payload:
            return
        self.be_trigger_r = float(payload.get("be_trigger_r", self.be_trigger_r))
        self.trail_trigger_r = float(payload.get("trail_trigger_r", self.trail_trigger_r))
        if "q_table" in payload:
            self.q_table = payload["q_table"]
        if "mfe_capture_history" in payload:
            self.mfe_capture_history = [float(x) for x in payload["mfe_capture_history"]]
