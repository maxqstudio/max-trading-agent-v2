from __future__ import annotations

import math

import numpy as np

from backend.core.models import Bar, DLPrediction, MarketRegime, Side, TradeSignal
from backend.features.indicators import IndicatorSnapshot, atr, ema
from backend.features.htf import alignment, session_of
from backend.features.structure import ALIGNED_NAMES, aligned_context

REGIME_INDEX: dict[MarketRegime, int] = {
    MarketRegime.TRENDING_BULL: 0,
    MarketRegime.TRENDING_BEAR: 1,
    MarketRegime.RANGING: 2,
    MarketRegime.HIGH_VOL_SPIKE: 3,
    MarketRegime.LOW_VOL_CHOP: 4,
}

STRATEGY_INDEX: dict[str, int] = {
    "liquidity_sweep": 0,
    "micro_breakout": 1,
    "ema_vwap_pullback": 2,
    "tick_imbalance": 3,
}
# Strategies added after the first 24 columns get their one-hot at the end instead.
LATE_STRATEGIES: tuple[str, ...] = ("structure_shift",)

META_FEATURE_NAMES: list[str] = [
    "regime_bull",
    "regime_bear",
    "regime_range",
    "regime_highvol",
    "regime_chop",
    "strat_sweep",
    "strat_breakout",
    "strat_pullback",
    "strat_imbalance",
    "side_sign",
    "raw_confidence",
    "adx_norm",
    "rsi14_norm",
    "vol_ratio",
    "bb_rank",
    "wick_align",
    "body_atr_ratio",
    "spread_atr_ratio",
    "session_sin",
    "session_cos",
    "dl_prob_align",
    "dl_prob_against",
    "dl_prob_flat",
    "dl_vol_expansion",
    # Appended later; models trained before these existed use only the first 24 columns.
    "htf_m15_align",
    "htf_h1_align",
    "session_asia",
    "session_london",
    "session_newyork",
    # Structure context (with the trade = +1, against = -1)
    "strat_structure_shift",
    "sd_zone_align",
    "kumo_align",
    "bb_pos_align",
    "engulf_align",
    # Ichimoku checklist (M1 and M15), Tenkan/Kijun cross, Chikou, future cloud, cloud thickness, Kijun stretch
    "ichi_score_align",
    "ichi_m15_align",
    "ichi_tk_cross_align",
    "ichi_chikou_align",
    "ichi_future_cloud_align",
    "ichi_cloud_thick_atr",
    "ichi_kijun_dist_align",
]
BASE_META_FEATURES = 24


def build_sequence_tensor(bars: list[Bar], seq_len: int = 32, point: float = 0.01) -> np.ndarray | None:
    """Build strictly causal normalized sequence tensor of shape (seq_len, 8) for 1D-CNN."""
    if len(bars) < seq_len + 16:
        return None
    window = bars[-(seq_len + 16) :]
    opens = np.array([b.open for b in window], dtype=np.float64)
    highs = np.array([b.high for b in window], dtype=np.float64)
    lows = np.array([b.low for b in window], dtype=np.float64)
    closes = np.array([b.close for b in window], dtype=np.float64)
    vols = np.array([max(b.volume, 1.0) for b in window], dtype=np.float64)
    spreads = np.array([b.spread_points * point for b in window], dtype=np.float64)

    atr_vals = atr(highs, lows, closes, 14)
    e21 = ema(closes, 21)

    seq = np.zeros((seq_len, 8), dtype=np.float64)
    start = len(window) - seq_len
    vol_mean = float(np.mean(vols[start:])) + 1e-9

    for idx in range(seq_len):
        i = start + idx
        a = max(float(atr_vals[i]), 1e-9)
        bar_rng = max(float(highs[i] - lows[i]), 1e-9)
        prev_c = float(closes[i - 1]) if i > 0 else float(opens[i])
        ret_atr = np.clip((closes[i] - prev_c) / a, -4.0, 4.0) / 4.0
        hl_atr = np.clip(bar_rng / a, 0.0, 4.0) / 4.0
        body_dir = np.clip((closes[i] - opens[i]) / a, -3.0, 3.0) / 3.0
        up_wick = (highs[i] - max(opens[i], closes[i])) / bar_rng
        dn_wick = (min(opens[i], closes[i]) - lows[i]) / bar_rng
        rel_vol = np.clip(vols[i] / vol_mean, 0.0, 4.0) / 4.0
        ema_dist = np.clip((closes[i] - e21[i]) / (3.0 * a), -1.0, 1.0)
        sp_atr = np.clip(spreads[i] / a, 0.0, 1.5) / 1.5

        seq[idx] = [ret_atr, hl_atr, body_dir, up_wick, dn_wick, rel_vol, ema_dist, sp_atr]
    return seq


def build_dl_training_dataset(
    bars: list[Bar],
    seq_len: int = 32,
    horizon: int = 8,
    point: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (X, y_dir, y_vol) strictly causally using Triple-Barrier style horizon labels.
    y_dir: 0=down, 1=flat, 2=up; y_vol: 1 if future range expands > 1.25x current ATR else 0.
    """
    min_needed = seq_len + 20 + horizon
    if len(bars) < min_needed:
        return (
            np.empty((0, seq_len, 8), dtype=np.float64),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.float64),
        )

    highs = np.array([b.high for b in bars], dtype=np.float64)
    lows = np.array([b.low for b in bars], dtype=np.float64)
    closes = np.array([b.close for b in bars], dtype=np.float64)
    atr_vals = atr(highs, lows, closes, 14)

    xs: list[np.ndarray] = []
    ys_dir: list[int] = []
    ys_vol: list[float] = []

    for end_idx in range(seq_len + 16, len(bars) - horizon):
        seq = build_sequence_tensor(bars[: end_idx + 1], seq_len=seq_len, point=point)
        if seq is None:
            continue
        cur_close = float(closes[end_idx])
        cur_atr = max(float(atr_vals[end_idx]), 1e-9)
        up_barrier = cur_close + 0.9 * cur_atr
        dn_barrier = cur_close - 0.9 * cur_atr

        label = 1  # flat default
        fut_hi = cur_close
        fut_lo = cur_close
        for j in range(end_idx + 1, end_idx + horizon + 1):
            fut_hi = max(fut_hi, float(highs[j]))
            fut_lo = min(fut_lo, float(lows[j]))
            hit_up = highs[j] >= up_barrier
            hit_dn = lows[j] <= dn_barrier
            if hit_up and not hit_dn:
                label = 2
                break
            if hit_dn and not hit_up:
                label = 0
                break
        vol_exp = 1.0 if (fut_hi - fut_lo) >= 1.35 * cur_atr else 0.0
        xs.append(seq)
        ys_dir.append(label)
        ys_vol.append(vol_exp)

    if not xs:
        return (
            np.empty((0, seq_len, 8), dtype=np.float64),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.float64),
        )
    return np.stack(xs, axis=0), np.array(ys_dir, dtype=np.int64), np.array(ys_vol, dtype=np.float64)


def extract_meta_features(
    signal: TradeSignal,
    snap: IndicatorSnapshot,
    spread_price: float,
    dl_pred: DLPrediction,
    timestamp_sec: int,
    htf: tuple[int, int] = (0, 0),
    server_offset_sec: int = 0,
    structure: dict[str, float] | None = None,
) -> np.ndarray:
    """Tabular feature vector: regime, strategy, microstructure, DL probabilities, then the
    M15/H1 direction (+1 with the trade, -1 against, 0 none) and the trading session."""
    reg_oh = [0.0] * 5
    reg_oh[REGIME_INDEX.get(signal.regime, 2)] = 1.0

    strat_oh = [0.0] * 4
    if signal.strategy_id in STRATEGY_INDEX:
        strat_oh[STRATEGY_INDEX[signal.strategy_id]] = 1.0
    elif signal.strategy_id not in LATE_STRATEGIES:
        strat_oh[0] = 1.0  # unknown ids keep the old behaviour

    is_buy = signal.side == Side.BUY
    side_sign = 1.0 if is_buy else -1.0
    wick_align = snap.lower_wick_ratio if is_buy else snap.upper_wick_ratio
    vol_ratio = snap.atr14 / max(snap.atr_baseline, 1e-9)
    spread_atr = spread_price / max(snap.atr14, 1e-9)

    hour_utc = ((timestamp_sec % 86400) / 3600.0) if timestamp_sec > 0 else 12.0
    angle = 2.0 * math.pi * hour_utc / 24.0

    dl_align = dl_pred.prob_up if is_buy else dl_pred.prob_down
    dl_against = dl_pred.prob_down if is_buy else dl_pred.prob_up

    vec = [
        *reg_oh,
        *strat_oh,
        side_sign,
        float(signal.confidence),
        min(snap.adx14 / 50.0, 1.5),
        (snap.rsi14 - 50.0) / 50.0,
        min(vol_ratio, 3.0),
        snap.bb_width_pct_rank,
        wick_align,
        min(snap.body_atr_ratio, 3.0),
        min(spread_atr, 1.5),
        math.sin(angle),
        math.cos(angle),
        dl_align,
        dl_against,
        dl_pred.prob_flat,
        dl_pred.vol_expansion,
        float(alignment(side_sign, htf[0])),
        float(alignment(side_sign, htf[1])),
    ]
    sess = session_of(timestamp_sec, server_offset_sec) if timestamp_sec > 0 else "london"
    vec += [
        1.0 if sess == "asia" else 0.0,
        1.0 if sess in ("london", "overlap") else 0.0,
        1.0 if sess in ("newyork", "overlap") else 0.0,
    ]
    vec += [1.0 if signal.strategy_id == "structure_shift" else 0.0, *aligned_context(structure, side_sign)]
    return np.array(vec, dtype=np.float64)


assert META_FEATURE_NAMES[-len(ALIGNED_NAMES) :] == list(ALIGNED_NAMES)
