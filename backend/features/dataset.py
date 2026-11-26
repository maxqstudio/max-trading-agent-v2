from __future__ import annotations

import math

import numpy as np

from backend.core.models import Bar, DLPrediction, MarketRegime, Side, TradeSignal
from backend.features.indicators import IndicatorSnapshot, atr, ema

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
]


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
) -> np.ndarray:
    """Extract 24-dim tabular feature vector combining regime, strategy, microstructure & DL probabilities."""
    reg_oh = [0.0] * 5
    reg_oh[REGIME_INDEX.get(signal.regime, 2)] = 1.0

    strat_oh = [0.0] * 4
    strat_oh[STRATEGY_INDEX.get(signal.strategy_id, 0)] = 1.0

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
    ]
    return np.array(vec, dtype=np.float64)
