from __future__ import annotations

from dataclasses import dataclass

from backend.core.models import Bar, MarketRegime
from backend.features.indicators import IndicatorSnapshot, compute_indicator_snapshot


@dataclass(slots=True)
class RegimeAssessment:
    regime: MarketRegime
    volatility_ratio: float
    trend_strength: float
    indicators: IndicatorSnapshot


def classify_regime(bars: list[Bar], snapshot: IndicatorSnapshot | None = None) -> RegimeAssessment | None:
    """Classify current market state strictly causally from closed bars."""
    snap = snapshot or compute_indicator_snapshot(bars)
    if snap is None:
        return None

    vol_ratio = snap.atr14 / max(snap.atr_baseline, 1e-9)
    trend_strength = snap.adx14

    if vol_ratio >= 2.2:
        regime = MarketRegime.HIGH_VOL_SPIKE
    elif vol_ratio <= 0.55 and snap.adx14 < 16.0 and snap.bb_width_pct_rank < 0.25:
        regime = MarketRegime.LOW_VOL_CHOP
    elif snap.adx14 >= 21.0 and snap.ema9 > snap.ema21 and snap.plus_di > snap.minus_di:
        regime = MarketRegime.TRENDING_BULL
    elif snap.adx14 >= 21.0 and snap.ema9 < snap.ema21 and snap.minus_di > snap.plus_di:
        regime = MarketRegime.TRENDING_BEAR
    else:
        regime = MarketRegime.RANGING

    return RegimeAssessment(
        regime=regime,
        volatility_ratio=round(vol_ratio, 4),
        trend_strength=round(trend_strength, 2),
        indicators=snap,
    )
