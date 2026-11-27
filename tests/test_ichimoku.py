"""Ichimoku checklist, M15 series, ML columns and the Ichimoku entry filter."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from backend.core.models import Bar, DLPrediction, MarketRegime, Side
from backend.engine.filters import FilterLearner, decide
from backend.features import ichimoku
from backend.features.dataset import META_FEATURE_NAMES, extract_meta_features
from backend.features.indicators import compute_indicator_snapshot
from backend.features.structure import structure_context

T0 = 1_795_000_000 - (1_795_000_000 % 3600)


def _arrays(closes: np.ndarray, wick: float = 0.3):
    return closes + wick, closes - wick, closes


def _bars(closes: np.ndarray, wick: float = 0.3) -> list[Bar]:
    out, prev = [], float(closes[0])
    for i, c in enumerate(closes):
        c = float(c)
        out.append(Bar(symbol="XAUUSD", timestamp=T0 + i * 60, open=prev, high=max(prev, c) + wick, low=min(prev, c) - wick, close=c))
        prev = c
    return out


def test_uptrend_scores_plus_five_and_downtrend_minus_five():
    up = np.linspace(100, 200, 200)
    r = ichimoku.read(*_arrays(up), atr_now=1.0)
    assert (r.price_cloud, r.tk, r.chikou, r.price_kijun, r.future_cloud) == (1, 1, 1, 1, 1) and r.score == 5
    assert r.kijun_dist_atr > 0 and r.cloud_thick_atr > 0
    down = ichimoku.read(*_arrays(-up), atr_now=1.0)
    assert down.score == -5
    assert ichimoku.read(*_arrays(up[:80]), atr_now=1.0).score == 0  # not enough bars


def test_flat_market_scores_near_zero_and_cross_quality():
    flat = 100 + 0.2 * np.sin(np.arange(200) / 3.0)
    assert abs(ichimoku.read(*_arrays(flat), atr_now=1.0).score) <= 2
    # Falling, then a sharp turn up: Tenkan crosses above Kijun while price is still below the cloud: a weak buy cross
    path = np.concatenate([np.linspace(200, 150, 170), np.linspace(150, 156, 12)])
    r = ichimoku.read(*_arrays(path), atr_now=1.0)
    assert r.tk_cross in (0.25, 0.5) and r.price_cloud == -1


def test_m15_score_series_is_causal_and_matches_the_live_value():
    bars = _bars(np.linspace(2000, 2100, 30 * 60))
    series = ichimoku.htf_score_series(bars, 15)
    assert len(series) == len(bars) and series[-1] == 5 and series[100] == 0
    for i in (1700, 1745, 1799):
        assert series[i] == ichimoku.htf_score(bars[: i + 1], 15)


def test_features_signed_with_the_trade():
    bars = _bars(np.linspace(2000, 2060, 400))
    ctx = structure_context(bars, ichi_m15=4)
    snap = compute_indicator_snapshot(bars)
    names = META_FEATURE_NAMES
    buy = extract_meta_features(SimpleNamespace(regime=MarketRegime.TRENDING_BULL, strategy_id="structure_shift", side=Side.BUY, confidence=0.6),
                                snap, 0.1, DLPrediction(), bars[-1].timestamp, structure=ctx)
    sell = extract_meta_features(SimpleNamespace(regime=MarketRegime.TRENDING_BULL, strategy_id="micro_breakout", side=Side.SELL, confidence=0.6),
                                 snap, 0.1, DLPrediction(), bars[-1].timestamp, structure=ctx)
    i = names.index
    assert buy[i("ichi_score_align")] == 1.0 and sell[i("ichi_score_align")] == -1.0
    assert buy[i("ichi_m15_align")] == 0.8 and sell[i("ichi_m15_align")] == -0.8
    assert buy[i("kumo_align")] == 1.0 and buy[i("ichi_cloud_thick_atr")] == sell[i("ichi_cloud_thick_atr")]


def test_ichimoku_filter_off_on_auto():
    fl = FilterLearner(min_samples=10)
    base = dict(htf_mode="off", htf_timeframe="H1", session_mode="off", blocked_sessions=[], m15=0, h1=0, session="london")
    assert decide(fl, ichimoku_mode="auto", ichi=-4, **base).code == ""
    assert decide(fl, ichimoku_mode="on", ichi=-4, **base).code == "ICHIMOKU_AGAINST"
    assert decide(fl, ichimoku_mode="on", ichi=-2, **base).code == ""  # default threshold is 3
    assert decide(fl, ichimoku_mode="on", ichi=-2, ichi_threshold=2, **base).code == "ICHIMOKU_AGAINST"
    for k in range(12):
        fl.record(0, 0, "london", -0.9 if k % 4 else 0.4, ichi=-4)
        fl.record(0, 0, "london", 0.7 if k % 3 else -1.0, ichi=4)
    assert decide(fl, ichimoku_mode="auto", ichi=-4, **base).code == "ICHIMOKU_AGAINST"
    assert decide(fl, ichimoku_mode="off", ichi=-4, **base).code == ""
    rows = {r["bucket"]: r for r in fl.table("ichi:3", ("with", "mixed", "against"))}
    assert rows["against"]["trades"] == 12 and rows["against"]["clearly_worse"]
