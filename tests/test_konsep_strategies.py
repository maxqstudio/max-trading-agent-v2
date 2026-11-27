"""Day Range Reversal, Band Reversion, yesterday's range, Stochastic and the new ML columns."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from backend.core.models import Bar, DLPrediction, MarketRegime, Side
from backend.features.dataset import META_FEATURE_NAMES, extract_meta_features
from backend.features.daylevels import prev_day_range, prev_day_series, range_position
from backend.features.indicators import compute_indicator_snapshot, stochastic
from backend.features.structure import structure_context
from backend.strategies.band_reversion import BandReversionStrategy
from backend.strategies.day_range_reversal import DayRangeReversalStrategy
from backend.strategies.registry import StrategySuite

DAY0 = 1_795_000_000 - (1_795_000_000 % 86400)


def _bars(closes, start=DAY0, wick=0.15) -> list[Bar]:
    out, prev = [], float(closes[0])
    for i, c in enumerate(closes):
        c = float(c)
        out.append(Bar(symbol="XAUUSD", timestamp=start + i * 60, open=prev, high=max(prev, c) + wick, low=min(prev, c) - wick, close=c))
        prev = c
    return out


def yesterday_then_poke(mirror: bool = False) -> list[Bar]:
    """A full day ranging 2000..2010, then today: drifts down through 2000 and turns back up."""
    day1 = [2005 + 5 * math.sin(i / 25.0) for i in range(1440)]
    today = [2004.0] * 60 + [2004 - 0.2 * i for i in range(1, 41)] + [1996.0, 1996.1, 1995.9]
    today += [1996 + 0.45 * i for i in range(1, 16)]
    path = np.array(day1 + today)
    if mirror:
        path = 4010 - path
    return _bars(path)


def _scan(strat, bars, context_fn, start):
    hits = []
    for i in range(start, len(bars) + 1):
        win = bars[:i]
        sig = strat.evaluate("XAUUSD", win[-600:], compute_indicator_snapshot(win[-600:]), MarketRegime.RANGING, context=context_fn(win))
        if sig:
            hits.append((i - 1, sig))
    return hits


def test_prev_day_range_and_series():
    bars = yesterday_then_poke()
    hi, lo = prev_day_range(bars)
    assert 2009.9 < hi < 2010.3 and 1999.7 < lo < 2000.1
    series = prev_day_series(bars)
    assert series[1439] is None and series[1440] == (hi, lo) and series[-1] == (hi, lo)
    assert prev_day_range(bars[:1440]) is None  # no earlier day yet
    assert prev_day_range(bars[600:]) is None  # yesterday starts mid-day in this buffer
    assert range_position(lo, (hi, lo)) == -1.0 and range_position(hi, (hi, lo)) == 1.0 and range_position(1.0, None) == 0.0


def test_stochastic_bounds():
    c = np.linspace(100, 110, 60)
    k, d = stochastic(c + 0.5, c - 0.5, c)
    assert k[-1] > 80 and d[-1] > 80 and 0 <= k.min() and k.max() <= 100


def test_day_range_reversal_buy_and_sell():
    for mirror, side in ((False, Side.BUY), (True, Side.SELL)):
        bars = yesterday_then_poke(mirror)
        hits = _scan(DayRangeReversalStrategy({}), bars, lambda w: {"prev_day": prev_day_range(w)}, 1440 + 100)
        assert hits, f"no {side} signal"
        _, sig = hits[0]
        assert sig.side == side and sig.strategy_id == "day_range_reversal"
        if side == Side.BUY:
            assert sig.sl_price < 1996.0 - 0.1 < sig.entry_price < sig.tp_price
        else:
            assert sig.tp_price < sig.entry_price < 4010 - 1996.0 + 0.1 < sig.sl_price
        assert len(hits) <= 2


def test_day_range_reversal_needs_yesterday():
    bars = yesterday_then_poke()
    assert _scan(DayRangeReversalStrategy({}), bars, lambda w: {}, 1440 + 100) == []


def band_dip(mirror: bool = False) -> list[Bar]:
    rng = np.random.RandomState(3)
    base = list(2000 + np.cumsum(rng.normal(0, 0.05, 120)))
    px = base[-1]
    base += [px - 0.6 * i for i in range(1, 7)]  # sharp dip outside the lower band
    px = base[-1]
    base += [px + 0.15, px + 0.9, px + 1.5]  # turns back inside
    path = np.array(base)
    if mirror:
        path = 4000 - path
    return _bars(path)


def test_band_reversion_respects_the_h1_direction():
    strat = BandReversionStrategy({"max_sl_atr": 6.0})  # the fixture is quiet before the dip, so ATR is small
    bars = band_dip()
    hits = _scan(strat, bars, lambda w: {"htf_h1": 0}, 100)
    assert hits and hits[0][1].side == Side.BUY and hits[0][1].tp_price > hits[0][1].entry_price > hits[0][1].sl_price
    assert _scan(strat, bars, lambda w: {"htf_h1": -1}, 100) == []
    assert _scan(BandReversionStrategy({}), bars, lambda w: {"htf_h1": 0}, 100) == []  # stop too wide for 3 ATR
    assert _scan(BandReversionStrategy({"trend_filter": "with", "max_sl_atr": 6.0}), bars, lambda w: {"htf_h1": 0}, 100) == []
    sells = _scan(strat, band_dip(mirror=True), lambda w: {"htf_h1": 0}, 100)
    assert sells and sells[0][1].side == Side.SELL


def test_suite_passes_context_only_to_strategies_that_use_it():
    bars = yesterday_then_poke()
    win = bars
    suite = StrategySuite({})
    snap = compute_indicator_snapshot(win[-600:])
    suite.scan_all("XAUUSD", win[-600:], snap, MarketRegime.RANGING, context={"prev_day": prev_day_range(win)})
    suite.scan_all("XAUUSD", win[-600:], snap, MarketRegime.RANGING)  # no context: still fine


def test_new_feature_columns():
    bars = yesterday_then_poke()
    ctx = structure_context(bars[-600:], 0, prev_day_range(bars))
    snap = compute_indicator_snapshot(bars[-600:])
    sig = SimpleNamespace(regime=MarketRegime.RANGING, strategy_id="day_range_reversal", side=Side.BUY, confidence=0.6)
    v = extract_meta_features(sig, snap, 0.1, DLPrediction(), bars[-1].timestamp, structure=ctx)
    i = META_FEATURE_NAMES.index
    assert len(v) == len(META_FEATURE_NAMES)
    assert v[i("strat_day_range_reversal")] == 1.0 and v[i("strat_structure_shift")] == 0.0 and v[i("strat_band_reversion")] == 0.0
    assert -1.5 <= v[i("prev_day_pos_align")] <= 0.0  # near yesterday's low: a buy sits at the low end
    assert -1.0 <= v[i("stoch_align")] <= 1.0 and v[i("ema20_100_align")] in (-1.0, -0.5, 0.5, 1.0)
