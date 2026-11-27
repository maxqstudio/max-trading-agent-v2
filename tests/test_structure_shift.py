"""Structure Shift strategy, structure features, staged stops and the bandit arm."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from backend.core.models import Bar, DLPrediction, MarketRegime, Side
from backend.engine.exit_policy import stage_stop
from backend.evolution.bandit import ContextualBanditRL
from backend.features.dataset import META_FEATURE_NAMES, extract_meta_features
from backend.features.indicators import compute_indicator_snapshot
from backend.features.structure import (
    aligned_context,
    engulfing_at,
    kumo,
    sd_zones,
    structure_context,
    swings,
)
from backend.strategies.structure_shift import StructureShiftStrategy

T0 = 1_795_000_000


def _bars(path: list[tuple[float, float]], wick: float = 0.15) -> list[Bar]:
    return [
        Bar(symbol="XAUUSD", timestamp=T0 + i * 60, open=o, high=max(o, c) + wick, low=min(o, c) - wick, close=c)
        for i, (o, c) in enumerate(path)
    ]


def buy_setup(mirror: bool = False) -> tuple[list[Bar], int]:
    """A falling market, a bullish engulfing, a break of the last lower high, a pullback to about
    half of the leg and a candle that turns back up. Returns (bars, index of the entry bar)."""
    seq: list[tuple[float, float]] = []
    px = 2100.0

    def move(d: float) -> None:
        nonlocal px
        seq.append((px, px + d))
        px += d

    for _ in range(34):  # lower highs and lower lows
        for d in (-0.8, -0.8, -0.8, -0.8, 0.5, 0.5, 0.5):
            move(d)
    move(-0.6)
    move(-0.6)
    # engulfing: opens at the last close, closes above both bearish opens
    move(2.0)
    move(0.9)
    move(0.9)  # breaks the last swing high (2 bars after the engulfing)
    for d in (-0.7, -0.7, -0.6):  # pullback
        move(d)
    move(1.0)  # turns back up, closes above the previous high
    if mirror:
        seq = [(4200.0 - o, 4200.0 - c) for o, c in seq]
    return _bars(seq), len(seq) - 1


def _scan(bars: list[Bar], strat: StructureShiftStrategy):
    hits = []
    for i in range(200, len(bars) + 1):
        win = bars[:i]
        snap = compute_indicator_snapshot(win)
        sig = strat.evaluate("XAUUSD", win, snap, MarketRegime.RANGING)
        if sig:
            hits.append((i - 1, sig))
    return hits


def test_buy_setup_fires_once_on_the_retest_with_extension_targets():
    bars, entry_i = buy_setup()
    strat = StructureShiftStrategy({})
    hits = _scan(bars, strat)
    assert [i for i, _ in hits] == [entry_i]
    sig = hits[0][1]
    assert sig.side == Side.BUY and sig.strategy_id == "structure_shift"
    assert sig.sl_price < bars[-2].low < sig.entry_price < sig.tp_price
    # default final target is the 2.618 extension; the 1.618 one is a stage before it
    assert len(sig.tp_stages) == 1 and sig.entry_price < sig.tp_stages[0] < sig.tp_price
    assert 0.382 <= sig.features["retest_depth"] <= 0.786


def test_sell_setup_is_the_mirror():
    bars, entry_i = buy_setup(mirror=True)
    hits = _scan(bars, StructureShiftStrategy({}))
    assert [i for i, _ in hits] == [entry_i]
    sig = hits[0][1]
    assert sig.side == Side.SELL and sig.tp_price < sig.tp_stages[0] < sig.entry_price < sig.sl_price


def test_no_entry_without_a_retest_or_after_too_deep_a_pullback():
    bars, _ = buy_setup()
    strat = StructureShiftStrategy({"retest_min": 0.75})  # pullback in the fixture is shallower
    assert _scan(bars, strat) == []
    strat = StructureShiftStrategy({"invalid_level": 0.2})
    assert _scan(bars, strat) == []
    assert _scan(bars, StructureShiftStrategy({"enabled": False})) == []


def test_target_stage_and_max_stop():
    bars, _ = buy_setup()
    one = _scan(bars, StructureShiftStrategy({"target_stage": 1}))[0][1]
    three = _scan(bars, StructureShiftStrategy({"target_stage": 3}))[0][1]
    assert one.tp_stages == [] and len(three.tp_stages) == 2 and three.tp_price > one.tp_price
    assert _scan(bars, StructureShiftStrategy({"max_sl_atr": 0.1})) == []


def test_building_blocks():
    o = np.array([10.0, 9.6, 9.2, 9.0])
    c = np.array([9.6, 9.2, 9.0, 10.2])
    assert engulfing_at(o, c, 3) == 1 and engulfing_at(-o, -c, 3) == -1 and engulfing_at(o, c, 1) == 0
    h = np.array([1, 2, 3, 2, 1, 2, 5, 2, 1], dtype=float)
    hi, lo = swings(h, h - 0.5, 2)
    assert [s.index for s in hi] == [2, 6] and hi[0].confirmed_at == 4 and [s.index for s in lo] == [4]
    # Demand base: rally, three small candles, rally out
    path = [(100 + i, 101 + i) for i in range(5)] + [(105, 105.1), (105.1, 105.0), (105.0, 105.1)] + [(105.1, 106.5 + i * 1.4) for i in range(3)]
    bars = _bars(path)
    ar = np.full(len(bars), 1.0)
    zs = sd_zones(np.array([b.open for b in bars]), np.array([b.high for b in bars]), np.array([b.low for b in bars]),
                  np.array([b.close for b in bars]), ar)
    assert zs and zs[0].kind == "demand" and zs[0].pattern == "RBR"
    up = np.linspace(100, 160, 120)
    assert kumo(up + 0.5, up - 0.5, up)[0] == 1 and kumo(-up + 0.5, -up - 0.5, -up)[0] == -1
    assert aligned_context({"demand_near": 1.0, "kumo": -1.0, "bb_pos": 0.5, "engulf": 1.0}, -1.0) == [-1.0, 1.0, -0.5, -1.0]


def test_meta_features_and_old_models():
    from sklearn.linear_model import LogisticRegression

    from backend.evolution.meta_labeler import CalibratedMetaLabeler

    bars, _ = buy_setup()
    ctx = structure_context(bars)
    sig = SimpleNamespace(regime=MarketRegime.RANGING, strategy_id="structure_shift", side=Side.BUY, confidence=0.7)
    snap = compute_indicator_snapshot(bars)
    v = extract_meta_features(sig, snap, 0.1, DLPrediction(), bars[-1].timestamp, structure=ctx)
    assert len(v) == len(META_FEATURE_NAMES) == 34
    assert list(v[5:9]) == [0.0, 0.0, 0.0, 0.0] and v[29] == 1.0  # its own column at the end
    assert v[33] in (-1.0, 0.0, 1.0)
    old = CalibratedMetaLabeler()
    old.model = LogisticRegression().fit(np.random.RandomState(0).rand(40, 29), [0, 1] * 20)
    old.trained = True
    assert 0.0 <= old.predict_win_probability(v) <= 1.0


def test_stage_stop_steps_the_stop_up():
    stages = [2010.0, 2020.0]
    assert stage_stop(side=Side.BUY, entry=2000.0, stages=stages, peak_price=2005.0, sl=1995.0) is None
    assert stage_stop(side=Side.BUY, entry=2000.0, stages=stages, peak_price=2011.0, sl=1995.0, cost_px=0.3) == 2000.3
    assert stage_stop(side=Side.BUY, entry=2000.0, stages=stages, peak_price=2021.0, sl=2000.3) == 2010.0
    assert stage_stop(side=Side.BUY, entry=2000.0, stages=stages, peak_price=2021.0, sl=2015.0) is None  # never loosens
    assert stage_stop(side=Side.SELL, entry=2000.0, stages=[1990.0], peak_price=1989.0, sl=2005.0) == 2000.0


def test_bandit_adds_the_new_arm_to_saved_state():
    old = {"gamma": 0.985, "arms": {"XAUUSD": {"RANGING": {s: {"alpha": 3.0, "beta": 2.0, "n": 5.0, "avg_r": 0.2} for s in (
        "liquidity_sweep", "micro_breakout", "ema_vwap_pullback", "tick_imbalance")}}}}
    b = ContextualBanditRL()
    b.load_dict(old)
    w = b.get_weights("XAUUSD", MarketRegime.RANGING)
    assert "structure_shift" in w and w["structure_shift"] > 0
    b.update("XAUUSD", MarketRegime.RANGING, "structure_shift", 1.5)
    assert any(r["trades"]["structure_shift"] == 1 for r in b.get_matrix("XAUUSD"))


def test_paper_normal_book_steps_its_stop_at_a_stage(tmp_path):
    import asyncio

    from backend.core.models import TickSnapshot, utc_now_ms
    from tests.test_filters import _orch

    orch, db = _orch(tmp_path)

    async def run():
        await db.connect()
        try:
            await orch.initialize_from_storage()
            orch.ticks["XAUUSD"] = TickSnapshot(symbol="XAUUSD", bid=2650.0, ask=2650.1, spread_points=10)
            sig = SimpleNamespace(side=Side.BUY, strategy_id="structure_shift", atr=1.0, tp_stages=[2655.0, 2700.0])
            await orch._open_paper_position(
                "XAUUSD", sig, SimpleNamespace(volume=0.1, adjusted_sl=2646.0, adjusted_tp=2662.0),
                SimpleNamespace(regime=MarketRegime.RANGING), orch.specs["XAUUSD"], 0.6, 0.6, 10.0,
                books=("normal", "hitrun"), veto="NONE",
            )
            p = orch.paper_books["normal"].positions["XAUUSD"]
            assert p.tp_stages == [2655.0]  # 2700 is beyond the final TP
            assert orch.paper_books["hitrun"].positions["XAUUSD"].tp_stages == []
            orch.config.exits.be_at_r = 99.0  # isolate the stage ladder from the profit lock
            orch.config.exits.trail_start_r = 99.0
            await orch._update_paper_positions_on_tick(
                TickSnapshot(symbol="XAUUSD", bid=2655.5, ask=2655.6, spread_points=10), int(utc_now_ms() // 1000)
            )
            return p
        finally:
            await db.close()

    p = asyncio.run(run())
    assert p.sl >= p.entry_price
