from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from backend.bridge.simulator import SimulatedMT5Bridge
from backend.core.models import MarketRegime
from backend.evolution.sequence_net import SequenceConvNet
from backend.features.dataset import build_dl_training_dataset, build_sequence_tensor
from backend.features.indicators import compute_indicator_snapshot
from backend.features.regime import classify_regime


def test_causal_indicators_and_sequence_have_zero_lookahead() -> None:
    sim = SimulatedMT5Bridge(symbols=["XAUUSD"], seed=101)
    bars = sim.generate_bootstrap_bars(count=120)["XAUUSD"]

    snap_before = compute_indicator_snapshot(bars[:90])
    tensor_before = build_sequence_tensor(bars[:90], seq_len=32, point=0.01)
    assert snap_before is not None
    assert tensor_before is not None
    assert tensor_before.shape == (32, 8)

    # Mutate future bar 95 dramatically — past snapshot & tensor at bar 89 MUST remain identical
    mutated = [b.model_copy(deep=True) for b in bars]
    mutated[95].close *= 1.5
    mutated[95].high *= 1.6

    snap_after = compute_indicator_snapshot(mutated[:90])
    tensor_after = build_sequence_tensor(mutated[:90], seq_len=32, point=0.01)
    assert snap_after == snap_before
    assert np.allclose(tensor_before, tensor_after)


def test_regime_classifier_returns_valid_regime() -> None:
    sim = SimulatedMT5Bridge(symbols=["XAUUSD"], seed=7)
    bars = sim.generate_bootstrap_bars(count=100)["XAUUSD"]
    assess = classify_regime(bars)
    assert assess is not None
    assert isinstance(assess.regime, MarketRegime)
    assert assess.volatility_ratio > 0


def test_sequence_convnet_trains_infers_fast_and_persists(tmp_path: Path) -> None:
    sim = SimulatedMT5Bridge(symbols=["XAUUSD"], seed=21)
    bars = sim.generate_bootstrap_bars(count=240)["XAUUSD"]
    x, y_dir, y_vol = build_dl_training_dataset(bars, seq_len=32, horizon=6, point=0.01)
    assert len(x) >= 120
    assert x.shape[1:] == (32, 8)

    net = SequenceConvNet(seed=42)
    summary = net.fit(x, y_dir, y_vol, epochs=10, batch_size=32, lr=0.01)
    assert net.trained is True
    assert summary["val_loss"] > 0

    t0 = time.perf_counter()
    pred = net.predict(x[-1])
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms < 25.0
    assert abs((pred.prob_down + pred.prob_flat + pred.prob_up) - 1.0) < 0.01
    assert 0.0 <= pred.vol_expansion <= 1.0

    weights_file = tmp_path / "dl_test.npz"
    net.save(weights_file)

    reloaded = SequenceConvNet(seed=999)
    assert reloaded.load(weights_file) is True
    pred2 = reloaded.predict(x[-1])
    assert pred2.prob_up == pred.prob_up
    assert pred2.prob_down == pred.prob_down
