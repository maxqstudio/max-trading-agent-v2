"""Retraining keeps what the models learned: warm start, longer window, champion kept unless beaten."""

from __future__ import annotations

import asyncio
import copy
from pathlib import Path

import numpy as np

from backend.bridge.simulator import SimulatedMT5Bridge
from backend.core.config import EnvSettings, load_trading_config
from backend.engine.orchestrator import TradingOrchestrator
from backend.evolution.meta_labeler import CalibratedMetaLabeler
from backend.evolution.optimizer import _pick_dl, _pick_ml
from backend.evolution.sequence_net import SequenceConvNet
from backend.features.dataset import build_dl_training_dataset
from backend.llm.router import LLMRouter
from backend.storage.database import Database


def _dl_data(seed: int, n: int = 500):
    sim = SimulatedMT5Bridge(symbols=["EURUSD"], seed=seed)
    bars = sim.generate_bootstrap_bars(count=n)["EURUSD"]
    return build_dl_training_dataset(bars, seq_len=32, horizon=8, point=0.00001)


def test_warm_start_continues_from_champion_weights():
    x, yd, yv = _dl_data(1)
    champ = SequenceConvNet(seed=3)
    champ.fit(x, yd, yv, epochs=4)
    warm = copy.deepcopy(champ)
    assert all(np.array_equal(warm.params[k], champ.params[k]) for k in champ.params)
    fresh = SequenceConvNet(seed=9)
    # A fresh net starts far from the champion; the warm copy starts exactly on it.
    assert any(not np.allclose(fresh.params[k], champ.params[k]) for k in champ.params)
    x2, yd2, yv2 = _dl_data(2)
    warm.fit(x2, yd2, yv2, epochs=2, lr=0.004)
    assert warm.trained and champ.evaluate(x2, yd2, yv2) is not None


def test_dl_champion_kept_when_new_model_is_worse():
    x, yd, yv = _dl_data(4)
    champ = SequenceConvNet(seed=3)
    champ.fit(x, yd, yv, epochs=6)
    bad = copy.deepcopy(champ)
    for k in bad.params:
        bad.params[k] = bad.params[k] + np.random.default_rng(0).normal(0, 2.0, bad.params[k].shape)
    picked, dec = _pick_dl(champ, bad, (x, yd, yv), warm=True)
    assert picked is champ and dec["adopted"] is False
    picked, dec = _pick_dl(bad, champ, (x, yd, yv), warm=True)
    assert picked is champ and dec["adopted"] is True
    # No champion yet: the first trained model is used.
    picked, dec = _pick_dl(SequenceConvNet(), champ, None, warm=False)
    assert picked is champ and dec["adopted"] is True


def test_ml_champion_kept_without_enough_evidence_or_when_better():
    rng = np.random.default_rng(7)
    x = rng.normal(0, 1, size=(200, 24))
    y = (x[:, 20] - 0.5 * x[:, 17] > 0).astype(int)
    good = CalibratedMetaLabeler()
    good.fit_purged_walk_forward(x[:150], y[:150])
    noise = CalibratedMetaLabeler()
    noise.fit_purged_walk_forward(x[:150], rng.integers(0, 2, 150))
    picked, dec = _pick_ml(good, noise, x[150:], y[150:])
    assert picked is good and dec["adopted"] is False
    picked, dec = _pick_ml(noise, good, x[150:], y[150:])
    assert picked is good and dec["adopted"] is True
    small = CalibratedMetaLabeler()
    small.fit_purged_walk_forward(x[:40], y[:40])
    picked, dec = _pick_ml(good, small, x[:3], y[:3])  # too few unseen setups: more training data wins
    assert picked is good and dec["adopted"] is False
    picked, dec = _pick_ml(small, good, x[:3], y[:3])
    assert picked is good and dec["adopted"] is True


def test_training_window_reads_days_of_bars_from_database(tmp_path: Path):
    cfg = load_trading_config(Path(__file__).resolve().parent.parent / "config" / "default.yaml", local_path=None)
    cfg.app.database_path = str(tmp_path / "t.db")
    cfg.app.snapshots_dir = str(tmp_path / "snaps")
    cfg.app.kill_switch_path = str(tmp_path / "kill.json")
    cfg.evolution.train_window_bars = 3000
    env = EnvSettings(bridge_auth_token="b" * 32, control_api_token="c" * 32, _env_file=None)

    async def go():
        db = Database(cfg.app.database_path)
        await db.connect()
        orch = TradingOrchestrator(cfg, env, db, llm_router=LLMRouter(resolved=[]))
        sym = cfg.instruments.symbols[0]
        bars = SimulatedMT5Bridge(symbols=[sym], seed=5).generate_bootstrap_bars(count=3500)[sym]
        await db.save_bars(bars)
        orch.bars[sym].extend(bars[-600:])
        out = await orch._training_bars()
        await db.close()
        return sym, out

    sym, out = asyncio.run(go())
    assert len(out[sym]) == 3000  # much more than the 600-bar live buffer
    assert out[sym][-1].timestamp == max(b.timestamp for b in out[sym])
