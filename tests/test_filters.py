"""M15/H1 direction, sessions, entry filters (off / on / auto), MT5 calendar and the ML features."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from backend.core.config import EnvSettings, load_trading_config
from backend.core.models import AccountState, Bar, BridgeSyncRequest, CalendarEvent, DLPrediction, MarketRegime, Side, TickSnapshot, utc_now_ms
from backend.engine.filters import FilterLearner, decide, htf_bucket
from backend.engine.orchestrator import TradingOrchestrator
from backend.features.dataset import META_FEATURE_NAMES, extract_meta_features
from backend.features.htf import alignment, htf_bias, htf_bias_series, session_of
from backend.llm.router import LLMRouter
from backend.storage.database import Database

T0 = 1_795_000_000 - (1_795_000_000 % 3600)  # an hour boundary


def trend_bars(n: int, step: float, wiggle: float = 0.3) -> list[Bar]:
    out, px = [], 2000.0
    for i in range(n):
        o = px
        px = px + step + (wiggle if i % 2 else -wiggle) * 0.5
        out.append(Bar(symbol="XAUUSD", timestamp=T0 + i * 60, open=o, high=max(o, px) + wiggle, low=min(o, px) - wiggle, close=px))
    return out


def test_htf_bias_up_down_and_not_enough_data():
    up = trend_bars(30 * 60, 0.05)
    down = trend_bars(30 * 60, -0.05)
    assert htf_bias(up, 60) == 1 and htf_bias(up, 15) == 1
    assert htf_bias(down, 60) == -1
    assert htf_bias(up[:600], 60) == 0  # only 10 H1 candles: no opinion yet
    flat = trend_bars(30 * 60, 0.0)
    assert htf_bias(flat, 60) == 0


def test_htf_series_is_causal():
    bars = trend_bars(30 * 60, 0.05)
    series = htf_bias_series(bars, 60)
    assert len(series) == len(bars)
    # Value at bar i only depends on bars up to i
    for i in (900, 1300, 1799):
        assert series[i] == htf_bias(bars[: i + 1], 60)


def test_sessions_and_alignment():
    day = T0 - (T0 % 86400)
    assert session_of(day + 3 * 3600) == "asia"
    assert session_of(day + 8 * 3600) == "london"
    assert session_of(day + 13 * 3600) == "overlap"
    assert session_of(day + 17 * 3600) == "newyork"
    assert session_of(day + 22 * 3600) == "late"
    # Bar times are broker server time: GMT+2 server, 10:00 on the server is 08:00 UTC
    assert session_of(day + 10 * 3600, server_offset_sec=7200) == "london"
    assert alignment(1.0, 1) == 1 and alignment(-1.0, 1) == -1 and alignment(-1.0, 0) == 0
    assert htf_bucket(1, -1, "M15") == "with" and htf_bucket(1, -1, "H1") == "against"
    assert htf_bucket(1, -1, "BOTH") == "against" and htf_bucket(1, 0, "BOTH") == "with"


def test_auto_mode_blocks_only_with_evidence():
    fl = FilterLearner(min_samples=10)
    args = dict(htf_timeframe="H1", session_mode="off", blocked_sessions=[], m15=0, h1=-1, session="london")
    assert decide(fl, htf_mode="auto", **args).code == ""
    assert decide(fl, htf_mode="on", **args).code == "HTF_AGAINST"
    assert decide(fl, htf_mode="off", **args).code == ""
    for i in range(12):
        fl.record(0, -1, "london", -0.9 if i % 4 else 0.5)  # against the H1 direction: mostly losers
        fl.record(0, 1, "london", 0.6 if i % 3 else -1.0)  # with it: mostly winners
    assert fl.clearly_worse("htf:H1", "against")
    assert decide(fl, htf_mode="auto", **args).code == "HTF_AGAINST"
    # Session auto: london is the only session, so it cannot be "worse than the rest"
    assert decide(fl, htf_mode="off", htf_timeframe="H1", session_mode="auto", blocked_sessions=[], m15=0, h1=0, session="london").code == ""
    assert decide(fl, htf_mode="off", htf_timeframe="H1", session_mode="on", blocked_sessions=["asia"], m15=0, h1=0, session="asia").code == "SESSION_FILTER"
    fl2 = FilterLearner(min_samples=10)
    fl2.load(fl.to_dict())
    assert fl2.clearly_worse("htf:H1", "against")


def test_meta_features_append_context_and_old_models_still_work():
    from backend.evolution.meta_labeler import CalibratedMetaLabeler
    from sklearn.linear_model import LogisticRegression

    sig = SimpleNamespace(regime=MarketRegime.RANGING, strategy_id="tick_imbalance", side=Side.SELL, confidence=0.7)
    snap = SimpleNamespace(lower_wick_ratio=0.1, upper_wick_ratio=0.2, atr14=1.0, atr_baseline=1.0, adx14=20, rsi14=50,
                           bb_width_pct_rank=0.5, body_atr_ratio=0.5)
    day = T0 - (T0 % 86400)
    v = extract_meta_features(sig, snap, 0.1, DLPrediction(), day + 13 * 3600, htf=(-1, 1))
    assert len(v) == len(META_FEATURE_NAMES) == 34
    assert list(v[24:29]) == [1.0, -1.0, 0.0, 1.0, 1.0]  # sell with M15 down, against H1 up, overlap session
    old = CalibratedMetaLabeler()
    old.model = LogisticRegression().fit(np.random.RandomState(0).rand(40, 24), [0, 1] * 20)
    old.trained = True
    assert 0.0 <= old.predict_win_probability(v) <= 1.0


def _orch(tmp_path: Path):
    cfg = load_trading_config(local_path=None)
    cfg.app.database_path = str(tmp_path / "f.db")
    cfg.app.snapshots_dir = str(tmp_path / "s")
    cfg.app.kill_switch_path = str(tmp_path / "k.json")
    cfg.shields.news_file = str(tmp_path / "none.json")
    env = EnvSettings(bridge_auth_token="b" * 32, control_api_token="c" * 32, _env_file=None)
    db = Database(cfg.app.database_path)
    return TradingOrchestrator(cfg, env, db, llm_router=LLMRouter(resolved=[])), db


def test_calendar_from_mt5_blocks_entries_around_news_and_offset_is_kept(tmp_path: Path):
    orch, db = _orch(tmp_path)
    now = int(utc_now_ms() // 1000)

    async def run():
        await db.connect()
        try:
            await orch.initialize_from_storage()
            req = BridgeSyncRequest(
                account=AccountState(balance=10000, equity=10000),
                server_utc_offset_sec=7200,
                calendar=[CalendarEvent(title="Non-Farm Payrolls", currency="USD", time_utc=now + 300)],
            )
            await orch.handle_bridge_sync(req)
            v1 = orch.shields.evaluate("XAUUSD", 10.0, datetime.now(timezone.utc))
            v2 = orch.shields.evaluate("EURJPY", 10.0, datetime.now(timezone.utc))
            view = orch.filters_view()
            offset_saved = await db.get_kv("server_offset_sec")
            return v1, v2, view, offset_saved
        finally:
            await db.close()

    v1, v2, view, offset_saved = asyncio.run(run())
    assert v1.reason_code == "NEWS_BLACKOUT" and "Non-Farm" in v1.detail
    assert v2.reason_code != "NEWS_BLACKOUT"
    assert offset_saved == 7200 and view["server_offset_sec"] == 7200
    assert view["news"]["from_mt5"] and view["news"]["upcoming"][0]["title"] == "USD Non-Farm Payrolls"


def test_no_veto_book_still_learns_when_a_filter_blocks(tmp_path: Path):
    orch, db = _orch(tmp_path)

    async def run():
        await db.connect()
        try:
            await orch.initialize_from_storage()
            orch.config.filters.htf_mode = "on"
            ctx = {"htf_m15": 0, "htf_h1": -1, "session": "london"}
            assert orch._entry_filter(ctx).code == "HTF_AGAINST"
            orch.ticks["XAUUSD"] = TickSnapshot(symbol="XAUUSD", bid=2650.0, ask=2650.1, spread_points=10)
            await orch._open_paper_position(
                "XAUUSD", SimpleNamespace(side=Side.BUY, strategy_id="s", atr=1.0),
                SimpleNamespace(volume=0.1, adjusted_sl=2645.0, adjusted_tp=2660.0),
                SimpleNamespace(regime=MarketRegime.RANGING), orch.specs["XAUUSD"], 0.6, 0.6, 10.0,
                books=("raw",), veto="NONE", context=ctx,
            )
            p = orch.paper_books["raw"].positions["XAUUSD"]
            assert (p.htf_h1, p.session) == (-1, "london")
            await orch._update_paper_positions_on_tick(
                TickSnapshot(symbol="XAUUSD", bid=2644.0, ask=2644.1, spread_points=10), int(utc_now_ms() // 1000)
            )
            return orch.filter_learner.table("htf:H1", ("with", "neutral", "against")), await db.get_kv("filter_stats")
        finally:
            await db.close()

    table, saved = asyncio.run(run())
    against = next(r for r in table if r["bucket"] == "against")
    assert against["trades"] == 1 and against["avg_r"] < 0 and saved
