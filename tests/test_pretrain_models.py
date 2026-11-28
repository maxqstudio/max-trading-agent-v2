from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import pretrain_models  # noqa: E402
from backend.core.models import BacktestMetrics, Bar, MarketRegime, Side, TradeRecord  # noqa: E402
from backend.storage.database import Database  # noqa: E402


def test_parse_mt5_tab_export_strips_broker_suffix_and_skips_bad_rows(tmp_path: Path) -> None:
    path = tmp_path / "XAUUSD.m_M1.tsv"
    path.write_text(
        "<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>\n"
        "2025.01.02\t03:04:00\t2650.0\t2651.0\t2649.0\t2650.5\t23\t0\t18\n"
        "invalid\trow\twith\tbroken\tprices\n",
        encoding="utf-8",
    )

    parsed = pretrain_models.parse_mt5_csv(path)

    assert parsed.symbol == "XAUUSD"
    assert len(parsed.bars) == 1
    assert parsed.bars[0].timeframe == "M1"
    assert parsed.bars[0].volume == 23
    assert parsed.bars[0].spread_points == 18
    assert parsed.skipped_rows == 1
    assert parsed.bars[0].timestamp == int(datetime(2025, 1, 2, 3, 4, tzinfo=timezone.utc).timestamp())


def test_parse_mt5_comma_export_and_combined_datetime(tmp_path: Path) -> None:
    path = tmp_path / "EURUSDm_M1.csv"
    path.write_text(
        "datetime,open,high,low,close,volume,spread\n"
        "2025-01-02 03:04:00,1.03,1.04,1.02,1.035,31,9\n"
        "2025-01-02 03:05:00,broken,1.04,1.02,1.035,31,9\n",
        encoding="utf-8",
    )

    parsed = pretrain_models.parse_mt5_csv(path)

    assert parsed.symbol == "EURUSD"
    assert len(parsed.bars) == 1
    assert parsed.bars[0].volume == 31
    assert parsed.bars[0].spread_points == 9
    assert parsed.skipped_rows == 1


def test_overfitting_diagnostics_flag_short_history_setup_count_loss_gap_and_calibration() -> None:
    start = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp())
    bars = [
        Bar(symbol="XAUUSD", timestamp=start, open=1, high=2, low=0.5, close=1.5),
        Bar(symbol="XAUUSD", timestamp=start + 2 * 86400, open=1.5, high=2, low=1, close=1.8),
    ]

    warnings = pretrain_models._diagnose(
        {"XAUUSD": bars},
        server_offset_sec=0,
        ml_oos_count=40,
        dl_summary={"train_loss": 0.7},
        dl_candidate_oos={"loss": 1.0},
        ml_train_brier=0.15,
        ml_oos_brier=0.23,
        calibration_error=0.10,
    )

    assert any("< 7 hari" in warning for warning in warnings)
    assert any("setup OOS < 100" in warning for warning in warnings)
    assert any("gap DL Train/OOS loss" in warning for warning in warnings)
    assert any("calibration error > 0.08" in warning for warning in warnings)


def test_oos_symbol_table_keeps_symbols_with_zero_trades() -> None:
    table = pretrain_models._format_group_table(
        [_trade(0, 1_735_689_600, 1)],
        group_key="symbol",
        risk_per_trade_pct=0.5,
        include_keys=["XAUUSD", "EURUSD", "GBPUSD", "USDJPY"],
    )

    assert any("| EURUSD | 0 | — | — | — | — |" in row for row in table)
    assert any("| GBPUSD | 0 | — | — | — | — |" in row for row in table)
    assert any("| USDJPY | 0 | — | — | — | — |" in row for row in table)


def _synthetic_bars(count: int = 520) -> list[Bar]:
    rng = np.random.default_rng(91)
    price = 2650.0
    bars: list[Bar] = []
    start = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp())
    for i in range(count):
        open_price = price
        close = open_price + float(rng.normal(0, 0.55))
        high = max(open_price, close) + float(rng.uniform(0.05, 0.6))
        low = min(open_price, close) - float(rng.uniform(0.05, 0.6))
        bars.append(
            Bar(
                symbol="XAUUSD",
                timeframe="M1",
                timestamp=start + i * 60,
                open=open_price,
                high=high,
                low=low,
                close=close,
                volume=float(rng.integers(20, 200)),
                spread_points=18,
            )
        )
        price = close
    return bars


def _seed_database(path: Path, bars: list[Bar]) -> None:
    async def save() -> None:
        db = Database(path)
        await db.connect()
        await db.save_bars(bars)
        await db.close()

    asyncio.run(save())


def _fake_backtest_run(self, bars_by_symbol, config, specs=None, dl_net=None, ml_labeler=None, collect_meta_samples=False):
    if collect_meta_samples:
        rng = np.random.default_rng(111 + len(next(iter(bars_by_symbol.values()), [])))
        x = rng.normal(size=(96, 24))
        y = (x[:, 20] + 0.25 * x[:, 17] > 0).astype(np.int64)
        trades = [
            _trade(i, int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp()) + i * 60, int(y[i]))
            for i in range(len(y))
        ]
        return BacktestMetrics(total_trades=len(trades)), trades, (x, y)

    trades = [_trade(i, 1_735_689_600 + i * 60, i % 2) for i in range(16)]
    return BacktestMetrics(total_trades=len(trades), win_rate=0.5, profit_factor=1.2), trades, (
        np.empty((0, 24)),
        np.empty((0,), dtype=np.int64),
    )


def _trade(index: int, timestamp: int, won: int) -> TradeRecord:
    pnl_r = 1.1 if won else -0.8
    return TradeRecord(
        symbol="XAUUSD",
        strategy_id="micro_breakout" if index % 2 else "liquidity_sweep",
        regime=MarketRegime.TRENDING_BULL,
        side=Side.BUY,
        volume=0.01,
        entry_price=2650.0,
        exit_price=2650.0 + pnl_r,
        sl_price=2649.0,
        tp_price=2652.0,
        open_time=timestamp,
        close_time=timestamp + 60,
        pnl_money=pnl_r * 10.0,
        pnl_r=pnl_r,
    )


def _write_test_config(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "instruments": {"symbols": ["XAUUSD"]},
                "ai": {"dl_epochs": 1, "dl_min_train_samples": 40},
                "evolution": {
                    "monte_carlo_iterations": 20,
                    "train_window_bars": 520,
                    "dl_min_oos_gain": 0.005,
                    "ml_min_oos_gain": 0.005,
                },
            }
        ),
        encoding="utf-8",
    )


def test_cli_dry_run_and_normal_training_write_only_expected_champions(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(pretrain_models.CausalBacktester, "run", _fake_backtest_run)
    bars = _synthetic_bars()
    config_path = tmp_path / "training.yaml"
    _write_test_config(config_path)

    dry_db = tmp_path / "dry.db"
    _seed_database(dry_db, bars)
    dry_snapshots = tmp_path / "dry_snapshots"
    assert pretrain_models.main(
        [
            "--db",
            str(dry_db),
            "--snapshots-dir",
            str(dry_snapshots),
            "--config",
            str(config_path),
            "--dry-run",
        ]
    ) == 0
    dry_report = capsys.readouterr().out
    assert not dry_snapshots.exists()
    assert "Scientist Overfitting & Calibration Audit Report" in dry_report
    assert "Data Coverage" in dry_report
    assert "Pillar 1 — 1D-CNN (DL)" in dry_report
    assert "Pillar 2 — Calibrated ML Meta-Labeler" in dry_report
    assert "Unseen OOS Backtest Audit" in dry_report
    assert "Diagnosis Risiko Overfitting" in dry_report

    live_db = tmp_path / "train.db"
    _seed_database(live_db, bars)
    live_snapshots = tmp_path / "snapshots"
    assert pretrain_models.main(
        [
            "--db",
            str(live_db),
            "--snapshots-dir",
            str(live_snapshots),
            "--config",
            str(config_path),
        ]
    ) == 0
    train_report = capsys.readouterr().out
    assert (live_snapshots / "dl_champion.npz").is_file()
    assert (live_snapshots / "ml_champion.pkl").is_file()
    for field in (
        "Train loss",
        "Val loss",
        "Val acc",
        "Unseen OOS",
        "Brier score",
        "Calibration Curve",
        "Permutation Feature Importances",
        "total_trades",
        "win_rate",
        "profit_factor",
        "sharpe_ratio",
        "expectancy_r",
        "max_drawdown_pct",
        "mc_p95_dd_pct",
        "per symbol",
        "per strategy",
    ):
        assert field in train_report


def test_research_api_import_csv_and_run_audit(tmp_path: Path, monkeypatch) -> None:
    from fastapi.testclient import TestClient

    from backend.api.routes import create_app
    from backend.core.config import EnvSettings, TradingConfig
    from backend.engine.orchestrator import TradingOrchestrator
    from backend.evolution.backtester import CausalBacktester

    monkeypatch.setattr(CausalBacktester, "run", _fake_backtest_run)

    cfg = TradingConfig()
    cfg.app.database_path = str(tmp_path / "research_api.db")
    cfg.app.snapshots_dir = str(tmp_path / "snaps")
    cfg.instruments.symbols = ["XAUUSD"]
    cfg.ai.dl_epochs = 1
    cfg.ai.dl_min_train_samples = 40
    cfg.evolution.monte_carlo_iterations = 20
    cfg.evolution.train_window_bars = 520

    token = "test_control_token_1234567890abcdef"
    env = EnvSettings(control_api_token=token, bridge_auth_token="test_bridge_token_1234567890abcdef")
    db = Database(cfg.app.database_path)
    asyncio.run(db.connect())
    orch = TradingOrchestrator(cfg, env, db)
    asyncio.run(orch.initialize_from_storage())
    app = create_app(orch)
    headers = {"X-Control-Token": token}

    try:
        client = TestClient(app)

        # 1. Import synthetic MT5 CSV directly via Research API
        bars = _synthetic_bars(420)
        csv_lines = ["<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>"]
        for b in bars:
            dt = datetime.fromtimestamp(b.timestamp, tz=timezone.utc)
            csv_lines.append(
                f"{dt.strftime('%Y.%m.%d')}\t{dt.strftime('%H:%M:%S')}\t"
                f"{b.open:.2f}\t{b.high:.2f}\t{b.low:.2f}\t{b.close:.2f}\t{int(b.volume)}\t0\t{int(b.spread_points)}"
            )
        imp = client.post(
            "/api/research/import-csv",
            headers=headers,
            json={"filename": "XAUUSD.m_M1.tsv", "content": "\n".join(csv_lines)},
        )
        assert imp.status_code == 200, imp.text
        assert imp.json()["symbol"] == "XAUUSD"
        assert imp.json()["imported"] == 420

        # 2. Check coverage via GET /api/research
        st = client.get("/api/research", headers=headers)
        assert st.status_code == 200
        body = st.json()
        assert body["last_audit"] is None
        xau_cov = next(r for r in body["coverage"] if r["symbol"] == "XAUUSD")
        assert xau_cov["bars"] == 420

        # 3. Run Training Ground audit via POST /api/research/run (dry_run=True)
        run_dry = client.post(
            "/api/research/run",
            headers=headers,
            json={"window_bars": 520, "dry_run": True, "force": False},
        )
        assert run_dry.status_code == 200, run_dry.text
        payload = run_dry.json()
        assert payload["status"] == "completed"
        assert payload["audit"]["dl"]["saved"] is False
        assert payload["audit"]["ml"]["saved"] is False
        assert not (tmp_path / "snaps" / "dl_champion.npz").exists()

        # 4. Run Training Ground gated save (dry_run=False) & verify persisted in GET /api/research
        run_save = client.post(
            "/api/research/run",
            headers=headers,
            json={"window_bars": 520, "dry_run": False, "force": False},
        )
        assert run_save.status_code == 200, run_save.text
        assert (tmp_path / "snaps" / "dl_champion.npz").is_file()
        assert (tmp_path / "snaps" / "ml_champion.pkl").is_file()

        st_after = client.get("/api/research", headers=headers).json()
        assert st_after["last_audit"] is not None
        assert st_after["dl_champion"]["trained"] is True
        assert st_after["ml_champion"]["trained"] is True

        # 5. Verify GET /api/report returns KPIs, breakdowns, leaderboard & OOS benchmark
        now_s = int(datetime.now(timezone.utc).timestamp())
        asyncio.run(
            db.save_trade(
                TradeRecord(
                    trade_id="rpt-1",
                    ticket=101,
                    is_shadow=False,
                    version_id="PAPER-RAW",
                    symbol="XAUUSD",
                    strategy_id="ema_vwap_pullback",
                    regime=MarketRegime.TRENDING_BULL,
                    side=Side.BUY,
                    volume=0.05,
                    entry_price=2650.0,
                    exit_price=2654.0,
                    sl_price=2648.0,
                    tp_price=2654.0,
                    open_time=now_s - 300,
                    close_time=now_s - 60,
                    pnl_money=20.0,
                    pnl_r=1.8,
                    mae_r=-0.2,
                    mfe_r=1.9,
                    ml_confidence=0.62,
                    dl_confidence=0.58,
                    exit_reason="TP",
                )
            )
        )
        rep = client.get("/api/report?book=RAW&days=7", headers=headers)
        assert rep.status_code == 200, rep.text
        rb = rep.json()
        assert rb["summary"]["total_trades"] == 1
        assert rb["summary"]["wins"] == 1
        assert rb["summary"]["net_pnl"] == 20.0
        assert rb["oos_benchmark"] is not None
        assert any(r["group"] == "ema_vwap_pullback" and r["total_trades"] == 1 for r in rb["by_strategy"])
        assert any(r["group"] == "TRENDING_BULL" and r["total_trades"] == 1 for r in rb["by_regime"])
        assert "# Performance Report" in rb["markdown_report"]
    finally:
        asyncio.run(db.close())
