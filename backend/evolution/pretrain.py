"""Offline historical DL (1D-CNN) & ML (Calibrated Meta-Labeler) pre-training & Scientist OOS Audit.

Strictly trains and audits Pillar 1 (SequenceConvNet) and Pillar 2 (CalibratedMetaLabeler) on real
historical M1 bars using purged walk-forward splits and exponential time-decay weighting.
Never touches RL learners (Bandit, RL Exit, ExitStyleSelector, HitRunTuner, FilterLearner) or paper books.
"""

from __future__ import annotations

import copy
import csv
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import brier_score_loss

from backend.core.config import TradingConfig
from backend.core.models import BacktestMetrics, Bar, SymbolSpec, TradeRecord
from backend.evolution.backtester import CausalBacktester
from backend.evolution.meta_labeler import CalibratedMetaLabeler
from backend.evolution.monte_carlo import run_monte_carlo_stress
from backend.evolution.optimizer import (
    PurgedWalkForwardOptimizer,
    _multiscale_indices,
    _pick_dl,
    _pick_ml,
    compute_timestamp_recency_weights,
)
from backend.evolution.sequence_net import SequenceConvNet
from backend.features.dataset import build_dl_training_dataset
from backend.risk.manager import DEFAULT_SPECS


@dataclass(slots=True)
class CSVImportResult:
    symbol: str
    bars: list[Bar]
    skipped_rows: int


@dataclass(slots=True)
class DLData:
    x: np.ndarray
    y_dir: np.ndarray
    y_vol: np.ndarray
    timestamps: np.ndarray
    sample_weight: np.ndarray | None = None


@dataclass(slots=True)
class PretrainRunResult:
    audit: dict[str, Any]
    markdown_report: str
    selected_dl: SequenceConvNet | None
    selected_ml: CalibratedMetaLabeler | None
    new_dl: SequenceConvNet
    new_ml: CalibratedMetaLabeler
    dl_saved: bool
    ml_saved: bool


def normalize_symbol_from_filename(path: str | Path) -> str:
    """Read the logical symbol prefix and remove common broker suffixes."""
    stem = Path(path).stem
    token = re.split(r"[_\s-]", stem, maxsplit=1)[0].strip()
    token = re.sub(r"(?i)(?:\.pro|\.raw|\.ecn|\.m|m)$", "", token)
    token = token.upper()
    if not token or not re.fullmatch(r"[A-Z0-9]+", token):
        raise ValueError("filename does not begin with a recognizable symbol")
    return token


def _normalize_header(value: str | None) -> str:
    return (value or "").strip().strip("<>").strip().upper().replace(" ", "_")


def _detect_delimiter(sample: str) -> str:
    try:
        return csv.Sniffer().sniff(sample, delimiters="\t,").delimiter
    except csv.Error:
        first = sample.splitlines()[0] if sample.splitlines() else ""
        return "\t" if first.count("\t") > first.count(",") else ","


def _first_value(row: dict[str, str | None], *names: str) -> str | None:
    for name in names:
        value = row.get(name)
        if value is not None and value.strip() != "":
            return value.strip()
    return None


def _parse_timestamp(date_value: str, time_value: str | None, server_offset_sec: int) -> int:
    raw = date_value.strip()
    if time_value and "T" not in raw and " " not in raw:
        raw = f"{raw} {time_value.strip()}"
    try:
        numeric = float(raw)
        if math.isfinite(numeric):
            if abs(numeric) > 100_000_000_000:
                numeric /= 1000.0
            return int(numeric)
    except ValueError:
        pass

    candidate = raw.replace("Z", "+00:00")
    parsed: datetime | None = None
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        for fmt in (
            "%Y.%m.%d %H:%M:%S",
            "%Y.%m.%d %H:%M",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%m/%d/%Y %H:%M:%S",
            "%m/%d/%Y %H:%M",
            "%Y.%m.%d",
            "%Y-%m-%d",
        ):
            try:
                parsed = datetime.strptime(raw, fmt)
                break
            except ValueError:
                continue
    if parsed is None:
        raise ValueError("unrecognized timestamp")

    # MT5 exports use broker-server wall time. Store in the same epoch convention as MT5 bars;
    # explicit timezone timestamps are converted back to broker-server time.
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
        return int(parsed.timestamp())
    return int(parsed.astimezone(timezone.utc).timestamp()) + int(server_offset_sec)


def parse_mt5_csv_text(filename: str | Path, text: str, server_offset_sec: int = 0) -> CSVImportResult:
    """Parse MT5 History Center CSV/TSV text, dropping malformed rows safely."""
    symbol = normalize_symbol_from_filename(filename)
    cleaned = text.lstrip("\ufeff")
    delimiter = _detect_delimiter(cleaned[:8192])
    reader = csv.DictReader(cleaned.splitlines(), delimiter=delimiter)
    if not reader.fieldnames:
        raise ValueError("missing CSV header")
    rows = [{_normalize_header(k): v for k, v in row.items() if k is not None} for row in reader]
    bars: list[Bar] = []
    skipped = 0
    for row in rows:
        try:
            combined = _first_value(row, "DATETIME", "DATE_TIME", "TIMESTAMP", "TIME_UTC")
            date_value = combined or _first_value(row, "DATE")
            time_value = None if combined else _first_value(row, "TIME")
            if date_value is None:
                raise ValueError("missing date")
            timestamp = _parse_timestamp(date_value, time_value, server_offset_sec)
            open_value = _first_value(row, "OPEN", "O")
            high_value = _first_value(row, "HIGH", "H")
            low_value = _first_value(row, "LOW", "L")
            close_value = _first_value(row, "CLOSE", "C")
            if None in (open_value, high_value, low_value, close_value):
                raise ValueError("missing OHLC")
            open_price, high, low, close = (float(v) for v in (open_value, high_value, low_value, close_value))
            values = (open_price, high, low, close)
            if (
                not all(math.isfinite(v) and v > 0 for v in values)
                or high < max(open_price, close, low)
                or low > min(open_price, close, high)
            ):
                raise ValueError("invalid OHLC")
            volume_text = _first_value(row, "TICKVOL", "TICK_VOLUME", "VOLUME", "VOL")
            spread_text = _first_value(row, "SPREAD", "SPREAD_POINTS")
            volume = max(0.0, float(volume_text)) if volume_text is not None else 1.0
            spread = max(0.0, float(spread_text)) if spread_text is not None else 10.0
            if not math.isfinite(volume) or not math.isfinite(spread):
                raise ValueError("invalid volume or spread")
            bars.append(
                Bar(
                    symbol=symbol,
                    timeframe="M1",
                    timestamp=timestamp,
                    open=open_price,
                    high=high,
                    low=low,
                    close=close,
                    volume=volume,
                    spread_points=spread,
                )
            )
        except (TypeError, ValueError, OverflowError):
            skipped += 1
    bars.sort(key=lambda bar: bar.timestamp)
    return CSVImportResult(symbol=symbol, bars=bars, skipped_rows=skipped)


def parse_mt5_csv(path: str | Path, server_offset_sec: int = 0) -> CSVImportResult:
    """Parse one MT5 History Center CSV/TSV file, dropping malformed data rows."""
    source = Path(path)
    text = source.read_text(encoding="utf-8-sig", errors="replace")
    return parse_mt5_csv_text(source, text, server_offset_sec=server_offset_sec)


def _dl_dataset(
    bars_by_symbol: dict[str, list[Bar]],
    config: TradingConfig,
    specs: dict[str, SymbolSpec],
    *,
    weighted: bool,
) -> DLData | None:
    xs_all: list[np.ndarray] = []
    yd_all: list[np.ndarray] = []
    yv_all: list[np.ndarray] = []
    ts_all: list[np.ndarray] = []
    seq_len = config.ai.dl_seq_len
    for symbol, bars in bars_by_symbol.items():
        spec = specs.get(symbol, DEFAULT_SPECS.get(symbol, DEFAULT_SPECS["XAUUSD"]))
        xs, yd, yv = build_dl_training_dataset(bars, seq_len=seq_len, horizon=8, point=spec.point)
        if len(xs) == 0:
            continue
        start_bar = seq_len + 16
        timestamps = np.asarray([bar.timestamp for bar in bars[start_bar : start_bar + len(xs)]], dtype=np.int64)
        if weighted and len(xs) > 3600:
            keep = _multiscale_indices(len(xs), recent_keep=2400, max_total=3600)
            xs, yd, yv, timestamps = xs[keep], yd[keep], yv[keep], timestamps[keep]
        xs_all.append(xs)
        yd_all.append(yd)
        yv_all.append(yv)
        ts_all.append(timestamps)
    if not xs_all:
        return None

    x = np.concatenate(xs_all, axis=0)
    y_dir = np.concatenate(yd_all, axis=0)
    y_vol = np.concatenate(yv_all, axis=0)
    timestamps = np.concatenate(ts_all, axis=0)
    order = np.argsort(timestamps, kind="stable")
    x, y_dir, y_vol, timestamps = x[order], y_dir[order], y_vol[order], timestamps[order]
    sample_weight = None
    if weighted:
        sample_weight = compute_timestamp_recency_weights(
            timestamps,
            half_life_days=config.evolution.recency_half_life_days,
        )
    return DLData(x=x, y_dir=y_dir, y_vol=y_vol, timestamps=timestamps, sample_weight=sample_weight)


def _sort_ml_samples(
    trades: Sequence[TradeRecord], x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(x) == 0 or len(trades) != len(x):
        return x, y, np.empty((0,), dtype=np.int64)
    timestamps = np.asarray([trade.open_time for trade in trades], dtype=np.int64)
    order = np.argsort(timestamps, kind="stable")
    return x[order], y[order], timestamps[order]


def _brier_and_accuracy(
    model: CalibratedMetaLabeler | None, x: np.ndarray, y: np.ndarray
) -> tuple[float | None, float | None, np.ndarray]:
    if model is None or not model.trained or len(x) == 0:
        return None, None, np.empty((0,), dtype=np.float64)
    probs = model.predict_proba_batch(x)
    brier = float(brier_score_loss(y, probs)) if len(y) else None
    accuracy = float(np.mean((probs >= 0.5).astype(np.int64) == y)) if len(y) else None
    return brier, accuracy, probs


def _calibration_curve(
    probs: np.ndarray, labels: np.ndarray, bins: int = 5
) -> tuple[list[dict[str, float | int]], float | None]:
    if len(probs) == 0 or len(probs) != len(labels):
        return [], None
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[dict[str, float | int]] = []
    weighted_error = 0.0
    count = len(probs)
    for idx in range(bins):
        lower, upper = edges[idx], edges[idx + 1]
        mask = (probs >= lower) & ((probs <= upper) if idx == bins - 1 else (probs < upper))
        n = int(np.sum(mask))
        if n == 0:
            continue
        predicted = float(np.mean(probs[mask]))
        actual = float(np.mean(labels[mask]))
        gap = actual - predicted
        weighted_error += n / count * abs(gap)
        rows.append(
            {
                "lower": round(float(lower), 3),
                "upper": round(float(upper), 3),
                "predicted": round(predicted, 4),
                "actual": round(actual, 4),
                "gap": round(gap, 4),
                "count": n,
            }
        )
    return rows, round(float(weighted_error), 5)


def _trade_metrics(trades: Sequence[TradeRecord], risk_per_trade_pct: float = 0.5) -> BacktestMetrics:
    if not trades:
        return BacktestMetrics()
    returns = np.asarray([trade.pnl_r for trade in trades], dtype=np.float64)
    wins = returns[returns > 0]
    losses = -returns[returns <= 0]
    gross_loss = float(np.sum(losses))
    profit_factor = float(np.sum(wins) / gross_loss) if gross_loss > 1e-12 else (9.99 if len(wins) else 0.0)
    expectancy = float(np.mean(returns))
    std = float(np.std(returns))
    sharpe = expectancy / std * math.sqrt(min(len(returns), 252)) if std > 1e-12 else 0.0
    equity = np.cumprod(1.0 + np.clip(returns * (risk_per_trade_pct / 100.0), -0.5, 0.5))
    equity = np.concatenate(([1.0], equity))
    peaks = np.maximum.accumulate(equity)
    max_dd = float(np.max((peaks - equity) / np.maximum(peaks, 1e-12)) * 100.0)
    return BacktestMetrics(
        total_trades=len(trades),
        win_rate=round(float(np.mean(returns > 0)), 4),
        profit_factor=round(min(profit_factor, 9.99), 4),
        sharpe_ratio=round(sharpe, 4),
        expectancy_r=round(expectancy, 4),
        max_drawdown_pct=round(max_dd, 4),
        net_return_pct=round(float((equity[-1] - 1.0) * 100.0), 4),
    )


def _utc_iso(timestamp: int, server_offset_sec: int) -> str:
    return datetime.fromtimestamp(int(timestamp) - int(server_offset_sec), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _fmt(value: float | int | None, decimals: int = 4) -> str:
    if value is None or not math.isfinite(float(value)):
        return "—"
    return f"{float(value):.{decimals}f}"


def _group_breakdown(
    trades: Sequence[TradeRecord],
    group_key: str,
    risk_per_trade_pct: float,
    include_keys: Sequence[str] = (),
) -> list[dict[str, Any]]:
    grouped: dict[str, list[TradeRecord]] = {key: [] for key in include_keys}
    for trade in trades:
        key = str(getattr(trade, group_key))
        grouped.setdefault(key, []).append(trade)
    out: list[dict[str, Any]] = []
    for key in sorted(grouped):
        items = grouped[key]
        if not items:
            out.append(
                {
                    "group": key,
                    "total_trades": 0,
                    "win_rate": None,
                    "profit_factor": None,
                    "expectancy_r": None,
                    "max_drawdown_pct": None,
                }
            )
            continue
        m = _trade_metrics(items, risk_per_trade_pct)
        out.append(
            {
                "group": key,
                "total_trades": m.total_trades,
                "win_rate": m.win_rate,
                "profit_factor": m.profit_factor,
                "expectancy_r": m.expectancy_r,
                "max_drawdown_pct": m.max_drawdown_pct,
            }
        )
    return out


def _format_group_table(
    trades: Sequence[TradeRecord],
    group_key: str,
    risk_per_trade_pct: float,
    include_keys: Sequence[str] = (),
) -> list[str]:
    rows = _group_breakdown(trades, group_key, risk_per_trade_pct, include_keys=include_keys)
    lines = ["| Group | Trades | Win rate | Profit factor | Expectancy (R) | Max DD (%) |", "|---|---:|---:|---:|---:|---:|"]
    if not rows:
        lines.append("| — | 0 | — | — | — | — |")
        return lines
    for row in rows:
        lines.append(
            f"| {row['group']} | {row['total_trades']} | {_fmt(row['win_rate'])} | {_fmt(row['profit_factor'])} | "
            f"{_fmt(row['expectancy_r'])} | {_fmt(row['max_drawdown_pct'])} |"
        )
    return lines


def diagnose_overfitting(
    bars_by_symbol: dict[str, list[Bar]],
    server_offset_sec: int,
    ml_oos_count: int,
    dl_summary: dict[str, float],
    dl_candidate_oos: dict[str, float] | None,
    ml_train_brier: float | None,
    ml_oos_brier: float | None,
    calibration_error: float | None,
) -> list[str]:
    del server_offset_sec
    warnings: list[str] = []
    short_symbols = []
    for symbol, bars in bars_by_symbol.items():
        if len(bars) > 1 and (max(b.timestamp for b in bars) - min(b.timestamp for b in bars)) / 86400.0 < 7.0:
            short_symbols.append(symbol)
    if short_symbols:
        warnings.append("rentang data < 7 hari untuk: " + ", ".join(short_symbols))
    if ml_oos_count < 100:
        warnings.append(f"setup OOS < 100 ({ml_oos_count})")
    train_loss = dl_summary.get("train_loss")
    oos_loss = (dl_candidate_oos or {}).get("loss")
    if train_loss is not None and oos_loss is not None and train_loss > 0 and oos_loss - train_loss > max(0.15, train_loss * 0.35):
        warnings.append(f"gap DL Train/OOS loss melebar ({train_loss:.4f} vs {oos_loss:.4f})")
    if ml_train_brier is not None and ml_oos_brier is not None and ml_oos_brier - ml_train_brier > max(0.05, ml_train_brier * 0.50):
        warnings.append(f"gap ML Train/OOS Brier melebar ({ml_train_brier:.4f} vs {ml_oos_brier:.4f})")
    if calibration_error is not None and calibration_error > 0.08:
        warnings.append(f"rata-rata calibration error > 0.08 ({calibration_error:.4f})")
    return warnings


def format_audit_markdown(audit: dict[str, Any]) -> str:
    """Render structured audit dictionary into the Markdown Scientist OOS Audit Report."""
    coverage = audit.get("coverage", [])
    dl = audit.get("dl", {})
    ml = audit.get("ml", {})
    bt = audit.get("oos_backtest", {})
    warnings = audit.get("warnings", [])

    lines = [
        "# Scientist Overfitting & Calibration Audit Report",
        "",
        f"- Generated: {audit.get('generated_utc', '')}",
        f"- Data window limit: {int(audit.get('window_bars', 0)):,} M1 bars per symbol",
        f"- Server offset: {int(audit.get('server_offset_sec', 0)):+d} seconds (broker server minus UTC)",
        f"- Mode: {audit.get('mode_label', 'TRAIN / SAVE GATED')}",
        "",
        "## Data Coverage",
        "",
        "| Symbol | Bars | From (UTC) | To (UTC) | Duration (days) |",
        "|---|---:|---|---|---:|",
    ]
    if coverage:
        for row in coverage:
            lines.append(
                f"| {row['symbol']} | {row['bars']} | {row['from_utc']} | {row['to_utc']} | {row['duration_days']:.2f} |"
            )
    else:
        lines.append("| — | 0 | — | — | 0.00 |")

    cand_oos = dl.get("candidate_oos") or {}
    champ_oos = dl.get("champion_oos") or {}
    lines += [
        "",
        "## Pillar 1 — 1D-CNN (DL)",
        "",
        f"- Warm-start: {'yes' if dl.get('warm_start') else 'no'}",
        f"- Train loss: {_fmt(dl.get('train_loss'))}",
        f"- Val loss: {_fmt(dl.get('val_loss'))}",
        f"- Val acc: {_fmt(dl.get('val_acc'))}",
        f"- New model Unseen OOS loss / acc / n: {_fmt(cand_oos.get('loss'))} / {_fmt(cand_oos.get('acc'))} / {_fmt(cand_oos.get('n'), 0)}",
        f"- Champion Unseen OOS loss / acc / n: {_fmt(champ_oos.get('loss'))} / {_fmt(champ_oos.get('acc'))} / {_fmt(champ_oos.get('n'), 0)}",
        f"- Gate: **{dl.get('status', 'KEPT')}** — {dl.get('reason', 'no candidate decision')}",
        f"- Snapshot action: {dl.get('save_action', '')}",
        "",
        "## Pillar 2 — Calibrated ML Meta-Labeler",
        "",
        f"- Setup train / OOS: {ml.get('train_setups', 0)} / {ml.get('oos_setups', 0)}",
        f"- Train Brier (in-sample diagnostic) / new OOS Brier score: {_fmt(ml.get('train_brier'))} / {_fmt(ml.get('candidate_brier'))}",
        f"- Champion OOS Brier: {_fmt(ml.get('champion_brier'))}",
        f"- New OOS accuracy: {_fmt(ml.get('candidate_acc'))}",
        f"- Gate: **{ml.get('status', 'KEPT')}** — {ml.get('reason', 'no candidate decision')}",
        f"- Snapshot action: {ml.get('save_action', '')}",
        "",
        "### Calibration Curve (new candidate on OOS setups)",
        "",
        "| Pred P(win) | Actual WR | Actual − Pred | Setups |",
        "|---:|---:|---:|---:|",
    ]
    cal_rows = ml.get("calibration_curve", [])
    if cal_rows:
        for row in cal_rows:
            lines.append(
                f"| {_fmt(row['predicted'], 3)} | {_fmt(row['actual'], 3)} | {_fmt(row['gap'], 3)} | {row['count']} |"
            )
    else:
        lines.append("| — | — | — | 0 |")

    lines += [
        "",
        f"- Mean absolute calibration error (ECE): {_fmt(ml.get('ece'))}",
        "",
        "### Top 8 Permutation Feature Importances (OOS)",
        "",
        "| Rank | Feature | Importance |",
        "|---:|---|---:|",
    ]
    fis = ml.get("feature_importances", [])
    if fis:
        for idx, item in enumerate(fis[:8], start=1):
            lines.append(f"| {idx} | {item.get('name', '—')} | {_fmt(item.get('importance'))} |")
    else:
        lines.append("| — | No OOS estimate | — |")

    lines += [
        "",
        "## Unseen OOS Backtest Audit",
        "",
        f"- total_trades: {bt.get('total_trades', 0)}",
        f"- win_rate: {_fmt(bt.get('win_rate'))}",
        f"- profit_factor: {_fmt(bt.get('profit_factor'))}",
        f"- sharpe_ratio: {_fmt(bt.get('sharpe_ratio'))}",
        f"- expectancy_r: {_fmt(bt.get('expectancy_r'))}",
        f"- max_drawdown_pct: {_fmt(bt.get('max_drawdown_pct'))}",
        f"- mc_p95_dd_pct: {_fmt(bt.get('mc_p95_dd_pct'))}",
        "",
        "### OOS per symbol",
        "",
        "| Group | Trades | Win rate | Profit factor | Expectancy (R) | Max DD (%) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in bt.get("by_symbol", []) or []:
        lines.append(
            f"| {row['group']} | {row['total_trades']} | {_fmt(row['win_rate'])} | {_fmt(row['profit_factor'])} | "
            f"{_fmt(row['expectancy_r'])} | {_fmt(row['max_drawdown_pct'])} |"
        )
    lines += [
        "",
        "### OOS per strategy",
        "",
        "| Group | Trades | Win rate | Profit factor | Expectancy (R) | Max DD (%) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    strat_rows = bt.get("by_strategy", []) or []
    if strat_rows:
        for row in strat_rows:
            lines.append(
                f"| {row['group']} | {row['total_trades']} | {_fmt(row['win_rate'])} | {_fmt(row['profit_factor'])} | "
                f"{_fmt(row['expectancy_r'])} | {_fmt(row['max_drawdown_pct'])} |"
            )
    else:
        lines.append("| — | 0 | — | — | — | — |")

    lines += [
        "",
        "## Diagnosis Risiko Overfitting",
        "",
    ]
    if warnings:
        lines.extend(f"- WARNING: {warning}" for warning in warnings)
    else:
        lines.append("- Tidak ada peringatan otomatis dari ambang audit ini.")
    lines += [
        "",
        "> OOS split belum dipakai sebagai final test terpisah: split ini juga menjadi gate adopsi DL/ML. Angka backtest OOS adalah evaluasi kandidat setelah gate, jadi tetap perlu data live/shadow untuk validasi lanjutan.",
    ]
    return "\n".join(lines) + "\n"


def run_pretrain_audit(
    *,
    bars_by_symbol: dict[str, list[Bar]],
    config: TradingConfig,
    specs: dict[str, SymbolSpec] | None = None,
    champion_dl: SequenceConvNet | None = None,
    champion_ml: CalibratedMetaLabeler | None = None,
    server_offset_sec: int = 0,
    window_bars: int | None = None,
    dry_run: bool = False,
    force: bool = False,
    snapshots_dir: str | Path | None = None,
    seed: int = 42,
) -> PretrainRunResult:
    """Train DL (1D-CNN) + Calibrated ML Meta-Labeler with purged OOS gating and return structured audit."""
    sym_specs = specs or DEFAULT_SPECS
    eff_window = int(window_bars or config.evolution.train_window_bars)
    optimizer = PurgedWalkForwardOptimizer(seed=seed)
    optimizer.backtester.server_offset_sec = server_offset_sec
    train_bars, oos_bars = optimizer._split_bars_purged(bars_by_symbol, train_ratio=0.65, embargo_bars=12)

    warm_start = champion_dl is not None and champion_dl.trained
    new_dl = copy.deepcopy(champion_dl) if (warm_start and champion_dl is not None) else SequenceConvNet(seed=int(optimizer.rng.integers(1, 10000)))

    dl_train_summary: dict[str, float] = {"train_loss": 0.0, "val_loss": 0.0, "val_acc": 0.0}
    train_dl = _dl_dataset(train_bars, config, sym_specs, weighted=True)
    oos_dl = _dl_dataset(oos_bars, config, sym_specs, weighted=False)
    oos_dl_tuple = (oos_dl.x, oos_dl.y_dir, oos_dl.y_vol) if oos_dl is not None else None
    if train_dl is not None and len(train_dl.x) >= config.ai.dl_min_train_samples:
        lr = config.ai.dl_lr * (config.evolution.dl_finetune_lr_mult if warm_start else 1.0)
        dl_train_summary = new_dl.fit(
            train_dl.x,
            train_dl.y_dir,
            train_dl.y_vol,
            epochs=config.ai.dl_epochs,
            lr=lr,
            sample_weight=train_dl.sample_weight,
        )
    selected_dl, dl_decision = _pick_dl(
        champion_dl,
        new_dl,
        oos_dl_tuple,
        warm_start,
        min_oos_gain=config.evolution.dl_min_oos_gain,
    )
    dl_candidate_oos = new_dl.evaluate(*oos_dl_tuple) if oos_dl_tuple is not None else None
    dl_champion_oos = champion_dl.evaluate(*oos_dl_tuple) if champion_dl is not None and oos_dl_tuple is not None else None

    _, train_trades, (mx, my) = optimizer.backtester.run(
        train_bars,
        config,
        specs=sym_specs,
        dl_net=selected_dl,
        ml_labeler=None,
        collect_meta_samples=True,
    )
    mx, my, ml_train_timestamps = _sort_ml_samples(train_trades, mx, my)
    ml_sample_weight = (
        compute_timestamp_recency_weights(
            ml_train_timestamps,
            half_life_days=config.evolution.recency_half_life_days,
        )
        if len(ml_train_timestamps)
        else None
    )
    new_ml = CalibratedMetaLabeler()
    new_ml.fit_purged_walk_forward(mx, my, sample_weight=ml_sample_weight)
    ml_train_brier: float | None = None
    if new_ml.trained and len(mx):
        train_probs = new_ml.predict_proba_batch(mx)
        ml_train_brier = round(float(brier_score_loss(my, train_probs)), 5)

    _, oos_setup_trades, (ox, oy) = optimizer.backtester.run(
        oos_bars,
        config,
        specs=sym_specs,
        dl_net=selected_dl,
        ml_labeler=None,
        collect_meta_samples=True,
    )
    ox, oy, _ = _sort_ml_samples(oos_setup_trades, ox, oy)
    if new_ml.trained:
        new_ml.feature_importances = []
        if len(ox) >= 8:
            new_ml._compute_feature_importances(ox, oy)
    ml_candidate_brier, ml_candidate_acc, ml_probs = _brier_and_accuracy(new_ml, ox, oy)
    ml_champion_brier, _, _ = _brier_and_accuracy(champion_ml, ox, oy)
    calibration_rows, calibration_error = _calibration_curve(ml_probs, oy)
    selected_ml, ml_decision = _pick_ml(
        champion_ml,
        new_ml,
        ox,
        oy,
        min_oos_gain=config.evolution.ml_min_oos_gain,
    )

    metrics, oos_trades, _ = optimizer.backtester.run(
        oos_bars,
        config,
        specs=sym_specs,
        dl_net=selected_dl if selected_dl and selected_dl.trained else None,
        ml_labeler=selected_ml if selected_ml and selected_ml.trained else None,
    )
    mc = run_monte_carlo_stress(
        [trade.pnl_r for trade in oos_trades],
        risk_per_trade_pct=config.risk.risk_per_trade_pct,
        iterations=config.evolution.monte_carlo_iterations,
    )
    metrics.mc_p95_dd_pct = mc.p95_max_dd_pct
    metrics.mc_ruin_prob = mc.ruin_probability

    dl_save_action = "not saved (dry-run)" if dry_run else "not saved (gate kept champion)"
    ml_save_action = "not saved (dry-run)" if dry_run else "not saved (gate kept champion)"
    dl_saved = False
    ml_saved = False
    if not dry_run and snapshots_dir is not None:
        snap_path = Path(snapshots_dir)
        if new_dl.trained and (dl_decision.get("adopted") or force):
            new_dl.save(snap_path / "dl_champion.npz")
            dl_saved = True
            selected_dl = new_dl
            dl_save_action = "saved candidate" if dl_decision.get("adopted") else "forced save despite OOS gate"
        elif not new_dl.trained:
            dl_save_action = "not saved (candidate was not trained)"
        if new_ml.trained and (ml_decision.get("adopted") or force):
            new_ml.save(snap_path / "ml_champion.pkl")
            ml_saved = True
            selected_ml = new_ml
            ml_save_action = "saved candidate" if ml_decision.get("adopted") else "forced save despite OOS gate"
        elif not new_ml.trained:
            ml_save_action = "not saved (candidate was not trained)"

    warnings = diagnose_overfitting(
        bars_by_symbol,
        server_offset_sec,
        len(oy),
        dl_train_summary,
        dl_candidate_oos,
        ml_train_brier,
        ml_candidate_brier,
        calibration_error,
    )

    coverage_rows: list[dict[str, Any]] = []
    for symbol, bars in bars_by_symbol.items():
        if not bars:
            continue
        stamps = [b.timestamp for b in bars]
        start, end = min(stamps), max(stamps)
        dur = round(max(0.0, (end - start) / 86400.0), 2)
        coverage_rows.append(
            {
                "symbol": symbol,
                "bars": len(bars),
                "from_utc": _utc_iso(start, server_offset_sec),
                "to_utc": _utc_iso(end, server_offset_sec),
                "duration_days": dur,
                "pct_of_window": round(min(100.0, (len(bars) / max(eff_window, 1)) * 100.0), 1),
            }
        )

    mode_label = f"{'DRY RUN' if dry_run else 'TRAIN / SAVE GATED'}{' + FORCE' if force else ''}"
    audit: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "window_bars": eff_window,
        "server_offset_sec": int(server_offset_sec),
        "dry_run": bool(dry_run),
        "force": bool(force),
        "mode_label": mode_label,
        "coverage": coverage_rows,
        "dl": {
            "warm_start": warm_start,
            "train_loss": dl_train_summary.get("train_loss"),
            "val_loss": dl_train_summary.get("val_loss"),
            "val_acc": dl_train_summary.get("val_acc"),
            "candidate_oos": dl_candidate_oos,
            "champion_oos": dl_champion_oos,
            "adopted": bool(dl_decision.get("adopted")),
            "status": "ADOPTED" if dl_decision.get("adopted") else "KEPT",
            "reason": str(dl_decision.get("reason", "no candidate decision")),
            "save_action": dl_save_action,
            "saved": dl_saved,
        },
        "ml": {
            "train_setups": int(len(mx)),
            "oos_setups": int(len(oy)),
            "train_brier": ml_train_brier,
            "candidate_brier": round(ml_candidate_brier, 5) if ml_candidate_brier is not None else None,
            "champion_brier": round(ml_champion_brier, 5) if ml_champion_brier is not None else None,
            "candidate_acc": round(ml_candidate_acc, 4) if ml_candidate_acc is not None else None,
            "adopted": bool(ml_decision.get("adopted")),
            "status": "ADOPTED" if ml_decision.get("adopted") else "KEPT",
            "reason": str(ml_decision.get("reason", "no candidate decision")),
            "save_action": ml_save_action,
            "saved": ml_saved,
            "calibration_curve": calibration_rows,
            "ece": calibration_error,
            "feature_importances": (new_ml.feature_importances[:8] if new_ml.trained else []),
        },
        "oos_backtest": {
            "total_trades": metrics.total_trades,
            "win_rate": metrics.win_rate,
            "profit_factor": metrics.profit_factor,
            "sharpe_ratio": metrics.sharpe_ratio,
            "expectancy_r": metrics.expectancy_r,
            "max_drawdown_pct": metrics.max_drawdown_pct,
            "net_return_pct": metrics.net_return_pct,
            "mc_p95_dd_pct": metrics.mc_p95_dd_pct,
            "mc_ruin_prob": metrics.mc_ruin_prob,
            "by_symbol": _group_breakdown(oos_trades, "symbol", config.risk.risk_per_trade_pct, include_keys=list(bars_by_symbol)),
            "by_strategy": _group_breakdown(oos_trades, "strategy_id", config.risk.risk_per_trade_pct),
        },
        "warnings": warnings,
    }
    markdown_report = format_audit_markdown(audit)
    audit["markdown_report"] = markdown_report

    return PretrainRunResult(
        audit=audit,
        markdown_report=markdown_report,
        selected_dl=selected_dl,
        selected_ml=selected_ml,
        new_dl=new_dl,
        new_ml=new_ml,
        dl_saved=dl_saved,
        ml_saved=ml_saved,
    )
