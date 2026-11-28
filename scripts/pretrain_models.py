"""CLI wrapper for offline DL (1D-CNN) + ML (Calibrated Meta-Labeler) pre-training & Scientist OOS Audit."""

from __future__ import annotations

import argparse
import asyncio
import csv
import sqlite3
import sys
from pathlib import Path
from typing import Sequence

import aiosqlite

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.core.config import TradingConfig, load_trading_config
from backend.core.models import Bar
from backend.evolution.backtester import CausalBacktester
from backend.evolution.meta_labeler import CalibratedMetaLabeler
from backend.evolution.pretrain import (
    CSVImportResult,
    _format_group_table,
    diagnose_overfitting as _diagnose,
    normalize_symbol_from_filename,
    parse_mt5_csv,
    run_pretrain_audit,
)
from backend.evolution.sequence_net import SequenceConvNet
from backend.risk.manager import DEFAULT_SPECS
from backend.storage.database import Database

__all__ = [
    "CSVImportResult",
    "CausalBacktester",
    "_format_group_table",
    "normalize_symbol_from_filename",
    "parse_mt5_csv",
    "_diagnose",
    "main",
]


def _connect_uri(path: Path, *, read_only: bool) -> str:
    uri = path.resolve().as_uri()
    return f"{uri}?mode=ro" if read_only else uri


async def _open_database(path: Path, *, read_only: bool, allow_create_bars: bool = False) -> Database:
    if read_only and not path.is_file():
        raise FileNotFoundError(f"database not found: {path}")
    if not read_only:
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = await aiosqlite.connect(_connect_uri(path, read_only=read_only), uri=True)
    conn.row_factory = aiosqlite.Row
    db = Database(path)
    db._conn = conn
    if allow_create_bars:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bars (
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                spread_points REAL NOT NULL,
                PRIMARY KEY (symbol, timeframe, timestamp)
            )
            """
        )
        await conn.commit()
    return db


async def _read_server_offset(db: Database) -> int:
    try:
        value = await db.get_kv("server_offset_sec")
    except sqlite3.OperationalError:
        return 0
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return 0


async def _import_csv_directory(csv_dir: Path, db_path: Path) -> tuple[int, int, list[str], int]:
    if not csv_dir.is_dir():
        raise FileNotFoundError(f"CSV directory not found: {csv_dir}")
    files = sorted([*csv_dir.glob("*.csv"), *csv_dir.glob("*.tsv")])
    if not files:
        return 0, 0, [], 0

    db = await _open_database(db_path, read_only=False, allow_create_bars=True)
    try:
        server_offset = await _read_server_offset(db)
        imported = 0
        skipped = 0
        symbols: list[str] = []
        for file_path in files:
            try:
                parsed = parse_mt5_csv(file_path, server_offset_sec=server_offset)
            except (OSError, ValueError, csv.Error):
                skipped += 1
                continue
            if parsed.symbol not in symbols:
                symbols.append(parsed.symbol)
            skipped += parsed.skipped_rows
            imported += len(parsed.bars)
            await db.save_bars(parsed.bars)
    finally:
        await db.close()
    return imported, skipped, symbols, server_offset


async def _db_symbols(db: Database) -> list[str]:
    try:
        cur = await db.conn.execute("SELECT DISTINCT symbol FROM bars WHERE timeframe = 'M1' ORDER BY symbol")
        rows = await cur.fetchall()
    except sqlite3.OperationalError:
        return []
    return [str(row["symbol"]) for row in rows if row["symbol"]]


async def _load_bars(
    db_path: Path,
    config: TradingConfig,
    window_bars: int,
) -> tuple[dict[str, list[Bar]], int]:
    db = await _open_database(db_path, read_only=True)
    try:
        server_offset = await _read_server_offset(db)
        symbols = list(dict.fromkeys([*config.instruments.symbols, *(await _db_symbols(db))]))
        bars_by_symbol: dict[str, list[Bar]] = {}
        for symbol in symbols:
            try:
                bars = await db.load_recent_bars(symbol, timeframe="M1", limit=window_bars)
            except sqlite3.OperationalError:
                bars = []
            if bars:
                bars_by_symbol[symbol] = bars
        return bars_by_symbol, server_offset
    finally:
        await db.close()


def _load_champions(snapshots_dir: Path) -> tuple[SequenceConvNet | None, CalibratedMetaLabeler | None]:
    dl: SequenceConvNet | None = None
    ml: CalibratedMetaLabeler | None = None
    dl_path = snapshots_dir / "dl_champion.npz"
    if dl_path.is_file():
        candidate = SequenceConvNet()
        if candidate.load(dl_path) and candidate.trained:
            dl = candidate
    ml_path = snapshots_dir / "ml_champion.pkl"
    if ml_path.is_file():
        candidate_ml = CalibratedMetaLabeler()
        if candidate_ml.load(ml_path) and candidate_ml.trained:
            ml = candidate_ml
    return dl, ml


def _train(args: argparse.Namespace) -> str:
    config_path = Path(args.config)
    forbidden_local = REPO_ROOT / "config" / "local.yaml"
    if config_path.name.casefold() == "local.yaml" or config_path.resolve() == forbidden_local.resolve():
        raise ValueError("refusing to read config/local.yaml; pass config/default.yaml or another public config")
    config = load_trading_config(config_path, local_path=None)
    window_bars = int(args.window_bars or config.evolution.train_window_bars)
    if window_bars <= 0:
        raise ValueError("window-bars must be greater than zero")

    csv_summary = ""
    if args.csv_dir:
        imported, skipped, symbols, offset = asyncio.run(_import_csv_directory(Path(args.csv_dir), Path(args.db)))
        csv_summary = f"CSV import: {imported} bar(s), {skipped} invalid row/file(s); symbols: {', '.join(symbols) if symbols else 'none'}; server offset: {offset}s.\n\n"

    bars_by_symbol, server_offset_sec = asyncio.run(_load_bars(Path(args.db), config, window_bars))
    champion_dl, champion_ml = _load_champions(Path(args.snapshots_dir))
    res = run_pretrain_audit(
        bars_by_symbol=bars_by_symbol,
        config=config,
        specs=DEFAULT_SPECS,
        champion_dl=champion_dl,
        champion_ml=champion_ml,
        server_offset_sec=server_offset_sec,
        window_bars=window_bars,
        dry_run=bool(args.dry_run),
        force=bool(args.force),
        snapshots_dir=Path(args.snapshots_dir),
    )
    report = csv_summary + res.markdown_report
    if args.report_out:
        out_path = Path(args.report_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline DL (1D-CNN) + Calibrated ML Meta-Labeler pre-training & Scientist OOS Audit."
    )
    parser.add_argument("--db", default="data/max_agent.db", help="Path to SQLite database (default: data/max_agent.db)")
    parser.add_argument("--snapshots-dir", default="data/snapshots", help="Directory for champion snapshots")
    parser.add_argument("--config", default="config/default.yaml", help="Path to YAML config (default: config/default.yaml)")
    parser.add_argument("--csv-dir", default=None, help="Optional folder of MT5 History Center CSV/TSV files to import")
    parser.add_argument("--window-bars", type=int, default=0, help="M1 bars per symbol (0 = config train_window_bars)")
    parser.add_argument("--dry-run", action="store_true", help="Evaluate & print report without overwriting snapshots")
    parser.add_argument("--force", action="store_true", help="Save trained snapshots even if OOS gate does not adopt")
    parser.add_argument("--report-out", default=None, help="Optional file path to save the Markdown audit report")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    report = _train(args)
    sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
