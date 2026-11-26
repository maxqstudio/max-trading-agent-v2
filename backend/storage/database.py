from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import aiosqlite

from backend.core.models import (
    Bar,
    ChallengerCandidate,
    ChampionSnapshot,
    DecisionEvent,
    ReflectionReport,
    TradeRecord,
)

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA busy_timeout=5000;

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
);
CREATE INDEX IF NOT EXISTS idx_bars_sym_tf_ts ON bars(symbol, timeframe, timestamp DESC);

CREATE TABLE IF NOT EXISTS trades (
    trade_id TEXT PRIMARY KEY,
    ticket INTEGER NOT NULL,
    is_shadow INTEGER NOT NULL DEFAULT 0,
    version_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    strategy_id TEXT NOT NULL,
    regime TEXT NOT NULL,
    side TEXT NOT NULL,
    volume REAL NOT NULL,
    entry_price REAL NOT NULL,
    exit_price REAL NOT NULL,
    sl_price REAL NOT NULL,
    tp_price REAL NOT NULL,
    open_time INTEGER NOT NULL,
    close_time INTEGER NOT NULL,
    pnl_money REAL NOT NULL,
    pnl_r REAL NOT NULL,
    mae_r REAL NOT NULL,
    mfe_r REAL NOT NULL,
    spread_at_entry REAL NOT NULL,
    slippage_points REAL NOT NULL,
    commission REAL NOT NULL,
    ml_confidence REAL NOT NULL,
    dl_confidence REAL NOT NULL,
    bandit_weight REAL NOT NULL,
    exit_reason TEXT NOT NULL,
    rl_exit_trace TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_trades_close_time ON trades(close_time DESC);
CREATE INDEX IF NOT EXISTS idx_trades_shadow ON trades(is_shadow, close_time DESC);

CREATE TABLE IF NOT EXISTS decisions (
    id TEXT PRIMARY KEY,
    timestamp_ms INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    category TEXT NOT NULL,
    tag TEXT NOT NULL,
    strategy_id TEXT NOT NULL,
    side TEXT NOT NULL,
    message TEXT NOT NULL,
    latency_ms REAL NOT NULL,
    metadata_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_decisions_ts ON decisions(timestamp_ms DESC);

CREATE TABLE IF NOT EXISTS kv_state (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    updated_at_ms INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS snapshots (
    version_id TEXT PRIMARY KEY,
    created_at_ms INTEGER NOT NULL,
    status TEXT NOT NULL,
    payload_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS challengers (
    version_id TEXT PRIMARY KEY,
    created_at_ms INTEGER NOT NULL,
    status TEXT NOT NULL,
    payload_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reflections (
    report_id TEXT PRIMARY KEY,
    created_at_ms INTEGER NOT NULL,
    payload_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS equity_curve (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_ms INTEGER NOT NULL,
    champion_equity REAL NOT NULL,
    shadow_equity REAL NOT NULL,
    daily_dd_pct REAL NOT NULL,
    event_label TEXT DEFAULT ''
);
"""


class Database:
    """Async SQLite WAL repository for high-frequency M1-M5 telemetry & evolution audit."""

    def __init__(self, db_path: str | Path = "data/max_agent.db") -> None:
        self.db_path = Path(db_path)
        self._conn: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(str(self.db_path))
        self._conn.row_factory = aiosqlite.Row
        await self._conn.executescript(SCHEMA_SQL)
        await self._conn.commit()

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._conn

    async def save_bars(self, bars: list[Bar]) -> None:
        if not bars:
            return
        await self.conn.executemany(
            """
            INSERT OR REPLACE INTO bars
            (symbol, timeframe, timestamp, open, high, low, close, volume, spread_points)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    b.symbol,
                    b.timeframe,
                    b.timestamp,
                    b.open,
                    b.high,
                    b.low,
                    b.close,
                    b.volume,
                    b.spread_points,
                )
                for b in bars
            ],
        )
        await self.conn.commit()

    async def load_recent_bars(self, symbol: str, timeframe: str = "M1", limit: int = 500) -> list[Bar]:
        cur = await self.conn.execute(
            """
            SELECT symbol, timeframe, timestamp, open, high, low, close, volume, spread_points
            FROM bars
            WHERE symbol = ? AND timeframe = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (symbol, timeframe, limit),
        )
        rows = await cur.fetchall()
        return [Bar(**dict(r)) for r in reversed(rows)]

    async def save_trade(self, trade: TradeRecord) -> None:
        await self.conn.execute(
            """
            INSERT OR REPLACE INTO trades (
                trade_id, ticket, is_shadow, version_id, symbol, strategy_id, regime, side,
                volume, entry_price, exit_price, sl_price, tp_price, open_time, close_time,
                pnl_money, pnl_r, mae_r, mfe_r, spread_at_entry, slippage_points, commission,
                ml_confidence, dl_confidence, bandit_weight, exit_reason, rl_exit_trace
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trade.trade_id,
                trade.ticket,
                1 if trade.is_shadow else 0,
                trade.version_id,
                trade.symbol,
                trade.strategy_id,
                trade.regime.value,
                trade.side.value,
                trade.volume,
                trade.entry_price,
                trade.exit_price,
                trade.sl_price,
                trade.tp_price,
                trade.open_time,
                trade.close_time,
                trade.pnl_money,
                trade.pnl_r,
                trade.mae_r,
                trade.mfe_r,
                trade.spread_at_entry,
                trade.slippage_points,
                trade.commission,
                trade.ml_confidence,
                trade.dl_confidence,
                trade.bandit_weight,
                trade.exit_reason,
                trade.rl_exit_trace,
            ),
        )
        await self.conn.commit()

    async def list_trades(
        self,
        limit: int = 100,
        is_shadow: bool | None = None,
        symbol: str | None = None,
    ) -> list[TradeRecord]:
        clauses: list[str] = []
        params: list[Any] = []
        if is_shadow is not None:
            clauses.append("is_shadow = ?")
            params.append(1 if is_shadow else 0)
        if symbol:
            clauses.append("symbol = ?")
            params.append(symbol)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        cur = await self.conn.execute(
            f"SELECT * FROM trades {where} ORDER BY close_time DESC LIMIT ?",
            params,
        )
        rows = await cur.fetchall()
        results: list[TradeRecord] = []
        for row in rows:
            d = dict(row)
            d["is_shadow"] = bool(d["is_shadow"])
            results.append(TradeRecord.model_validate(d))
        return results

    async def save_decision(self, event: DecisionEvent) -> None:
        await self.conn.execute(
            """
            INSERT OR REPLACE INTO decisions
            (id, timestamp_ms, symbol, category, tag, strategy_id, side, message, latency_ms, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.id,
                event.timestamp_ms,
                event.symbol,
                event.category,
                event.tag,
                event.strategy_id,
                event.side,
                event.message,
                event.latency_ms,
                json.dumps(event.metadata),
            ),
        )
        await self.conn.commit()

    async def list_decisions(self, limit: int = 80, category: str | None = None) -> list[DecisionEvent]:
        if category:
            cur = await self.conn.execute(
                "SELECT * FROM decisions WHERE category = ? ORDER BY timestamp_ms DESC LIMIT ?",
                (category, limit),
            )
        else:
            cur = await self.conn.execute(
                "SELECT * FROM decisions ORDER BY timestamp_ms DESC LIMIT ?",
                (limit,),
            )
        rows = await cur.fetchall()
        out: list[DecisionEvent] = []
        for r in rows:
            d = dict(r)
            d["metadata"] = json.loads(d.pop("metadata_json") or "{}")
            out.append(DecisionEvent.model_validate(d))
        return out

    async def set_kv(self, key: str, value: Any, updated_at_ms: int) -> None:
        await self.conn.execute(
            "INSERT OR REPLACE INTO kv_state (key, value_json, updated_at_ms) VALUES (?, ?, ?)",
            (key, json.dumps(value), updated_at_ms),
        )
        await self.conn.commit()

    async def get_kv(self, key: str) -> Any | None:
        cur = await self.conn.execute("SELECT value_json FROM kv_state WHERE key = ?", (key,))
        row = await cur.fetchone()
        return json.loads(row["value_json"]) if row else None

    async def save_snapshot(self, snap: ChampionSnapshot) -> None:
        await self.conn.execute(
            "INSERT OR REPLACE INTO snapshots (version_id, created_at_ms, status, payload_json) VALUES (?, ?, ?, ?)",
            (snap.version_id, snap.created_at_ms, snap.status, snap.model_dump_json()),
        )
        await self.conn.commit()

    async def list_snapshots(self) -> list[ChampionSnapshot]:
        cur = await self.conn.execute("SELECT payload_json FROM snapshots ORDER BY created_at_ms DESC")
        rows = await cur.fetchall()
        return [ChampionSnapshot.model_validate_json(r["payload_json"]) for r in rows]

    async def save_challenger(self, cand: ChallengerCandidate) -> None:
        await self.conn.execute(
            "INSERT OR REPLACE INTO challengers (version_id, created_at_ms, status, payload_json) VALUES (?, ?, ?, ?)",
            (cand.version_id, cand.created_at_ms, cand.status, cand.model_dump_json()),
        )
        await self.conn.commit()

    async def list_challengers(self, limit: int = 20) -> list[ChallengerCandidate]:
        cur = await self.conn.execute(
            "SELECT payload_json FROM challengers ORDER BY created_at_ms DESC LIMIT ?",
            (limit,),
        )
        rows = await cur.fetchall()
        return [ChallengerCandidate.model_validate_json(r["payload_json"]) for r in rows]

    async def save_reflection(self, report: ReflectionReport) -> None:
        await self.conn.execute(
            "INSERT OR REPLACE INTO reflections (report_id, created_at_ms, payload_json) VALUES (?, ?, ?)",
            (report.report_id, report.created_at_ms, report.model_dump_json()),
        )
        await self.conn.commit()

    async def list_reflections(self, limit: int = 20) -> list[ReflectionReport]:
        cur = await self.conn.execute(
            "SELECT payload_json FROM reflections ORDER BY created_at_ms DESC LIMIT ?",
            (limit,),
        )
        rows = await cur.fetchall()
        return [ReflectionReport.model_validate_json(r["payload_json"]) for r in rows]

    async def record_equity_point(
        self,
        timestamp_ms: int,
        champion_equity: float,
        shadow_equity: float,
        daily_dd_pct: float,
        event_label: str = "",
    ) -> None:
        await self.conn.execute(
            """
            INSERT INTO equity_curve (timestamp_ms, champion_equity, shadow_equity, daily_dd_pct, event_label)
            VALUES (?, ?, ?, ?, ?)
            """,
            (timestamp_ms, champion_equity, shadow_equity, daily_dd_pct, event_label),
        )
        await self.conn.commit()

    async def list_equity_curve(self, limit: int = 200) -> list[dict[str, Any]]:
        cur = await self.conn.execute(
            """
            SELECT timestamp_ms, champion_equity, shadow_equity, daily_dd_pct, event_label
            FROM equity_curve ORDER BY id DESC LIMIT ?
            """,
            (limit,),
        )
        rows = await cur.fetchall()
        return [dict(r) for r in reversed(rows)]
