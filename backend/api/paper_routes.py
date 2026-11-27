"""Side-by-side comparison of the paper books (Normal vs Hit & Run) and the real MT5 account."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from backend.core.models import TradeRecord
from backend.engine.orchestrator import PAPER_BOOKS, PAPER_LABELS, TradingOrchestrator


def summarize(trades: list[TradeRecord], start_equity: float) -> dict[str, Any]:
    """Win rate, profit factor, max drawdown and averages from closed trades (oldest first)."""
    trades = sorted(trades, key=lambda t: t.close_time)
    wins = [t.pnl_money for t in trades if t.pnl_money > 0]
    losses = [t.pnl_money for t in trades if t.pnl_money <= 0]
    gross_win, gross_loss = sum(wins), -sum(losses)
    eq = peak = start_equity
    max_dd = 0.0
    for t in trades:
        eq += t.pnl_money
        peak = max(peak, eq)
        max_dd = max(max_dd, (peak - eq) / peak * 100.0 if peak > 0 else 0.0)
    n = len(trades)
    return {
        "trades": n,
        "win_rate": round(len(wins) / n * 100.0, 1) if n else 0.0,
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else (None if not wins else 99.0),
        "net_pnl": round(gross_win - gross_loss, 2),
        "avg_win": round(gross_win / len(wins), 2) if wins else 0.0,
        "avg_loss": round(-gross_loss / len(losses), 2) if losses else 0.0,
        "avg_r": round(sum(t.pnl_r for t in trades) / n, 3) if n else 0.0,
        "max_dd_pct": round(max_dd, 2),
        "losers_after_half_r": sum(1 for t in trades if t.pnl_money <= 0 and t.mfe_r >= 0.5),
    }


def register_paper_routes(api: APIRouter, orch: TradingOrchestrator) -> None:
    @api.get("/paper/compare")
    async def compare() -> dict[str, Any]:
        start = orch.config.paper.start_equity
        ai_cost = orch.supervisor.status().get("spent_today_usd", 0.0)
        rows = []
        for name, version in PAPER_BOOKS.items():
            book = orch.paper_books[name]
            trades = await orch.db.list_trades(limit=5000, version_id=version)
            rows.append({
                "book": name,
                "label": PAPER_LABELS[name],
                "equity": book.equity,
                "open_positions": len(book.positions),
                **summarize(trades, start),
            })
        mt5 = await orch.db.list_trades(limit=5000, is_shadow=False, exclude_version_prefix="PAPER")
        rows.append({
            "book": "mt5",
            "label": "MT5 (akun)",
            "equity": orch.account.equity,
            "open_positions": len(orch.open_positions),
            **summarize(mt5, orch.account.balance or start),
        })
        return {
            "rows": rows,
            "ai_cost_today_usd": ai_cost,
            "note": "Normal dan Hit & Run memakai setup yang sama (lolos risk + supervisor). Tanpa Veto membuka semua setup yang lolos risk, termasuk yang ditolak Claude.",
            "learning": {
                "adaptive": orch.config.exits.adaptive,
                "min_samples": orch.exit_selector.min_samples,
                "groups": orch.exit_selector.table(),
            },
            "veto": orch.veto_scoreboard(),
            "settings": {"exits": orch.config.exits.model_dump(), "hit_run": orch.config.paper.hit_run.model_dump()},
        }
