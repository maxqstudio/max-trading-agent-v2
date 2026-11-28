"""Detailed Performance Report API for Live, Paper, and Shadow Challenger books.

Provides institutional-grade KPIs, Monte Carlo stress metrics, execution/excursion forensics,
live-vs-OOS decay diagnostics, live ML confidence buckets, AI Supervisor veto impact, and
breakdowns by symbol, strategy, market regime, and exit reason, plus Markdown export.
"""

from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from typing import Any, Sequence

import numpy as np
from fastapi import APIRouter

from backend.core.models import TradeRecord
from backend.engine.orchestrator import (
    HITRUN_VERSION,
    PAPER_BOOKS,
    PAPER_LABELS,
    PAPER_VERSION,
    RAW_HITRUN_VERSION,
    RAW_VERSION,
    TradingOrchestrator,
)
from backend.evolution.monte_carlo import run_monte_carlo_stress

BOOK_LABELS: dict[str, str] = {
    "PAPER": "Paper · Normal (AI Supervised)",
    "HITRUN": "Paper · Hit & Run (AI Supervised)",
    "RAW": "Paper · No Veto (Normal)",
    "RAW_HITRUN": "Paper · No Veto (Hit & Run)",
    "SHADOW": "Challenger · AI Veto",
    "SHADOW_RAW": "Challenger · Raw",
    "LIVE": "MT5 Account (Demo / Live)",
    "ALL_PAPER": "All Paper Books Combined",
}


def _compute_stats(trades: Sequence[TradeRecord], start_equity: float, risk_pct: float) -> dict[str, Any]:
    ordered = sorted(trades, key=lambda t: t.close_time)
    n = len(ordered)
    if n == 0:
        return {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "breakeven_win_rate": None,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "expectancy_r": 0.0,
            "net_pnl": 0.0,
            "net_return_pct": 0.0,
            "gross_win": 0.0,
            "gross_loss": 0.0,
            "avg_win_money": 0.0,
            "avg_loss_money": 0.0,
            "avg_win_r": 0.0,
            "avg_loss_r": 0.0,
            "payoff_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "mc_p95_dd_pct": 0.0,
            "mc_ruin_prob": 0.0,
            "avg_mfe_r": 0.0,
            "avg_mae_r": 0.0,
            "mfe_capture_pct": 0.0,
            "total_commission": 0.0,
            "avg_spread_points": 0.0,
            "avg_slippage_points": 0.0,
            "avg_hold_min": 0.0,
            "losers_after_half_r": 0,
            "best_trade_r": 0.0,
            "worst_trade_r": 0.0,
            "max_win_streak": 0,
            "max_loss_streak": 0,
        }

    rs = np.asarray([t.pnl_r for t in ordered], dtype=np.float64)
    moneys = np.asarray([t.pnl_money for t in ordered], dtype=np.float64)
    win_mask = moneys > 0
    loss_mask = ~win_mask
    n_wins = int(np.sum(win_mask))
    n_losses = int(np.sum(loss_mask))

    gross_win = float(np.sum(moneys[win_mask])) if n_wins else 0.0
    gross_loss = float(-np.sum(moneys[loss_mask])) if n_losses else 0.0
    net_pnl = gross_win - gross_loss

    pf = (gross_win / gross_loss) if gross_loss > 1e-9 else (9.99 if gross_win > 0 else 0.0)
    avg_win_m = gross_win / n_wins if n_wins else 0.0
    avg_loss_m = gross_loss / n_losses if n_losses else 0.0
    avg_win_r = float(np.mean(rs[win_mask])) if n_wins else 0.0
    avg_loss_r = float(np.mean(rs[loss_mask])) if n_losses else 0.0
    payoff = (avg_win_m / avg_loss_m) if avg_loss_m > 1e-9 else (9.99 if avg_win_m > 0 else 0.0)
    be_wr = round(avg_loss_m / (avg_win_m + avg_loss_m), 4) if (avg_win_m > 0 and avg_loss_m > 0) else None

    eq = peak = max(start_equity, 1.0)
    max_dd = 0.0
    for m in moneys:
        eq += float(m)
        peak = max(peak, eq)
        dd = ((peak - eq) / peak) * 100.0 if peak > 0 else 0.0
        max_dd = max(max_dd, dd)

    expectancy_r = float(np.mean(rs))
    std_r = float(np.std(rs))
    sharpe = (expectancy_r / std_r) * math.sqrt(min(n, 252)) if std_r > 1e-9 else 0.0

    mc = run_monte_carlo_stress(
        [float(r) for r in rs],
        risk_per_trade_pct=risk_pct,
        iterations=500,
    )

    mfe_vals = [t.mfe_r for t in ordered]
    mae_vals = [t.mae_r for t in ordered]
    pos_mfe = [t for t in ordered if t.mfe_r > 0.05]
    mfe_cap = (
        float(np.mean([max(0.0, min(1.5, t.pnl_r / t.mfe_r)) for t in pos_mfe])) * 100.0
        if pos_mfe
        else 0.0
    )

    max_w = max_l = cur_w = cur_l = 0
    for is_w in win_mask:
        if is_w:
            cur_w += 1
            cur_l = 0
            max_w = max(max_w, cur_w)
        else:
            cur_l += 1
            cur_w = 0
            max_l = max(max_l, cur_l)

    hold_mins = [max(0.0, (t.close_time - t.open_time) / 60.0) for t in ordered if t.close_time >= t.open_time]

    return {
        "total_trades": n,
        "wins": n_wins,
        "losses": n_losses,
        "win_rate": round(n_wins / n, 4),
        "breakeven_win_rate": be_wr,
        "profit_factor": round(min(pf, 9.99), 4),
        "sharpe_ratio": round(sharpe, 4),
        "expectancy_r": round(expectancy_r, 4),
        "net_pnl": round(net_pnl, 2),
        "net_return_pct": round((net_pnl / max(start_equity, 1.0)) * 100.0, 2),
        "gross_win": round(gross_win, 2),
        "gross_loss": round(gross_loss, 2),
        "avg_win_money": round(avg_win_m, 2),
        "avg_loss_money": round(-avg_loss_m, 2),
        "avg_win_r": round(avg_win_r, 3),
        "avg_loss_r": round(avg_loss_r, 3),
        "payoff_ratio": round(min(payoff, 9.99), 2),
        "max_drawdown_pct": round(max_dd, 2),
        "mc_p95_dd_pct": round(mc.p95_max_dd_pct, 2),
        "mc_ruin_prob": round(mc.ruin_probability, 4),
        "avg_mfe_r": round(float(np.mean(mfe_vals)), 3),
        "avg_mae_r": round(float(np.mean(mae_vals)), 3),
        "mfe_capture_pct": round(mfe_cap, 1),
        "total_commission": round(sum(abs(t.commission) for t in ordered), 2),
        "avg_spread_points": round(float(np.mean([t.spread_at_entry for t in ordered])), 2),
        "avg_slippage_points": round(float(np.mean([t.slippage_points for t in ordered])), 2),
        "avg_hold_min": round(float(np.mean(hold_mins)), 1) if hold_mins else 0.0,
        "losers_after_half_r": sum(1 for t in ordered if t.pnl_money <= 0 and t.mfe_r >= 0.5),
        "best_trade_r": round(float(np.max(rs)), 3),
        "worst_trade_r": round(float(np.min(rs)), 3),
        "max_win_streak": max_w,
        "max_loss_streak": max_l,
    }


def _breakdown(
    trades: Sequence[TradeRecord],
    key_fn,
    start_equity: float,
    include_keys: Sequence[str] = (),
) -> list[dict[str, Any]]:
    groups: dict[str, list[TradeRecord]] = {k: [] for k in include_keys}
    for t in trades:
        k = str(key_fn(t) or "UNKNOWN")
        groups.setdefault(k, []).append(t)

    total_n = max(len(trades), 1)
    rows: list[dict[str, Any]] = []
    for k in sorted(groups.keys()):
        items = sorted(groups[k], key=lambda t: t.close_time)
        n = len(items)
        if n == 0:
            rows.append(
                {
                    "group": k,
                    "total_trades": 0,
                    "share_pct": 0.0,
                    "win_rate": None,
                    "profit_factor": None,
                    "expectancy_r": None,
                    "net_pnl": 0.0,
                    "max_drawdown_pct": None,
                }
            )
            continue
        wins = [t.pnl_money for t in items if t.pnl_money > 0]
        losses = [-t.pnl_money for t in items if t.pnl_money <= 0]
        gw, gl = sum(wins), sum(losses)
        pf = (gw / gl) if gl > 1e-9 else (9.99 if gw > 0 else 0.0)
        eq = peak = max(start_equity, 1.0)
        max_dd = 0.0
        for t in items:
            eq += t.pnl_money
            peak = max(peak, eq)
            max_dd = max(max_dd, ((peak - eq) / peak) * 100.0 if peak > 0 else 0.0)
        rows.append(
            {
                "group": k,
                "total_trades": n,
                "share_pct": round((n / total_n) * 100.0, 1),
                "win_rate": round(len(wins) / n, 4),
                "profit_factor": round(min(pf, 9.99), 4),
                "expectancy_r": round(sum(t.pnl_r for t in items) / n, 4),
                "net_pnl": round(gw - gl, 2),
                "max_drawdown_pct": round(max_dd, 2),
            }
        )
    rows.sort(key=lambda r: (r["total_trades"] == 0, -r["total_trades"], r["group"]))
    return rows


def _ml_live_buckets(trades: Sequence[TradeRecord]) -> list[dict[str, Any]]:
    buckets = [
        ("< 45%", 0.0, 0.45),
        ("45% – 55%", 0.45, 0.55),
        ("55% – 65%", 0.55, 0.65),
        ("≥ 65%", 0.65, 1.01),
    ]
    out: list[dict[str, Any]] = []
    for label, lo, hi in buckets:
        items = [t for t in trades if lo <= t.ml_confidence < hi]
        if not items:
            continue
        n = len(items)
        wins = sum(1 for t in items if t.pnl_money > 0)
        avg_pred = sum(t.ml_confidence for t in items) / n
        actual_wr = wins / n
        out.append(
            {
                "bucket": label,
                "trades": n,
                "avg_pred": round(avg_pred, 4),
                "actual_wr": round(actual_wr, 4),
                "gap": round(actual_wr - avg_pred, 4),
                "avg_r": round(sum(t.pnl_r for t in items) / n, 3),
            }
        )
    return out


def _build_markdown(report: dict[str, Any]) -> str:
    s = report["summary"]
    lines = [
        f"# Performance Report — {report['book_label']}",
        "",
        f"- Generated: {report['generated_utc']}",
        f"- Window: {report['days_label']}",
        f"- Total Trades: {s['total_trades']} ({s['wins']}W / {s['losses']}L)",
        f"- Win Rate: {s['win_rate'] * 100:.1f}% | Profit Factor: {s['profit_factor']:.2f} | Sharpe: {s['sharpe_ratio']:.2f}",
        f"- Net PnL: ${s['net_pnl']:+.2f} ({s['net_return_pct']:+.2f}%) | Expectancy: {s['expectancy_r']:+.3f}R",
        f"- Max DD: {s['max_drawdown_pct']:.2f}% | MC P95 DD: {s['mc_p95_dd_pct']:.2f}%",
        f"- Avg MFE / MAE: +{s['avg_mfe_r']:.2f}R / {s['avg_mae_r']:.2f}R | MFE Capture: {s['mfe_capture_pct']:.1f}%",
        f"- Commission: ${s['total_commission']:.2f} | Avg Spread: {s['avg_spread_points']:.1f}pt | Avg Hold: {s['avg_hold_min']:.1f}m",
        "",
    ]

    for title, key in (
        ("## Breakdown by Symbol", "by_symbol"),
        ("## Breakdown by Strategy", "by_strategy"),
        ("## Breakdown by Regime", "by_regime"),
        ("## Breakdown by Exit Reason", "by_exit"),
    ):
        lines += [
            title,
            "",
            "| Group | Trades | Win % | PF | Exp (R) | Net ($) | Max DD (%) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        rows = report.get(key, [])
        if not rows:
            lines.append("| — | 0 | — | — | — | 0.00 | — |")
        else:
            for r in rows:
                wr = f"{r['win_rate'] * 100:.1f}%" if r["win_rate"] is not None else "—"
                pf = f"{r['profit_factor']:.2f}" if r["profit_factor"] is not None else "—"
                ex = f"{r['expectancy_r']:+.3f}" if r["expectancy_r"] is not None else "—"
                dd = f"{r['max_drawdown_pct']:.2f}%" if r["max_drawdown_pct"] is not None else "—"
                lines.append(
                    f"| {r['group']} | {r['total_trades']} | {wr} | {pf} | {ex} | ${r['net_pnl']:+.2f} | {dd} |"
                )
        lines.append("")
    return "\n".join(lines)


async def _fetch_book_trades(orch: TradingOrchestrator, book: str) -> list[TradeRecord]:
    if book == "LIVE":
        return await orch.db.list_trades(limit=5000, is_shadow=False, exclude_version_prefix="PAPER")
    if book == "SHADOW":
        all_sh = await orch.db.list_trades(limit=5000, is_shadow=True)
        return [t for t in all_sh if not t.version_id.endswith("-raw")]
    if book == "SHADOW_RAW":
        all_sh = await orch.db.list_trades(limit=5000, is_shadow=True)
        return [t for t in all_sh if t.version_id.endswith("-raw")]
    if book == "ALL_PAPER":
        out: list[TradeRecord] = []
        for ver in PAPER_BOOKS.values():
            out.extend(await orch.db.list_trades(limit=5000, version_id=ver))
        return out
    ver_map = {
        "PAPER": PAPER_VERSION,
        "HITRUN": HITRUN_VERSION,
        "RAW": RAW_VERSION,
        "RAW_HITRUN": RAW_HITRUN_VERSION,
    }
    ver = ver_map.get(book, PAPER_VERSION)
    return await orch.db.list_trades(limit=5000, version_id=ver)


def register_report_routes(api: APIRouter, orch: TradingOrchestrator) -> None:
    @api.get("/report")
    async def get_performance_report(book: str = "PAPER", days: int = 0) -> dict[str, Any]:
        book_key = book.upper() if book.upper() in BOOK_LABELS else "PAPER"
        trades = await _fetch_book_trades(orch, book_key)
        cutoff = (int(time.time()) - days * 86400) if days > 0 else 0
        if cutoff > 0:
            trades = [t for t in trades if t.close_time >= cutoff]

        start_eq = (
            (orch.account.balance or orch.config.paper.start_equity)
            if book_key == "LIVE"
            else orch.config.paper.start_equity
        )
        risk_pct = orch.config.risk.risk_per_trade_pct
        summary = _compute_stats(trades, start_eq, risk_pct)

        # Compact leaderboard across all books for instant switching
        leaderboard: list[dict[str, Any]] = []
        for b_key, b_label in (
            ("PAPER", PAPER_LABELS["normal"]),
            ("HITRUN", PAPER_LABELS["hitrun"]),
            ("RAW", PAPER_LABELS["raw"]),
            ("RAW_HITRUN", PAPER_LABELS["raw_hitrun"]),
            ("SHADOW", "Challenger (AI Veto)"),
            ("SHADOW_RAW", "Challenger (Raw)"),
            ("LIVE", "MT5 Account"),
        ):
            bt = await _fetch_book_trades(orch, b_key)
            if cutoff > 0:
                bt = [t for t in bt if t.close_time >= cutoff]
            wins = [t.pnl_money for t in bt if t.pnl_money > 0]
            losses = [-t.pnl_money for t in bt if t.pnl_money <= 0]
            gw, gl = sum(wins), sum(losses)
            pf = (gw / gl) if gl > 1e-9 else (9.99 if gw > 0 else 0.0)
            leaderboard.append(
                {
                    "book": b_key,
                    "label": b_label,
                    "trades": len(bt),
                    "win_rate": round(len(wins) / len(bt), 4) if bt else 0.0,
                    "profit_factor": round(min(pf, 9.99), 2) if bt else 0.0,
                    "expectancy_r": round(sum(t.pnl_r for t in bt) / len(bt), 3) if bt else 0.0,
                    "net_pnl": round(gw - gl, 2),
                }
            )

        # Compare against latest Training Ground OOS benchmark if available
        oos_ref = None
        if orch.last_research_audit and isinstance(orch.last_research_audit.get("oos_backtest"), dict):
            ob = orch.last_research_audit["oos_backtest"]
            oos_ref = {
                "generated_utc": orch.last_research_audit.get("generated_utc", ""),
                "total_trades": ob.get("total_trades", 0),
                "win_rate": ob.get("win_rate", 0.0),
                "profit_factor": ob.get("profit_factor", 0.0),
                "expectancy_r": ob.get("expectancy_r", 0.0),
                "max_drawdown_pct": ob.get("max_drawdown_pct", 0.0),
            }

        report: dict[str, Any] = {
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "book": book_key,
            "book_label": BOOK_LABELS[book_key],
            "days": days,
            "days_label": f"Last {days} days" if days > 0 else "All Time",
            "summary": summary,
            "leaderboard": leaderboard,
            "by_symbol": _breakdown(trades, lambda t: t.symbol, start_eq, include_keys=orch.config.instruments.symbols),
            "by_strategy": _breakdown(trades, lambda t: t.strategy_id, start_eq),
            "by_regime": _breakdown(trades, lambda t: t.regime.value, start_eq),
            "by_exit": _breakdown(trades, lambda t: t.exit_reason, start_eq),
            "ml_buckets": _ml_live_buckets(trades),
            "veto_scoreboard": orch.veto_scoreboard(),
            "oos_benchmark": oos_ref,
        }
        report["markdown_report"] = _build_markdown(report)
        return report
