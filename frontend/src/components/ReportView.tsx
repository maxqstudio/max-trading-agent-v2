import React, { useEffect, useState } from 'react';
import { BarChart3, Copy, FileCode2, RefreshCw } from 'lucide-react';
import { api } from '../api/client';
import { human } from '../labels';

interface Props {
  onToast: (msg: string) => void;
}

interface BreakdownRow {
  group: string;
  total_trades: number;
  share_pct: number;
  win_rate: number | null;
  profit_factor: number | null;
  expectancy_r: number | null;
  net_pnl: number;
  max_drawdown_pct: number | null;
}

interface LeaderboardRow {
  book: string;
  label: string;
  trades: number;
  win_rate: number;
  profit_factor: number;
  expectancy_r: number;
  net_pnl: number;
}

interface MLBucketRow {
  bucket: string;
  trades: number;
  avg_pred: number;
  actual_wr: number;
  gap: number;
  avg_r: number;
}

interface VetoRow {
  verdict: string;
  trades: number;
  win_rate: number;
  avg_r: number;
  net_pnl: number;
}

interface ReportPayload {
  generated_utc: string;
  book: string;
  book_label: string;
  days: number;
  days_label: string;
  summary: {
    total_trades: number;
    wins: number;
    losses: number;
    win_rate: number;
    breakeven_win_rate: number | null;
    profit_factor: number;
    sharpe_ratio: number;
    expectancy_r: number;
    net_pnl: number;
    net_return_pct: number;
    gross_win: number;
    gross_loss: number;
    avg_win_money: number;
    avg_loss_money: number;
    avg_win_r: number;
    avg_loss_r: number;
    payoff_ratio: number;
    max_drawdown_pct: number;
    mc_p95_dd_pct: number;
    mc_ruin_prob: number;
    avg_mfe_r: number;
    avg_mae_r: number;
    mfe_capture_pct: number;
    total_commission: number;
    avg_spread_points: number;
    avg_slippage_points: number;
    avg_hold_min: number;
    losers_after_half_r: number;
    best_trade_r: number;
    worst_trade_r: number;
    max_win_streak: number;
    max_loss_streak: number;
  };
  leaderboard: LeaderboardRow[];
  by_symbol: BreakdownRow[];
  by_strategy: BreakdownRow[];
  by_regime: BreakdownRow[];
  by_exit: BreakdownRow[];
  ml_buckets: MLBucketRow[];
  veto_scoreboard: VetoRow[];
  oos_benchmark: {
    generated_utc: string;
    total_trades: number;
    win_rate: number;
    profit_factor: number;
    expectancy_r: number;
    max_drawdown_pct: number;
  } | null;
  markdown_report: string;
}

const fmtPct = (v: number | null | undefined, d = 1) =>
  v == null || !Number.isFinite(v) ? '—' : `${(v * 100).toFixed(d)}%`;

const fmtNum = (v: number | null | undefined, d = 2) =>
  v == null || !Number.isFinite(v) ? '—' : v.toFixed(d);

const fmtSignedR = (v: number | null | undefined) =>
  v == null || !Number.isFinite(v) ? '—' : `${v >= 0 ? '+' : ''}${v.toFixed(3)}R`;

const fmtMoney = (v: number | null | undefined) =>
  v == null || !Number.isFinite(v) ? '—' : `${v >= 0 ? '+' : '-'}$${Math.abs(v).toFixed(2)}`;

const BreakdownTable: React.FC<{
  title: string;
  colLabel: string;
  rows: BreakdownRow[];
  formatGroup?: (g: string) => string;
}> = ({ title, colLabel, rows, formatGroup }) => (
  <div className="p-2.5 rounded cockpit-sub space-y-2">
    <div className="text-[11px] font-bold uppercase tracking-wider text-[#8A9BB5]">{title}</div>
    <div className="overflow-x-auto">
      <table className="w-full text-xs text-left border-collapse mono">
        <thead>
          <tr className="border-b border-[#1F2D45] text-[#8A9BB5] font-sans">
            <th className="py-1">{colLabel}</th>
            <th className="py-1 text-right">Trades</th>
            <th className="py-1 text-right">Win %</th>
            <th className="py-1 text-right">PF</th>
            <th className="py-1 text-right">Exp (R)</th>
            <th className="py-1 text-right">Net $</th>
            <th className="py-1 text-right">Max DD</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-[#1F2D45]/70">
          {rows.length === 0 ? (
            <tr>
              <td colSpan={7} className="py-3 text-center text-[#8A9BB5] font-sans">
                No closed trades in this slice yet.
              </td>
            </tr>
          ) : (
            rows.map((row) => (
              <tr key={row.group}>
                <td className="py-1.5 font-sans font-semibold text-white">
                  {formatGroup ? formatGroup(row.group) : row.group}
                </td>
                <td className="py-1.5 text-right">{row.total_trades}</td>
                <td className="py-1.5 text-right">{fmtPct(row.win_rate)}</td>
                <td
                  className={`py-1.5 text-right font-bold ${
                    row.profit_factor == null
                      ? 'text-[#8A9BB5]'
                      : row.profit_factor >= 1.0
                        ? 'text-[#10B981]'
                        : 'text-[#EF4444]'
                  }`}
                >
                  {fmtNum(row.profit_factor, 2)}
                </td>
                <td
                  className={`py-1.5 text-right ${
                    row.expectancy_r == null
                      ? 'text-[#8A9BB5]'
                      : row.expectancy_r >= 0
                        ? 'text-[#10B981]'
                        : 'text-[#EF4444]'
                  }`}
                >
                  {row.expectancy_r == null
                    ? '—'
                    : `${row.expectancy_r >= 0 ? '+' : ''}${row.expectancy_r.toFixed(3)}`}
                </td>
                <td
                  className={`py-1.5 text-right font-bold ${
                    row.total_trades === 0
                      ? 'text-[#8A9BB5]'
                      : row.net_pnl >= 0
                        ? 'text-[#10B981]'
                        : 'text-[#EF4444]'
                  }`}
                >
                  {row.total_trades === 0 ? '—' : fmtMoney(row.net_pnl)}
                </td>
                <td className="py-1.5 text-right text-[#8A9BB5]">
                  {row.max_drawdown_pct == null ? '—' : `${row.max_drawdown_pct.toFixed(2)}%`}
                </td>
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  </div>
);

export const ReportView: React.FC<Props> = ({ onToast }) => {
  const [book, setBook] = useState<string>('RAW');
  const [days, setDays] = useState<number>(0);
  const [report, setReport] = useState<ReportPayload | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [showMarkdown, setShowMarkdown] = useState<boolean>(false);

  const load = async (b = book, d = days) => {
    setLoading(true);
    try {
      const res = (await api.getReport(b, d)) as ReportPayload;
      setReport(res);
    } catch {
      // ignore transient errors
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load(book, days);
    const t = window.setInterval(() => load(book, days), 8000);
    return () => window.clearInterval(t);
  }, [book, days]);

  const copyMarkdown = async () => {
    if (!report?.markdown_report) return;
    try {
      await navigator.clipboard.writeText(report.markdown_report);
      onToast('Performance report copied to clipboard');
    } catch {
      onToast('Could not copy report');
    }
  };

  const s = report?.summary;

  return (
    <div className="space-y-3">
      {/* Header & Controls */}
      <div className="cockpit-card rounded-lg p-3.5 space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2.5">
          <div>
            <div className="flex items-center gap-2">
              <BarChart3 className="w-4 h-4 text-[#06B6D4]" />
              <h2 className="text-sm font-bold">Performance Report — Live, Paper &amp; Challenger Analytics</h2>
            </div>
            <p className="text-xs text-[#8A9BB5]">
              Detailed quant breakdown by book, symbol, strategy, market regime, exit reason, execution costs, and live-vs-OOS decay.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2 text-xs">
            <select
              value={book}
              onChange={(e) => setBook(e.target.value)}
              className="mono font-semibold px-2.5 py-1.5 rounded-md cockpit-sub border border-[#1F2D45] cursor-pointer focus:outline-none"
            >
              <option value="PAPER">Paper · Normal (AI Supervised)</option>
              <option value="HITRUN">Paper · Hit &amp; Run (AI Supervised)</option>
              <option value="RAW">Paper · No Veto (Normal)</option>
              <option value="RAW_HITRUN">Paper · No Veto (Hit &amp; Run)</option>
              <option value="SHADOW">Challenger · AI Veto</option>
              <option value="SHADOW_RAW">Challenger · Raw</option>
              <option value="LIVE">MT5 Account (Demo / Live)</option>
              <option value="ALL_PAPER">All Paper Books Combined</option>
            </select>

            <select
              value={days}
              onChange={(e) => setDays(Number(e.target.value))}
              className="mono font-semibold px-2.5 py-1.5 rounded-md cockpit-sub border border-[#1F2D45] cursor-pointer focus:outline-none"
            >
              <option value={0}>All Time</option>
              <option value={1}>Last 24 Hours</option>
              <option value={7}>Last 7 Days</option>
              <option value={14}>Last 14 Days</option>
              <option value={30}>Last 30 Days</option>
            </select>

            <button
              onClick={() => setShowMarkdown((v) => !v)}
              className="px-2.5 py-1.5 rounded-md cockpit-sub border border-[#1F2D45] font-semibold flex items-center gap-1 cursor-pointer hover:bg-[#162236]"
            >
              <FileCode2 className="w-3.5 h-3.5 text-[#06B6D4]" />
              <span>{showMarkdown ? 'Hide Raw' : 'Raw Markdown'}</span>
            </button>

            <button
              onClick={copyMarkdown}
              className="px-2.5 py-1.5 rounded-md cockpit-sub border border-[#1F2D45] font-semibold flex items-center gap-1 cursor-pointer hover:bg-[#162236]"
            >
              <Copy className="w-3.5 h-3.5 text-[#8A9BB5]" />
              <span>Copy</span>
            </button>

            <button
              onClick={() => load(book, days)}
              disabled={loading}
              className="px-2.5 py-1.5 rounded-md bg-[#3B82F6] hover:bg-[#2563EB] text-white font-bold flex items-center gap-1 cursor-pointer disabled:opacity-50"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
              <span>Refresh</span>
            </button>
          </div>
        </div>

        {/* All-Books Quick Leaderboard (click any book to drill down) */}
        {report?.leaderboard && (
          <div className="pt-2 border-t border-[#1F2D45] overflow-x-auto">
            <table className="w-full text-xs text-left border-collapse mono">
              <thead>
                <tr className="border-b border-[#1F2D45] text-[#8A9BB5] font-sans">
                  <th className="py-1 px-2">Book (click to inspect)</th>
                  <th className="py-1 px-2 text-right">Trades</th>
                  <th className="py-1 px-2 text-right">Win %</th>
                  <th className="py-1 px-2 text-right">PF</th>
                  <th className="py-1 px-2 text-right">Exp (R)</th>
                  <th className="py-1 px-2 text-right">Net P&amp;L</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#1F2D45]/70">
                {report.leaderboard.map((row) => {
                  const selected = row.book === book;
                  return (
                    <tr
                      key={row.book}
                      onClick={() => setBook(row.book)}
                      className={`cursor-pointer transition-colors ${
                        selected ? 'bg-[#3B82F6]/15' : 'hover:bg-[#121B2B]'
                      }`}
                    >
                      <td className="py-1.5 px-2 font-sans font-semibold text-white">
                        {row.label}
                        {selected && (
                          <span className="ml-2 px-1.5 py-0.5 rounded bg-[#3B82F6]/25 text-[#60A5FA] mono text-[10px] font-bold">
                            ACTIVE
                          </span>
                        )}
                      </td>
                      <td className="py-1.5 px-2 text-right">{row.trades}</td>
                      <td className="py-1.5 px-2 text-right">{row.trades ? fmtPct(row.win_rate) : '—'}</td>
                      <td
                        className={`py-1.5 px-2 text-right font-bold ${
                          !row.trades ? 'text-[#8A9BB5]' : row.profit_factor >= 1.0 ? 'text-[#10B981]' : 'text-[#EF4444]'
                        }`}
                      >
                        {row.trades ? row.profit_factor.toFixed(2) : '—'}
                      </td>
                      <td
                        className={`py-1.5 px-2 text-right ${
                          !row.trades ? 'text-[#8A9BB5]' : row.expectancy_r >= 0 ? 'text-[#10B981]' : 'text-[#EF4444]'
                        }`}
                      >
                        {row.trades ? fmtSignedR(row.expectancy_r) : '—'}
                      </td>
                      <td
                        className={`py-1.5 px-2 text-right font-bold ${
                          !row.trades ? 'text-[#8A9BB5]' : row.net_pnl >= 0 ? 'text-[#10B981]' : 'text-[#EF4444]'
                        }`}
                      >
                        {row.trades ? fmtMoney(row.net_pnl) : '—'}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {report && s && (
        <>
          {/* Executive KPI Strip */}
          <div className="cockpit-card rounded-lg p-3.5 space-y-3">
            <div className="flex items-center justify-between flex-wrap gap-2">
              <div className="flex items-center gap-2">
                <h3 className="text-xs font-bold uppercase tracking-wider">{report.book_label}</h3>
                <span className="mono text-[11px] px-2 py-0.5 rounded bg-[#070B12] text-[#8A9BB5]">
                  {report.days_label} · {report.generated_utc}
                </span>
              </div>
              <span
                className={`mono text-xs font-bold px-2 py-0.5 rounded ${
                  s.total_trades === 0
                    ? 'bg-[#1F2D45] text-[#8A9BB5]'
                    : s.profit_factor >= 1.15
                      ? 'bg-[#10B981]/15 text-[#10B981]'
                      : s.profit_factor >= 1.0
                        ? 'bg-[#F59E0B]/15 text-[#F59E0B]'
                        : 'bg-[#EF4444]/15 text-[#EF4444]'
                }`}
              >
                PF {s.profit_factor.toFixed(2)} · Exp {fmtSignedR(s.expectancy_r)} · {fmtMoney(s.net_pnl)}
              </span>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-2 text-xs">
              <div className="p-2.5 rounded cockpit-sub">
                <div className="text-[#8A9BB5]">Trades (W/L)</div>
                <div className="mono font-bold text-sm mt-0.5">
                  {s.total_trades}{' '}
                  <span className="text-[11px] text-[#8A9BB5]">
                    ({s.wins}/{s.losses})
                  </span>
                </div>
              </div>
              <div className="p-2.5 rounded cockpit-sub">
                <div className="text-[#8A9BB5]">Win Rate</div>
                <div className="mono font-bold text-sm mt-0.5">
                  {fmtPct(s.win_rate)}
                  {s.breakeven_win_rate != null && (
                    <span className="text-[10px] text-[#8A9BB5] block">BE {fmtPct(s.breakeven_win_rate)}</span>
                  )}
                </div>
              </div>
              <div className="p-2.5 rounded cockpit-sub">
                <div className="text-[#8A9BB5]">Profit Factor</div>
                <div className={`mono font-bold text-sm mt-0.5 ${s.profit_factor >= 1.0 ? 'text-[#10B981]' : 'text-[#EF4444]'}`}>
                  {s.profit_factor.toFixed(2)}
                </div>
              </div>
              <div className="p-2.5 rounded cockpit-sub">
                <div className="text-[#8A9BB5]">Net P&amp;L</div>
                <div className={`mono font-bold text-sm mt-0.5 ${s.net_pnl >= 0 ? 'text-[#10B981]' : 'text-[#EF4444]'}`}>
                  {fmtMoney(s.net_pnl)}
                  <span className="text-[10px] text-[#8A9BB5] block">
                    {s.net_return_pct >= 0 ? '+' : ''}{s.net_return_pct.toFixed(2)}%
                  </span>
                </div>
              </div>
              <div className="p-2.5 rounded cockpit-sub">
                <div className="text-[#8A9BB5]">Expectancy</div>
                <div className={`mono font-bold text-sm mt-0.5 ${s.expectancy_r >= 0 ? 'text-[#10B981]' : 'text-[#EF4444]'}`}>
                  {fmtSignedR(s.expectancy_r)}
                </div>
              </div>
              <div className="p-2.5 rounded cockpit-sub">
                <div className="text-[#8A9BB5]">Sharpe</div>
                <div className={`mono font-bold text-sm mt-0.5 ${s.sharpe_ratio >= 0 ? 'text-[#10B981]' : 'text-[#EF4444]'}`}>
                  {s.sharpe_ratio.toFixed(2)}
                </div>
              </div>
              <div className="p-2.5 rounded cockpit-sub">
                <div className="text-[#8A9BB5]">Max DD</div>
                <div className="mono font-bold text-sm mt-0.5 text-[#F59E0B]">{s.max_drawdown_pct.toFixed(2)}%</div>
              </div>
              <div className="p-2.5 rounded cockpit-sub">
                <div className="text-[#8A9BB5]">MC P95 DD</div>
                <div className="mono font-bold text-sm mt-0.5 text-[#EF4444]">{s.mc_p95_dd_pct.toFixed(2)}%</div>
              </div>
            </div>
          </div>

          {/* 2-Column Forensics & AI Impact */}
          <div className="grid grid-cols-1 xl:grid-cols-12 gap-3">
            {/* Execution & Excursion Forensics */}
            <div className="xl:col-span-6 cockpit-card rounded-lg p-3.5 space-y-3">
              <h3 className="text-xs font-bold uppercase tracking-wider">Execution, Excursion &amp; Risk Profile</h3>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 text-xs">
                <div className="p-2.5 rounded cockpit-sub">
                  <div className="text-[#8A9BB5]">Avg Win</div>
                  <div className="mono font-bold text-sm text-[#10B981] mt-0.5">{fmtMoney(s.avg_win_money)}</div>
                  <div className="mono text-[11px] text-[#8A9BB5]">+{s.avg_win_r.toFixed(2)}R</div>
                </div>
                <div className="p-2.5 rounded cockpit-sub">
                  <div className="text-[#8A9BB5]">Avg Loss</div>
                  <div className="mono font-bold text-sm text-[#EF4444] mt-0.5">{fmtMoney(s.avg_loss_money)}</div>
                  <div className="mono text-[11px] text-[#8A9BB5]">{s.avg_loss_r.toFixed(2)}R · Payoff {s.payoff_ratio.toFixed(2)}×</div>
                </div>
                <div className="p-2.5 rounded cockpit-sub">
                  <div className="text-[#8A9BB5]">MFE / MAE (Avg)</div>
                  <div className="mono font-bold text-sm mt-0.5">
                    <span className="text-[#10B981]">+{s.avg_mfe_r.toFixed(2)}R</span> /{' '}
                    <span className="text-[#EF4444]">{s.avg_mae_r.toFixed(2)}R</span>
                  </div>
                  <div className="mono text-[11px] text-[#06B6D4]">Capture {s.mfe_capture_pct.toFixed(1)}%</div>
                </div>
                <div className="p-2.5 rounded cockpit-sub">
                  <div className="text-[#8A9BB5]">Commission / Spread</div>
                  <div className="mono font-bold text-sm mt-0.5">${s.total_commission.toFixed(2)}</div>
                  <div className="mono text-[11px] text-[#8A9BB5]">Sp {s.avg_spread_points.toFixed(1)}pt · Slip {s.avg_slippage_points.toFixed(1)}pt</div>
                </div>
                <div className="p-2.5 rounded cockpit-sub">
                  <div className="text-[#8A9BB5]">Best / Worst Trade</div>
                  <div className="mono font-bold text-sm mt-0.5">
                    <span className="text-[#10B981]">+{s.best_trade_r.toFixed(2)}R</span> /{' '}
                    <span className="text-[#EF4444]">{s.worst_trade_r.toFixed(2)}R</span>
                  </div>
                  <div className="mono text-[11px] text-[#8A9BB5]">Avg hold {s.avg_hold_min.toFixed(1)}m</div>
                </div>
                <div className="p-2.5 rounded cockpit-sub">
                  <div className="text-[#8A9BB5]">Streaks &amp; Givebacks</div>
                  <div className="mono font-bold text-sm mt-0.5">
                    <span className="text-[#10B981]">{s.max_win_streak}W</span> /{' '}
                    <span className="text-[#EF4444]">{s.max_loss_streak}L</span>
                  </div>
                  <div className="mono text-[11px] text-[#F59E0B]">Losers ≥+0.5R: {s.losers_after_half_r}</div>
                </div>
              </div>

              {/* Live vs OOS Scientist Decay Check */}
              {report.oos_benchmark && (
                <div className="p-2.5 rounded cockpit-sub space-y-1.5">
                  <div className="flex items-center justify-between text-[11px]">
                    <span className="font-bold uppercase tracking-wider text-[#8A9BB5]">Live Book vs Training Ground OOS Benchmark</span>
                    <span className="mono text-[#8A9BB5]">{report.oos_benchmark.generated_utc}</span>
                  </div>
                  <div className="grid grid-cols-3 gap-2 text-xs mono">
                    <div>
                      <span className="text-[#8A9BB5] font-sans block text-[11px]">Profit Factor</span>
                      <b>{s.profit_factor.toFixed(2)}</b> <span className="text-[#8A9BB5]">vs OOS {report.oos_benchmark.profit_factor.toFixed(2)}</span>
                    </div>
                    <div>
                      <span className="text-[#8A9BB5] font-sans block text-[11px]">Win Rate</span>
                      <b>{fmtPct(s.win_rate)}</b> <span className="text-[#8A9BB5]">vs OOS {fmtPct(report.oos_benchmark.win_rate)}</span>
                    </div>
                    <div>
                      <span className="text-[#8A9BB5] font-sans block text-[11px]">Expectancy</span>
                      <b>{fmtSignedR(s.expectancy_r)}</b> <span className="text-[#8A9BB5]">vs OOS {fmtSignedR(report.oos_benchmark.expectancy_r)}</span>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* AI Supervisor Veto Impact & Live ML Calibration */}
            <div className="xl:col-span-6 cockpit-card rounded-lg p-3.5 space-y-3">
              <h3 className="text-xs font-bold uppercase tracking-wider">AI Supervisor Veto &amp; Live ML Calibration</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {/* Veto Scoreboard */}
                <div className="space-y-1.5">
                  <div className="text-[11px] font-bold uppercase tracking-wider text-[#8A9BB5]">AI Supervisor Verdicts (No Veto Ground Truth)</div>
                  <table className="w-full text-xs text-left border-collapse mono">
                    <thead>
                      <tr className="border-b border-[#1F2D45] text-[#8A9BB5] font-sans">
                        <th className="py-1">Verdict</th>
                        <th className="py-1 text-right">n</th>
                        <th className="py-1 text-right">Win %</th>
                        <th className="py-1 text-right">Avg R</th>
                        <th className="py-1 text-right">Net $</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-[#1F2D45]/70">
                      {report.veto_scoreboard.length === 0 ? (
                        <tr><td colSpan={5} className="py-3 text-center text-[#8A9BB5] font-sans">No verdict samples yet</td></tr>
                      ) : (
                        report.veto_scoreboard.map((v) => (
                          <tr key={v.verdict}>
                            <td className="py-1.5 font-bold text-white">{v.verdict}</td>
                            <td className="py-1.5 text-right">{v.trades}</td>
                            <td className="py-1.5 text-right">{v.win_rate.toFixed(1)}%</td>
                            <td className={`py-1.5 text-right ${v.avg_r >= 0 ? 'text-[#10B981]' : 'text-[#EF4444]'}`}>
                              {v.avg_r >= 0 ? '+' : ''}{v.avg_r.toFixed(3)}
                            </td>
                            <td className={`py-1.5 text-right font-bold ${v.net_pnl >= 0 ? 'text-[#10B981]' : 'text-[#EF4444]'}`}>
                              {fmtMoney(v.net_pnl)}
                            </td>
                          </tr>
                        ))
                      )}
                    </tbody>
                  </table>
                </div>

                {/* Live ML Confidence Buckets */}
                <div className="space-y-1.5">
                  <div className="text-[11px] font-bold uppercase tracking-wider text-[#8A9BB5]">Live ML P(win) Calibration</div>
                  <table className="w-full text-xs text-left border-collapse mono">
                    <thead>
                      <tr className="border-b border-[#1F2D45] text-[#8A9BB5] font-sans">
                        <th className="py-1">ML Bucket</th>
                        <th className="py-1 text-right">n</th>
                        <th className="py-1 text-right">Actual WR</th>
                        <th className="py-1 text-right">Gap</th>
                        <th className="py-1 text-right">Avg R</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-[#1F2D45]/70">
                      {report.ml_buckets.length === 0 ? (
                        <tr><td colSpan={5} className="py-3 text-center text-[#8A9BB5] font-sans">No trades in this book</td></tr>
                      ) : (
                        report.ml_buckets.map((b) => (
                          <tr key={b.bucket}>
                            <td className="py-1.5 font-semibold text-white">{b.bucket}</td>
                            <td className="py-1.5 text-right">{b.trades}</td>
                            <td className="py-1.5 text-right">{fmtPct(b.actual_wr)}</td>
                            <td className={`py-1.5 text-right ${Math.abs(b.gap) <= 0.08 ? 'text-[#10B981]' : 'text-[#F59E0B]'}`}>
                              {b.gap >= 0 ? '+' : ''}{(b.gap * 100).toFixed(1)}%
                            </td>
                            <td className={`py-1.5 text-right ${b.avg_r >= 0 ? 'text-[#10B981]' : 'text-[#EF4444]'}`}>
                              {b.avg_r >= 0 ? '+' : ''}{b.avg_r.toFixed(3)}
                            </td>
                          </tr>
                        ))
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>

          {/* 2x2 Detailed Breakdown Tables */}
          <div className="cockpit-card rounded-lg p-3.5 space-y-3">
            <h3 className="text-xs font-bold uppercase tracking-wider">Granular Performance Breakdowns</h3>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
              <BreakdownTable title="Breakdown by Symbol" colLabel="Symbol" rows={report.by_symbol} />
              <BreakdownTable
                title="Breakdown by Strategy"
                colLabel="Strategy"
                rows={report.by_strategy}
                formatGroup={(g) => human(g)}
              />
              <BreakdownTable
                title="Breakdown by Market Regime"
                colLabel="Regime"
                rows={report.by_regime}
                formatGroup={(g) => human(g)}
              />
              <BreakdownTable
                title="Breakdown by Exit Reason"
                colLabel="Exit Reason"
                rows={report.by_exit}
                formatGroup={(g) => human(g)}
              />
            </div>
          </div>

          {/* Collapsible Raw Markdown Report */}
          {showMarkdown && report.markdown_report && (
            <div className="cockpit-card rounded-lg p-3.5 space-y-2">
              <div className="flex items-center justify-between text-xs font-bold uppercase tracking-wider text-[#8A9BB5]">
                <span>Raw Performance Report (Markdown)</span>
                <button
                  onClick={copyMarkdown}
                  className="px-2 py-0.5 rounded cockpit-sub text-xs font-semibold cursor-pointer hover:bg-[#162236]"
                >
                  Copy
                </button>
              </div>
              <pre className="p-3 rounded bg-[#070B12] border border-[#1F2D45] text-[11px] mono text-[#CBD5E1] overflow-x-auto whitespace-pre-wrap max-h-96 overflow-y-auto">
                {report.markdown_report}
              </pre>
            </div>
          )}
        </>
      )}
    </div>
  );
};
