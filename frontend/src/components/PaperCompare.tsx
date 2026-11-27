import React, { useEffect, useState } from 'react';
import { Brain, RotateCcw, Scale } from 'lucide-react';
import { api } from '../api/client';
import { human } from '../labels';

// Normal vs Hit & Run on the same setups, plus the real MT5 account, and what the
// exit-style learner has concluded so far per pair/regime.

interface Row {
  book: string;
  label: string;
  equity: number;
  open_positions: number;
  trades: number;
  win_rate: number;
  profit_factor: number | null;
  net_pnl: number;
  avg_win: number;
  avg_loss: number;
  avg_r: number;
  max_dd_pct: number;
  losers_after_half_r: number;
  breakeven_win_rate: number | null;
}

interface Group {
  key: string;
  normal_n: number;
  normal_avg_r: number;
  normal_wr: number;
  hitrun_n: number;
  hitrun_avg_r: number;
  hitrun_wr: number;
  pick: 'normal' | 'hitrun' | null;
  need: number;
}

interface Compare {
  rows: Row[];
  ai_cost_today_usd: number;
  note: string;
  learning: { adaptive: boolean; min_samples: number; groups: Group[] };
  veto?: { verdict: string; trades: number; win_rate: number; avg_r: number; net_pnl: number }[];
  hitrun_learning?: {
    adaptive: boolean;
    min_samples: number;
    default: string;
    groups: { key: string; trades: number; best: string; best_avg: number; best_wr: number; default_avg: number | null; using: string | null }[];
  };
}

const COLOR: Record<string, string> = { normal: '#F59E0B', hitrun: '#EC4899', raw: '#14B8A6', raw_hitrun: '#A855F7', mt5: '#3B82F6' };
const VERDICT: Record<string, string> = {
  APPROVE: 'Approved by AI Supervisor',
  REDUCE: 'Reduced by AI Supervisor',
  REJECT: 'Rejected by AI Supervisor',
  NONE: 'Not reviewed',
};
const groupLabel = (k: string) => k.split(' · ').map((part) => human(part)).join(' · ');
const money = (v: number) => `${v >= 0 ? '+' : '-'}$${Math.abs(v).toFixed(2)}`;
const signed = (v: number, d = 2) => `${v >= 0 ? '+' : ''}${v.toFixed(d)}`;

export const PaperCompare: React.FC = () => {
  const [data, setData] = useState<Compare | null>(null);
  const [err, setErr] = useState('');
  const [busy, setBusy] = useState(false);
  const [resetNote, setResetNote] = useState('');

  const load = async () => {
    try {
      setData(await api.getPaperCompare());
      setErr('');
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    }
  };

  useEffect(() => {
    load();
    const id = window.setInterval(load, 15000);
    return () => window.clearInterval(id);
  }, []);

  const toggleAdaptive = async () => {
    if (!data) return;
    setBusy(true);
    try {
      await api.saveExits({ adaptive: !data.learning.adaptive });
      await load();
    } finally {
      setBusy(false);
    }
  };

  const resetPaperBooks = async () => {
    const ok = window.confirm(
      'Reset all Paper & Shadow books to $10,000, clear old virtual paper trades, and reset Bandit / RL / Filter learners?\n\nReal MT5 trades and M1 market bars will be kept.',
    );
    if (!ok) return;
    setBusy(true);
    setResetNote('');
    try {
      const res = await api.resetPaper(true);
      await load();
      setResetNote(`Cleared ${res.cleared_trades} old paper trades · reset to $${res.start_equity}`);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const paper = data?.rows.filter((r) => r.book === 'normal' || r.book === 'hitrun') ?? [];
  const veto = data?.veto ?? [];
  const rej = veto.find((v) => v.verdict === 'REJECT');
  const ok = veto.find((v) => v.verdict === 'APPROVE');
  const best =
    paper.length === 2 && paper.every((r) => r.trades > 0) ? (paper[0].net_pnl >= paper[1].net_pnl ? paper[0].book : paper[1].book) : null;
  const groups = (data?.learning.groups ?? []).filter((g) => g.normal_n + g.hitrun_n > 0);

  return (
    <div className="cockpit-card rounded-lg">
      <div className="px-3 py-2 border-b border-[#1F2D45] flex flex-wrap items-center justify-between gap-2 text-xs">
        <span className="font-bold uppercase tracking-wider flex items-center gap-1.5">
          <Scale className="w-3.5 h-3.5 text-[#06B6D4]" /> Exit styles: Normal vs Hit &amp; Run
        </span>
        <div className="flex items-center gap-2.5">
          <span className="mono text-[#8A9BB5]" title={data?.note}>
            AI cost today ${data?.ai_cost_today_usd?.toFixed(2) ?? '0.00'} (shared)
          </span>
          <button
            disabled={busy}
            onClick={resetPaperBooks}
            title="Start a clean forward test: wipes old paper/shadow trades and resets paper equity to $10,000 while keeping real MT5 trades and M1 bars"
            className="px-2 py-0.5 rounded border border-[#F59E0B]/40 text-[#F59E0B] bg-[#F59E0B]/10 hover:bg-[#F59E0B]/20 font-bold flex items-center gap-1 cursor-pointer disabled:opacity-50"
          >
            <RotateCcw className="w-3 h-3" /> Reset Paper Books
          </button>
        </div>
      </div>
      {resetNote ? <div className="px-3 py-1.5 text-xs text-[#10B981] bg-[#10B981]/10 border-b border-[#1F2D45] mono">{resetNote}</div> : null}
      {err ? <div className="px-3 py-2 text-xs text-[#EF4444]">Could not load: {err}</div> : null}
      <div className="overflow-x-auto p-2">
        <table className="w-full text-xs text-left border-collapse">
          <thead>
            <tr className="border-b border-[#1F2D45] text-[#8A9BB5]">
              <th className="py-1.5 px-2">Book</th>
              <th className="py-1.5 px-2 text-right">Equity</th>
              <th className="py-1.5 px-2 text-right">Trade</th>
              <th className="py-1.5 px-2 text-right">Win rate</th>
              <th className="py-1.5 px-2 text-right" title="Win rate needed to break even with these average win and loss sizes">
                Break-even WR
              </th>
              <th className="py-1.5 px-2 text-right">PF</th>
              <th className="py-1.5 px-2 text-right">Net</th>
              <th className="py-1.5 px-2 text-right">Avg win / loss</th>
              <th className="py-1.5 px-2 text-right">Avg R</th>
              <th className="py-1.5 px-2 text-right">Max DD</th>
              <th className="py-1.5 px-2 text-right" title="Losing trades that were at least +0.5R in profit first">
                Profit → loss
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-[#1F2D45] mono">
            {(data?.rows ?? []).map((r) => (
              <tr key={r.book} className={r.book === 'mt5' ? 'text-[#8A9BB5]' : ''}>
                <td className="py-1.5 px-2 font-sans font-bold whitespace-nowrap" style={{ color: COLOR[r.book] }}>
                  {r.label}
                  {best === r.book ? <span className="ml-1.5 px-1 rounded bg-[#10B981]/15 text-[#10B981] text-[10px]">leading</span> : null}
                  {r.open_positions ? <span className="ml-1.5 text-[#8A9BB5] font-normal">({r.open_positions} open)</span> : null}
                </td>
                <td className="py-1.5 px-2 text-right">${r.equity.toFixed(0)}</td>
                <td className="py-1.5 px-2 text-right">{r.trades}</td>
                <td className="py-1.5 px-2 text-right">{r.trades ? `${r.win_rate.toFixed(1)}%` : '—'}</td>
                <td
                  className={`py-1.5 px-2 text-right ${
                    r.breakeven_win_rate == null ? '' : r.win_rate >= r.breakeven_win_rate ? 'text-[#10B981]' : 'text-[#EF4444]'
                  }`}
                >
                  {r.breakeven_win_rate == null ? '—' : `${r.breakeven_win_rate.toFixed(1)}%`}
                </td>
                <td className="py-1.5 px-2 text-right">{r.profit_factor == null ? '—' : r.profit_factor.toFixed(2)}</td>
                <td className={`py-1.5 px-2 text-right font-bold ${r.net_pnl >= 0 ? 'text-[#10B981]' : 'text-[#EF4444]'}`}>{money(r.net_pnl)}</td>
                <td className="py-1.5 px-2 text-right">{r.trades ? `${money(r.avg_win)} / ${money(r.avg_loss)}` : '—'}</td>
                <td className="py-1.5 px-2 text-right">{r.trades ? signed(r.avg_r) : '—'}</td>
                <td className="py-1.5 px-2 text-right">{r.max_dd_pct.toFixed(2)}%</td>
                <td className="py-1.5 px-2 text-right">{r.losers_after_half_r}</td>
              </tr>
            ))}
            {!data && !err ? (
              <tr>
                <td colSpan={11} className="py-4 text-center text-[#8A9BB5] font-sans">
                  Loading...
                </td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>

      <div className="px-3 py-2 border-t border-[#1F2D45] text-xs space-y-2">
        <div className="font-bold flex items-center gap-1.5">
          <Brain className="w-3.5 h-3.5 text-[#14B8A6]" /> AI Supervisor veto accuracy (from the No Veto book)
        </div>
        <p className="text-[#8A9BB5]">
          The No Veto book opens every setup that passed risk, including the ones AI Supervisor rejected. The bandit and RL exit learn from this
          book's market results, never from AI Supervisor's decisions. This table only scores the veto: if rejected setups do worse, the veto helps.
        </p>
        {veto.length ? (
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b border-[#1F2D45] text-[#8A9BB5]">
                  <th className="py-1 px-2">Decision</th>
                  <th className="py-1 px-2 text-right">Trade</th>
                  <th className="py-1 px-2 text-right">Win rate</th>
                  <th className="py-1 px-2 text-right">Avg R</th>
                  <th className="py-1 px-2 text-right">Net</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#1F2D45] mono">
                {veto.map((v) => (
                  <tr key={v.verdict}>
                    <td className="py-1 px-2 font-sans font-bold">{VERDICT[v.verdict] ?? v.verdict}</td>
                    <td className="py-1 px-2 text-right">{v.trades}</td>
                    <td className="py-1 px-2 text-right">{v.win_rate.toFixed(1)}%</td>
                    <td className="py-1 px-2 text-right">{signed(v.avg_r)}</td>
                    <td className={`py-1 px-2 text-right font-bold ${v.net_pnl >= 0 ? 'text-[#10B981]' : 'text-[#EF4444]'}`}>{money(v.net_pnl)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {rej && ok ? (
              <p className="mt-1 font-bold" style={{ color: rej.avg_r < ok.avg_r ? '#10B981' : '#F59E0B' }}>
                {rej.avg_r < ok.avg_r
                  ? `The veto helps: rejected setups average ${signed(rej.avg_r)}R, approved ones ${signed(ok.avg_r)}R.`
                  : `The veto has not helped yet: rejected setups average ${signed(rej.avg_r)}R, approved ones ${signed(ok.avg_r)}R.`}
              </p>
            ) : null}
          </div>
        ) : (
          <p className="text-[#8A9BB5]">No closed No Veto trades yet.</p>
        )}
      </div>

      <div className="px-3 py-2 border-t border-[#1F2D45] text-xs space-y-2">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <span className="font-bold flex items-center gap-1.5">
            <Brain className="w-3.5 h-3.5 text-[#8B5CF6]" /> Exit style learning per market &amp; regime
          </span>
          <button
            disabled={!data || busy}
            onClick={toggleAdaptive}
            title="When on, MT5 positions use the winning exit style for that market and regime. When off, MT5 always uses Normal."
            className={`px-2 py-0.5 rounded border font-bold cursor-pointer disabled:opacity-50 ${
              data?.learning.adaptive ? 'border-[#10B981]/50 text-[#10B981] bg-[#10B981]/10' : 'border-[#1F2D45] text-[#8A9BB5]'
            }`}
          >
            Apply to MT5: {data?.learning.adaptive ? 'ON' : 'OFF'}
          </button>
        </div>
        <p className="text-[#8A9BB5]">
          Hit &amp; Run closes as soon as the trade is in net profit (after spread, commission and swap), with its own tighter SL as the loss
          limit. Both books open the same setups, so every closed trade is a direct comparison. MT5 switches to the winner only after each
          style has at least {data?.learning.min_samples ?? 20} trades in that group and Hit &amp; Run is clearly better. Until then MT5 uses Normal.
        </p>
        {groups.length ? (
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b border-[#1F2D45] text-[#8A9BB5]">
                  <th className="py-1 px-2">Market · regime</th>
                  <th className="py-1 px-2 text-right">Normal (n · avg R · WR)</th>
                  <th className="py-1 px-2 text-right">Hit &amp; Run (n · avg R · WR)</th>
                  <th className="py-1 px-2 text-right">Pick</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#1F2D45] mono">
                {groups.map((g) => (
                  <tr key={g.key}>
                    <td className="py-1 px-2 font-bold">{groupLabel(g.key)}</td>
                    <td className="py-1 px-2 text-right">
                      {g.normal_n} · {signed(g.normal_avg_r)} · {g.normal_wr.toFixed(0)}%
                    </td>
                    <td className="py-1 px-2 text-right">
                      {g.hitrun_n} · {signed(g.hitrun_avg_r)} · {g.hitrun_wr.toFixed(0)}%
                    </td>
                    <td className="py-1 px-2 text-right font-bold" style={{ color: g.pick ? COLOR[g.pick] : '#8A9BB5' }}>
                      {g.pick === 'hitrun' ? 'Hit & Run' : g.pick === 'normal' ? 'Normal' : `needs ${g.need} more`}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-[#8A9BB5]">No closed paper trades yet.</p>
        )}
      </div>

      {data?.hitrun_learning ? (
        <div className="px-3 py-2 border-t border-[#1F2D45] text-xs space-y-2">
          <div className="font-bold flex items-center gap-1.5">
            <Brain className="w-3.5 h-3.5 text-[#EC4899]" /> Hit &amp; Run stop learning
          </div>
          <p className="text-[#8A9BB5]">
            Every setup in the No Veto book is followed by ghost trades with different stop distances and time stops, all with the same
            money cap. Default: {data.hitrun_learning.default}. A different setting is used only after {data.hitrun_learning.min_samples}{' '}
            ghost trades each and a clear edge{data.hitrun_learning.adaptive ? '' : ' (learning is off in Setup)'}.
          </p>
          {data.hitrun_learning.groups.length ? (
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="border-b border-[#1F2D45] text-[#8A9BB5]">
                    <th className="py-1 px-2">Market · regime</th>
                    <th className="py-1 px-2 text-right">Ghost trades</th>
                    <th className="py-1 px-2">Best so far</th>
                    <th className="py-1 px-2 text-right">Best $/trade · WR</th>
                    <th className="py-1 px-2 text-right">Default $/trade</th>
                    <th className="py-1 px-2 text-right">In use</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#1F2D45] mono">
                  {data.hitrun_learning.groups.map((g) => (
                    <tr key={g.key}>
                      <td className="py-1 px-2 font-bold">{groupLabel(g.key)}</td>
                      <td className="py-1 px-2 text-right">{g.trades}</td>
                      <td className="py-1 px-2 font-sans">{g.best}</td>
                      <td className="py-1 px-2 text-right">
                        {money(g.best_avg)} · {g.best_wr.toFixed(0)}%
                      </td>
                      <td className="py-1 px-2 text-right">{g.default_avg == null ? '—' : money(g.default_avg)}</td>
                      <td className="py-1 px-2 text-right font-sans">{g.using ?? '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="text-[#8A9BB5]">No ghost trades finished yet.</p>
          )}
        </div>
      ) : null}
    </div>
  );
};
