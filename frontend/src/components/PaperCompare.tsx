import React, { useEffect, useState } from 'react';
import { Brain, Scale } from 'lucide-react';
import { api } from '../api/client';

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
}

const COLOR: Record<string, string> = { normal: '#F59E0B', hitrun: '#EC4899', mt5: '#3B82F6' };
const money = (v: number) => `${v >= 0 ? '+' : '-'}$${Math.abs(v).toFixed(2)}`;
const signed = (v: number, d = 2) => `${v >= 0 ? '+' : ''}${v.toFixed(d)}`;

export const PaperCompare: React.FC = () => {
  const [data, setData] = useState<Compare | null>(null);
  const [err, setErr] = useState('');
  const [busy, setBusy] = useState(false);

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

  const paper = data?.rows.filter((r) => r.book !== 'mt5') ?? [];
  const best =
    paper.length === 2 && paper.every((r) => r.trades > 0) ? (paper[0].net_pnl >= paper[1].net_pnl ? paper[0].book : paper[1].book) : null;
  const groups = (data?.learning.groups ?? []).filter((g) => g.normal_n + g.hitrun_n > 0);

  return (
    <div className="cockpit-card rounded-lg">
      <div className="px-3 py-2 border-b border-[#1F2D45] flex flex-wrap items-center justify-between gap-2 text-xs">
        <span className="font-bold uppercase tracking-wider flex items-center gap-1.5">
          <Scale className="w-3.5 h-3.5 text-[#06B6D4]" /> Perbandingan Pola: Normal vs Hit &amp; Run
        </span>
        <span className="mono text-[#8A9BB5]" title={data?.note}>
          Biaya AI hari ini ${data?.ai_cost_today_usd?.toFixed(2) ?? '0.00'} (dipakai bersama)
        </span>
      </div>
      {err ? <div className="px-3 py-2 text-xs text-[#EF4444]">Gagal memuat: {err}</div> : null}
      <div className="overflow-x-auto p-2">
        <table className="w-full text-xs text-left border-collapse">
          <thead>
            <tr className="border-b border-[#1F2D45] text-[#8A9BB5]">
              <th className="py-1.5 px-2">Pola</th>
              <th className="py-1.5 px-2 text-right">Equity</th>
              <th className="py-1.5 px-2 text-right">Trade</th>
              <th className="py-1.5 px-2 text-right">Win rate</th>
              <th className="py-1.5 px-2 text-right">PF</th>
              <th className="py-1.5 px-2 text-right">Net</th>
              <th className="py-1.5 px-2 text-right">Avg win / loss</th>
              <th className="py-1.5 px-2 text-right">Avg R</th>
              <th className="py-1.5 px-2 text-right">Max DD</th>
              <th className="py-1.5 px-2 text-right" title="Trade rugi yang sempat profit minimal +0.5R">
                Untung → rugi
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-[#1F2D45] mono">
            {(data?.rows ?? []).map((r) => (
              <tr key={r.book} className={r.book === 'mt5' ? 'text-[#8A9BB5]' : ''}>
                <td className="py-1.5 px-2 font-sans font-bold whitespace-nowrap" style={{ color: COLOR[r.book] }}>
                  {r.label}
                  {best === r.book ? <span className="ml-1.5 px-1 rounded bg-[#10B981]/15 text-[#10B981] text-[10px]">unggul</span> : null}
                  {r.open_positions ? <span className="ml-1.5 text-[#8A9BB5] font-normal">({r.open_positions} open)</span> : null}
                </td>
                <td className="py-1.5 px-2 text-right">${r.equity.toFixed(0)}</td>
                <td className="py-1.5 px-2 text-right">{r.trades}</td>
                <td className="py-1.5 px-2 text-right">{r.trades ? `${r.win_rate.toFixed(1)}%` : '—'}</td>
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
                <td colSpan={10} className="py-4 text-center text-[#8A9BB5] font-sans">
                  Memuat...
                </td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>

      <div className="px-3 py-2 border-t border-[#1F2D45] text-xs space-y-2">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <span className="font-bold flex items-center gap-1.5">
            <Brain className="w-3.5 h-3.5 text-[#8B5CF6]" /> Belajar gaya exit per pasar &amp; kondisi
          </span>
          <button
            disabled={!data || busy}
            onClick={toggleAdaptive}
            title="Kalau aktif, posisi MT5 memakai gaya exit pemenang untuk pasar + kondisi itu. Kalau mati, MT5 selalu Normal."
            className={`px-2 py-0.5 rounded border font-bold cursor-pointer disabled:opacity-50 ${
              data?.learning.adaptive ? 'border-[#10B981]/50 text-[#10B981] bg-[#10B981]/10' : 'border-[#1F2D45] text-[#8A9BB5]'
            }`}
          >
            Terapkan ke MT5: {data?.learning.adaptive ? 'AKTIF' : 'MATI'}
          </button>
        </div>
        <p className="text-[#8A9BB5]">
          Dua pola membuka setup yang sama, jadi setiap trade yang selesai adalah perbandingan langsung. Gaya pemenang baru dipakai di MT5
          setelah masing-masing punya minimal {data?.learning.min_samples ?? 20} trade di grup itu dan Hit &amp; Run jelas lebih baik. Sebelum
          itu MT5 tetap Normal.
        </p>
        {groups.length ? (
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b border-[#1F2D45] text-[#8A9BB5]">
                  <th className="py-1 px-2">Pasar · kondisi</th>
                  <th className="py-1 px-2 text-right">Normal (n · avg R · WR)</th>
                  <th className="py-1 px-2 text-right">Hit &amp; Run (n · avg R · WR)</th>
                  <th className="py-1 px-2 text-right">Pilihan</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#1F2D45] mono">
                {groups.map((g) => (
                  <tr key={g.key}>
                    <td className="py-1 px-2 font-bold">{g.key}</td>
                    <td className="py-1 px-2 text-right">
                      {g.normal_n} · {signed(g.normal_avg_r)} · {g.normal_wr.toFixed(0)}%
                    </td>
                    <td className="py-1 px-2 text-right">
                      {g.hitrun_n} · {signed(g.hitrun_avg_r)} · {g.hitrun_wr.toFixed(0)}%
                    </td>
                    <td className="py-1 px-2 text-right font-bold" style={{ color: g.pick ? COLOR[g.pick] : '#8A9BB5' }}>
                      {g.pick === 'hitrun' ? 'Hit & Run' : g.pick === 'normal' ? 'Normal' : `butuh ${g.need} lagi`}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-[#8A9BB5]">Belum ada trade paper yang selesai.</p>
        )}
      </div>
    </div>
  );
};
