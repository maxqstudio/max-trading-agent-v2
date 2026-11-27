import React, { useState } from 'react';
import { Check } from 'lucide-react';
import { api } from '../api/client';

// Broker costs used by the paper books: learned from closed MT5 trades, or set by hand.

interface CostRow {
  symbol: string;
  commission_per_lot: number;
  commission_source: 'manual' | 'mt5' | 'default';
  slippage_points: number;
  slippage_source: 'manual' | 'mt5' | 'default';
  mt5_trades: number;
}

export interface CostsView {
  manual: { commission_per_lot: number | null; slippage_points: number | null };
  symbols: CostRow[];
}

const SRC: Record<string, { label: string; color: string }> = {
  mt5: { label: 'otomatis dari MT5', color: '#10B981' },
  manual: { label: 'manual', color: '#3B82F6' },
  default: { label: 'default (belum ada trade MT5)', color: '#F59E0B' },
};

const inputCls = 'w-full px-2 py-1 rounded cockpit-sub mono text-xs';

export const BrokerCostsCard: React.FC<{ costs: CostsView; onSaved: (c: CostsView) => void; onToast: (m: string) => void }> = ({
  costs,
  onSaved,
  onToast,
}) => {
  const [commAuto, setCommAuto] = useState(costs.manual.commission_per_lot == null);
  const [comm, setComm] = useState(String(costs.manual.commission_per_lot ?? costs.symbols[0]?.commission_per_lot ?? 7));
  const [slipAuto, setSlipAuto] = useState(costs.manual.slippage_points == null);
  const [slip, setSlip] = useState(String(costs.manual.slippage_points ?? costs.symbols[0]?.slippage_points ?? 0));
  const [busy, setBusy] = useState(false);

  const save = async () => {
    setBusy(true);
    try {
      const r = await api.saveCosts({
        commission_per_lot: commAuto ? null : Number(comm),
        slippage_points: slipAuto ? null : Number(slip),
      });
      onSaved(r);
      onToast('Biaya broker tersimpan');
    } catch (e) {
      onToast(`Gagal simpan biaya: ${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="cockpit-card rounded-lg p-3.5 space-y-3 text-xs">
      <div className="flex items-center justify-between gap-2">
        <div>
          <h3 className="font-bold uppercase tracking-wider">Biaya Broker (Paper)</h3>
          <p className="text-[#8A9BB5] mt-0.5">
            Dipotong dari setiap trade paper. Spread sudah otomatis (beli di ask, jual di bid). Komisi dan slippage dipelajari dari trade MT5
            yang sudah ditutup.
          </p>
        </div>
        <button
          onClick={save}
          disabled={busy}
          className="px-2.5 py-1 rounded border font-bold flex items-center gap-1 cursor-pointer disabled:opacity-50 bg-[#3B82F6] text-white border-[#3B82F6] shrink-0"
        >
          <Check className="w-3.5 h-3.5" /> Simpan
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-2.5">
        <div className="space-y-1">
          <label className="flex items-center gap-2 cursor-pointer">
            <input type="checkbox" checked={commAuto} onChange={(e) => setCommAuto(e.target.checked)} />
            <span>Komisi otomatis dari MT5</span>
          </label>
          <label className="space-y-1 block">
            <span className="text-[#8A9BB5]">Komisi pulang-pergi per 1 lot (mata uang akun)</span>
            <input type="number" step={0.5} min={0} disabled={commAuto} value={comm} onChange={(e) => setComm(e.target.value)} className={inputCls} />
          </label>
        </div>
        <div className="space-y-1">
          <label className="flex items-center gap-2 cursor-pointer">
            <input type="checkbox" checked={slipAuto} onChange={(e) => setSlipAuto(e.target.checked)} />
            <span>Slippage otomatis dari MT5</span>
          </label>
          <label className="space-y-1 block">
            <span className="text-[#8A9BB5]">Slippage per eksekusi (point, saat masuk dan keluar)</span>
            <input type="number" step={1} min={0} disabled={slipAuto} value={slip} onChange={(e) => setSlip(e.target.value)} className={inputCls} />
          </label>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="border-b border-[#1F2D45] text-[#8A9BB5]">
              <th className="py-1 px-2">Pasar</th>
              <th className="py-1 px-2 text-right">Komisi / lot</th>
              <th className="py-1 px-2">Sumber</th>
              <th className="py-1 px-2 text-right">Slippage</th>
              <th className="py-1 px-2">Sumber</th>
              <th className="py-1 px-2 text-right">Trade MT5</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-[#1F2D45] mono">
            {costs.symbols.map((r) => (
              <tr key={r.symbol}>
                <td className="py-1 px-2 font-bold">{r.symbol}</td>
                <td className="py-1 px-2 text-right">{r.commission_per_lot.toFixed(2)}</td>
                <td className="py-1 px-2 font-sans" style={{ color: SRC[r.commission_source].color }}>
                  {SRC[r.commission_source].label}
                </td>
                <td className="py-1 px-2 text-right">{r.slippage_points.toFixed(1)} pt</td>
                <td className="py-1 px-2 font-sans" style={{ color: SRC[r.slippage_source].color }}>
                  {SRC[r.slippage_source].label}
                </td>
                <td className="py-1 px-2 text-right">{r.mt5_trades}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-[#8A9BB5]">
        Komisi otomatis = komisi entry yang dilaporkan MT5 × 2 (masuk + keluar), median dari 30 trade terakhir per pasar.
      </p>
    </div>
  );
};
