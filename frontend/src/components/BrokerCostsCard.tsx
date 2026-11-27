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
  mt5: { label: 'auto from MT5', color: '#10B981' },
  manual: { label: 'manual', color: '#3B82F6' },
  default: { label: 'default (no MT5 trades yet)', color: '#F59E0B' },
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
      onToast('Broker costs saved');
    } catch (e) {
      onToast(`Could not save costs: ${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="cockpit-card rounded-lg p-3.5 space-y-3 text-xs">
      <div className="flex items-center justify-between gap-2">
        <div>
          <h3 className="font-bold uppercase tracking-wider">Broker costs (paper)</h3>
          <p className="text-[#8A9BB5] mt-0.5">
            Charged on every paper trade. Spread is already included (buy at ask, sell at bid). Commission and slippage are learned from
            closed MT5 trades.
          </p>
        </div>
        <button
          onClick={save}
          disabled={busy}
          className="px-2.5 py-1 rounded border font-bold flex items-center gap-1 cursor-pointer disabled:opacity-50 bg-[#3B82F6] text-white border-[#3B82F6] shrink-0"
        >
          <Check className="w-3.5 h-3.5" /> Save
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-2.5">
        <div className="space-y-1">
          <label className="flex items-center gap-2 cursor-pointer">
            <input type="checkbox" checked={commAuto} onChange={(e) => setCommAuto(e.target.checked)} />
            <span>Commission auto from MT5</span>
          </label>
          <label className="space-y-1 block">
            <span className="text-[#8A9BB5]">Round-trip commission per 1 lot (account currency)</span>
            <input type="number" step={0.5} min={0} disabled={commAuto} value={comm} onChange={(e) => setComm(e.target.value)} className={inputCls} />
          </label>
        </div>
        <div className="space-y-1">
          <label className="flex items-center gap-2 cursor-pointer">
            <input type="checkbox" checked={slipAuto} onChange={(e) => setSlipAuto(e.target.checked)} />
            <span>Slippage auto from MT5</span>
          </label>
          <label className="space-y-1 block">
            <span className="text-[#8A9BB5]">Slippage per fill (points, on entry and on exit)</span>
            <input type="number" step={1} min={0} disabled={slipAuto} value={slip} onChange={(e) => setSlip(e.target.value)} className={inputCls} />
          </label>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="border-b border-[#1F2D45] text-[#8A9BB5]">
              <th className="py-1 px-2">Market</th>
              <th className="py-1 px-2 text-right">Commission / lot</th>
              <th className="py-1 px-2">Source</th>
              <th className="py-1 px-2 text-right">Slippage</th>
              <th className="py-1 px-2">Source</th>
              <th className="py-1 px-2 text-right">MT5 trades</th>
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
        Auto commission = entry commission reported by MT5 × 2 (in + out), median of the last 30 trades per market.
      </p>
    </div>
  );
};
