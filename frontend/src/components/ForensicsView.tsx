import React, { useEffect, useState } from 'react';
import { api } from '../api/client';
import type { TradeRecord } from '../types/trading';
import { human, humanText } from '../labels';

export const ForensicsView: React.FC<{ symbols: string[] }> = ({ symbols }) => {
  const [mode, setMode] = useState('ALL');
  const [sym, setSym] = useState('ALL');
  const [trades, setTrades] = useState<TradeRecord[]>([]);
  const [heatmap, setHeatmap] = useState<any[]>([]);
  const [expanded, setExpanded] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    const load = async () => {
      try {
        const res = await api.getJournal(mode, sym);
        if (alive) {
          setTrades(res.trades || []);
          setHeatmap(res.session_heatmap || []);
        }
      } catch {
        // ignore
      }
    };
    load();
    const t = window.setInterval(load, 5000);
    return () => {
      alive = false;
      window.clearInterval(t);
    };
  }, [mode, sym]);

  return (
    <div className="space-y-3">
      <div className="cockpit-card rounded-lg p-3.5 space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <h2 className="text-sm font-bold">Trade Forensics Journal (Click Row for MAE/MFE & RL Exit Trace)</h2>
            <p className="text-xs text-[#8A9BB5]">Actual slippage, commission, AI scores at entry, and MAE/MFE profile</p>
          </div>
          <div className="flex gap-2 text-xs">
            <select value={mode} onChange={(e) => setMode(e.target.value)} className="px-2.5 py-1 rounded cockpit-sub">
              <option value="ALL">All</option>
              <option value="LIVE">MT5 (Demo/Live)</option>
              <option value="PAPER">Paper Normal</option>
              <option value="HITRUN">Paper Hit &amp; Run</option>
              <option value="RAW">Paper No Veto</option>
              <option value="SHADOW">Kandidat (Evolusi)</option>
            </select>
            <select value={sym} onChange={(e) => setSym(e.target.value)} className="px-2.5 py-1 rounded cockpit-sub mono">
              <option value="ALL">All Pairs</option>
              {symbols.map((s) => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left border-collapse">
            <thead>
              <tr className="border-b border-[#1F2D45] text-[#8A9BB5]">
                <th className="py-1.5 pr-2">ID</th>
                <th className="py-1.5 px-2">Mode</th>
                <th className="py-1.5 px-2">Pair</th>
                <th className="py-1.5 px-2">Strategy / Regime</th>
                <th className="py-1.5 px-2">Side</th>
                <th className="py-1.5 px-2">MAE / MFE</th>
                <th className="py-1.5 px-2">ML / DL</th>
                <th className="py-1.5 pl-2 text-right">Net P&L (R)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#1F2D45]">
              {trades.length === 0 ? (
                <tr><td colSpan={8} className="py-6 text-center text-[#8A9BB5]">No closed trades recorded yet.</td></tr>
              ) : (
                trades.map((t) => {
                  const win = t.pnl_money >= 0;
                  const isOpen = expanded === t.trade_id;
                  return (
                    <React.Fragment key={t.trade_id}>
                      <tr onClick={() => setExpanded(isOpen ? null : t.trade_id)} className="cursor-pointer hover:bg-[#121B2B]">
                        <td className="py-2 pr-2 mono text-[#8A9BB5]">{t.trade_id}</td>
                        <td className="py-2 px-2"><span className={`px-1.5 py-0.5 rounded font-bold ${t.is_shadow ? 'bg-[#8B5CF6]/20 text-[#8B5CF6]' : t.version_id === 'PAPER-HITRUN' ? 'bg-[#EC4899]/20 text-[#EC4899]' : t.version_id === 'PAPER-RAW' ? 'bg-[#14B8A6]/20 text-[#14B8A6]' : t.version_id === 'PAPER' ? 'bg-[#F59E0B]/20 text-[#F59E0B]' : 'bg-[#3B82F6]/20 text-[#3B82F6]'}`}>{t.is_shadow ? 'KANDIDAT' : t.version_id === 'PAPER-HITRUN' ? 'HIT&RUN' : t.version_id === 'PAPER-RAW' ? 'TANPA VETO' : t.version_id === 'PAPER' ? 'NORMAL' : 'MT5'}</span></td>
                        <td className="py-2 px-2 mono font-bold">{t.symbol}</td>
                        <td className="py-2 px-2">{human(t.strategy_id)} <span className="text-[#8A9BB5]">• {human(t.regime)}</span></td>
                        <td className={`py-2 px-2 mono font-bold ${t.side === 'BUY' ? 'text-[#10B981]' : 'text-[#EF4444]'}`}>{t.side}</td>
                        <td className="py-2 px-2 mono"><span className="text-[#EF4444]">{t.mae_r.toFixed(2)}R</span> / <span className="text-[#10B981]">+{t.mfe_r.toFixed(2)}R</span></td>
                        <td className="py-2 px-2 mono">{Math.round(t.ml_confidence * 100)}% / {Math.round(t.dl_confidence * 100)}%</td>
                        <td className={`py-2 pl-2 text-right mono font-bold ${win ? 'text-[#10B981]' : 'text-[#EF4444]'}`}>{win ? '+' : ''}${t.pnl_money.toFixed(2)} ({win ? '+' : ''}{t.pnl_r.toFixed(2)}R)</td>
                      </tr>
                      {isOpen && (
                        <tr className="bg-[#0B111C]">
                          <td colSpan={8} className="p-3">
                            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
                              <div>
                                <div className="font-bold mb-1">MAE vs MFE Excursion</div>
                                <div className="flex items-center gap-2 mono">
                                  <span className="text-[#EF4444]">{t.mae_r.toFixed(2)}R</span>
                                  <div className="flex-1 h-2 rounded-full bg-[#121B2B] overflow-hidden flex">
                                    <div style={{ width: `${Math.min(48, Math.abs(t.mae_r) * 30)}%` }} className="bg-[#EF4444]" />
                                    <div style={{ width: `${Math.min(52, Math.max(0, t.mfe_r) * 28)}%` }} className="bg-[#10B981]" />
                                  </div>
                                  <span className="text-[#10B981]">+{t.mfe_r.toFixed(2)}R</span>
                                </div>
                              </div>
                              <div>
                                <div className="font-bold mb-1">Execution Costs & Exit</div>
                                <div className="mono text-[#8A9BB5]">Slip: {t.slippage_points}pt • Comm: ${t.commission.toFixed(2)} • Exit: {human(t.exit_reason)}</div>
                              </div>
                              <div>
                                <div className="font-bold mb-1">RL-2 Exit Trace</div>
                                <div className="mono text-[#06B6D4]">{humanText(t.rl_exit_trace)}</div>
                              </div>
                            </div>
                          </td>
                        </tr>
                      )}
                    </React.Fragment>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>

      <div className="cockpit-card rounded-lg p-3.5 space-y-2.5">
        <h3 className="text-xs font-bold uppercase tracking-wider">Session Expectancy Heatmap (00:00 - 23:00 UTC)</h3>
        <div className="grid grid-cols-6 sm:grid-cols-12 gap-1.5">
          {heatmap.map((h) => {
            const hh = String(h.hour_utc).padStart(2, '0');
            if (h.is_blackout) {
              return (
                <div key={h.hour_utc} className="p-2 rounded text-center mono text-xs bg-[#F59E0B]/20 text-[#F59E0B] border border-[#F59E0B]/40">
                  <b>{hh}h</b><div>BLK</div>
                </div>
              );
            }
            const pos = h.total_r >= 0;
            return (
              <div
                key={h.hour_utc}
                className="p-2 rounded text-center mono text-xs border border-[#1F2D45]"
                style={{
                  backgroundColor: h.trades === 0
                    ? '#121B2B'
                    : pos
                      ? 'rgba(16, 185, 129, 0.22)'
                      : 'rgba(239, 68, 68, 0.22)',
                }}
              >
                <b>{hh}h</b>
                <div>{h.trades === 0 ? '0.0R' : `${pos ? '+' : ''}${h.total_r.toFixed(1)}R`}</div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
