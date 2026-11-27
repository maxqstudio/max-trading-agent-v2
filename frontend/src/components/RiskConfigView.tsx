import React, { useEffect, useState } from 'react';
import { api } from '../api/client';

export const RiskConfigView: React.FC<{ onToast: (msg: string) => void }> = ({ onToast }) => {
  const [cfg, setCfg] = useState<any>(null);
  const [riskPct, setRiskPct] = useState(0.5);
  const [maxDd, setMaxDd] = useState(3.0);
  const [spAtr, setSpAtr] = useState(0.3);
  const [costRisk, setCostRisk] = useState(0.18);
  const [maxPos, setMaxPos] = useState(3);
  const [mlProb, setMlProb] = useState(0.56);

  useEffect(() => {
    api.getConfig().then((c) => {
      setCfg(c);
      setRiskPct(c.risk.risk_per_trade_pct);
      setMaxDd(c.risk.max_daily_drawdown_pct);
      setSpAtr(c.risk.max_spread_atr_ratio);
      setCostRisk(c.risk.max_cost_to_risk_ratio ?? 0.18);
      setMaxPos(c.risk.max_concurrent_positions);
      setMlProb(c.ai.ml_min_probability);
    }).catch(() => {});
  }, []);

  const saveRisk = async () => {
    await api.updateRisk({
      risk_per_trade_pct: riskPct,
      max_daily_drawdown_pct: maxDd,
      max_spread_atr_ratio: spAtr,
      max_cost_to_risk_ratio: costRisk,
      max_concurrent_positions: maxPos,
      ml_min_probability: mlProb,
    });
    onToast('Risk guard parameters clamped & applied');
  };

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-1 xl:grid-cols-12 gap-3">
        <div className="xl:col-span-6 cockpit-card rounded-lg p-3.5 space-y-3">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-xs font-bold uppercase tracking-wider">Python Portfolio Risk Controls</h3>
              <p className="text-xs text-[#8A9BB5]">Hard-clamped by Pydantic absolute safety bounds</p>
            </div>
            <button onClick={saveRisk} className="px-3 py-1 rounded bg-[#3B82F6] text-white text-xs font-bold cursor-pointer">Apply</button>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5 text-xs">
            <label className="p-2.5 rounded cockpit-sub space-y-1">
              <span className="text-[#8A9BB5]">Risk / Trade % [Max 1.5%]</span>
              <input type="number" step="0.05" value={riskPct} onChange={(e) => setRiskPct(Number(e.target.value))} className="w-full mono font-bold bg-transparent border-b border-[#1F2D45] py-0.5 focus:outline-none" />
            </label>
            <label className="p-2.5 rounded cockpit-sub space-y-1">
              <span className="text-[#8A9BB5]">Max Daily DD %</span>
              <input type="number" step="0.25" value={maxDd} onChange={(e) => setMaxDd(Number(e.target.value))} className="w-full mono font-bold bg-transparent border-b border-[#1F2D45] py-0.5 focus:outline-none" />
            </label>
            <label className="p-2.5 rounded cockpit-sub space-y-1">
              <span className="text-[#8A9BB5]">Max Spread / ATR Ratio</span>
              <input type="number" step="0.02" value={spAtr} onChange={(e) => setSpAtr(Number(e.target.value))} className="w-full mono font-bold bg-transparent border-b border-[#1F2D45] py-0.5 focus:outline-none" />
            </label>
            <label className="p-2.5 rounded cockpit-sub space-y-1">
              <span className="text-[#8A9BB5]">Max Concurrent Positions</span>
              <input type="number" value={maxPos} onChange={(e) => setMaxPos(Number(e.target.value))} className="w-full mono font-bold bg-transparent border-b border-[#1F2D45] py-0.5 focus:outline-none" />
            </label>
            <label
              className="p-2.5 rounded cockpit-sub space-y-1"
              title="Blocks trades where (spread + round-trip broker commission) exceeds this fraction of SL distance"
            >
              <span className="text-[#8A9BB5]">Max Total Cost / SL Risk Ratio</span>
              <input type="number" step="0.01" value={costRisk} onChange={(e) => setCostRisk(Number(e.target.value))} className="w-full mono font-bold bg-transparent border-b border-[#1F2D45] py-0.5 focus:outline-none" />
            </label>
            <label className="p-2.5 rounded cockpit-sub space-y-1">
              <span className="text-[#8A9BB5]">Calibrated ML Min Win Probability Cutoff</span>
              <input type="number" step="0.01" value={mlProb} onChange={(e) => setMlProb(Number(e.target.value))} className="w-full mono font-bold bg-transparent border-b border-[#1F2D45] py-0.5 focus:outline-none" />
            </label>
          </div>
        </div>

        <div className="xl:col-span-6 cockpit-card rounded-lg p-3.5 space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-bold uppercase tracking-wider">MQL5 EA Hard Risk Guard & Shields</h3>
            <span className={`px-2 py-0.5 rounded mono text-xs font-bold ${cfg?.hard_guard?.synced ? 'bg-[#10B981]/15 text-[#10B981]' : 'bg-[#F59E0B]/15 text-[#F59E0B]'}`}>
              {cfg?.hard_guard?.synced ? `SYNCED (${cfg.hard_guard.ea_version})` : 'WAITING EA'}
            </span>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs mono">
            <div className="p-2 rounded cockpit-sub"><div className="font-sans text-[#8A9BB5]">Hard DD</div><div className="font-bold text-sm text-[#EF4444]">{cfg?.hard_guard?.hard_max_daily_dd_pct ?? 3.5}%</div></div>
            <div className="p-2 rounded cockpit-sub"><div className="font-sans text-[#8A9BB5]">Hard Lot</div><div className="font-bold text-sm">{cfg?.hard_guard?.hard_max_lot ?? 0.5}</div></div>
            <div className="p-2 rounded cockpit-sub"><div className="font-sans text-[#8A9BB5]">Watchdog</div><div className="font-bold text-sm">{cfg?.hard_guard?.heartbeat_timeout_sec ?? 15}s</div></div>
            <div className="p-2 rounded cockpit-sub"><div className="font-sans text-[#8A9BB5]">Order TTL</div><div className="font-bold text-sm">{cfg?.hard_guard?.order_ttl_ms ?? 1500}ms</div></div>
          </div>
          <div className="p-3 rounded cockpit-sub text-xs space-y-1 text-[#8A9BB5]">
            <div>LLM Provider: <b className="text-white mono">{cfg?.llm_provider || 'deterministic'}</b></div>
            <div>Telegram Control Bot: <b className="text-white">{cfg?.telegram_configured ? 'Connected' : 'Not configured (.env)'}</b></div>
            <div>Bridge Auth Token: <b className="text-white mono">{cfg?.bridge_token_masked || '***'}</b></div>
          </div>
        </div>
      </div>

      <div className="cockpit-card rounded-lg p-3.5 space-y-2.5">
        <h3 className="text-xs font-bold uppercase tracking-wider">Live Broker SymbolSpec (Auto-Discovered from MT5)</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left border-collapse mono">
            <thead>
              <tr className="border-b border-[#1F2D45] text-[#8A9BB5] font-sans">
                <th className="py-1.5 pr-2">Logical → Broker</th>
                <th className="py-1.5 px-2">Digits</th>
                <th className="py-1.5 px-2">Tick Value</th>
                <th className="py-1.5 px-2">Vol Min / Step</th>
                <th className="py-1.5 px-2">Stops Lvl</th>
                <th className="py-1.5 pl-2">Fill Mode</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#1F2D45]">
              {(cfg?.symbol_specs || []).map((s: any) => (
                <tr key={s.symbol}>
                  <td className="py-2 pr-2 font-bold">{s.symbol} → {s.broker_symbol}</td>
                  <td className="py-2 px-2">{s.digits}</td>
                  <td className="py-2 px-2">${s.tick_value}</td>
                  <td className="py-2 px-2">{s.volume_min} / {s.volume_step}</td>
                  <td className="py-2 px-2">{s.stops_level} pt</td>
                  <td className="py-2 pl-2"><span className="px-1.5 py-0.5 rounded bg-[#10B981]/15 text-[#10B981] font-bold">{s.filling_mode}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
