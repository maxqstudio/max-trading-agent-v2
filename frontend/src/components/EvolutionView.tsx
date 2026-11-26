import React, { useEffect, useState } from 'react';
import { RefreshCw, Check, X, RotateCcw } from 'lucide-react';
import { api } from '../api/client';
import { EquityCurveCanvas } from './EquityCurveCanvas';

interface Props {
  onToast: (msg: string) => void;
}

export const EvolutionView: React.FC<Props> = ({ onToast }) => {
  const [evo, setEvo] = useState<any>(null);
  const [busy, setBusy] = useState(false);

  const load = async () => {
    try {
      const res = await api.getEvolution();
      setEvo(res);
    } catch {
      // ignore
    }
  };

  useEffect(() => {
    load();
    const t = window.setInterval(load, 5000);
    return () => window.clearInterval(t);
  }, []);

  const runCycle = async () => {
    setBusy(true);
    try {
      const res = await api.runEvolution();
      onToast(`Evolution cycle: ${res.status}`);
      await load();
    } catch (e: any) {
      onToast(`Evolution error: ${e.message}`);
    } finally {
      setBusy(false);
    }
  };

  const cand = evo?.active_challenger;
  const activeSnap = evo?.active_snapshot;

  return (
    <div className="space-y-3">
      <div className="cockpit-card rounded-lg p-3.5 space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <h2 className="text-sm font-bold">Champion vs Shadow Challenger — 4-Stage Anti-Overfit Gate</h2>
            <p className="text-xs text-[#8A9BB5]">Purged WFO + DL/ML Retrain → 1,000× Monte Carlo Stress → Live Shadow Arena → Promote & Auto-Rollback</p>
          </div>
          <div className="flex items-center gap-2">
            <label className="flex items-center gap-2 px-2.5 py-1.5 rounded cockpit-sub text-xs cursor-pointer">
              <input
                type="checkbox"
                checked={Boolean(evo?.auto_promote)}
                onChange={async (e) => {
                  await api.updateRisk({ auto_promote_challenger: e.target.checked });
                  onToast(e.target.checked ? 'Auto-Promote ARMED' : 'Manual Approval Mode Active');
                  load();
                }}
                className="accent-[#3B82F6]"
              />
              <span className="font-semibold">Auto-Promote</span>
            </label>
            <button
              disabled={busy || evo?.running}
              onClick={runCycle}
              className="px-3 py-1.5 rounded-md bg-[#3B82F6] text-white text-xs font-bold flex items-center gap-1.5 cursor-pointer disabled:opacity-50"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${busy || evo?.running ? 'animate-spin' : ''}`} />
              <span>{busy || evo?.running ? 'Running WFO + MC...' : 'Run Evolution Cycle'}</span>
            </button>
          </div>
        </div>

        <EquityCurveCanvas
          points={evo?.equity_curve || []}
          championVersion={activeSnap?.version_id || 'v1.0'}
        />
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-12 gap-3">
        <div className="xl:col-span-7 cockpit-card rounded-lg p-3.5 space-y-3">
          <div className="flex items-center justify-between flex-wrap gap-2">
            <h3 className="text-xs font-bold uppercase tracking-wider">Head-to-Head Arena & Parameter Diff</h3>
            {cand && (
              <div className="flex gap-2">
                <button
                  onClick={async () => {
                    await api.approveChallenger();
                    onToast(`Promoted ${cand.version_id} to Champion!`);
                    load();
                  }}
                  className="px-2.5 py-1 rounded bg-[#10B981] text-white text-xs font-bold flex items-center gap-1 cursor-pointer"
                >
                  <Check className="w-3.5 h-3.5" /> Approve & Promote
                </button>
                <button
                  onClick={async () => {
                    await api.rejectChallenger();
                    onToast('Challenger rejected');
                    load();
                  }}
                  className="px-2.5 py-1 rounded cockpit-sub text-xs font-bold flex items-center gap-1 cursor-pointer"
                >
                  <X className="w-3.5 h-3.5" /> Reject
                </button>
              </div>
            )}
          </div>

          {cand ? (
            <>
              <table className="w-full text-xs text-left border-collapse mono">
                <thead>
                  <tr className="border-b border-[#1F2D45] text-[#8A9BB5] font-sans">
                    <th className="py-1.5">Metric</th>
                    <th className="py-1.5">Champion ({cand.parent_version_id})</th>
                    <th className="py-1.5">Challenger ({cand.version_id})</th>
                    <th className="py-1.5">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#1F2D45]">
                  <tr><td className="py-1.5 font-sans">OOS Profit Factor</td><td>{cand.champion_oos_metrics.profit_factor.toFixed(2)}</td><td className="text-[#8B5CF6] font-bold">{cand.oos_metrics.profit_factor.toFixed(2)}</td><td>{cand.status}</td></tr>
                  <tr><td className="py-1.5 font-sans">OOS Sharpe</td><td>{cand.champion_oos_metrics.sharpe_ratio.toFixed(2)}</td><td className="text-[#8B5CF6] font-bold">{cand.oos_metrics.sharpe_ratio.toFixed(2)}</td><td>{cand.oos_metrics.total_trades} trades</td></tr>
                  <tr><td className="py-1.5 font-sans">Monte Carlo P95 DD</td><td>{cand.champion_oos_metrics.mc_p95_dd_pct.toFixed(2)}%</td><td className="text-[#10B981] font-bold">{cand.oos_metrics.mc_p95_dd_pct.toFixed(2)}%</td><td>Ruin {(cand.oos_metrics.mc_ruin_prob * 100).toFixed(1)}%</td></tr>
                  <tr><td className="py-1.5 font-sans">Live Shadow Arena</td><td>—</td><td className="text-[#06B6D4] font-bold">{cand.shadow_trades}t ({cand.shadow_pnl_r >= 0 ? '+' : ''}{cand.shadow_pnl_r}R)</td><td>PF {cand.shadow_pf.toFixed(2)}</td></tr>
                </tbody>
              </table>
              <div className="p-2.5 rounded cockpit-sub mono text-xs space-y-1">
                {(cand.patches || []).map((p: any) => (
                  <div key={p.path}>
                    <span className="text-[#8A9BB5]">{p.path}: </span>
                    <span className="text-[#EF4444]">{p.old_value}</span>
                    <span> → </span>
                    <span className="text-[#10B981] font-bold">{p.new_value}</span>
                  </div>
                ))}
              </div>
            </>
          ) : (
            <div className="p-4 rounded cockpit-sub text-xs text-[#8A9BB5]">
              No active Challenger in Shadow Arena. Click <b>Run Evolution Cycle</b> to execute Purged WFO + DL/ML retrain + LLM post-mortem reflection.
            </div>
          )}
        </div>

        <div className="xl:col-span-5 cockpit-card rounded-lg p-3.5 space-y-2.5">
          <h3 className="text-xs font-bold uppercase tracking-wider">Immutable Champion Snapshots</h3>
          <div className="space-y-2 max-h-72 overflow-y-auto">
            {(evo?.snapshots || []).map((s: any) => (
              <div key={s.version_id} className="p-2.5 rounded cockpit-sub flex items-center justify-between gap-2 text-xs">
                <div>
                  <div className="flex items-center gap-2">
                    <span className="mono font-bold">{s.version_id}</span>
                    <span className={`px-1.5 py-0.5 rounded font-bold ${s.status === 'ACTIVE' ? 'bg-[#10B981]/15 text-[#10B981]' : 'bg-[#3B82F6]/15 text-[#3B82F6]'}`}>{s.status}</span>
                  </div>
                  <div className="text-[#8A9BB5] mt-0.5">{s.note}</div>
                </div>
                {s.status !== 'ACTIVE' && (
                  <button
                    onClick={async () => {
                      await api.rollbackChampion(s.version_id);
                      onToast(`Rolled back instantly to ${s.version_id}`);
                      load();
                    }}
                    className="px-2 py-1 rounded cockpit-card font-bold flex items-center gap-1 cursor-pointer shrink-0"
                  >
                    <RotateCcw className="w-3 h-3" /> Rollback
                  </button>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="cockpit-card rounded-lg p-3.5 space-y-2.5">
        <h3 className="text-xs font-bold uppercase tracking-wider">LLM Post-Mortem Reflection Reports</h3>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-2.5">
          {(evo?.reflections || []).length === 0 ? (
            <div className="text-xs text-[#8A9BB5] p-3 rounded cockpit-sub">No post-mortem reflections yet. Trigger an evolution cycle to generate one.</div>
          ) : (
            (evo?.reflections || []).map((r: any) => (
              <div key={r.report_id} className="p-3 rounded cockpit-sub space-y-1.5 text-xs">
                <div className="flex justify-between mono">
                  <span className="font-bold">{r.report_id} • {r.provider}</span>
                  <span className="px-1.5 py-0.5 rounded bg-[#8B5CF6]/20 text-[#8B5CF6] font-bold">{r.gate_status}</span>
                </div>
                <div className="font-medium">{r.finding}</div>
                <div className="text-[#8A9BB5] mono">{r.evidence}</div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
};
