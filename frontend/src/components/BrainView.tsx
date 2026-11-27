import React, { useEffect, useRef, useState } from 'react';
import { api } from '../api/client';
import { human } from '../labels';

export const BrainView: React.FC<{ symbols: string[] }> = ({ symbols }) => {
  const [sym, setSym] = useState(symbols[0] || 'XAUUSD');
  const [data, setData] = useState<any>(null);
  const calibRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    let alive = true;
    const load = async () => {
      try {
        const res = await api.getBrain(sym);
        if (alive) setData(res);
      } catch {
        // ignore
      }
    };
    load();
    const t = window.setInterval(load, 4000);
    return () => {
      alive = false;
      window.clearInterval(t);
    };
  }, [sym]);

  useEffect(() => {
    const canvas = calibRef.current;
    if (!canvas || !data?.ml_diagnostics) return;
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.offsetWidth || 280;
    const h = 165;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);

    const pad = 22;
    const cw = w - pad * 2;
    const ch = h - pad * 2;
    ctx.strokeStyle = '#1F2D45';
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(pad, h - pad);
    ctx.lineTo(w - pad, pad);
    ctx.stroke();
    ctx.setLineDash([]);

    const pts: [number, number][] = data.ml_diagnostics.calibration_points || [];
    if (pts.length >= 2) {
      ctx.strokeStyle = '#3B82F6';
      ctx.lineWidth = 2;
      ctx.beginPath();
      pts.forEach(([x, y], i) => {
        const px = pad + x * cw;
        const py = h - pad - y * ch;
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });
      ctx.stroke();
    }

    const cut = data.ml_min_probability || 0.56;
    ctx.strokeStyle = '#F59E0B';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(pad + cut * cw, pad);
    ctx.lineTo(pad + cut * cw, h - pad);
    ctx.stroke();
  }, [data]);

  const strats = ['liquidity_sweep', 'micro_breakout', 'ema_vwap_pullback', 'tick_imbalance', 'structure_shift'];
  const stratLabels: Record<string, string> = {
    liquidity_sweep: 'LiquiditySweep',
    micro_breakout: 'MicroBreakout',
    ema_vwap_pullback: 'EmaVwapPull',
    tick_imbalance: 'TickImbalance',
    structure_shift: 'StructureShift',
  };

  return (
    <div className="space-y-3">
      <div className="cockpit-card rounded-lg px-4 py-3 flex flex-wrap items-center justify-between gap-2">
        <div>
          <h2 className="text-sm font-bold">4-Pillar Hybrid Brain Inspector (DL + ML + Dual RL)</h2>
          <p className="text-xs text-[#8A9BB5]">Live Thompson Sampling Bandit weights, NumPy 1D-CNN Sequence Net, Calibrated Meta-Labeler & RL-2 Q-Table</p>
        </div>
        <select
          value={sym}
          onChange={(e) => setSym(e.target.value)}
          className="px-2.5 py-1.5 rounded-md cockpit-sub mono text-xs font-bold cursor-pointer"
        >
          {symbols.map((s) => (
            <option key={s} value={s}>{s}</option>
          ))}
        </select>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-12 gap-3">
        <div className="xl:col-span-7 cockpit-card rounded-lg p-3.5 space-y-2.5">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-bold uppercase tracking-wider">RL-1: Contextual Thompson Sampling Bandit ({sym})</h3>
            <span className="px-2 py-0.5 rounded bg-[#10B981]/15 text-[#10B981] border border-[#10B981]/40 mono text-xs font-bold">ONLINE ADAPT</span>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-xs text-left border-collapse">
              <thead>
                <tr className="border-b border-[#1F2D45] text-[#8A9BB5]">
                  <th className="py-1.5 pr-2">Regime</th>
                  {strats.map((s) => (
                    <th key={s} className="py-1.5 px-2 text-center">{stratLabels[s]}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-[#1F2D45] mono">
                {(data?.bandit_matrix || []).map((row: any) => (
                  <tr key={row.regime}>
                    <td className="py-2 pr-2 font-sans font-semibold">{human(row.regime)}</td>
                    {strats.map((s) => {
                      const pct = Math.round((row.weights?.[s] ?? 0.25) * 100);
                      return (
                        <td key={s} className="py-1.5 px-2 text-center">
                          <div
                            className="py-1 rounded font-bold"
                            style={{ backgroundColor: `rgba(59, 130, 246, ${Math.min(0.45, pct / 100)})` }}
                          >
                            {pct}%
                          </div>
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="xl:col-span-5 cockpit-card rounded-lg p-3.5 space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-bold uppercase tracking-wider">DL Pillar: NumPy 1D-CNN Sequence Net</h3>
            <span className="px-2 py-0.5 rounded bg-[#06B6D4]/15 text-[#06B6D4] border border-[#06B6D4]/40 mono text-xs font-bold">
              {data?.dl_trained ? `Val Acc ${Math.round((data.dl_val_acc || 0) * 100)}%` : 'WARMUP'}
            </span>
          </div>
          <div className="grid grid-cols-4 gap-2 mono text-xs text-center">
            <div className="p-2.5 rounded cockpit-sub">
              <div className="font-sans text-[#8A9BB5]">P(Down)</div>
              <div className="text-sm font-bold text-[#EF4444] mt-0.5">{Math.round((data?.dl_prediction?.prob_down ?? 0.33) * 100)}%</div>
            </div>
            <div className="p-2.5 rounded cockpit-sub">
              <div className="font-sans text-[#8A9BB5]">P(Flat)</div>
              <div className="text-sm font-bold mt-0.5">{Math.round((data?.dl_prediction?.prob_flat ?? 0.34) * 100)}%</div>
            </div>
            <div className="p-2.5 rounded cockpit-sub">
              <div className="font-sans text-[#8A9BB5]">P(Up)</div>
              <div className="text-sm font-bold text-[#10B981] mt-0.5">{Math.round((data?.dl_prediction?.prob_up ?? 0.33) * 100)}%</div>
            </div>
            <div className="p-2.5 rounded cockpit-sub">
              <div className="font-sans text-[#8A9BB5]">Vol Exp</div>
              <div className="text-sm font-bold text-[#06B6D4] mt-0.5">{Math.round((data?.dl_prediction?.vol_expansion ?? 0.5) * 100)}%</div>
            </div>
          </div>
          <div className="p-3 rounded cockpit-sub text-xs text-[#8A9BB5] leading-relaxed">
            Strictly causal 1D-CNN encoder (32 closed M1 bars × 8 microstructure channels) with dilated residual block, AdamW cosine schedule, and sub-millisecond CPU inference. Output feeds directly into the calibrated ML Meta-Labeler.
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-12 gap-3">
        <div className="xl:col-span-6 cockpit-card rounded-lg p-3.5 space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-bold uppercase tracking-wider">ML Pillar: Calibrated HistGradientBoosting</h3>
            <span className="mono text-xs text-[#8A9BB5]">
              Brier: <b className="text-white">{data?.ml_diagnostics?.brier_score ?? '0.25'}</b> • Cutoff: <b className="text-[#F59E0B]">{data?.ml_min_probability ?? 0.56}</b>
            </span>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <canvas ref={calibRef} className="w-full rounded cockpit-sub" style={{ height: 165 }} />
            <div className="space-y-2 text-xs">
              {(data?.ml_diagnostics?.feature_importances || []).slice(0, 6).map((f: any) => (
                <div key={f.name} className="space-y-0.5">
                  <div className="flex justify-between mono">
                    <span>{human(f.name)}</span>
                    <b>{Math.round(f.importance * 100)}%</b>
                  </div>
                  <div className="h-1.5 rounded-full bg-[#070B12] overflow-hidden">
                    <div className="h-full bg-[#06B6D4]" style={{ width: `${Math.min(100, f.importance * 360)}%` }} />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="xl:col-span-6 cockpit-card rounded-lg p-3.5 space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-bold uppercase tracking-wider">RL-2: Q-Learning Dynamic Exit Policy</h3>
            <span className="px-2 py-0.5 rounded bg-[#10B981]/15 text-[#10B981] border border-[#10B981]/40 mono text-xs font-bold">
              MFE Capture {data?.rl_mfe_capture_pct ?? 68.5}%
            </span>
          </div>
          <table className="w-full text-xs text-left border-collapse mono">
            <thead>
              <tr className="border-b border-[#1F2D45] text-[#8A9BB5] font-sans">
                <th className="py-1.5 pr-2">Unrealized R</th>
                <th className="py-1.5 px-2 text-center">1-3 Bars</th>
                <th className="py-1.5 px-2 text-center">4-8 Bars</th>
                <th className="py-1.5 px-2 text-center">9+ Bars</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#1F2D45]">
              {(data?.rl_exit_grid || []).map((r: any) => (
                <tr key={r.r_bucket}>
                  <td className="py-2 pr-2 font-sans font-semibold">{r.r_bucket}</td>
                  {(r.bars || []).map((act: string, idx: number) => (
                    <td key={idx} className="py-2 px-2 text-center">
                      <span className="px-2 py-0.5 rounded cockpit-sub font-bold text-[#06B6D4]">{act}</span>
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
