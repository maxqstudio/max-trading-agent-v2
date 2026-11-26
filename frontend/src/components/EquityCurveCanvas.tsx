import React, { useEffect, useRef, useState } from 'react';
import type { EquityPoint } from '../types/trading';

interface Props {
  points: EquityPoint[];
  championVersion: string;
}

export const EquityCurveCanvas: React.FC<Props> = ({ points, championVersion }) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [hoverInfo, setHoverInfo] = useState<string>('Hover chart to inspect Champion vs Shadow');

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const data =
      points.length >= 2
        ? points
        : [
            { timestamp_ms: 0, champion_equity: 10000, shadow_equity: 10000, daily_dd_pct: 0, event_label: '' },
            { timestamp_ms: 1, champion_equity: 10000, shadow_equity: 10000, daily_dd_pct: 0, event_label: '' },
          ];

    const dpr = window.devicePixelRatio || 1;
    const w = canvas.offsetWidth || 640;
    const h = 210;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);

    const all = data.flatMap((d) => [d.champion_equity, d.shadow_equity]);
    const minV = Math.min(...all) - 20;
    const maxV = Math.max(...all) + 20;

    const padL = 58;
    const padR = 14;
    const padT = 14;
    const padB = 20;
    const cw = w - padL - padR;
    const ch = h - padT - padB;
    const xOf = (i: number) => padL + (i / Math.max(data.length - 1, 1)) * cw;
    const yOf = (v: number) => padT + ((maxV - v) / Math.max(maxV - minV, 1)) * ch;

    ctx.font = '11px "JetBrains Mono", monospace';
    for (let i = 0; i <= 3; i++) {
      const y = padT + (ch / 3) * i;
      ctx.strokeStyle = '#1F2D45';
      ctx.setLineDash([2, 3]);
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(w - padR, y);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = '#8A9BB5';
      ctx.fillText('$' + (maxV - ((maxV - minV) / 3) * i).toFixed(0), 6, y + 4);
    }

    // Shadow Challenger curve
    ctx.strokeStyle = '#8B5CF6';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 3]);
    ctx.beginPath();
    data.forEach((p, i) => (i === 0 ? ctx.moveTo(xOf(i), yOf(p.shadow_equity)) : ctx.lineTo(xOf(i), yOf(p.shadow_equity))));
    ctx.stroke();
    ctx.setLineDash([]);

    // Champion curve + subtle gradient fill
    ctx.strokeStyle = '#10B981';
    ctx.lineWidth = 2.4;
    ctx.beginPath();
    data.forEach((p, i) => (i === 0 ? ctx.moveTo(xOf(i), yOf(p.champion_equity)) : ctx.lineTo(xOf(i), yOf(p.champion_equity))));
    ctx.stroke();
    ctx.lineTo(xOf(data.length - 1), padT + ch);
    ctx.lineTo(xOf(0), padT + ch);
    ctx.closePath();
    ctx.fillStyle = 'rgba(16, 185, 129, 0.10)';
    ctx.fill();
  }, [points]);

  return (
    <div className="space-y-1.5">
      <div className="flex flex-wrap items-center gap-4 text-xs">
        <span className="font-bold">Equity Curve Comparison</span>
        <span className="flex items-center gap-1.5 text-[#8A9BB5]">
          <span className="w-3 h-0.5 bg-[#10B981] inline-block" /> Champion ({championVersion})
        </span>
        <span className="flex items-center gap-1.5 text-[#8A9BB5]">
          <span className="w-3 h-0.5 bg-[#8B5CF6] inline-block" /> Shadow Challenger
        </span>
        <span className="ml-auto mono text-[#8A9BB5]">{hoverInfo}</span>
      </div>
      <canvas
        ref={canvasRef}
        className="w-full rounded-md cockpit-sub"
        style={{ height: 210 }}
        onMouseMove={(e) => {
          if (points.length < 2) return;
          const rect = e.currentTarget.getBoundingClientRect();
          const idx = Math.max(0, Math.min(points.length - 1, Math.round(((e.clientX - rect.left - 58) / Math.max(rect.width - 72, 1)) * (points.length - 1))));
          const pt = points[idx];
          setHoverInfo(`Champ: $${pt.champion_equity.toFixed(2)} | Shadow: $${pt.shadow_equity.toFixed(2)} | DD: ${pt.daily_dd_pct.toFixed(2)}%`);
        }}
      />
    </div>
  );
};
