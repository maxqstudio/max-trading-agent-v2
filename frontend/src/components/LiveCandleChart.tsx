import React, { useEffect, useRef, useState } from 'react';
import type { LivePosition, RadarSymbol } from '../types/trading';

interface Props {
  symbolData: RadarSymbol | undefined;
  position: LivePosition | undefined;
}

export const LiveCandleChart: React.FC<Props> = ({ symbolData, position }) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [showEma, setShowEma] = useState(true);
  const [showZones, setShowZones] = useState(true);
  const [showOrders, setShowOrders] = useState(true);
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !symbolData) return;
    const bars = symbolData.bars.slice(-52);
    if (bars.length < 2) return;

    const dpr = window.devicePixelRatio || 1;
    const w = canvas.offsetWidth || 680;
    const h = 320;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);

    const digs = symbolData.digits;
    let hi = Math.max(...bars.map((b) => b.high), symbolData.bid || 0);
    let lo = Math.min(...bars.map((b) => b.low), symbolData.bid || Infinity);
    if (showOrders && position) {
      if (position.tp > 0) {
        hi = Math.max(hi, position.tp);
        lo = Math.min(lo, position.tp);
      }
      if (position.sl > 0) {
        hi = Math.max(hi, position.sl);
        lo = Math.min(lo, position.sl);
      }
      hi = Math.max(hi, position.open_price);
      lo = Math.min(lo, position.open_price);
    }
    const padRange = Math.max((hi - lo) * 0.14, symbolData.point * 18);
    hi += padRange;
    lo -= padRange;

    const padL = 10;
    const padR = 76;
    const padT = 34;
    const padB = 22;
    const cw = w - padL - padR;
    const ch = h - padT - padB;
    const yOf = (p: number) => padT + ((hi - p) / Math.max(hi - lo, 1e-9)) * ch;
    const stepX = cw / bars.length;

    // Hairline price grid
    ctx.font = '11px "JetBrains Mono", ui-monospace, monospace';
    for (let i = 0; i <= 4; i++) {
      const y = padT + (ch / 4) * i;
      ctx.strokeStyle = '#1F2D45';
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 4]);
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(w - padR, y);
      ctx.stroke();
      ctx.setLineDash([]);
      const pVal = hi - ((hi - lo) / 4) * i;
      ctx.fillStyle = '#8A9BB5';
      ctx.fillText(pVal.toFixed(digs), w - padR + 6, y + 4);
    }

    // AI Liquidity Sweep / Micro S/R Zone
    if (showZones) {
      const recentLow = Math.min(...bars.slice(-20).map((b) => b.low));
      const zTop = yOf(recentLow + (hi - lo) * 0.07);
      const zBot = yOf(recentLow);
      ctx.fillStyle = 'rgba(6, 182, 212, 0.10)';
      ctx.fillRect(padL + cw * 0.35, zTop, cw * 0.65, Math.max(12, zBot - zTop));
      ctx.fillStyle = '#06B6D4';
      ctx.fillText('AI LIQUIDITY SWEEP ZONE', padL + cw * 0.37, zTop + 11);
    }

    // EMA9 & Session VWAP
    if (showEma) {
      let e9 = bars[0].close;
      const k = 2 / 10;
      ctx.strokeStyle = '#F59E0B';
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      bars.forEach((b, i) => {
        e9 = b.close * k + e9 * (1 - k);
        const x = padL + i * stepX + stepX / 2;
        if (i === 0) ctx.moveTo(x, yOf(e9));
        else ctx.lineTo(x, yOf(e9));
      });
      ctx.stroke();

      let cumPv = 0;
      let cumV = 0;
      ctx.strokeStyle = '#8B5CF6';
      ctx.lineWidth = 1.4;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      bars.forEach((b, i) => {
        const vol = Math.max(b.volume, 1);
        cumPv += ((b.high + b.low + b.close) / 3) * vol;
        cumV += vol;
        const x = padL + i * stepX + stepX / 2;
        if (i === 0) ctx.moveTo(x, yOf(cumPv / cumV));
        else ctx.lineTo(x, yOf(cumPv / cumV));
      });
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Candlesticks
    bars.forEach((b, i) => {
      const x = padL + i * stepX + stepX / 2;
      const bull = b.close >= b.open;
      const col = bull ? '#10B981' : '#EF4444';
      ctx.strokeStyle = col;
      ctx.fillStyle = col;
      ctx.lineWidth = 1.25;
      ctx.beginPath();
      ctx.moveTo(x, yOf(b.high));
      ctx.lineTo(x, yOf(b.low));
      ctx.stroke();

      const top = yOf(Math.max(b.open, b.close));
      const bot = yOf(Math.min(b.open, b.close));
      const bw = Math.max(3, stepX * 0.62);
      ctx.fillRect(x - bw / 2, top, bw, Math.max(2, bot - top));
    });

    // Live Position Entry / SL / TP Rails
    if (showOrders && position) {
      const drawRail = (px: number, label: string, color: string) => {
        if (px <= 0) return;
        const y = yOf(px);
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.3;
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.moveTo(padL, y);
        ctx.lineTo(w - padR, y);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = color;
        ctx.fillRect(w - padR + 2, y - 8, padR - 4, 16);
        ctx.fillStyle = '#FFFFFF';
        ctx.fillText(label, w - padR + 5, y + 4);
      };
      drawRail(position.tp, 'TP ' + position.tp.toFixed(digs > 3 ? 4 : digs), '#10B981');
      drawRail(position.open_price, `${position.side} ${position.volume}`, '#3B82F6');
      drawRail(
        position.sl,
        (position.rl_exit_action === 'MOVE_BE' ? 'BE ' : 'SL ') + position.sl.toFixed(digs > 3 ? 4 : digs),
        position.rl_exit_action === 'MOVE_BE' ? '#06B6D4' : '#EF4444'
      );
    }

    // Live Bid price rail
    if (symbolData.bid > 0) {
      const bidY = yOf(symbolData.bid);
      ctx.strokeStyle = '#3B82F6';
      ctx.lineWidth = 1.1;
      ctx.setLineDash([2, 2]);
      ctx.beginPath();
      ctx.moveTo(padL, bidY);
      ctx.lineTo(w - padR, bidY);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = '#3B82F6';
      ctx.fillRect(w - padR + 2, bidY - 9, padR - 4, 18);
      ctx.fillStyle = '#FFFFFF';
      ctx.fillText(symbolData.bid.toFixed(digs), w - padR + 5, bidY + 4);
    }

    // Hover crosshair
    if (hoverIdx !== null && bars[hoverIdx]) {
      const x = padL + hoverIdx * stepX + stepX / 2;
      ctx.strokeStyle = 'rgba(230,237,247,0.35)';
      ctx.setLineDash([2, 2]);
      ctx.beginPath();
      ctx.moveTo(x, padT);
      ctx.lineTo(x, h - padB);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }, [symbolData, position, showEma, showZones, showOrders, hoverIdx]);

  const bars = symbolData?.bars.slice(-52) || [];
  const activeBar = hoverIdx !== null && bars[hoverIdx] ? bars[hoverIdx] : bars[bars.length - 1];
  const digs = symbolData?.digits ?? 2;

  return (
    <div className="flex flex-col">
      <div className="px-3 py-2 border-b border-[#1F2D45] flex flex-wrap items-center justify-between gap-2 text-xs">
        <div className="flex items-center gap-2 mono">
          <span className="px-2 py-0.5 rounded cockpit-sub font-bold">M1 LIVE</span>
          <span className="font-bold text-sm text-[#10B981]">
            {symbolData?.bid ? symbolData.bid.toFixed(digs) : '---'}
          </span>
          <span className="text-[#8A9BB5]">
            Sp {symbolData?.spread_points?.toFixed(1) ?? '0'}pt ({symbolData?.spread_atr_pct ?? 0}% ATR)
          </span>
        </div>
        <div className="flex items-center gap-3 text-xs text-[#8A9BB5]">
          <label className="flex items-center gap-1.5 cursor-pointer select-none">
            <input type="checkbox" checked={showEma} onChange={(e) => setShowEma(e.target.checked)} className="accent-[#3B82F6]" />
            <span>EMA / VWAP</span>
          </label>
          <label className="flex items-center gap-1.5 cursor-pointer select-none">
            <input type="checkbox" checked={showZones} onChange={(e) => setShowZones(e.target.checked)} className="accent-[#3B82F6]" />
            <span>AI Zones</span>
          </label>
          <label className="flex items-center gap-1.5 cursor-pointer select-none">
            <input type="checkbox" checked={showOrders} onChange={(e) => setShowOrders(e.target.checked)} className="accent-[#3B82F6]" />
            <span>SL/TP Rails</span>
          </label>
        </div>
      </div>

      <div className="relative bg-[#0B111C]">
        {activeBar && symbolData && (
          <div className="absolute top-2 left-2.5 z-10 pointer-events-none flex flex-wrap items-center gap-2 text-xs mono">
            <span className="px-2 py-0.5 rounded bg-[#0E1522]/90 border border-[#1F2D45]">
              O {activeBar.open.toFixed(digs)} H {activeBar.high.toFixed(digs)} L {activeBar.low.toFixed(digs)} C {activeBar.close.toFixed(digs)}
            </span>
            <span className="px-2 py-0.5 rounded bg-[#10B981]/15 text-[#10B981] border border-[#10B981]/40 font-bold">
              {symbolData.regime}
            </span>
            <span className="px-2 py-0.5 rounded bg-[#06B6D4]/15 text-[#06B6D4] border border-[#06B6D4]/40 font-bold">
              DL {symbolData.dl.prob_up >= symbolData.dl.prob_down ? `${Math.round(symbolData.dl.prob_up * 100)}% UP` : `${Math.round(symbolData.dl.prob_down * 100)}% DN`} • VOL {Math.round(symbolData.dl.vol_expansion * 100)}%
            </span>
          </div>
        )}
        <canvas
          ref={canvasRef}
          className="w-full block cursor-crosshair"
          style={{ height: 320 }}
          onMouseMove={(e) => {
            const rect = e.currentTarget.getBoundingClientRect();
            const rel = (e.clientX - rect.left - 10) / Math.max(rect.width - 86, 1);
            const idx = Math.max(0, Math.min(bars.length - 1, Math.floor(rel * bars.length)));
            setHoverIdx(idx);
          }}
          onMouseLeave={() => setHoverIdx(null)}
        />
      </div>
    </div>
  );
};
