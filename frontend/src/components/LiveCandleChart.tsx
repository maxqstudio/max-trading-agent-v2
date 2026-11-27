import React, { useEffect, useRef, useState } from 'react';
import type { Bar, LivePosition, RadarSymbol } from '../types/trading';
import { human } from '../labels';

interface Props {
  symbolData: RadarSymbol | undefined;
  position: LivePosition | undefined;
}

const VISIBLE_BARS = 52;
const FUTURE_CLOUD_SLOTS = 8;

function midpoint(bars: Bar[], endIdx: number, period: number): number | null {
  if (endIdx + 1 < period) return null;
  let hi = -Infinity;
  let lo = Infinity;
  for (let j = endIdx - period + 1; j <= endIdx; j++) {
    if (bars[j].high > hi) hi = bars[j].high;
    if (bars[j].low < lo) lo = bars[j].low;
  }
  return (hi + lo) / 2;
}

export const LiveCandleChart: React.FC<Props> = ({ symbolData, position }) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [showEma, setShowEma] = useState(true);
  const [showIchimoku, setShowIchimoku] = useState(true);
  const [showZones, setShowZones] = useState(true);
  const [showOrders, setShowOrders] = useState(true);
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !symbolData) return;
    const allBars = symbolData.bars;
    const bars = allBars.slice(-VISIBLE_BARS);
    if (bars.length < 2) return;
    const startIdx = allBars.length - bars.length;

    // Precompute Ichimoku (9, 26, 52) on full history so the cloud & Kijun are accurate across all visible bars
    const tenkan: (number | null)[] = [];
    const kijun: (number | null)[] = [];
    const spanA: (number | null)[] = [];
    const spanB: (number | null)[] = [];
    if (showIchimoku) {
      const totalSlots = bars.length + FUTURE_CLOUD_SLOTS;
      for (let slot = 0; slot < totalSlots; slot++) {
        const curAbs = startIdx + slot;
        if (slot < bars.length) {
          tenkan.push(midpoint(allBars, curAbs, 9));
          kijun.push(midpoint(allBars, curAbs, 26));
        }
        // Senkou Spans plotted at curAbs are computed 26 bars earlier (standard +26 forward shift)
        const srcAbs = curAbs - 26;
        if (srcAbs >= 0 && srcAbs < allBars.length) {
          const tSrc = midpoint(allBars, srcAbs, 9);
          const kSrc = midpoint(allBars, srcAbs, 26);
          spanA.push(tSrc !== null && kSrc !== null ? (tSrc + kSrc) / 2 : null);
          spanB.push(midpoint(allBars, srcAbs, 52));
        } else {
          spanA.push(null);
          spanB.push(null);
        }
      }
    }

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
    if (showIchimoku) {
      for (const v of [...spanA, ...spanB, ...kijun]) {
        if (v !== null) {
          hi = Math.max(hi, v);
          lo = Math.min(lo, v);
        }
      }
    }
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
    const totalSlots = bars.length + (showIchimoku ? FUTURE_CLOUD_SLOTS : 0);
    const stepX = cw / totalSlots;
    const xOf = (slot: number) => padL + slot * stepX + stepX / 2;

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

    // Ichimoku Kumo Cloud (Senkou Span A/B) + Tenkan (9) + Kijun (26)
    if (showIchimoku) {
      // Shaded Kumo segments (green when Span A >= Span B, red when bearish)
      for (let i = 0; i < spanA.length - 1; i++) {
        const a0 = spanA[i];
        const b0 = spanB[i];
        const a1 = spanA[i + 1];
        const b1 = spanB[i + 1];
        if (a0 === null || b0 === null || a1 === null || b1 === null) continue;
        const bullCloud = (a0 + a1) >= (b0 + b1);
        ctx.fillStyle = bullCloud ? 'rgba(16, 185, 129, 0.14)' : 'rgba(239, 68, 68, 0.14)';
        ctx.beginPath();
        ctx.moveTo(xOf(i), yOf(a0));
        ctx.lineTo(xOf(i + 1), yOf(a1));
        ctx.lineTo(xOf(i + 1), yOf(b1));
        ctx.lineTo(xOf(i), yOf(b0));
        ctx.closePath();
        ctx.fill();
      }

      const drawLine = (vals: (number | null)[], color: string, width: number, dash: number[] = []) => {
        ctx.strokeStyle = color;
        ctx.lineWidth = width;
        ctx.setLineDash(dash);
        ctx.beginPath();
        let started = false;
        vals.forEach((v, idx) => {
          if (v === null) {
            started = false;
            return;
          }
          const x = xOf(idx);
          const y = yOf(v);
          if (!started) {
            ctx.moveTo(x, y);
            started = true;
          } else {
            ctx.lineTo(x, y);
          }
        });
        ctx.stroke();
        ctx.setLineDash([]);
      };

      drawLine(spanA, 'rgba(16, 185, 129, 0.55)', 1.0);
      drawLine(spanB, 'rgba(239, 68, 68, 0.55)', 1.0);
      drawLine(tenkan, '#38BDF8', 1.15);
      drawLine(kijun, '#FB7185', 1.25);
    }

    // AI Liquidity Sweep / Micro S/R Zone
    if (showZones) {
      const recentLow = Math.min(...bars.slice(-20).map((b) => b.low));
      const zTop = yOf(recentLow + (hi - lo) * 0.07);
      const zBot = yOf(recentLow);
      const zoneW = stepX * bars.length;
      ctx.fillStyle = 'rgba(6, 182, 212, 0.10)';
      ctx.fillRect(padL + zoneW * 0.35, zTop, zoneW * 0.65, Math.max(12, zBot - zTop));
      ctx.fillStyle = '#06B6D4';
      ctx.fillText('AI LIQUIDITY SWEEP ZONE', padL + zoneW * 0.37, zTop + 11);
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
        const x = xOf(i);
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
        const x = xOf(i);
        if (i === 0) ctx.moveTo(x, yOf(cumPv / cumV));
        else ctx.lineTo(x, yOf(cumPv / cumV));
      });
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Candlesticks
    bars.forEach((b, i) => {
      const x = xOf(i);
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
      const x = xOf(hoverIdx);
      ctx.strokeStyle = 'rgba(230,237,247,0.35)';
      ctx.setLineDash([2, 2]);
      ctx.beginPath();
      ctx.moveTo(x, padT);
      ctx.lineTo(x, h - padB);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }, [symbolData, position, showEma, showIchimoku, showZones, showOrders, hoverIdx]);

  const bars = symbolData?.bars.slice(-VISIBLE_BARS) || [];
  const activeBar = hoverIdx !== null && bars[hoverIdx] ? bars[hoverIdx] : bars[bars.length - 1];
  const digs = symbolData?.digits ?? 2;
  const ichiM15 = symbolData?.ichi_m15 ?? 0;

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
        <div className="flex flex-wrap items-center gap-3 text-xs text-[#8A9BB5]">
          <label className="flex items-center gap-1.5 cursor-pointer select-none" title="Tenkan (9), Kijun (26), and Kumo Cloud (52)">
            <input type="checkbox" checked={showIchimoku} onChange={(e) => setShowIchimoku(e.target.checked)} className="accent-[#38BDF8]" />
            <span>Ichimoku</span>
          </label>
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
              {human(symbolData.regime)}
            </span>
            <span className="px-2 py-0.5 rounded bg-[#06B6D4]/15 text-[#06B6D4] border border-[#06B6D4]/40 font-bold">
              DL {symbolData.dl.prob_up >= symbolData.dl.prob_down ? `${Math.round(symbolData.dl.prob_up * 100)}% UP` : `${Math.round(symbolData.dl.prob_down * 100)}% DN`} • VOL {Math.round(symbolData.dl.vol_expansion * 100)}%
            </span>
            <span
              className={`px-2 py-0.5 rounded border font-bold ${
                ichiM15 >= 2
                  ? 'bg-[#10B981]/15 text-[#10B981] border-[#10B981]/40'
                  : ichiM15 <= -2
                    ? 'bg-[#EF4444]/15 text-[#EF4444] border-[#EF4444]/40'
                    : 'bg-[#0E1522]/90 text-[#8A9BB5] border-[#1F2D45]'
              }`}
              title="M15 Ichimoku 5-point checklist score (-5 to +5)"
            >
              ICHI M15 {ichiM15 > 0 ? `+${ichiM15}` : ichiM15}
            </span>
          </div>
        )}
        <canvas
          ref={canvasRef}
          className="w-full block cursor-crosshair"
          style={{ height: 320 }}
          onMouseMove={(e) => {
            const rect = e.currentTarget.getBoundingClientRect();
            const slots = bars.length + (showIchimoku ? FUTURE_CLOUD_SLOTS : 0);
            const rel = (e.clientX - rect.left - 10) / Math.max(rect.width - 86, 1);
            const idx = Math.max(0, Math.min(bars.length - 1, Math.floor(rel * slots)));
            setHoverIdx(idx);
          }}
          onMouseLeave={() => setHoverIdx(null)}
        />
      </div>
    </div>
  );
};
