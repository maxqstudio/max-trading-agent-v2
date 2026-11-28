import React, { useEffect, useRef, useState } from 'react';
import type { Bar, LivePosition, PaperPosition, RadarSymbol } from '../types/trading';
import { human } from '../labels';

export interface ChartPosition {
  symbol: string;
  side: 'BUY' | 'SELL';
  volume: number;
  open_price: number;
  sl: number;
  tp: number;
  profit: number;
  label: string;
  is_be?: boolean;
}

interface Props {
  symbolData: RadarSymbol | undefined;
  position: ChartPosition | undefined;
}

const VISIBLE_BARS = 60;
const FUTURE_CLOUD_SLOTS = 6;

export function toChartPosition(
  mt5Pos: LivePosition | undefined,
  paperPos: PaperPosition | undefined
): ChartPosition | undefined {
  if (mt5Pos) {
    return {
      symbol: mt5Pos.symbol,
      side: mt5Pos.side,
      volume: mt5Pos.volume,
      open_price: mt5Pos.open_price,
      sl: mt5Pos.sl,
      tp: mt5Pos.tp,
      profit: mt5Pos.profit,
      label: 'MT5',
      is_be: mt5Pos.rl_exit_action === 'MOVE_BE',
    };
  }
  if (paperPos) {
    const bookTag =
      paperPos.book === 'hitrun'
        ? 'PAPER H&R'
        : paperPos.book === 'raw_hitrun'
          ? 'NO VETO H&R'
          : paperPos.book === 'raw'
            ? 'NO VETO'
            : 'PAPER';
    return {
      symbol: paperPos.symbol,
      side: paperPos.side,
      volume: paperPos.volume,
      open_price: paperPos.entry_price,
      sl: paperPos.sl,
      tp: paperPos.tp,
      profit: paperPos.profit,
      label: bookTag,
      is_be: false,
    };
  }
  return undefined;
}

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

function fmtTimeHHMM(tsSec: number): string {
  const d = new Date(tsSec * 1000);
  const hh = String(d.getUTCHours()).padStart(2, '0');
  const mm = String(d.getUTCMinutes()).padStart(2, '0');
  return `${hh}:${mm}`;
}

export const LiveCandleChart: React.FC<Props> = ({ symbolData, position }) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [chartWidth, setChartWidth] = useState(680);
  const [showEma, setShowEma] = useState(true);
  const [showIchimoku, setShowIchimoku] = useState(true);
  const [showZones, setShowZones] = useState(false);
  const [showOrders, setShowOrders] = useState(true);
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);
  const [hoverY, setHoverY] = useState<number | null>(null);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const update = () => {
      if (el.clientWidth > 100) setChartWidth(el.clientWidth);
    };
    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Build visible bars + live forming M1 candle connecting last closed bar to current Bid
  const allBars = symbolData?.bars || [];
  const closedVisible = allBars.slice(-VISIBLE_BARS);
  const lastClosed = closedVisible[closedVisible.length - 1];
  const hasLiveCandle = Boolean(
    symbolData &&
      lastClosed &&
      symbolData.bid > 0 &&
      Math.abs(symbolData.bid - lastClosed.close) >= symbolData.point * 0.5
  );
  const liveBar: Bar | null =
    hasLiveCandle && symbolData && lastClosed
      ? {
          symbol: symbolData.symbol,
          timeframe: 'M1',
          timestamp: lastClosed.timestamp + 60,
          open: lastClosed.close,
          high: Math.max(lastClosed.close, symbolData.bid),
          low: Math.min(lastClosed.close, symbolData.bid),
          close: symbolData.bid,
          volume: 0,
          spread_points: symbolData.spread_points,
        }
      : null;
  const displayBars = liveBar ? [...closedVisible, liveBar] : closedVisible;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !symbolData || displayBars.length < 2) return;

    const fullBars = liveBar ? [...allBars, liveBar] : allBars;
    const startIdx = fullBars.length - displayBars.length;

    // Precompute Ichimoku (9, 26, 52) on full history
    const tenkan: (number | null)[] = [];
    const kijun: (number | null)[] = [];
    const spanA: (number | null)[] = [];
    const spanB: (number | null)[] = [];
    const futureSlots = showIchimoku ? FUTURE_CLOUD_SLOTS : 2;
    if (showIchimoku) {
      const totalSlots = displayBars.length + futureSlots;
      for (let slot = 0; slot < totalSlots; slot++) {
        const curAbs = startIdx + slot;
        if (slot < displayBars.length) {
          tenkan.push(midpoint(fullBars, curAbs, 9));
          kijun.push(midpoint(fullBars, curAbs, 26));
        }
        const srcAbs = curAbs - 26;
        if (srcAbs >= 0 && srcAbs < fullBars.length) {
          const tSrc = midpoint(fullBars, srcAbs, 9);
          const kSrc = midpoint(fullBars, srcAbs, 26);
          spanA.push(tSrc !== null && kSrc !== null ? (tSrc + kSrc) / 2 : null);
          spanB.push(midpoint(fullBars, srcAbs, 52));
        } else {
          spanA.push(null);
          spanB.push(null);
        }
      }
    }

    const dpr = window.devicePixelRatio || 1;
    const w = chartWidth;
    const h = 340;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const digs = symbolData.digits;
    let hi = Math.max(...displayBars.map((b) => b.high), symbolData.bid || 0);
    let lo = Math.min(...displayBars.map((b) => b.low), symbolData.bid > 0 ? symbolData.bid : Infinity);

    // Include Ichimoku only softly if within reasonable range so clouds never squash candles
    if (showIchimoku) {
      const candleRange = Math.max(hi - lo, symbolData.point * 25);
      for (const v of [...spanA, ...spanB, ...kijun]) {
        if (v !== null && v <= hi + candleRange * 0.6 && v >= lo - candleRange * 0.6) {
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
      if (position.open_price > 0) {
        hi = Math.max(hi, position.open_price);
        lo = Math.min(lo, position.open_price);
      }
    }
    const padRange = Math.max((hi - lo) * 0.12, symbolData.point * 15);
    hi += padRange;
    lo -= padRange;

    const padL = 10;
    const padR = 86;
    const padT = 36;
    const padB = 26;
    const cw = w - padL - padR;
    const ch = h - padT - padB;
    const yOf = (p: number) => padT + ((hi - p) / Math.max(hi - lo, 1e-9)) * ch;
    const pOf = (y: number) => hi - ((y - padT) / Math.max(ch, 1)) * (hi - lo);
    const totalSlots = displayBars.length + futureSlots;
    const stepX = cw / Math.max(totalSlots, 1);
    const xOf = (slot: number) => padL + slot * stepX + stepX / 2;

    // 1. Hairline Price Grid + Right Axis Labels
    ctx.font = '11px "JetBrains Mono", ui-monospace, monospace';
    for (let i = 0; i <= 5; i++) {
      const y = Math.round(padT + (ch / 5) * i) + 0.5;
      ctx.strokeStyle = 'rgba(31, 45, 69, 0.75)';
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 4]);
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(w - padR, y);
      ctx.stroke();
      ctx.setLineDash([]);
      const pVal = hi - ((hi - lo) / 5) * i;
      ctx.fillStyle = '#8A9BB5';
      ctx.fillText(pVal.toFixed(digs), w - padR + 8, y + 4);
    }

    // Vertical time grid + X-axis timestamps (HH:MM UTC)
    const tickEvery = Math.max(6, Math.round(displayBars.length / 7));
    for (let i = 0; i < displayBars.length; i += tickEvery) {
      const x = Math.round(xOf(i)) + 0.5;
      ctx.strokeStyle = 'rgba(31, 45, 69, 0.45)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, padT);
      ctx.lineTo(x, h - padB);
      ctx.stroke();
      ctx.fillStyle = '#64748B';
      ctx.fillText(fmtTimeHHMM(displayBars[i].timestamp), x - 16, h - 8);
    }

    // Axis separator lines
    ctx.strokeStyle = '#1F2D45';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(w - padR + 0.5, padT);
    ctx.lineTo(w - padR + 0.5, h - padB);
    ctx.moveTo(padL, h - padB + 0.5);
    ctx.lineTo(w - padR, h - padB + 0.5);
    ctx.stroke();

    // 2. TradingView-style Position Risk/Reward Shaded Zones (behind candles)
    if (showOrders && position && position.open_price > 0) {
      const yOpen = yOf(position.open_price);
      if (position.tp > 0) {
        const yTp = yOf(position.tp);
        ctx.fillStyle = 'rgba(16, 185, 129, 0.08)';
        ctx.fillRect(padL, Math.min(yOpen, yTp), cw, Math.abs(yOpen - yTp));
      }
      if (position.sl > 0) {
        const ySl = yOf(position.sl);
        ctx.fillStyle = 'rgba(239, 68, 68, 0.08)';
        ctx.fillRect(padL, Math.min(yOpen, ySl), cw, Math.abs(yOpen - ySl));
      }
    }

    // 3. Ichimoku Kumo Cloud (subtle opacity so candles stay crisp)
    if (showIchimoku) {
      for (let i = 0; i < spanA.length - 1; i++) {
        const a0 = spanA[i];
        const b0 = spanB[i];
        const a1 = spanA[i + 1];
        const b1 = spanB[i + 1];
        if (a0 === null || b0 === null || a1 === null || b1 === null) continue;
        const bullCloud = a0 + a1 >= b0 + b1;
        ctx.fillStyle = bullCloud ? 'rgba(16, 185, 129, 0.09)' : 'rgba(239, 68, 68, 0.09)';
        ctx.beginPath();
        ctx.moveTo(xOf(i), yOf(a0));
        ctx.lineTo(xOf(i + 1), yOf(a1));
        ctx.lineTo(xOf(i + 1), yOf(b1));
        ctx.lineTo(xOf(i), yOf(b0));
        ctx.closePath();
        ctx.fill();
      }

      const drawLine = (vals: (number | null)[], color: string, width: number) => {
        ctx.strokeStyle = color;
        ctx.lineWidth = width;
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
      };

      drawLine(spanA, 'rgba(16, 185, 129, 0.38)', 0.9);
      drawLine(spanB, 'rgba(239, 68, 68, 0.38)', 0.9);
      drawLine(tenkan, 'rgba(56, 189, 248, 0.85)', 1.1);
      drawLine(kijun, 'rgba(251, 113, 133, 0.85)', 1.15);
    }

    // 4. Clean Swing High / Swing Low S/R Lines (optional)
    if (showZones) {
      const look = displayBars.slice(-24);
      const swingHi = Math.max(...look.map((b) => b.high));
      const swingLo = Math.min(...look.map((b) => b.low));
      const drawLevel = (px: number, label: string, color: string) => {
        const y = Math.round(yOf(px)) + 0.5;
        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.moveTo(padL, y);
        ctx.lineTo(w - padR, y);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = color;
        ctx.fillText(label, padL + 6, y - 4);
      };
      drawLevel(swingHi, `SWING HIGH ${swingHi.toFixed(digs)}`, 'rgba(245, 158, 11, 0.75)');
      drawLevel(swingLo, `SWING LOW ${swingLo.toFixed(digs)}`, 'rgba(6, 182, 212, 0.75)');
    }

    // 5. EMA9 & Session VWAP
    if (showEma) {
      let e9 = displayBars[0].close;
      const k = 2 / 10;
      ctx.strokeStyle = '#F59E0B';
      ctx.lineWidth = 1.25;
      ctx.beginPath();
      displayBars.forEach((b, i) => {
        e9 = b.close * k + e9 * (1 - k);
        const x = xOf(i);
        if (i === 0) ctx.moveTo(x, yOf(e9));
        else ctx.lineTo(x, yOf(e9));
      });
      ctx.stroke();

      let cumPv = 0;
      let cumV = 0;
      ctx.strokeStyle = 'rgba(139, 92, 246, 0.85)';
      ctx.lineWidth = 1.25;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      displayBars.forEach((b, i) => {
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

    // 6. Crisp Pixel-Aligned Candlesticks (last bar = live forming candle if active)
    const bodyW = Math.max(3, Math.floor(stepX * 0.64));
    displayBars.forEach((b, i) => {
      const isLive = Boolean(liveBar && i === displayBars.length - 1);
      const cx = Math.round(xOf(i)) + 0.5;
      const bull = b.close >= b.open;
      const col = bull ? '#10B981' : '#EF4444';

      // Wick
      ctx.strokeStyle = col;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(cx, yOf(b.high));
      ctx.lineTo(cx, yOf(b.low));
      ctx.stroke();

      // Body
      const top = Math.round(yOf(Math.max(b.open, b.close)));
      const bot = Math.round(yOf(Math.min(b.open, b.close)));
      const bh = Math.max(1.5, bot - top);
      const bx = Math.round(cx - bodyW / 2);
      if (isLive) {
        ctx.fillStyle = bull ? 'rgba(16, 185, 129, 0.45)' : 'rgba(239, 68, 68, 0.45)';
        ctx.fillRect(bx, top, bodyW, bh);
        ctx.strokeStyle = col;
        ctx.lineWidth = 1.2;
        ctx.strokeRect(bx + 0.5, top + 0.5, Math.max(1, bodyW - 1), Math.max(1, bh - 1));
      } else {
        ctx.fillStyle = col;
        ctx.fillRect(bx, top, bodyW, bh);
      }
    });

    // 7. Live / Paper Position Open, TP, SL Rails (with crisp in-chart tags + right-axis badges)
    if (showOrders && position) {
      const drawRail = (px: number, leftTag: string, rightPrice: string, color: string, solid = false) => {
        if (px <= 0) return;
        const y = Math.round(yOf(px)) + 0.5;
        ctx.strokeStyle = color;
        ctx.lineWidth = solid ? 1.5 : 1.3;
        ctx.setLineDash(solid ? [] : [5, 3]);
        ctx.beginPath();
        ctx.moveTo(padL, y);
        ctx.lineTo(w - padR, y);
        ctx.stroke();
        ctx.setLineDash([]);

        // Left in-chart tag
        const textW = ctx.measureText(leftTag).width + 12;
        ctx.fillStyle = color;
        ctx.fillRect(padL + 4, y - 9, textW, 17);
        ctx.fillStyle = '#FFFFFF';
        ctx.fillText(leftTag, padL + 10, y + 3.5);

        // Right axis price badge
        ctx.fillStyle = color;
        ctx.fillRect(w - padR + 2, y - 9, padR - 4, 17);
        ctx.fillStyle = '#FFFFFF';
        ctx.fillText(rightPrice, w - padR + 6, y + 3.5);
      };

      const pnlTxt = `${position.profit >= 0 ? '+' : '-'}$${Math.abs(position.profit).toFixed(2)}`;
      if (position.tp > 0) {
        drawRail(position.tp, `TP ${position.tp.toFixed(digs)}`, position.tp.toFixed(digs), '#10B981');
      }
      if (position.open_price > 0) {
        drawRail(
          position.open_price,
          `${position.label} ${position.side} ${position.volume} (${pnlTxt})`,
          position.open_price.toFixed(digs),
          '#3B82F6',
          true
        );
      }
      if (position.sl > 0) {
        drawRail(
          position.sl,
          `${position.is_be ? 'BE' : 'SL'} ${position.sl.toFixed(digs)}`,
          position.sl.toFixed(digs),
          position.is_be ? '#06B6D4' : '#EF4444'
        );
      }
    }

    // 8. Live Bid Price Line
    if (symbolData.bid > 0) {
      const bidY = Math.round(yOf(symbolData.bid)) + 0.5;
      const bidCol = lastClosed && symbolData.bid >= lastClosed.close ? '#10B981' : '#EF4444';
      ctx.strokeStyle = bidCol;
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 3]);
      ctx.beginPath();
      ctx.moveTo(padL, bidY);
      ctx.lineTo(w - padR, bidY);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = '#3B82F6';
      ctx.fillRect(w - padR + 2, bidY - 9, padR - 4, 18);
      ctx.fillStyle = '#FFFFFF';
      ctx.fillText(symbolData.bid.toFixed(digs), w - padR + 6, bidY + 4);
    }

    // 9. Interactive Crosshair (X + Y + right-axis hovered price)
    if (hoverIdx !== null && displayBars[hoverIdx]) {
      const x = Math.round(xOf(hoverIdx)) + 0.5;
      ctx.strokeStyle = 'rgba(230, 237, 247, 0.30)';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(x, padT);
      ctx.lineTo(x, h - padB);
      if (hoverY !== null && hoverY >= padT && hoverY <= h - padB) {
        const hy = Math.round(hoverY) + 0.5;
        ctx.moveTo(padL, hy);
        ctx.lineTo(w - padR, hy);
      }
      ctx.stroke();
      ctx.setLineDash([]);

      if (hoverY !== null && hoverY >= padT && hoverY <= h - padB) {
        const hp = pOf(hoverY);
        ctx.fillStyle = '#1E293B';
        ctx.fillRect(w - padR + 2, hoverY - 8, padR - 4, 16);
        ctx.fillStyle = '#E6EDF7';
        ctx.fillText(hp.toFixed(digs), w - padR + 6, hoverY + 4);
      }
    }
  }, [symbolData, position, chartWidth, showEma, showIchimoku, showZones, showOrders, hoverIdx, hoverY, displayBars.length]);

  const activeBar = hoverIdx !== null && displayBars[hoverIdx] ? displayBars[hoverIdx] : displayBars[displayBars.length - 1];
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
          {position && (
            <span className="px-1.5 py-0.5 rounded bg-[#3B82F6]/15 text-[#60A5FA] border border-[#3B82F6]/35 font-bold">
              {position.label} {position.side}
            </span>
          )}
        </div>
        <div className="flex flex-wrap items-center gap-3 text-xs text-[#8A9BB5]">
          <label className="flex items-center gap-1.5 cursor-pointer select-none" title="Tenkan (9), Kijun (26), and Kumo Cloud (52)">
            <input type="checkbox" checked={showIchimoku} onChange={(e) => setShowIchimoku(e.target.checked)} className="accent-[#38BDF8]" />
            <span>Ichimoku</span>
          </label>
          <label className="flex items-center gap-1.5 cursor-pointer select-none">
            <input type="checkbox" checked={showEma} onChange={(e) => setShowEma(e.target.checked)} className="accent-[#F59E0B]" />
            <span>EMA / VWAP</span>
          </label>
          <label className="flex items-center gap-1.5 cursor-pointer select-none" title="24-bar Swing High / Swing Low levels">
            <input type="checkbox" checked={showZones} onChange={(e) => setShowZones(e.target.checked)} className="accent-[#06B6D4]" />
            <span>Swing S/R</span>
          </label>
          <label className="flex items-center gap-1.5 cursor-pointer select-none" title="Show Open, TP, and SL rails for active MT5 or Paper position">
            <input type="checkbox" checked={showOrders} onChange={(e) => setShowOrders(e.target.checked)} className="accent-[#10B981]" />
            <span>Open/TP/SL</span>
          </label>
        </div>
      </div>

      <div ref={containerRef} className="relative bg-[#0B111C]">
        {activeBar && symbolData && (
          <div className="absolute top-2 left-2.5 z-10 pointer-events-none flex flex-wrap items-center gap-2 text-xs mono">
            <span className="px-2 py-0.5 rounded bg-[#0E1522]/90 border border-[#1F2D45]">
              <span className="text-[#64748B] mr-1.5">{fmtTimeHHMM(activeBar.timestamp)}</span>
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
          style={{ height: 340 }}
          onMouseMove={(e) => {
            const rect = e.currentTarget.getBoundingClientRect();
            const futureSlots = showIchimoku ? FUTURE_CLOUD_SLOTS : 2;
            const slots = displayBars.length + futureSlots;
            const rel = (e.clientX - rect.left - 10) / Math.max(rect.width - 96, 1);
            const idx = Math.max(0, Math.min(displayBars.length - 1, Math.floor(rel * slots)));
            setHoverIdx(idx);
            setHoverY(e.clientY - rect.top);
          }}
          onMouseLeave={() => {
            setHoverIdx(null);
            setHoverY(null);
          }}
        />
      </div>
    </div>
  );
};
