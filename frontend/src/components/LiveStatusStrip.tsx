import React from 'react';
import { Bot, Cable, Gavel, Lock, Zap } from 'lucide-react';
import type { LiveStateSnapshot } from '../types/trading';
import { human, humanText } from '../labels';

// Real-time status from the backend snapshot (replaces the old animated "engine loop").

const BLOCK_LABELS: Record<string, string> = {
  KILL_SWITCH_ACTIVE: 'Kill switch on',
  SHADOW_ONLY_PAPER_MODE: 'Paper (no orders sent)',
  LIVE_TRADING_NOT_ENABLED: 'Live locked (.env)',
  REAL_ACCOUNT_NOT_AUTHORIZED: 'Real account not allowed',
};

function ago(ts?: number): string {
  if (!ts) return '';
  const s = Math.max(0, Math.round((Date.now() - ts) / 1000));
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.round(s / 60)}m ago`;
  return `${Math.round(s / 3600)}h ago`;
}

const Chip: React.FC<{ icon: React.ElementType; label: string; tone: 'ok' | 'warn' | 'bad' | 'info'; children: React.ReactNode; title?: string }> = ({
  icon: Icon,
  label,
  tone,
  children,
  title,
}) => {
  const color = { ok: '#10B981', warn: '#F59E0B', bad: '#EF4444', info: '#06B6D4' }[tone];
  return (
    <div title={title} className="flex items-center gap-1.5 px-2 py-1 rounded border border-[#1F2D45] bg-[#121B2B] min-w-0">
      <Icon className="w-3.5 h-3.5 shrink-0" style={{ color }} />
      <span className="mono font-bold text-[#8A9BB5] shrink-0">{label}</span>
      <span className="mono truncate" style={{ color }}>
        {children}
      </span>
    </div>
  );
};

export const LiveStatusStrip: React.FC<{ state: LiveStateSnapshot | null; onDemoSignal: () => void }> = ({ state, onDemoSignal }) => {
  const sup = state?.supervisor;
  const llm = state?.llm;
  const last = sup?.last;
  const block = state?.entry_block_reason || '';
  const nSym = state?.radar?.length ?? 0;

  const bridgeTone = state?.bridge_connected ? 'ok' : 'bad';
  const gateTone = block === 'KILL_SWITCH_ACTIVE' ? 'bad' : block ? 'warn' : 'ok';
  const supTone = !sup?.active ? 'warn' : 'ok';
  const lastTone = last?.decision === 'REJECT' ? 'bad' : last?.decision === 'REDUCE' ? 'warn' : 'ok';

  return (
    <div className="bg-[#0E1522] border-b border-[#1F2D45] px-3.5 py-2 flex flex-wrap items-center gap-2 text-xs">
      <Chip icon={Cable} label="MT5" tone={bridgeTone} title="MaxTradingBridge EA connection">
        {state?.bridge_connected
          ? `connected · ${nSym} markets · ${state?.bridge_latency_ms?.toFixed(0) ?? 0}ms`
          : `disconnected ${state?.bridge_stale_sec && state.bridge_stale_sec < 900 ? `(${state.bridge_stale_sec}s)` : ''}`}
      </Chip>

      <Chip icon={Lock} label="ENTRY" tone={gateTone} title="Whether new orders may be sent to MT5 right now">
        {block ? BLOCK_LABELS[block] || human(block) : `OPEN · ${human(state?.trading_mode ?? '')}`}
      </Chip>

      <Chip
        icon={Bot}
        label="AI"
        tone={llm?.enabled ? 'info' : 'warn'}
        title={llm?.fallbacks?.length ? `Fallbacks: ${llm.fallbacks.join(' → ')}` : 'No fallback provider'}
      >
        {llm?.enabled ? `${llm.provider} · ${llm.fast_model}${llm.fallbacks.length ? ` (+${llm.fallbacks.length} fallback)` : ''}` : 'not set (open Setup)'}
      </Chip>

      <Chip icon={Gavel} label="SUPERVISOR" tone={supTone} title="Claude decisions today: approve / reduce / reject / unreachable">
        {sup?.active
          ? `✓${sup.approve} ↓${sup.reduce} ✗${sup.reject}${sup.unavailable ? ` ⚠${sup.unavailable}` : ''} · $${sup.spent_today_usd.toFixed(2)}/$${sup.daily_budget_usd.toFixed(0)}`
          : 'off'}
      </Chip>

      <div className="flex-1 min-w-[12rem] flex items-center gap-1.5 px-2 py-1 rounded border border-[#1F2D45] bg-[#121B2B] min-w-0">
        {last?.decision ? (
          <>
            <span
              className="px-1.5 rounded mono font-bold shrink-0"
              style={{
                color: { ok: '#10B981', warn: '#F59E0B', bad: '#EF4444', info: '#06B6D4' }[lastTone],
                background: 'rgba(255,255,255,0.05)',
              }}
            >
              {human(last.decision)}
            </span>
            <span className="mono font-bold shrink-0">{last.symbol}</span>
            <span className="truncate text-[#E6EDF7]" title={last.reason}>
              {humanText(last.reason)}
            </span>
            <span className="mono text-[#8A9BB5] shrink-0">{ago(last.ts_ms)}</span>
          </>
        ) : (
          <span className="mono truncate text-[#8A9BB5]">{state?.active_thought || 'Menunggu setup pertama...'}</span>
        )}
      </div>

      <button
        onClick={onDemoSignal}
        className="px-2.5 py-1 rounded bg-[#3B82F6]/15 text-[#3B82F6] border border-[#3B82F6]/40 font-bold flex items-center gap-1 cursor-pointer shrink-0 hover:bg-[#3B82F6]/25"
      >
        <Zap className="w-3 h-3" /> Demo Signal
      </button>
    </div>
  );
};
