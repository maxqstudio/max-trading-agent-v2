import { useState } from 'react';
import {
  Activity,
  AlertTriangle,
  Cpu,
  FileText,
  LayoutGrid,
  OctagonAlert,
  Pause,
  Play,
  RefreshCw,
  Settings,
  Shield,
  ShieldAlert,
} from 'lucide-react';
import { api, useLiveTelemetry } from './api/client';
import { BrainView } from './components/BrainView';
import { EvolutionView } from './components/EvolutionView';
import { ForensicsView } from './components/ForensicsView';
import { KillSwitchModal } from './components/KillSwitchModal';
import { LiveCandleChart } from './components/LiveCandleChart';
import { LiveStatusStrip } from './components/LiveStatusStrip';
import { PaperCompare } from './components/PaperCompare';
import { RiskConfigView } from './components/RiskConfigView';
import { SetupView } from './components/SetupView';
import type { DecisionEvent } from './types/trading';
import { BullishLogo } from './components/BullishLogo';
import { human, humanText } from './labels';

type TabId = 'cockpit' | 'brain' | 'evolution' | 'journal' | 'risk' | 'setup';

// Result of a closed trade in the execution stream: from metadata, or parsed from older messages.
function tradeResult(d: DecisionEvent): number | null {
  if (typeof d.metadata?.pnl === 'number') return d.metadata.pnl;
  if (d.category !== 'EXEC' || !/closed/i.test(`${d.tag} ${d.message}`)) return null;
  const m = d.message.match(/\$([+-]\d+(?:\.\d+)?)|([+-]\d+(?:\.\d+)?)\$/);
  return m ? parseFloat(m[1] ?? m[2]) : null;
}

export default function App() {
  const { state, wsConnected } = useLiveTelemetry();
  const [tab, setTab] = useState<TabId>('cockpit');
  const [selectedSym, setSelectedSym] = useState('XAUUSD');
  const [logFilter, setLogFilter] = useState<'ALL' | 'EXEC' | 'BLOCK' | 'AI'>('ALL');
  const [killOpen, setKillOpen] = useState(false);
  const [toast, setToast] = useState<string | null>(null);

  const showToast = (msg: string) => {
    setToast(msg);
    window.setTimeout(() => setToast(null), 3200);
  };

  const symbols = state?.radar?.map((r) => r.symbol) || ['XAUUSD', 'EURUSD', 'GBPUSD', 'USDJPY'];
  const activeRadar = state?.radar?.find((r) => r.symbol === selectedSym) || state?.radar?.[0];
  const activePosForChart = state?.positions?.find((p) => p.symbol === activeRadar?.symbol);
  const eq = state?.account?.equity ?? 10000;
  const dayPnl = state?.account?.daily_pnl ?? 0;
  const ddPct = state?.account?.daily_dd_pct ?? 0;
  const staleBridge = state ? !state.bridge_connected : !wsConnected;
  const paperPositions = state?.paper?.positions || [];
  const normalPaperPositions = state?.paper?.books?.normal?.positions || paperPositions.filter((p) => !p.book || p.book === 'normal');
  const hasMt5Positions = (state?.positions?.length || 0) > 0;
  const floatingTotal = hasMt5Positions
    ? (state?.positions || []).reduce((acc, p) => acc + p.profit, 0)
    : normalPaperPositions.reduce((acc, p) => acc + p.profit, 0);
  const rawEaVer = state?.hard_guard?.ea_version;
  const displayVersion = rawEaVer ? `v${rawEaVer.replace(/^v/i, '').replace(/\.0$/, '')}` : 'v2.2';

  const filteredDecisions = (state?.decisions || []).filter(
    (d) => logFilter === 'ALL' || d.category === logFilter
  );

  return (
    <div className="min-h-screen flex flex-col bg-[#070B12] text-[#E6EDF7]">
      {/* Stale Watchdog / Kill / Shield Banner */}
      {(staleBridge || state?.killed || state?.manual_shield_override) && (
        <div className="px-4 py-2 text-xs font-semibold flex items-center justify-between border-b border-[#1F2D45] bg-[#EF4444]/15 text-[#EF4444]">
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 shrink-0" />
            <span>
              {state?.killed
                ? 'HARD KILL-SWITCH LOCKED — All MT5 positions flattened. Click RESUME only when safe.'
                : state?.manual_shield_override
                  ? 'MANUAL SHIELD OVERRIDE ACTIVE — New scalping entries paused.'
                  : `BRIDGE HEARTBEAT STALE (${state?.bridge_stale_sec ?? '?'}s) — Waiting for MT5 EA or Simulator stream...`}
            </span>
          </div>
          {state?.manual_shield_override && (
            <button onClick={() => api.toggleShield()} className="px-2.5 py-0.5 rounded border border-current cursor-pointer">
              Clear Shield
            </button>
          )}
        </div>
      )}

      {/* PERSISTENT TOP COMMAND BAR */}
      <header className="sticky top-0 z-30 bg-[#0A101B] border-b border-[#1F2D45] px-3.5 py-2 flex flex-wrap items-center justify-between gap-2.5">
        <div className="flex items-center gap-2.5 flex-wrap">
          <div className="flex items-center gap-2 pr-2.5 border-r border-[#1F2D45]">
            <BullishLogo size={36} />
            <div>
              <div className="flex items-center gap-1.5 leading-none">
                <span className="font-bold tracking-wider text-xs">MAX QUANT</span>
                <span className="mono text-xs px-1.5 py-0.5 rounded bg-[#10B981]/15 text-[#10B981] border border-[#10B981]/40 font-bold" title={rawEaVer ? `MT5 EA ${rawEaVer}` : 'EA v2.2'}>
                  {displayVersion}
                </span>
              </div>
              <div className="text-xs text-[#8A9BB5] mt-0.5">MT5 trading robot</div>
            </div>
          </div>

          <select
            value={state?.trading_mode || 'SHADOW_ONLY'}
            onChange={async (e) => {
              const next = e.target.value;
              let code = '';
              if (next === 'MT5_LIVE') {
                code = (window.prompt('REAL MONEY. Type LIVE to confirm:') || '').trim();
                if (code.toUpperCase() !== 'LIVE') return;
              }
              try {
                await api.setMode(next, code);
                showToast(`Mode switched to ${human(next)}`);
              } catch (err) {
                showToast(`Mode change refused: ${(err as Error).message}`);
              }
            }}
            className="mono text-xs font-bold px-2.5 py-1.5 rounded-md cockpit-card cursor-pointer focus:outline-none"
          >
            <option value="SHADOW_ONLY">{human('SHADOW_ONLY')}</option>
            <option value="SIMULATOR">{human('SIMULATOR')}</option>
            <option value="MT5_DEMO">{human('MT5_DEMO')}</option>
            <option value="MT5_LIVE">{human('MT5_LIVE')}</option>
          </select>

          <div className="flex items-center gap-2 px-2.5 py-1.5 rounded-md cockpit-card text-xs">
            <span className="relative flex h-2 w-2">
              <span className={`live-pulse absolute inline-flex h-full w-full rounded-full ${staleBridge ? 'bg-[#EF4444]' : 'bg-[#10B981]'}`} />
              <span className={`relative inline-flex rounded-full h-2 w-2 ${staleBridge ? 'bg-[#EF4444]' : 'bg-[#10B981]'}`} />
            </span>
            <span className="font-semibold">BRIDGE</span>
            <span className="mono font-bold text-[#10B981]">{state?.bridge_latency_ms?.toFixed(1) ?? '0.0'}ms</span>
            <span className="text-[#8A9BB5]">|</span>
            <span className="mono text-[#8A9BB5]">{state?.ticks_per_sec?.toFixed(1) ?? '0.0'} t/s</span>
          </div>

          <button
            onClick={async () => {
              const r = await api.toggleShield();
              showToast(r.manual_shield_override ? 'Manual Shield Freeze ENGAGED' : 'Market Shields CLEAR');
            }}
            className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-xs font-bold cursor-pointer border ${
              state?.manual_shield_override
                ? 'bg-[#F59E0B]/15 text-[#F59E0B] border-[#F59E0B]/40'
                : 'bg-[#10B981]/15 text-[#10B981] border-[#10B981]/40'
            }`}
          >
            {state?.manual_shield_override ? <ShieldAlert className="w-3.5 h-3.5" /> : <Shield className="w-3.5 h-3.5" />}
            <span>{state?.manual_shield_override ? 'SHIELD FREEZE' : 'SHIELD CLEAR'}</span>
          </button>
        </div>

        <div className="flex items-center gap-2.5 flex-wrap">
          <div className="flex items-center gap-3.5 px-3 py-1 rounded-md cockpit-card text-xs">
            <div>
              <div className="text-[#8A9BB5]">Equity</div>
              <div className="mono font-bold text-sm">${eq.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</div>
            </div>
            <div className="h-6 w-px bg-[#1F2D45]" />
            <div>
              <div className="text-[#8A9BB5]">Day P&L</div>
              <div className={`mono font-bold text-sm ${dayPnl >= 0 ? 'text-[#10B981]' : 'text-[#EF4444]'}`}>
                {dayPnl >= 0 ? '+' : ''}${dayPnl.toFixed(2)}
              </div>
            </div>
            <div className="h-6 w-px bg-[#1F2D45] hidden md:block" />
            <div className="hidden md:block w-28">
              <div className="flex justify-between mono text-xs">
                <span className="font-sans text-[#8A9BB5]">DD</span>
                <span className="font-bold">{ddPct.toFixed(2)}% / 3.0%</span>
              </div>
              <div className="w-full h-1.5 rounded-full bg-[#070B12] mt-1 overflow-hidden">
                <div className="h-full bg-[#10B981]" style={{ width: `${Math.min(100, (ddPct / 3.0) * 100)}%` }} />
              </div>
            </div>
          </div>

          <button
            onClick={async () => {
              if (state?.killed) {
                const code = (window.prompt('Kill switch is on. Type RELEASE to release it (engine stays PAUSED):') || '').trim();
                if (code.toUpperCase() !== 'RELEASE') return;
                await api.releaseKill(code);
                showToast('Kill switch released — engine still PAUSED');
                return;
              }
              const r = await api.togglePause();
              showToast(r.paused ? 'Engine PAUSED' : 'Engine RESUMED');
            }}
            className="px-3 py-1.5 rounded-md text-xs font-bold cockpit-card hover:bg-[#121B2B] cursor-pointer flex items-center gap-1.5"
          >
            {state?.paused ? <Play className="w-3.5 h-3.5 text-[#10B981]" /> : <Pause className="w-3.5 h-3.5 text-[#F59E0B]" />}
            <span>{state?.paused ? 'RESUME' : 'PAUSE'}</span>
          </button>

          <button
            onClick={() => setKillOpen(true)}
            className="px-3 py-1.5 rounded-md text-xs font-bold bg-[#EF4444] hover:bg-[#DC2626] text-white cursor-pointer flex items-center gap-1.5"
          >
            <OctagonAlert className="w-3.5 h-3.5" />
            <span>KILL SWITCH</span>
          </button>
        </div>
      </header>

      {/* LIVE STATUS: real values from the backend (MT5, entry gate, AI, supervisor, last verdict) */}
      <LiveStatusStrip
        state={state}
        onDemoSignal={async () => {
          try {
            const r = await api.triggerDemoSignal(selectedSym);
            showToast(r.ok ? `Signal fired on ${selectedSym}` : `Skipped: ${r.reason || 'inflight'}`);
          } catch (err) {
            showToast(`Demo signal refused: ${(err as Error).message}`);
          }
        }}
      />

      {/* MAIN WORKSPACE */}
      <div className="flex-1 flex flex-col lg:flex-row">
        {/* Left Icon Rail */}
        <aside className="lg:w-52 bg-[#0A101B] border-b lg:border-b-0 lg:border-r border-[#1F2D45] p-2 flex lg:flex-col gap-1 overflow-x-auto shrink-0">
          {[
            { id: 'cockpit' as TabId, label: 'Execution Cockpit', icon: LayoutGrid },
            { id: 'brain' as TabId, label: 'AI Brain & Regimes', icon: Cpu },
            { id: 'evolution' as TabId, label: 'Evolution & Challenger', icon: RefreshCw, badge: state?.active_challenger ? 1 : 0 },
            { id: 'journal' as TabId, label: 'Trade Forensics', icon: FileText },
            { id: 'risk' as TabId, label: 'Risk, Shields & MT5', icon: Shield },
            { id: 'setup' as TabId, label: 'Setup (AI, Supervisor)', icon: Settings },
          ].map((item) => {
            const Icon = item.icon;
            const active = tab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setTab(item.id)}
                className={`flex items-center gap-2.5 px-3 py-2 rounded-md text-xs font-semibold whitespace-nowrap cursor-pointer border transition-colors ${
                  active
                    ? 'bg-[#3B82F6]/15 text-[#3B82F6] border-[#3B82F6]/45'
                    : 'border-transparent text-[#8A9BB5] hover:text-white hover:bg-white/5'
                }`}
              >
                <Icon className="w-4 h-4 shrink-0" />
                <span>{item.label}</span>
                {item.badge ? (
                  <span className="ml-auto px-1.5 py-0.5 rounded-full bg-[#8B5CF6]/20 text-[#8B5CF6] mono font-bold">{item.badge}</span>
                ) : null}
              </button>
            );
          })}

          <div className="hidden lg:block mt-auto pt-3 border-t border-[#1F2D45] space-y-2 text-xs">
            <div className="p-2.5 rounded-md cockpit-card space-y-1.5">
              <div className="flex items-center justify-between">
                <span className="text-[#8A9BB5] font-semibold">WS STREAM</span>
                <span className={`px-1.5 py-0.5 rounded mono font-bold ${wsConnected ? 'bg-[#10B981]/15 text-[#10B981]' : 'bg-[#F59E0B]/15 text-[#F59E0B]'}`}>
                  {wsConnected ? 'LIVE' : 'POLL'}
                </span>
              </div>
              <div className="flex justify-between"><span className="text-[#8A9BB5]">AI Scans</span><span className="mono font-bold">{state?.scans_count ?? 0}</span></div>
              <div className="pt-1 border-t border-[#1F2D45]/70 text-[10px] uppercase tracking-wider text-[#64748B] font-bold">Paper · AI Supervised</div>
              <div className="flex justify-between"><span className="text-[#8A9BB5]">Normal</span><span className="mono font-bold text-[#F59E0B]">${(state?.paper?.books?.normal?.equity ?? state?.paper?.equity ?? 10000).toFixed(0)}</span></div>
              <div className="flex justify-between"><span className="text-[#8A9BB5]">Hit &amp; Run</span><span className="mono font-bold text-[#EC4899]">${(state?.paper?.books?.hitrun?.equity ?? 10000).toFixed(0)}</span></div>
              <div className="pt-1 border-t border-[#1F2D45]/70 text-[10px] uppercase tracking-wider text-[#64748B] font-bold">Paper · No Veto</div>
              <div className="flex justify-between" title="Every risk-approved setup (Normal exit), ignoring filters and AI Supervisor"><span className="text-[#8A9BB5]">No Veto</span><span className="mono font-bold text-[#14B8A6]">${(state?.paper?.books?.raw?.equity ?? 10000).toFixed(0)}</span></div>
              <div className="flex justify-between" title="Every risk-approved setup (Hit & Run exit), ignoring filters and AI Supervisor"><span className="text-[#8A9BB5]">No Veto H&amp;R</span><span className="mono font-bold text-[#38BDF8]">${(state?.paper?.books?.raw_hitrun?.equity ?? 10000).toFixed(0)}</span></div>
              <div className="pt-1 border-t border-[#1F2D45]/70 text-[10px] uppercase tracking-wider text-[#64748B] font-bold">Evolution · Challenger</div>
              <div className="flex justify-between" title="Evolved challenger WITH P0 filters and AI Supervisor veto"><span className="text-[#8A9BB5]">AI Veto</span><span className="mono font-bold text-[#8B5CF6]">${(state?.shadow_equity ?? 10000).toFixed(0)}</span></div>
              <div className="flex justify-between" title="Pure evolved challenger WITHOUT filters or AI Supervisor veto"><span className="text-[#8A9BB5]">Raw</span><span className="mono font-bold text-[#A3E635]">${(state?.shadow_raw_equity ?? 10000).toFixed(0)}</span></div>
            </div>
          </div>
        </aside>

        {/* Main Panel */}
        <main className="flex-1 p-3 space-y-3 overflow-x-hidden">
          {tab === 'cockpit' && (
            <>
              <div className="grid grid-cols-1 xl:grid-cols-12 gap-3">
                {/* Live M1 Chart Panel */}
                <div className="xl:col-span-8 cockpit-card rounded-lg overflow-hidden flex flex-col">
                  <div className="px-3 py-2 border-b border-[#1F2D45] flex items-center justify-between flex-wrap gap-2">
                    <div className="flex rounded-md overflow-hidden border border-[#1F2D45] mono text-xs font-bold">
                      {symbols.map((s) => (
                        <button
                          key={s}
                          onClick={() => setSelectedSym(s)}
                          className={`px-2.5 py-1 cursor-pointer ${
                            selectedSym === s ? 'bg-[#3B82F6] text-white' : 'bg-[#121B2B] text-[#8A9BB5] hover:text-white'
                          }`}
                        >
                          {s}
                        </button>
                      ))}
                    </div>
                    <div className="flex items-center gap-3 text-xs mono">
                      <span className="text-[#8A9BB5]">Readiness: <b className="text-[#10B981]">{activeRadar?.readiness ?? 0}%</b></span>
                      <span className="text-[#8A9BB5]">Pressure: <b className="text-[#06B6D4]">{activeRadar?.buy_pressure ?? 50}% BUY</b></span>
                    </div>
                  </div>
                  <LiveCandleChart symbolData={activeRadar} position={activePosForChart} />
                </div>

                {/* Multi-Pair AI Radar */}
                <div className="xl:col-span-4 cockpit-card rounded-lg flex flex-col">
                  <div className="px-3 py-2 border-b border-[#1F2D45] flex items-center justify-between text-xs">
                    <div className="flex items-center gap-2 font-bold uppercase tracking-wider">
                      <Activity className="w-3.5 h-3.5 text-[#06B6D4]" />
                      <span>Multi-Pair AI Radar</span>
                    </div>
                    <span className="mono text-[#8A9BB5]">Click to focus</span>
                  </div>
                  <div className="p-2.5 space-y-2 flex-1 overflow-y-auto">
                    {(state?.radar || []).map((r) => (
                      <div
                        key={r.symbol}
                        onClick={() => setSelectedSym(r.symbol)}
                        className={`p-2.5 rounded-md cockpit-sub cursor-pointer space-y-1.5 text-xs ${
                          r.symbol === selectedSym ? 'border-[#3B82F6] ring-1 ring-[#3B82F6]/40' : ''
                        }`}
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-1.5">
                            <span className="mono font-bold text-sm">{r.symbol}</span>
                            <span className="px-1.5 py-0.5 rounded bg-[#10B981]/15 text-[#10B981] font-bold">{human(r.regime)}</span>
                          </div>
                          <span className="mono font-bold">{r.bid.toFixed(r.digits)}</span>
                        </div>
                        <div className="flex justify-between mono text-[#8A9BB5]">
                          <span>Ask {r.ask.toFixed(r.digits)}</span>
                          <span>Sp <b>{r.spread_points.toFixed(1)}pt</b> ({r.spread_atr_pct}% ATR)</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="text-[#8A9BB5]">Readiness</span>
                          <div className="flex-1 h-1.5 rounded-full bg-[#070B12] overflow-hidden">
                            <div className="h-full bg-[#3B82F6] transition-all duration-300" style={{ width: `${r.readiness}%` }} />
                          </div>
                          <span className="mono font-bold">{r.readiness}%</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              <PaperCompare />

              {/* Row 2: Live Positions & Cognitive Stream */}
              <div className="grid grid-cols-1 xl:grid-cols-12 gap-3">
                <div className="xl:col-span-7 cockpit-card rounded-lg flex flex-col">
                  <div className="px-3 py-2 border-b border-[#1F2D45] flex items-center justify-between text-xs">
                    <span className="font-bold uppercase tracking-wider">
                      MT5 positions ({state?.positions?.length ?? 0}){paperPositions.length ? ` + Paper (${paperPositions.length})` : ''}
                    </span>
                    <span className={`mono font-bold ${floatingTotal >= 0 ? 'text-[#10B981]' : 'text-[#EF4444]'}`}>
                      Floating: {floatingTotal >= 0 ? '+' : ''}${floatingTotal.toFixed(2)}
                    </span>
                  </div>
                  <div className="overflow-x-auto p-2">
                    <table className="w-full text-xs text-left border-collapse">
                      <thead>
                        <tr className="border-b border-[#1F2D45] text-[#8A9BB5]">
                          <th className="py-1.5 px-2">Pair / Strat</th>
                          <th className="py-1.5 px-2">Side / Lot</th>
                          <th className="py-1.5 px-2">Entry → Live</th>
                          <th className="py-1.5 px-2">P&L</th>
                          <th className="py-1.5 px-2">RL-2 Exit</th>
                          <th className="py-1.5 px-2 text-right">Action</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-[#1F2D45]">
                        {paperPositions.map((p) => (
                          <tr key={`paper-${p.book ?? 'normal'}-${p.symbol}`}>
                            <td className="py-2 px-2"><div className="mono font-bold">{p.symbol} <span className={`px-1 rounded text-[10px] ${p.book === 'hitrun' ? 'bg-[#EC4899]/15 text-[#EC4899]' : p.book === 'raw_hitrun' ? 'bg-[#38BDF8]/15 text-[#38BDF8]' : p.book === 'raw' ? 'bg-[#14B8A6]/15 text-[#14B8A6]' : 'bg-[#F59E0B]/15 text-[#F59E0B]'}`}>{p.book === 'hitrun' ? 'HIT&RUN' : p.book === 'raw_hitrun' ? 'NO VETO H&R' : p.book === 'raw' ? 'NO VETO' : 'NORMAL'}</span></div><div className="text-[#8A9BB5]">{human(p.strategy_id)}</div></td>
                            <td className="py-2 px-2 mono"><span className={`px-1.5 py-0.5 rounded font-bold ${p.side === 'BUY' ? 'bg-[#10B981]/15 text-[#10B981]' : 'bg-[#EF4444]/15 text-[#EF4444]'}`}>{p.side}</span> {p.volume}</td>
                            <td className="py-2 px-2 mono">{p.entry_price} → <b>{p.current_price}</b></td>
                            <td className={`py-2 px-2 mono font-bold ${p.profit >= 0 ? 'text-[#10B981]' : 'text-[#EF4444]'}`}>{p.profit >= 0 ? '+' : ''}${p.profit.toFixed(2)}</td>
                            <td className="py-2 px-2 mono text-[#8A9BB5]">SL {p.sl} / TP {p.tp}</td>
                            <td className="py-2 px-2 text-right text-[#8A9BB5]">virtual</td>
                          </tr>
                        ))}
                        {(state?.positions || []).length === 0 && paperPositions.length === 0 ? (
                          <tr><td colSpan={6} className="py-6 text-center text-[#8A9BB5]">No open positions — AI actively hunting M1 setups...</td></tr>
                        ) : (
                          (state?.positions || []).map((p) => (
                            <tr key={p.ticket}>
                              <td className="py-2 px-2"><div className="mono font-bold">{p.symbol}</div><div className="text-[#8A9BB5]">{human(p.strategy_id)}</div></td>
                              <td className="py-2 px-2 mono"><span className={`px-1.5 py-0.5 rounded font-bold ${p.side === 'BUY' ? 'bg-[#10B981]/15 text-[#10B981]' : 'bg-[#EF4444]/15 text-[#EF4444]'}`}>{p.side}</span> {p.volume}</td>
                              <td className="py-2 px-2 mono">{p.open_price} → <b>{p.current_price}</b></td>
                              <td className={`py-2 px-2 mono font-bold ${p.profit >= 0 ? 'text-[#10B981]' : 'text-[#EF4444]'}`}>{p.profit >= 0 ? '+' : ''}${p.profit.toFixed(2)}</td>
                              <td className="py-2 px-2"><span className="px-1.5 py-0.5 rounded bg-[#06B6D4]/15 text-[#06B6D4] mono font-bold">{human(p.rl_exit_action)}</span></td>
                              <td className="py-2 px-2 text-right">
                                <button
                                  onClick={async () => {
                                    await api.closePosition(p.ticket);
                                    showToast(`Close queued for #${p.ticket}`);
                                  }}
                                  className="px-2 py-0.5 rounded cockpit-sub font-bold cursor-pointer hover:bg-[#1F2D45]"
                                >
                                  Close
                                </button>
                              </td>
                            </tr>
                          ))
                        )}
                      </tbody>
                    </table>
                  </div>
                </div>

                <div className="xl:col-span-5 cockpit-card rounded-lg flex flex-col">
                  <div className="px-3 py-2 border-b border-[#1F2D45] flex items-center justify-between gap-2 text-xs">
                    <span className="font-bold uppercase tracking-wider">Cognitive Execution Stream</span>
                    <div className="flex gap-1">
                      {(['ALL', 'EXEC', 'AI', 'BLOCK'] as const).map((f) => (
                        <button
                          key={f}
                          onClick={() => setLogFilter(f)}
                          className={`px-2 py-0.5 rounded cursor-pointer font-semibold ${
                            logFilter === f ? 'bg-[#3B82F6] text-white' : 'cockpit-sub text-[#8A9BB5]'
                          }`}
                        >
                          {f}
                        </button>
                      ))}
                    </div>
                  </div>
                  <div className="p-2 space-y-1.5 max-h-64 overflow-y-auto">
                    {filteredDecisions.length === 0 ? (
                      <div className="py-6 text-center text-xs text-[#8A9BB5]">Awaiting live decision telemetry...</div>
                    ) : (
                      filteredDecisions.map((d) => {
                        const pnl = tradeResult(d);
                        const tone = pnl == null ? '' : pnl > 0 ? 'win' : 'loss';
                        return (
                        <div
                          key={d.id}
                          className={`p-2 rounded cockpit-sub text-xs space-y-0.5 ${tone ? 'border-l-4' : ''}`}
                          style={tone ? { borderLeftColor: tone === 'win' ? '#10B981' : '#EF4444' } : undefined}
                        >
                          <div className="flex items-center justify-between mono">
                            <span>
                              <span className="text-[#8A9BB5] mr-1.5">{new Date(d.timestamp_ms).toISOString().substring(11, 19)}</span>
                              <b>{d.symbol}</b>
                            </span>
                            <span className={`px-1.5 py-0.5 rounded font-bold ${tone === 'win' ? 'bg-[#10B981]/20 text-[#10B981]' : tone === 'loss' ? 'bg-[#EF4444]/20 text-[#EF4444]' : d.category === 'EXEC' ? 'bg-[#06B6D4]/15 text-[#06B6D4]' : d.category === 'BLOCK' ? 'bg-[#F59E0B]/15 text-[#F59E0B]' : 'bg-[#06B6D4]/15 text-[#06B6D4]'}`}>
                              {humanText(d.tag)}
                              {pnl != null ? ` ${pnl > 0 ? '+' : pnl < 0 ? '-' : ''}$${Math.abs(pnl).toFixed(2)}` : ''}
                            </span>
                          </div>
                          <div className={`leading-snug ${tone === 'win' ? 'text-[#6EE7B7]' : tone === 'loss' ? 'text-[#FCA5A5]' : 'text-[#8A9BB5]'}`}>{humanText(d.message)}</div>
                        </div>
                        );
                      })
                    )}
                  </div>
                </div>
              </div>
            </>
          )}

          {tab === 'brain' && <BrainView symbols={symbols} />}
          {tab === 'evolution' && <EvolutionView onToast={showToast} />}
          {tab === 'journal' && <ForensicsView symbols={symbols} />}
          {tab === 'risk' && <RiskConfigView onToast={showToast} />}
          {tab === 'setup' && <SetupView onToast={showToast} />}
        </main>
      </div>

      <KillSwitchModal
        open={killOpen}
        onClose={() => setKillOpen(false)}
        onSoftPause={async () => {
          await api.killSwitch('SOFT_PAUSE');
          showToast('Soft Pause engaged');
        }}
        onHardKill={async (code) => {
          await api.killSwitch('FLATTEN_KILL', code);
          showToast('FLATTEN ALL & HARD KILL dispatched');
        }}
      />

      {toast && (
        <div className="fixed bottom-4 right-4 z-50 px-3.5 py-2 rounded-lg cockpit-card border border-[#3B82F6] shadow-xl text-xs font-semibold flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-[#3B82F6]" />
          <span>{toast}</span>
        </div>
      )}
    </div>
  );
}
