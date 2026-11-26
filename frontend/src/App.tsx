import React, { useEffect, useState } from 'react';
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
  Shield,
  ShieldAlert,
  TrendingUp,
  Zap,
} from 'lucide-react';
import { api, useLiveTelemetry } from './api/client';
import { BrainView } from './components/BrainView';
import { EvolutionView } from './components/EvolutionView';
import { ForensicsView } from './components/ForensicsView';
import { KillSwitchModal } from './components/KillSwitchModal';
import { LiveCandleChart } from './components/LiveCandleChart';
import { RiskConfigView } from './components/RiskConfigView';

type TabId = 'cockpit' | 'brain' | 'evolution' | 'journal' | 'risk';

const PIPELINE_NODES = ['TICK', 'REGIME', '4 STRATS', 'DL 1D-CNN', 'ML META', 'RL BANDIT', 'EXEC + SHADOW'];

export default function App() {
  const { state, wsConnected } = useLiveTelemetry();
  const [tab, setTab] = useState<TabId>('cockpit');
  const [selectedSym, setSelectedSym] = useState('XAUUSD');
  const [logFilter, setLogFilter] = useState<'ALL' | 'EXEC' | 'BLOCK' | 'AI'>('ALL');
  const [pipeStep, setPipeStep] = useState(0);
  const [killOpen, setKillOpen] = useState(false);
  const [toast, setToast] = useState<string | null>(null);

  const showToast = (msg: string) => {
    setToast(msg);
    window.setTimeout(() => setToast(null), 3200);
  };

  // Continuous neural pipeline heartbeat synced with live engine
  useEffect(() => {
    const t = window.setInterval(() => {
      setPipeStep((p) => (p + 1) % PIPELINE_NODES.length);
    }, 360);
    return () => window.clearInterval(t);
  }, []);

  const symbols = state?.radar?.map((r) => r.symbol) || ['XAUUSD', 'EURUSD', 'GBPUSD', 'USDJPY'];
  const activeRadar = state?.radar?.find((r) => r.symbol === selectedSym) || state?.radar?.[0];
  const activePosForChart = state?.positions?.find((p) => p.symbol === activeRadar?.symbol);
  const eq = state?.account?.equity ?? 10000;
  const dayPnl = state?.account?.daily_pnl ?? 0;
  const ddPct = state?.account?.daily_dd_pct ?? 0;
  const staleBridge = state ? !state.bridge_connected : !wsConnected;
  const floatingTotal = (state?.positions || []).reduce((acc, p) => acc + p.profit, 0);

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
            <div className="w-8 h-8 rounded-lg flex items-center justify-center bg-[#3B82F6]/15 border border-[#3B82F6]/45 text-[#3B82F6]">
              <TrendingUp className="w-4 h-4" />
            </div>
            <div>
              <div className="flex items-center gap-1.5 leading-none">
                <span className="font-bold tracking-wider text-xs">MAX QUANT</span>
                <span className="mono text-xs px-1.5 py-0.5 rounded bg-[#10B981]/15 text-[#10B981] border border-[#10B981]/40 font-bold">
                  {state?.champion_version || 'v1.0'}
                </span>
              </div>
              <div className="text-xs text-[#8A9BB5] mt-0.5">MT5 Autonomous Desk</div>
            </div>
          </div>

          <select
            value={state?.trading_mode || 'MT5_DEMO'}
            onChange={async (e) => {
              await api.setMode(e.target.value);
              showToast(`Mode switched to ${e.target.value}`);
            }}
            className="mono text-xs font-bold px-2.5 py-1.5 rounded-md cockpit-card cursor-pointer focus:outline-none"
          >
            <option value="MT5_DEMO">MT5 DEMO</option>
            <option value="SIMULATOR">SIMULATOR</option>
            <option value="SHADOW_ONLY">SHADOW ONLY</option>
            <option value="MT5_LIVE">MT5 LIVE</option>
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

      {/* ANIMATED NEURAL PIPELINE PULSE BAR (Active Cognitive Heartbeat) */}
      <div className="bg-[#0E1522] border-b border-[#1F2D45] px-3.5 py-2 flex flex-wrap items-center justify-between gap-2 text-xs">
        <div className="flex items-center gap-1.5 overflow-x-auto">
          <span className="mono font-bold text-[#8A9BB5] mr-1">ENGINE LOOP</span>
          {PIPELINE_NODES.map((node, idx) => (
            <React.Fragment key={node}>
              <div
                className={`px-2 py-0.5 rounded mono border transition-all ${
                  idx === pipeStep
                    ? 'border-[#3B82F6] bg-[#3B82F6]/20 text-white font-bold'
                    : 'border-[#1F2D45] bg-[#121B2B] text-[#8A9BB5]'
                }`}
              >
                {node}
              </div>
              {idx < PIPELINE_NODES.length - 1 && <span className="text-[#8A9BB5]">›</span>}
            </React.Fragment>
          ))}
        </div>

        <div className="flex items-center gap-2.5 min-w-0">
          <span className="px-1.5 py-0.5 rounded bg-[#06B6D4]/15 text-[#06B6D4] border border-[#06B6D4]/40 mono font-bold shrink-0">HUNTING</span>
          <span className="mono text-xs truncate max-w-md text-[#E6EDF7]">{state?.active_thought || 'Scanning M1 microstructure...'}</span>
          <button
            onClick={async () => {
              const r = await api.triggerDemoSignal(selectedSym);
              showToast(r.ok ? `Signal fired on ${selectedSym}` : `Skipped: ${r.reason || 'inflight'}`);
            }}
            className="px-2.5 py-1 rounded bg-[#3B82F6]/15 text-[#3B82F6] border border-[#3B82F6]/40 font-bold flex items-center gap-1 cursor-pointer shrink-0 hover:bg-[#3B82F6]/25"
          >
            <Zap className="w-3 h-3" /> Demo Signal
          </button>
        </div>
      </div>

      {/* MAIN WORKSPACE */}
      <div className="flex-1 flex flex-col lg:flex-row">
        {/* Left Icon Rail */}
        <aside className="lg:w-52 bg-[#0A101B] border-b lg:border-b-0 lg:border-r border-[#1F2D45] p-2 flex lg:flex-col gap-1 overflow-x-auto shrink-0">
          {[
            { id: 'cockpit' as TabId, label: 'Execution Cockpit', icon: LayoutGrid },
            { id: 'brain' as TabId, label: 'AI Brain & Regimes', icon: Cpu },
            { id: 'evolution' as TabId, label: 'Evolution & Shadow', icon: RefreshCw, badge: state?.active_challenger ? 1 : 0 },
            { id: 'journal' as TabId, label: 'Trade Forensics', icon: FileText },
            { id: 'risk' as TabId, label: 'Risk, Shields & MT5', icon: Shield },
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
              <div className="flex justify-between"><span className="text-[#8A9BB5]">Shadow Eq</span><span className="mono font-bold text-[#8B5CF6]">${(state?.shadow_equity ?? 10000).toFixed(0)}</span></div>
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
                            <span className="px-1.5 py-0.5 rounded bg-[#10B981]/15 text-[#10B981] font-bold">{r.regime}</span>
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

              {/* Row 2: Live Positions & Cognitive Stream */}
              <div className="grid grid-cols-1 xl:grid-cols-12 gap-3">
                <div className="xl:col-span-7 cockpit-card rounded-lg flex flex-col">
                  <div className="px-3 py-2 border-b border-[#1F2D45] flex items-center justify-between text-xs">
                    <span className="font-bold uppercase tracking-wider">Live MT5 Positions ({state?.positions?.length ?? 0})</span>
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
                        {(state?.positions || []).length === 0 ? (
                          <tr><td colSpan={6} className="py-6 text-center text-[#8A9BB5]">No open positions — AI actively hunting M1 setups...</td></tr>
                        ) : (
                          (state?.positions || []).map((p) => (
                            <tr key={p.ticket}>
                              <td className="py-2 px-2"><div className="mono font-bold">{p.symbol}</div><div className="text-[#8A9BB5]">{p.strategy_id}</div></td>
                              <td className="py-2 px-2 mono"><span className={`px-1.5 py-0.5 rounded font-bold ${p.side === 'BUY' ? 'bg-[#10B981]/15 text-[#10B981]' : 'bg-[#EF4444]/15 text-[#EF4444]'}`}>{p.side}</span> {p.volume}</td>
                              <td className="py-2 px-2 mono">{p.open_price} → <b>{p.current_price}</b></td>
                              <td className={`py-2 px-2 mono font-bold ${p.profit >= 0 ? 'text-[#10B981]' : 'text-[#EF4444]'}`}>{p.profit >= 0 ? '+' : ''}${p.profit.toFixed(2)}</td>
                              <td className="py-2 px-2"><span className="px-1.5 py-0.5 rounded bg-[#06B6D4]/15 text-[#06B6D4] mono font-bold">{p.rl_exit_action}</span></td>
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
                      filteredDecisions.map((d) => (
                        <div key={d.id} className="p-2 rounded cockpit-sub text-xs space-y-0.5">
                          <div className="flex items-center justify-between mono">
                            <span>
                              <span className="text-[#8A9BB5] mr-1.5">{new Date(d.timestamp_ms).toISOString().substring(11, 19)}</span>
                              <b>{d.symbol}</b>
                            </span>
                            <span className={`px-1.5 py-0.5 rounded font-bold ${d.category === 'EXEC' ? 'bg-[#10B981]/15 text-[#10B981]' : d.category === 'BLOCK' ? 'bg-[#F59E0B]/15 text-[#F59E0B]' : 'bg-[#06B6D4]/15 text-[#06B6D4]'}`}>
                              {d.tag}
                            </span>
                          </div>
                          <div className="text-[#8A9BB5] leading-snug">{d.message}</div>
                        </div>
                      ))
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
