import React, { useEffect, useState } from 'react';
import { ArrowDown, ArrowUp, Check, Loader2, Plus, RefreshCw, Trash2, X } from 'lucide-react';
import { api } from '../api/client';
import { BrokerCostsCard } from './BrokerCostsCard';
import { HitRunCard } from './HitRunCard';
import { FiltersCard } from './FiltersCard';
import { human } from '../labels';
import { LOGO_FULL_URL } from './BullishLogo';

// Setup page:
// 1. API Providers (credentials & connection checks)
// 2. Cross-provider Main Model Chain (Reflection / Evolution) & Fast Model Chain (AI Supervisor)
// 3. AI Supervisor, MT5, Broker Costs, Hit & Run, Market Filters, and Safety info.
// Keys are written to local .env only and returned masked. Live trading cannot be unlocked here.

type Provider = {
  name: string;
  label: string;
  note: string;
  key_url: string;
  needs_key: boolean;
  configured: boolean;
  api_key_masked: string;
  model: string;
  fast_model: string;
  fallback_models: string[];
  base_url: string;
  default_model: string;
  default_fast_model: string;
  default_base_url: string;
  locked_by_environment: string[];
  stats: { calls?: number; failures?: number; last_error?: string; total_cost_usd?: number };
};

type Draft = { api_key: string; clear_key: boolean; base_url: string };

const inputCls = 'w-full px-2 py-1.5 rounded bg-[#070B12] border border-[#1F2D45] mono text-xs focus:outline-none focus:border-[#3B82F6]';
const btnCls = 'px-2.5 py-1.5 rounded text-xs font-bold cursor-pointer border flex items-center gap-1 disabled:opacity-50';
const LOCAL = new Set(['ollama', 'llamacpp', 'custom']);

export const SetupView: React.FC<{ onToast: (msg: string) => void }> = ({ onToast }) => {
  const [setup, setSetup] = useState<any>(null);
  const [chain, setChain] = useState<string[]>([]);
  const [mainChain, setMainChain] = useState<string[]>([]);
  const [fastChain, setFastChain] = useState<string[]>([]);
  const [drafts, setDrafts] = useState<Record<string, Draft>>({});
  const [models, setModels] = useState<Record<string, string[]>>({});
  const [busy, setBusy] = useState<string>('');
  const [testResults, setTestResults] = useState<Record<string, string>>({});
  const [sup, setSup] = useState<any>(null);
  const [addPick, setAddPick] = useState('');

  const load = async () => {
    const s = await api.getSetup();
    setSetup(s);
    setChain(s.llm.chain);
    setMainChain(s.llm.main_chain || []);
    setFastChain(s.llm.fast_chain || []);
    const d: Record<string, Draft> = {};
    for (const p of s.llm.providers as Provider[]) {
      d[p.name] = { api_key: '', clear_key: false, base_url: p.base_url };
    }
    setDrafts(d);
    setSup(s.supervisor);
  };

  useEffect(() => {
    load().catch((e) => onToast(`Could not load setup: ${(e as Error).message}`));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (!setup) return <div className="cockpit-card rounded-lg p-6 text-xs text-[#8A9BB5]">Loading setup...</div>;

  const providers: Provider[] = setup.llm.providers;
  const byName = Object.fromEntries(providers.map((p) => [p.name, p]));
  const setDraft = (name: string, patch: Partial<Draft>) => setDrafts((d) => ({ ...d, [name]: { ...d[name], ...patch } }));

  const fetchModels = async (name: string) => {
    setBusy(`models:${name}`);
    try {
      const d = drafts[name];
      const r = await api.listModels(name, d.api_key, LOCAL.has(name) ? d.base_url : '');
      const ids: string[] = r.models.map((m: any) => m.id);
      setModels((m) => ({ ...m, [name]: ids }));
      onToast(`${ids.length} models found at ${byName[name].label}`);
    } catch (e) {
      onToast(`Could not fetch models: ${(e as Error).message}`);
    } finally {
      setBusy('');
    }
  };

  const saveLlm = async () => {
    setBusy('save');
    try {
      const body: any = { chain, main_chain: mainChain, fast_chain: fastChain, providers: {} };
      for (const p of providers) {
        const d = drafts[p.name];
        body.providers[p.name] = {
          api_key: d.api_key || null,
          clear_key: d.clear_key,
          base_url: d.base_url === p.default_base_url ? '' : d.base_url,
        };
      }
      await api.saveLlmSetup(body);
      await load();
      onToast('AI keys and cross-provider model chains saved to .env and active now');
    } catch (e) {
      onToast(`Could not save: ${(e as Error).message}`);
    } finally {
      setBusy('');
    }
  };

  const test = async (name: string, model?: string) => {
    const key = model ? `${name}:${model}` : name;
    setBusy(`test:${key}`);
    try {
      const d = drafts[name];
      const r = await api.testLlm(name, model, d?.api_key || '', LOCAL.has(name) ? d?.base_url || '' : '');
      const x = r.results[0];
      setTestResults((t) => ({ ...t, [key]: x.ok ? `OK · ${x.model} · ${x.latency_ms}ms` : `FAILED · ${x.error}` }));
    } catch (e) {
      setTestResults((t) => ({ ...t, [key]: `FAILED · ${(e as Error).message}` }));
    } finally {
      setBusy('');
    }
  };

  const saveSup = async () => {
    setBusy('sup');
    try {
      const r = await api.saveSupervisorSetup({
        enabled: sup.enabled,
        mode: sup.mode || 'periodic',
        brief_interval_min: Number(sup.brief_interval_min || 15),
        timeout_s: Number(sup.timeout_s),
        use_fast_model: sup.use_fast_model,
        effort: sup.effort,
        on_unavailable: sup.on_unavailable,
        max_calls_per_hour: Number(sup.max_calls_per_hour),
        daily_budget_usd: Number(sup.daily_budget_usd),
        max_price_drift_atr: Number(sup.max_price_drift_atr),
        review_in_shadow: sup.review_in_shadow,
      });
      setSup(r.supervisor);
      onToast('AI Supervisor settings saved');
    } catch (e) {
      onToast(`Could not save supervisor: ${(e as Error).message}`);
    } finally {
      setBusy('');
    }
  };

  const refreshBrief = async () => {
    setBusy('brief');
    try {
      const r = await api.refreshSupervisorBrief();
      setSup(r.supervisor);
      onToast('Desk Brief refreshed');
    } catch (e) {
      onToast(`Could not refresh Desk Brief: ${(e as Error).message}`);
    } finally {
      setBusy('');
    }
  };

  const notInChain = providers.filter((p) => !chain.includes(p.name));
  const availableProviders = chain.map((n) => byName[n]).filter(Boolean);

  return (
    <div className="space-y-3">
      <div className="rounded-lg p-3 flex justify-center bg-[#F4F7FB]">
        <img src={LOGO_FULL_URL} alt="Max Trading Agent" className="h-24 w-auto object-contain" />
      </div>

      {/* 1. API PROVIDERS (CREDENTIALS) */}
      <div className="cockpit-card rounded-lg p-3.5 space-y-3">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <div>
            <h3 className="text-xs font-bold uppercase tracking-wider">1. AI API Providers (Keys &amp; Connection)</h3>
            <p className="text-xs text-[#8A9BB5] mt-0.5">
              Enter your API keys below, click <b>Check models</b> to load each provider&apos;s model list, then arrange cross-provider order in the
              Main &amp; Fast Model chains below. Keys are stored only in local .env.
            </p>
          </div>
          <button onClick={saveLlm} disabled={busy === 'save'} className={`${btnCls} bg-[#3B82F6] text-white border-[#3B82F6]`}>
            {busy === 'save' ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Check className="w-3.5 h-3.5" />} Save AI
          </button>
        </div>

        {chain.length === 0 && <div className="text-xs text-[#F59E0B]">No API provider added yet: the robot runs without AI.</div>}

        <div className="grid grid-cols-1 xl:grid-cols-2 gap-2.5">
          {chain.map((name) => {
            const p = byName[name];
            const d = drafts[name];
            if (!p || !d) return null;
            return (
              <div key={name} className="cockpit-sub rounded-md p-3 space-y-2 text-xs">
                <div className="flex items-center justify-between gap-2 flex-wrap">
                  <div className="flex items-center gap-2">
                    <span className="font-bold text-sm">{p.label}</span>
                    <span
                      className={`px-1.5 py-0.5 rounded mono font-bold text-[11px] ${p.configured ? 'bg-[#10B981]/15 text-[#10B981]' : 'bg-[#F59E0B]/15 text-[#F59E0B]'}`}
                    >
                      {p.configured ? 'READY' : 'NO KEY YET'}
                    </span>
                  </div>
                  <button
                    onClick={() => setChain(chain.filter((n) => n !== name))}
                    className={`${btnCls} border-[#EF4444]/40 text-[#EF4444]`}
                    title="Remove provider"
                  >
                    <X className="w-3.5 h-3.5" />
                  </button>
                </div>

                {p.needs_key && (
                  <label className="space-y-1 block">
                    <span className="text-[#8A9BB5]">
                      API key {p.api_key_masked && <b className="mono text-[#E6EDF7]">saved {p.api_key_masked}</b>}{' '}
                      {p.key_url && (
                        <a href={p.key_url} target="_blank" rel="noreferrer" className="text-[#3B82F6] underline">
                          get key
                        </a>
                      )}
                    </span>
                    <div className="flex gap-1.5">
                      <input
                        type="password"
                        autoComplete="off"
                        value={d.api_key}
                        onChange={(e) => setDraft(name, { api_key: e.target.value, clear_key: false })}
                        placeholder={p.api_key_masked ? 'leave empty = keep saved key' : 'paste API key here'}
                        className={inputCls}
                      />
                      {p.api_key_masked && (
                        <button
                          onClick={() => setDraft(name, { clear_key: !d.clear_key, api_key: '' })}
                          className={`${btnCls} ${d.clear_key ? 'bg-[#EF4444] text-white border-[#EF4444]' : 'border-[#1F2D45]'}`}
                          title="Delete key on save"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      )}
                    </div>
                  </label>
                )}

                {(LOCAL.has(name) || !p.needs_key) && (
                  <label className="space-y-1 block">
                    <span className="text-[#8A9BB5]">Server address (base URL)</span>
                    <input value={d.base_url} onChange={(e) => setDraft(name, { base_url: e.target.value })} className={inputCls} />
                  </label>
                )}

                <div className="flex flex-wrap items-center gap-1.5 pt-0.5">
                  <button onClick={() => fetchModels(name)} disabled={busy === `models:${name}`} className={`${btnCls} border-[#06B6D4]/50 text-[#06B6D4]`}>
                    {busy === `models:${name}` ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <RefreshCw className="w-3.5 h-3.5" />} Check models
                    {models[name]?.length ? ` (${models[name].length})` : ''}
                  </button>
                  <button onClick={() => test(name)} disabled={busy === `test:${name}`} className={`${btnCls} border-[#10B981]/50 text-[#10B981]`}>
                    {busy === `test:${name}` ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Check className="w-3.5 h-3.5" />} Test key
                  </button>
                </div>
                {testResults[name] && (
                  <div className={`mono ${testResults[name].startsWith('OK') ? 'text-[#10B981]' : 'text-[#EF4444]'}`}>{testResults[name]}</div>
                )}
                {p.locked_by_environment.length > 0 && (
                  <div className="text-[#F59E0B]">Locked by env vars ({p.locked_by_environment.join(', ')})</div>
                )}
                {p.stats?.last_error && <div className="text-[#EF4444] mono truncate">Last error: {p.stats.last_error}</div>}
              </div>
            );
          })}
        </div>

        <div className="flex items-center gap-2 text-xs">
          <select value={addPick} onChange={(e) => setAddPick(e.target.value)} className={`${inputCls} max-w-xs`}>
            <option value="">+ Add API provider...</option>
            {notInChain.map((p) => (
              <option key={p.name} value={p.name}>
                {p.label}
              </option>
            ))}
          </select>
          <button
            disabled={!addPick}
            onClick={() => {
              setChain([...chain, addPick]);
              setAddPick('');
            }}
            className={`${btnCls} border-[#3B82F6]/50 text-[#3B82F6]`}
          >
            <Plus className="w-3.5 h-3.5" /> Add
          </button>
        </div>
      </div>

      {/* 2. CROSS-PROVIDER MODEL CHAINS (MAIN & FAST) */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-3">
        <ModelChainCard
          title="2A. Main Model Chain (Reflection & Evolution)"
          subtitle="Tried top to bottom across any provider for deep trade analysis every 4h."
          entries={mainChain}
          onChange={setMainChain}
          providers={availableProviders}
          modelsByProvider={models}
          defaultKind="main"
          busy={busy}
          testResults={testResults}
          onTest={test}
          onSave={saveLlm}
        />
        <ModelChainCard
          title="2B. Fast Model Chain (AI Supervisor & Desk Brief)"
          subtitle="Tried top to bottom across any provider for fast Desk Brief / veto calls."
          entries={fastChain}
          onChange={setFastChain}
          providers={availableProviders}
          modelsByProvider={models}
          defaultKind="fast"
          busy={busy}
          testResults={testResults}
          onTest={test}
          onSave={saveLlm}
        />
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-12 gap-3">
        {/* SUPERVISOR */}
        <div className="xl:col-span-7 cockpit-card rounded-lg p-3.5 space-y-3 text-xs">
          <div className="flex items-center justify-between flex-wrap gap-2">
            <div>
              <h3 className="font-bold uppercase tracking-wider">AI Supervisor</h3>
              <p className="text-[#8A9BB5] mt-0.5">Can only approve, reduce or reject a trade. Never raises risk.</p>
            </div>
            <div className="flex items-center gap-1.5">
              {sup?.mode !== 'per_trade' && (
                <button
                  onClick={refreshBrief}
                  disabled={busy === 'brief' || !sup?.enabled}
                  className={`${btnCls} border-[#06B6D4]/50 text-[#06B6D4]`}
                  title="Force an immediate multi-symbol Desk Brief update from the active AI provider"
                >
                  {busy === 'brief' ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <RefreshCw className="w-3.5 h-3.5" />} Refresh Brief
                </button>
              )}
              <button onClick={saveSup} disabled={busy === 'sup'} className={`${btnCls} bg-[#3B82F6] text-white border-[#3B82F6]`}>
                <Check className="w-3.5 h-3.5" /> Save
              </button>
            </div>
          </div>
          {sup && (
            <div className="grid grid-cols-2 md:grid-cols-3 gap-2.5">
              <Toggle label="On" value={sup.enabled} onChange={(v) => setSup({ ...sup, enabled: v })} />
              <Toggle label="Use the fast model" value={sup.use_fast_model} onChange={(v) => setSup({ ...sup, use_fast_model: v })} />
              <Toggle label="Also review in paper mode" value={sup.review_in_shadow} onChange={(v) => setSup({ ...sup, review_in_shadow: v })} />
              <label className="space-y-1 md:col-span-2">
                <span className="text-[#8A9BB5]">Supervision mode</span>
                <select value={sup.mode || 'periodic'} onChange={(e) => setSup({ ...sup, mode: e.target.value })} className={inputCls}>
                  <option value="periodic">Periodic desk brief (0ms per trade, zero M1 drift)</option>
                  <option value="hybrid">Hybrid (brief for M1 scalps, per-trade veto for slow setups)</option>
                  <option value="per_trade">Per-trade veto (live LLM call on every signal)</option>
                </select>
              </label>
              <Num
                label="Brief interval (min)"
                value={sup.brief_interval_min ?? 15}
                step={5}
                onChange={(v) => setSup({ ...sup, brief_interval_min: Math.max(1, Math.round(v)) })}
              />
              <Num label="Timeout (seconds)" value={sup.timeout_s} step={0.5} onChange={(v) => setSup({ ...sup, timeout_s: v })} />
              <Num label="Daily budget ($)" value={sup.daily_budget_usd} step={0.5} onChange={(v) => setSup({ ...sup, daily_budget_usd: v })} />
              <Num label="Max calls / hour" value={sup.max_calls_per_hour} step={10} onChange={(v) => setSup({ ...sup, max_calls_per_hour: v })} />
              <label className="space-y-1">
                <span className="text-[#8A9BB5]">Effort</span>
                <select value={sup.effort} onChange={(e) => setSup({ ...sup, effort: e.target.value })} className={inputCls}>
                  <option value="low">low (fastest)</option>
                  <option value="medium">medium</option>
                  <option value="high">high</option>
                </select>
              </label>
              <label className="space-y-1">
                <span className="text-[#8A9BB5]">If the AI does not answer</span>
                <select value={sup.on_unavailable} onChange={(e) => setSup({ ...sup, on_unavailable: e.target.value })} className={inputCls}>
                  <option value="block">Cancel the trade (safe)</option>
                  <option value="allow">Trade anyway without veto</option>
                </select>
              </label>
              <Num label="Max price drift (ATR)" value={sup.max_price_drift_atr} step={0.05} onChange={(v) => setSup({ ...sup, max_price_drift_atr: v })} />
            </div>
          )}
          {sup?.status && (
            <div className="space-y-2">
              <div className="mono text-[#8A9BB5]">
                Mode: <b className="text-[#E6EDF7] uppercase">{sup.status.mode || 'periodic'}</b> · Today: {sup.status.calls} calls · ✓{sup.status.approve} ↓{sup.status.reduce} ✗{sup.status.reject} · ${sup.status.spent_today_usd?.toFixed(3)}
              </div>
              {sup.status.brief && Object.keys(sup.status.brief).length > 0 && (
                <div className="cockpit-sub rounded-md p-2.5 space-y-1.5">
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-bold text-[#06B6D4]">Active Desk Brief</span>
                    {sup.status.brief_summary && <span className="text-[#8A9BB5] truncate">{sup.status.brief_summary}</span>}
                  </div>
                  <div className="flex flex-wrap gap-1.5 mono">
                    {Object.entries(sup.status.brief).map(([sym, b]: [string, any]) => {
                      const tone =
                        b.bias === 'LONG_ONLY'
                          ? 'border-[#10B981]/40 text-[#10B981] bg-[#10B981]/10'
                          : b.bias === 'SHORT_ONLY'
                            ? 'border-[#EF4444]/40 text-[#EF4444] bg-[#EF4444]/10'
                            : b.bias === 'FLAT'
                              ? 'border-[#F59E0B]/40 text-[#F59E0B] bg-[#F59E0B]/10'
                              : 'border-[#06B6D4]/40 text-[#06B6D4] bg-[#06B6D4]/10';
                      return (
                        <span key={sym} title={b.note || ''} className={`px-2 py-0.5 rounded border text-[11px] font-bold ${tone}`}>
                          {sym}: {b.bias}
                          {b.risk_multiplier < 0.99 ? ` ×${b.risk_multiplier.toFixed(1)}` : ''}
                        </span>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* MT5 + SAFETY */}
        <div className="xl:col-span-5 space-y-3 text-xs">
          <div className="cockpit-card rounded-lg p-3.5 space-y-2">
            <h3 className="font-bold uppercase tracking-wider">MetaTrader 5</h3>
            <Row k="Status EA" v={setup.mt5.bridge_connected ? 'CONNECTED' : 'not connected'} ok={setup.mt5.bridge_connected} />
            <Row k="Account" v={`${setup.mt5.account_is_demo ? 'DEMO' : 'REAL'} · ${setup.mt5.account_currency}`} ok={setup.mt5.account_is_demo} />
            <Row k="Markets" v={setup.mt5.symbols.join(', ')} />
            <Row k="Bridge URL" v={setup.mt5.bridge_url} />
            <Row k="Bridge token" v={setup.mt5.bridge_token_masked} />
            <p className="text-[#8A9BB5]">
              Broker symbol names (e.g. XAUUSD.m) are detected by the EA. start_all.bat installs and compiles the EA.
            </p>
          </div>
          {setup.costs && (
            <BrokerCostsCard costs={setup.costs} onSaved={(c) => setSetup({ ...setup, costs: c })} onToast={onToast} />
          )}
          {setup.hit_run && (
            <HitRunCard settings={setup.hit_run} onSaved={(h) => setSetup({ ...setup, hit_run: h })} onToast={onToast} />
          )}
          {setup.filters && (
            <FiltersCard view={setup.filters} onSaved={(f) => setSetup({ ...setup, filters: f })} onToast={onToast} />
          )}
          <div className="cockpit-card rounded-lg p-3.5 space-y-2">
            <h3 className="font-bold uppercase tracking-wider">Safety</h3>
            <Row k="Current mode" v={human(setup.safety.trading_mode)} />
            <Row k="Real money" v={setup.safety.live_trading_enabled ? 'ALLOWED in .env' : 'LOCKED'} ok={!setup.safety.live_trading_enabled} />
            <p className="text-[#8A9BB5]">
              Real money cannot be unlocked from this page. To do it, set ALLOW_LIVE_TRADING=true in the .env file by hand and restart.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

const ModelChainCard: React.FC<{
  title: string;
  subtitle: string;
  entries: string[];
  onChange: (next: string[]) => void;
  providers: Provider[];
  modelsByProvider: Record<string, string[]>;
  defaultKind: 'main' | 'fast';
  busy: string;
  testResults: Record<string, string>;
  onTest: (provider: string, model: string) => void;
  onSave: () => void;
}> = ({ title, subtitle, entries, onChange, providers, modelsByProvider, defaultKind, busy, testResults, onTest, onSave }) => {
  const [pickProv, setPickProv] = useState(providers[0]?.name || '');
  const [pickModel, setPickModel] = useState('');

  useEffect(() => {
    if (!pickProv && providers.length) setPickProv(providers[0].name);
  }, [providers, pickProv]);

  const activeProv = providers.find((p) => p.name === pickProv) || providers[0];
  const opts = activeProv ? modelsByProvider[activeProv.name] || [] : [];

  const move = (i: number, dir: -1 | 1) => {
    const j = i + dir;
    if (j < 0 || j >= entries.length) return;
    const next = [...entries];
    [next[i], next[j]] = [next[j], next[i]];
    onChange(next);
  };

  const addEntry = () => {
    if (!activeProv) return;
    const m = (pickModel.trim() || (defaultKind === 'fast' ? activeProv.fast_model : activeProv.model)).trim();
    if (!m) return;
    const item = `${activeProv.name}:${m}`;
    if (!entries.includes(item)) onChange([...entries, item]);
    setPickModel('');
  };

  return (
    <div className="cockpit-card rounded-lg p-3.5 space-y-2.5 text-xs">
      <div className="flex items-start justify-between gap-2">
        <div>
          <h3 className="font-bold uppercase tracking-wider">{title}</h3>
          <p className="text-[#8A9BB5] mt-0.5">{subtitle}</p>
        </div>
        <button onClick={onSave} disabled={busy === 'save'} className={`${btnCls} bg-[#3B82F6] text-white border-[#3B82F6] shrink-0`}>
          {busy === 'save' ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Check className="w-3.5 h-3.5" />} Save
        </button>
      </div>

      {entries.length === 0 ? (
        <div className="cockpit-sub rounded-md p-2.5 text-[#8A9BB5]">No models in chain yet — add one below, then click Save AI.</div>
      ) : (
        <div className="space-y-1.5">
          {entries.map((item, i) => {
            const idx = item.indexOf(':');
            const prov = idx >= 0 ? item.slice(0, idx) : item;
            const mod = idx >= 0 ? item.slice(idx + 1) : '';
            const res = testResults[item];
            return (
              <div key={item} className="cockpit-sub rounded-md px-2.5 py-2 space-y-1">
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-2 min-w-0">
                    <span className="mono font-bold text-[#8A9BB5] shrink-0">#{i + 1}</span>
                    <span className="px-1.5 py-0.5 rounded bg-[#3B82F6]/15 text-[#3B82F6] mono font-bold uppercase text-[10px] shrink-0">
                      {prov}
                    </span>
                    <span className="mono font-bold truncate" title={mod}>
                      {mod}
                    </span>
                  </div>
                  <div className="flex items-center gap-1 shrink-0">
                    <button
                      onClick={() => onTest(prov, mod)}
                      disabled={busy === `test:${item}`}
                      className={`${btnCls} py-1 px-2 border-[#10B981]/40 text-[#10B981]`}
                      title="Test this provider:model"
                    >
                      {busy === `test:${item}` ? <Loader2 className="w-3 h-3 animate-spin" /> : <Check className="w-3 h-3" />}
                    </button>
                    <button onClick={() => move(i, -1)} className={`${btnCls} py-1 px-1.5 border-[#1F2D45]`} title="Move up">
                      <ArrowUp className="w-3 h-3" />
                    </button>
                    <button onClick={() => move(i, 1)} className={`${btnCls} py-1 px-1.5 border-[#1F2D45]`} title="Move down">
                      <ArrowDown className="w-3 h-3" />
                    </button>
                    <button
                      onClick={() => onChange(entries.filter((x) => x !== item))}
                      className={`${btnCls} py-1 px-1.5 border-[#EF4444]/40 text-[#EF4444]`}
                      title="Remove"
                    >
                      <X className="w-3 h-3" />
                    </button>
                  </div>
                </div>
                {res && <div className={`mono text-[11px] ${res.startsWith('OK') ? 'text-[#10B981]' : 'text-[#EF4444]'}`}>{res}</div>}
              </div>
            );
          })}
        </div>
      )}

      <div className="grid grid-cols-1 sm:grid-cols-12 gap-1.5 pt-1">
        <select
          value={activeProv?.name || ''}
          onChange={(e) => {
            setPickProv(e.target.value);
            setPickModel('');
          }}
          className={`${inputCls} sm:col-span-4`}
        >
          {providers.map((p) => (
            <option key={p.name} value={p.name}>
              {p.name}
            </option>
          ))}
        </select>
        {opts.length > 0 ? (
          <select value={pickModel} onChange={(e) => setPickModel(e.target.value)} className={`${inputCls} sm:col-span-6`}>
            <option value="">pick fetched model...</option>
            {opts.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        ) : (
          <input
            value={pickModel}
            onChange={(e) => setPickModel(e.target.value)}
            placeholder={activeProv ? (defaultKind === 'fast' ? activeProv.fast_model : activeProv.model) : 'model id'}
            className={`${inputCls} sm:col-span-6`}
          />
        )}
        <button
          disabled={!activeProv}
          onClick={addEntry}
          className={`${btnCls} sm:col-span-2 justify-center border-[#3B82F6]/50 text-[#3B82F6]`}
        >
          <Plus className="w-3.5 h-3.5" /> Add
        </button>
      </div>
    </div>
  );
};

const Toggle: React.FC<{ label: string; value: boolean; onChange: (v: boolean) => void }> = ({ label, value, onChange }) => (
  <label className="flex items-center gap-2 cursor-pointer">
    <input type="checkbox" checked={value} onChange={(e) => onChange(e.target.checked)} />
    <span>{label}</span>
  </label>
);

const Num: React.FC<{ label: string; value: number; step: number; onChange: (v: number) => void }> = ({ label, value, step, onChange }) => (
  <label className="space-y-1">
    <span className="text-[#8A9BB5]">{label}</span>
    <input type="number" step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} className={inputCls} />
  </label>
);

const Row: React.FC<{ k: string; v: string; ok?: boolean }> = ({ k, v, ok }) => (
  <div className="flex justify-between gap-2">
    <span className="text-[#8A9BB5]">{k}</span>
    <span className={`mono font-bold text-right truncate ${ok === undefined ? '' : ok ? 'text-[#10B981]' : 'text-[#F59E0B]'}`}>{v}</span>
  </div>
);
