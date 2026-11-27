import React, { useEffect, useState } from 'react';
import { ArrowDown, ArrowUp, Check, Loader2, Plus, RefreshCw, Trash2, X } from 'lucide-react';
import { api } from '../api/client';
import { BrokerCostsCard } from './BrokerCostsCard';
import { human } from '../labels';
import { LOGO_FULL_URL } from './BullishLogo';

// Setup page: AI providers (order, key, model from the provider's list, fallback models),
// Claude supervisor settings, MT5 and safety info. Keys are written to the local .env only
// and come back masked. Live trading can't be unlocked here.

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

type Draft = { api_key: string; clear_key: boolean; model: string; fast_model: string; base_url: string; fallback_models: string[] };

const inputCls = 'w-full px-2 py-1.5 rounded bg-[#070B12] border border-[#1F2D45] mono text-xs focus:outline-none focus:border-[#3B82F6]';
const btnCls = 'px-2.5 py-1.5 rounded text-xs font-bold cursor-pointer border flex items-center gap-1 disabled:opacity-50';
const LOCAL = new Set(['ollama', 'llamacpp', 'custom']);

const ModelPicker: React.FC<{ value: string; options: string[]; onChange: (v: string) => void; placeholder: string }> = ({
  value,
  options,
  onChange,
  placeholder,
}) =>
  options.length ? (
    <select value={value} onChange={(e) => onChange(e.target.value)} className={inputCls}>
      {!options.includes(value) && value && <option value={value}>{value} (manual)</option>}
      {options.map((m) => (
        <option key={m} value={m}>
          {m}
        </option>
      ))}
    </select>
  ) : (
    <input value={value} onChange={(e) => onChange(e.target.value)} placeholder={placeholder} className={inputCls} />
  );

export const SetupView: React.FC<{ onToast: (msg: string) => void }> = ({ onToast }) => {
  const [setup, setSetup] = useState<any>(null);
  const [chain, setChain] = useState<string[]>([]);
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
    const d: Record<string, Draft> = {};
    for (const p of s.llm.providers as Provider[]) {
      d[p.name] = { api_key: '', clear_key: false, model: p.model, fast_model: p.fast_model, base_url: p.base_url, fallback_models: p.fallback_models };
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

  const move = (i: number, dir: -1 | 1) => {
    const j = i + dir;
    if (j < 0 || j >= chain.length) return;
    const c = [...chain];
    [c[i], c[j]] = [c[j], c[i]];
    setChain(c);
  };

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
      const body: any = { chain, providers: {} };
      for (const p of providers) {
        const d = drafts[p.name];
        body.providers[p.name] = {
          api_key: d.api_key || null,
          clear_key: d.clear_key,
          model: d.model === p.default_model ? '' : d.model,
          fast_model: d.fast_model === p.default_fast_model ? '' : d.fast_model,
          base_url: d.base_url === p.default_base_url ? '' : d.base_url,
          fallback_models: d.fallback_models,
        };
      }
      await api.saveLlmSetup(body);
      await load();
      onToast('AI setup saved to .env and active now');
    } catch (e) {
      onToast(`Could not save: ${(e as Error).message}`);
    } finally {
      setBusy('');
    }
  };

  const test = async (name: string) => {
    setBusy(`test:${name}`);
    try {
      const r = await api.testLlm(name);
      const x = r.results[0];
      setTestResults((t) => ({ ...t, [name]: x.ok ? `OK · ${x.model} · ${x.latency_ms}ms` : `GAGAL · ${x.error}` }));
    } catch (e) {
      setTestResults((t) => ({ ...t, [name]: `GAGAL · ${(e as Error).message}` }));
    } finally {
      setBusy('');
    }
  };

  const saveSup = async () => {
    setBusy('sup');
    try {
      const r = await api.saveSupervisorSetup({
        enabled: sup.enabled,
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
      onToast('Supervisor settings saved');
    } catch (e) {
      onToast(`Could not save supervisor: ${(e as Error).message}`);
    } finally {
      setBusy('');
    }
  };

  const notInChain = providers.filter((p) => !chain.includes(p.name));

  return (
    <div className="space-y-3">
      <div className="rounded-lg p-3 flex justify-center bg-[#F4F7FB]">
        <img src={LOGO_FULL_URL} alt="Max Trading Agent" className="h-24 w-auto object-contain" />
      </div>
      {/* AI PROVIDERS */}
      <div className="cockpit-card rounded-lg p-3.5 space-y-3">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <div>
            <h3 className="text-xs font-bold uppercase tracking-wider">AI providers & fallback order</h3>
            <p className="text-xs text-[#8A9BB5] mt-0.5">
              Tried top to bottom. If the main model fails, the fallback models of the same provider are tried first, then the next provider.
              Keys are stored only in the local .env file.
            </p>
          </div>
          <button onClick={saveLlm} disabled={busy === 'save'} className={`${btnCls} bg-[#3B82F6] text-white border-[#3B82F6]`}>
            {busy === 'save' ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Check className="w-3.5 h-3.5" />} Save AI
          </button>
        </div>

        {chain.length === 0 && <div className="text-xs text-[#F59E0B]">No provider yet: the robot runs without AI (supervisor off).</div>}

        {chain.map((name, i) => {
          const p = byName[name];
          const d = drafts[name];
          if (!p || !d) return null;
          const opts = models[name] || [];
          return (
            <div key={name} className="cockpit-sub rounded-md p-3 space-y-2.5">
              <div className="flex items-center justify-between gap-2 flex-wrap">
                <div className="flex items-center gap-2">
                  <span className="mono font-bold text-[#8A9BB5]">#{i + 1}</span>
                  <span className="font-bold text-sm">{p.label}</span>
                  <span
                    className={`px-1.5 py-0.5 rounded mono font-bold text-xs ${p.configured ? 'bg-[#10B981]/15 text-[#10B981]' : 'bg-[#F59E0B]/15 text-[#F59E0B]'}`}
                  >
                    {p.configured ? 'SIAP' : 'BELUM ADA KEY'}
                  </span>
                </div>
                <div className="flex items-center gap-1">
                  <button onClick={() => move(i, -1)} className={`${btnCls} border-[#1F2D45]`} title="Move up">
                    <ArrowUp className="w-3.5 h-3.5" />
                  </button>
                  <button onClick={() => move(i, 1)} className={`${btnCls} border-[#1F2D45]`} title="Move down">
                    <ArrowDown className="w-3.5 h-3.5" />
                  </button>
                  <button onClick={() => setChain(chain.filter((n) => n !== name))} className={`${btnCls} border-[#EF4444]/40 text-[#EF4444]`} title="Remove from the order">
                    <X className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>
              {p.note && <div className="text-xs text-[#8A9BB5]">{p.note}</div>}

              <div className="grid grid-cols-1 md:grid-cols-2 gap-2.5 text-xs">
                {p.needs_key && (
                  <label className="space-y-1">
                    <span className="text-[#8A9BB5]">
                      API key {p.api_key_masked && <b className="mono text-[#E6EDF7]">saved {p.api_key_masked}</b>}{' '}
                      {p.key_url && (
                        <a href={p.key_url} target="_blank" rel="noreferrer" className="text-[#3B82F6] underline">
                          get a key
                        </a>
                      )}
                    </span>
                    <div className="flex gap-1.5">
                      <input
                        type="password"
                        autoComplete="off"
                        value={d.api_key}
                        onChange={(e) => setDraft(name, { api_key: e.target.value, clear_key: false })}
                        placeholder={p.api_key_masked ? 'leave empty = keep the saved key' : 'paste key here'}
                        className={inputCls}
                      />
                      {p.api_key_masked && (
                        <button
                          onClick={() => setDraft(name, { clear_key: !d.clear_key, api_key: '' })}
                          className={`${btnCls} ${d.clear_key ? 'bg-[#EF4444] text-white border-[#EF4444]' : 'border-[#1F2D45]'}`}
                          title="Delete the key on save"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      )}
                    </div>
                  </label>
                )}
                {(LOCAL.has(name) || !p.needs_key) && (
                  <label className="space-y-1">
                    <span className="text-[#8A9BB5]">Server address (base URL)</span>
                    <input value={d.base_url} onChange={(e) => setDraft(name, { base_url: e.target.value })} className={inputCls} />
                  </label>
                )}
                <div className="space-y-1 flex flex-col justify-end">
                  <div className="flex gap-1.5">
                    <button onClick={() => fetchModels(name)} disabled={busy === `models:${name}`} className={`${btnCls} border-[#06B6D4]/50 text-[#06B6D4]`}>
                      {busy === `models:${name}` ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <RefreshCw className="w-3.5 h-3.5" />} Check models
                    </button>
                    <button onClick={() => test(name)} disabled={busy === `test:${name}`} className={`${btnCls} border-[#10B981]/50 text-[#10B981]`}>
                      {busy === `test:${name}` ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Check className="w-3.5 h-3.5" />} Test connection
                    </button>
                  </div>
                  {testResults[name] && (
                    <div className={`mono ${testResults[name].startsWith('OK') ? 'text-[#10B981]' : 'text-[#EF4444]'}`}>{testResults[name]}</div>
                  )}
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-2.5 text-xs">
                <label className="space-y-1">
                  <span className="text-[#8A9BB5]">Main model (reflection)</span>
                  <ModelPicker value={d.model} options={opts} onChange={(v) => setDraft(name, { model: v })} placeholder={p.default_model} />
                </label>
                <label className="space-y-1">
                  <span className="text-[#8A9BB5]">Fast model (scalping supervisor)</span>
                  <ModelPicker value={d.fast_model} options={opts} onChange={(v) => setDraft(name, { fast_model: v })} placeholder={p.default_fast_model} />
                </label>
                <div className="space-y-1">
                  <span className="text-[#8A9BB5]">Fallback models (in order)</span>
                  <div className="flex flex-wrap gap-1">
                    {d.fallback_models.map((m) => (
                      <span key={m} className="px-1.5 py-0.5 rounded bg-[#121B2B] border border-[#1F2D45] mono flex items-center gap-1">
                        {m}
                        <X
                          className="w-3 h-3 cursor-pointer text-[#EF4444]"
                          onClick={() => setDraft(name, { fallback_models: d.fallback_models.filter((x) => x !== m) })}
                        />
                      </span>
                    ))}
                  </div>
                  <AddFallback
                    options={opts.filter((m) => !d.fallback_models.includes(m) && m !== d.model && m !== d.fast_model)}
                    onAdd={(m) => setDraft(name, { fallback_models: [...d.fallback_models, m] })}
                  />
                </div>
              </div>
              {p.locked_by_environment.length > 0 && (
                <div className="text-xs text-[#F59E0B]">
                  Set by system environment variables ({p.locked_by_environment.join(', ')}), so changes here do not apply to those.
                </div>
              )}
              {p.stats?.last_error && <div className="text-xs text-[#EF4444] mono truncate">Last error: {p.stats.last_error}</div>}
            </div>
          );
        })}

        <div className="flex items-center gap-2 text-xs">
          <select value={addPick} onChange={(e) => setAddPick(e.target.value)} className={`${inputCls} max-w-xs`}>
            <option value="">+ Add fallback provider...</option>
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

      <div className="grid grid-cols-1 xl:grid-cols-12 gap-3">
        {/* SUPERVISOR */}
        <div className="xl:col-span-7 cockpit-card rounded-lg p-3.5 space-y-3 text-xs">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="font-bold uppercase tracking-wider">Supervisor Claude (veto)</h3>
              <p className="text-[#8A9BB5] mt-0.5">Can only approve, reduce or reject a trade. Never raises risk.</p>
            </div>
            <button onClick={saveSup} disabled={busy === 'sup'} className={`${btnCls} bg-[#3B82F6] text-white border-[#3B82F6]`}>
              <Check className="w-3.5 h-3.5" /> Save
            </button>
          </div>
          {sup && (
            <div className="grid grid-cols-2 md:grid-cols-3 gap-2.5">
              <Toggle label="On" value={sup.enabled} onChange={(v) => setSup({ ...sup, enabled: v })} />
              <Toggle label="Use the fast model" value={sup.use_fast_model} onChange={(v) => setSup({ ...sup, use_fast_model: v })} />
              <Toggle label="Also review in paper mode" value={sup.review_in_shadow} onChange={(v) => setSup({ ...sup, review_in_shadow: v })} />
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
            <div className="mono text-[#8A9BB5]">
              Today: {sup.status.calls} calls · ✓{sup.status.approve} ↓{sup.status.reduce} ✗{sup.status.reject} · ${sup.status.spent_today_usd?.toFixed(3)}
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

const AddFallback: React.FC<{ options: string[]; onAdd: (m: string) => void }> = ({ options, onAdd }) => {
  const [v, setV] = useState('');
  return (
    <div className="flex gap-1">
      {options.length ? (
        <select value={v} onChange={(e) => setV(e.target.value)} className={inputCls}>
          <option value="">pick a model...</option>
          {options.map((m) => (
            <option key={m} value={m}>
              {m}
            </option>
          ))}
        </select>
      ) : (
        <input value={v} onChange={(e) => setV(e.target.value)} placeholder="click Check models, or type" className={inputCls} />
      )}
      <button
        disabled={!v.trim()}
        onClick={() => {
          onAdd(v.trim());
          setV('');
        }}
        className={`${btnCls} border-[#3B82F6]/50 text-[#3B82F6]`}
      >
        <Plus className="w-3.5 h-3.5" />
      </button>
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
