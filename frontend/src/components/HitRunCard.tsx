import React, { useState } from 'react';
import { Check } from 'lucide-react';
import { api } from '../api/client';

// Hit & Run settings: the net profit to take, and its own (tighter) stop loss.

export interface HitRunSettings {
  min_net_profit: number;
  min_cost_multiple?: number;
  own_stop: boolean;
  max_loss_mult: number;
  sl_atr_mult: number;
  sl_min_spread_mult: number;
  time_stop_bars: number;
  adaptive: boolean;
  adaptive_min_samples: number;
}

const inputCls = 'w-full px-2 py-1 rounded cockpit-sub mono text-xs';

const Field: React.FC<{ label: string; hint?: string; value: number; step: number; min?: number; disabled?: boolean; onChange: (v: number) => void }> = ({
  label,
  hint,
  value,
  step,
  min = 0,
  disabled,
  onChange,
}) => (
  <label className="space-y-1 block">
    <span className="text-[#8A9BB5]">{label}</span>
    <input type="number" step={step} min={min} disabled={disabled} value={value} onChange={(e) => onChange(Number(e.target.value))} className={inputCls} />
    {hint ? <span className="text-[10px] text-[#8A9BB5] block">{hint}</span> : null}
  </label>
);

export const HitRunCard: React.FC<{ settings: HitRunSettings; onSaved: (s: HitRunSettings) => void; onToast: (m: string) => void }> = ({
  settings,
  onSaved,
  onToast,
}) => {
  const [s, setS] = useState<HitRunSettings>(settings);
  const [busy, setBusy] = useState(false);
  const set = (patch: Partial<HitRunSettings>) => setS({ ...s, ...patch });
  const maxLoss = s.max_loss_mult * s.min_net_profit;
  const beWr = maxLoss + s.min_net_profit > 0 ? (maxLoss / (maxLoss + s.min_net_profit)) * 100 : 0;

  const save = async () => {
    setBusy(true);
    try {
      const r = await api.saveExits({
        hit_min_net_profit: s.min_net_profit,
        hit_min_cost_multiple: s.min_cost_multiple ?? 2.0,
        hit_own_stop: s.own_stop,
        hit_max_loss_mult: s.max_loss_mult,
        hit_sl_atr_mult: s.sl_atr_mult,
        hit_sl_min_spread_mult: s.sl_min_spread_mult,
        hit_time_stop_bars: s.time_stop_bars,
        hit_adaptive: s.adaptive,
      });
      onSaved(r.hit_run);
      setS(r.hit_run);
      onToast('Hit & Run settings saved');
    } catch (e) {
      onToast(`Could not save Hit & Run: ${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="cockpit-card rounded-lg p-3.5 space-y-3 text-xs">
      <div className="flex items-center justify-between gap-2">
        <div>
          <h3 className="font-bold uppercase tracking-wider">Hit &amp; Run</h3>
          <p className="text-[#8A9BB5] mt-0.5">
            Closes as soon as the trade is in net profit. It has its own stop loss, tighter than Normal&apos;s, and a smaller lot so one loss
            cannot wipe out many small wins.
          </p>
        </div>
        <button
          onClick={save}
          disabled={busy}
          className="px-2.5 py-1 rounded border font-bold flex items-center gap-1 cursor-pointer disabled:opacity-50 bg-[#3B82F6] text-white border-[#3B82F6] shrink-0"
        >
          <Check className="w-3.5 h-3.5" /> Save
        </button>
      </div>

      <div className="grid grid-cols-2 gap-2.5">
        <Field label="Take at net profit ($)" value={s.min_net_profit} step={0.5} onChange={(v) => set({ min_net_profit: v })} />
        <Field
          label="Min profit floor (× cost)"
          hint="Net profit must also beat N × (spread + commission)"
          value={s.min_cost_multiple ?? 2.0}
          step={0.5}
          min={0}
          onChange={(v) => set({ min_cost_multiple: v })}
        />
        <Field
          label="Max loss per trade (× take)"
          hint={`= $${maxLoss.toFixed(2)} incl. commission; lot is sized to fit`}
          value={s.max_loss_mult}
          step={0.5}
          min={1}
          disabled={!s.own_stop}
          onChange={(v) => set({ max_loss_mult: v })}
        />
        <Field
          label="Stop distance (× ATR M1)"
          value={s.sl_atr_mult}
          step={0.1}
          min={0.2}
          disabled={!s.own_stop}
          onChange={(v) => set({ sl_atr_mult: v })}
        />
        <Field
          label="Stop at least (× spread)"
          value={s.sl_min_spread_mult}
          step={0.5}
          disabled={!s.own_stop}
          onChange={(v) => set({ sl_min_spread_mult: v })}
        />
        <Field
          label="Time stop (M1 bars, 0 = off)"
          hint="Close if still not in net profit after this many minutes"
          value={s.time_stop_bars}
          step={1}
          disabled={!s.own_stop}
          onChange={(v) => set({ time_stop_bars: Math.round(v) })}
        />
        <div className="space-y-1.5 pt-4">
          <label className="flex items-center gap-2 cursor-pointer">
            <input type="checkbox" checked={s.own_stop} onChange={(e) => set({ own_stop: e.target.checked })} />
            <span>Own stop loss (off = same SL and lot as Normal)</span>
          </label>
          <label className="flex items-center gap-2 cursor-pointer">
            <input type="checkbox" checked={s.adaptive} disabled={!s.own_stop} onChange={(e) => set({ adaptive: e.target.checked })} />
            <span>Learn the best stop per market and regime</span>
          </label>
        </div>
      </div>

      {s.own_stop ? (
        <p className="text-[#8A9BB5]">
          Risking about ${maxLoss.toFixed(2)} to make ${s.min_net_profit.toFixed(2)} needs a win rate above about{' '}
          <b className="text-[#E6EDF7]">{beWr.toFixed(0)}%</b> to break even. Wins are often a bit larger than the take level, which lowers
          this. Learning uses paper market results only and needs {s.adaptive_min_samples} ghost trades per setting before it changes
          anything.
        </p>
      ) : null}
    </div>
  );
};
