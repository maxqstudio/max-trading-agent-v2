import React, { useState } from 'react';
import { Check } from 'lucide-react';
import { api } from '../api/client';

// Direction (M15/H1), session and news filters, with what the No Veto book says about each.

type Mode = 'off' | 'on' | 'auto';

interface BucketRow {
  bucket: string;
  trades: number;
  win_rate: number;
  avg_r: number;
  clearly_worse: boolean;
}

export interface FiltersView {
  settings: {
    htf_mode: Mode;
    htf_timeframe: 'M15' | 'H1' | 'BOTH';
    trend_strategies_require_htf?: boolean;
    block_breakout_in_ranging?: boolean;
    session_mode: Mode;
    blocked_sessions: string[];
    ichimoku_mode: Mode;
    ichimoku_threshold: 2 | 3 | 4;
    min_samples: number;
  };
  news: { enabled: boolean; buffer_minutes: number; from_mt5: boolean; mt5_updated_ts: number; upcoming: { title: string; time_utc: number }[] };
  htf: Record<string, BucketRow[]>;
  session: BucketRow[];
  ichimoku?: BucketRow[];
  server_offset_sec: number;
}

const inputCls = 'w-full px-2 py-1 rounded cockpit-sub mono text-xs';
const SESSION_LABEL: Record<string, string> = {
  asia: 'Asia (23:00 to 07:00 UTC)',
  london: 'London (07:00 to 12:00)',
  overlap: 'London + New York (12:00 to 16:00)',
  newyork: 'New York (16:00 to 21:00)',
  late: 'Late (21:00 to 23:00)',
};
const ICHI_LABEL: Record<string, string> = { with: 'Checklist with the trade', mixed: 'Mixed', against: 'Checklist against the trade' };
const BUCKET_LABEL: Record<string, string> = { with: 'With the direction', neutral: 'No clear direction', against: 'Against the direction' };
const MODE_HELP: Record<Mode, string> = {
  off: 'never blocks',
  on: 'always blocks',
  auto: 'blocks only after the paper results show it clearly loses more',
};

const signed = (v: number) => `${v >= 0 ? '+' : ''}${v.toFixed(2)}`;

const StatsTable: React.FC<{ rows: BucketRow[]; label: (b: string) => string; minSamples: number }> = ({ rows, label, minSamples }) => (
  <table className="w-full text-left border-collapse">
    <thead>
      <tr className="border-b border-[#1F2D45] text-[#8A9BB5]">
        <th className="py-1 px-2">Setups</th>
        <th className="py-1 px-2 text-right">Trades</th>
        <th className="py-1 px-2 text-right">Win rate</th>
        <th className="py-1 px-2 text-right">Avg R</th>
        <th className="py-1 px-2 text-right">Verdict</th>
      </tr>
    </thead>
    <tbody className="divide-y divide-[#1F2D45] mono">
      {rows.map((r) => (
        <tr key={r.bucket}>
          <td className="py-1 px-2 font-sans">{label(r.bucket)}</td>
          <td className="py-1 px-2 text-right">{r.trades}</td>
          <td className="py-1 px-2 text-right">{r.trades ? `${r.win_rate.toFixed(0)}%` : '—'}</td>
          <td className={`py-1 px-2 text-right ${r.avg_r >= 0 ? 'text-[#10B981]' : 'text-[#EF4444]'}`}>{r.trades ? signed(r.avg_r) : '—'}</td>
          <td className="py-1 px-2 text-right font-sans">
            {r.clearly_worse ? (
              <span className="text-[#EF4444] font-bold">clearly worse</span>
            ) : r.trades < minSamples ? (
              <span className="text-[#8A9BB5]">needs {minSamples - r.trades} more</span>
            ) : (
              <span className="text-[#8A9BB5]">no clear gap</span>
            )}
          </td>
        </tr>
      ))}
    </tbody>
  </table>
);

export const FiltersCard: React.FC<{ view: FiltersView; onSaved: (v: FiltersView) => void; onToast: (m: string) => void }> = ({
  view,
  onSaved,
  onToast,
}) => {
  const [s, setS] = useState(view.settings);
  const [newsOn, setNewsOn] = useState(view.news.enabled);
  const [newsMin, setNewsMin] = useState(view.news.buffer_minutes);
  const [busy, setBusy] = useState(false);

  const save = async () => {
    setBusy(true);
    try {
      const r = await api.saveFilters({ ...s, news_enabled: newsOn, news_buffer_minutes: newsMin });
      onSaved(r);
      onToast('Filters saved');
    } catch (e) {
      onToast(`Could not save filters: ${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  };

  const toggleSession = (name: string) =>
    setS({
      ...s,
      blocked_sessions: s.blocked_sessions.includes(name) ? s.blocked_sessions.filter((x) => x !== name) : [...s.blocked_sessions, name],
    });

  const fmtTime = (ts: number) => new Date(ts * 1000).toLocaleString(undefined, { weekday: 'short', hour: '2-digit', minute: '2-digit' });

  return (
    <div className="cockpit-card rounded-lg p-3.5 space-y-3 text-xs">
      <div className="flex items-center justify-between gap-2">
        <div>
          <h3 className="font-bold uppercase tracking-wider">Market filters</h3>
          <p className="text-[#8A9BB5] mt-0.5">
            The No Veto paper book always takes every setup, so the robot keeps measuring whether each filter helps. On auto, a filter starts
            blocking only after {s.min_samples} trades on each side show a clear gap.
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

      {/* Direction */}
      <div className="cockpit-sub rounded-md p-2.5 space-y-2">
        <div className="font-bold">Higher-timeframe direction</div>
        <p className="text-[#8A9BB5]">
          Direction = EMA 20 slope and higher highs / higher lows (or lower lows / lower highs) on completed M15 or H1 candles. Blocks trades
          against it.
        </p>
        <div className="grid grid-cols-2 gap-2.5">
          <label className="space-y-1">
            <span className="text-[#8A9BB5]">Mode ({MODE_HELP[s.htf_mode]})</span>
            <select value={s.htf_mode} onChange={(e) => setS({ ...s, htf_mode: e.target.value as Mode })} className={inputCls}>
              <option value="auto">Auto</option>
              <option value="on">On</option>
              <option value="off">Off</option>
            </select>
          </label>
          <label className="space-y-1">
            <span className="text-[#8A9BB5]">Timeframe</span>
            <select value={s.htf_timeframe} onChange={(e) => setS({ ...s, htf_timeframe: e.target.value as FiltersView['settings']['htf_timeframe'] })} className={inputCls}>
              <option value="H1">H1</option>
              <option value="M15">M15</option>
              <option value="BOTH">M15 and H1 (against either one)</option>
            </select>
          </label>
        </div>
        <div className="flex flex-wrap gap-x-4 gap-y-1.5 pt-1">
          <label className="flex items-center gap-1.5 cursor-pointer">
            <input
              type="checkbox"
              checked={s.trend_strategies_require_htf ?? true}
              onChange={(e) => setS({ ...s, trend_strategies_require_htf: e.target.checked })}
            />
            <span>Trend strategies require HTF direction</span>
          </label>
          <label className="flex items-center gap-1.5 cursor-pointer">
            <input
              type="checkbox"
              checked={s.block_breakout_in_ranging ?? true}
              onChange={(e) => setS({ ...s, block_breakout_in_ranging: e.target.checked })}
            />
            <span>Block Micro Breakout in Ranging regime</span>
          </label>
        </div>
        <StatsTable rows={view.htf[s.htf_timeframe] ?? []} label={(b) => BUCKET_LABEL[b] ?? b} minSamples={s.min_samples} />
      </div>

      {/* Session */}
      <div className="cockpit-sub rounded-md p-2.5 space-y-2">
        <div className="font-bold">Trading session</div>
        <div className="grid grid-cols-2 gap-2.5">
          <label className="space-y-1">
            <span className="text-[#8A9BB5]">Mode ({MODE_HELP[s.session_mode]})</span>
            <select value={s.session_mode} onChange={(e) => setS({ ...s, session_mode: e.target.value as Mode })} className={inputCls}>
              <option value="auto">Auto</option>
              <option value="on">On (ticked sessions)</option>
              <option value="off">Off</option>
            </select>
          </label>
          <div className="space-y-1">
            <span className="text-[#8A9BB5]">Blocked when On</span>
            <div className="flex flex-wrap gap-x-3 gap-y-1">
              {Object.keys(SESSION_LABEL).map((name) => (
                <label key={name} className="flex items-center gap-1 cursor-pointer">
                  <input type="checkbox" checked={s.blocked_sessions.includes(name)} onChange={() => toggleSession(name)} />
                  <span className="capitalize">{name === 'newyork' ? 'New York' : name === 'overlap' ? 'Overlap' : name}</span>
                </label>
              ))}
            </div>
          </div>
        </div>
        <StatsTable rows={view.session} label={(b) => SESSION_LABEL[b] ?? b} minSamples={s.min_samples} />
        <p className="text-[#8A9BB5]">
          Hours are UTC. Broker server time offset: {view.server_offset_sec >= 0 ? '+' : ''}
          {(view.server_offset_sec / 3600).toFixed(1)} h{view.server_offset_sec === 0 ? ' (EA 2.10 or newer reports it)' : ''}.
        </p>
      </div>

      {/* Ichimoku */}
      <div className="cockpit-sub rounded-md p-2.5 space-y-2">
        <div className="font-bold">Ichimoku checklist (M15)</div>
        <p className="text-[#8A9BB5]">
          Five points on completed M15 candles: price against the cloud, Tenkan against Kijun, Chikou against the price and cloud of 26
          candles ago, price against Kijun, and the colour of the cloud ahead. Each point is +1 with the trade or -1 against it; the score is their sum.
        </p>
        <div className="grid grid-cols-2 gap-2.5">
          <label className="space-y-1">
            <span className="text-[#8A9BB5]">Mode ({MODE_HELP[s.ichimoku_mode ?? 'auto']})</span>
            <select value={s.ichimoku_mode ?? 'auto'} onChange={(e) => setS({ ...s, ichimoku_mode: e.target.value as Mode })} className={inputCls}>
              <option value="auto">Auto</option>
              <option value="on">On</option>
              <option value="off">Off</option>
            </select>
          </label>
          <label className="space-y-1">
            <span className="text-[#8A9BB5]">Against when the score (+5 to -5) is</span>
            <select
              value={s.ichimoku_threshold ?? 3}
              onChange={(e) => setS({ ...s, ichimoku_threshold: Number(e.target.value) as 2 | 3 | 4 })}
              className={inputCls}
            >
              <option value={2}>-2 or lower</option>
              <option value={3}>-3 or lower</option>
              <option value={4}>-4 or lower</option>
            </select>
          </label>
        </div>
        <StatsTable rows={view.ichimoku ?? []} label={(b) => ICHI_LABEL[b] ?? b} minSamples={s.min_samples} />
      </div>

      {/* News */}
      <div className="cockpit-sub rounded-md p-2.5 space-y-2">
        <div className="flex items-center justify-between gap-2">
          <div className="font-bold">High-impact news pause</div>
          <span className={view.news.from_mt5 ? 'text-[#10B981]' : 'text-[#F59E0B]'}>
            {view.news.from_mt5 ? 'calendar from MT5' : 'MT5 calendar not received yet (EA 2.10+)'}
          </span>
        </div>
        <div className="grid grid-cols-2 gap-2.5">
          <label className="flex items-center gap-2 cursor-pointer">
            <input type="checkbox" checked={newsOn} onChange={(e) => setNewsOn(e.target.checked)} />
            <span>No new entries around high-impact news</span>
          </label>
          <label className="space-y-1">
            <span className="text-[#8A9BB5]">Minutes before and after</span>
            <input type="number" min={0} step={5} value={newsMin} onChange={(e) => setNewsMin(Number(e.target.value))} className={inputCls} />
          </label>
        </div>
        {view.news.upcoming.length ? (
          <ul className="space-y-0.5">
            {view.news.upcoming.map((ev) => (
              <li key={`${ev.time_utc}-${ev.title}`} className="flex justify-between gap-2">
                <span className="truncate">{ev.title}</span>
                <span className="mono text-[#8A9BB5] shrink-0">{fmtTime(ev.time_utc)}</span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-[#8A9BB5]">No high-impact news in the next 36 hours.</p>
        )}
      </div>
    </div>
  );
};
