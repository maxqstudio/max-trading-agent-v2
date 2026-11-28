import React, { useEffect, useRef, useState } from 'react';
import {
  AlertTriangle,
  CheckCircle2,
  Copy,
  Database,
  FileCode2,
  FlaskConical,
  Play,
  RefreshCw,
  Upload,
} from 'lucide-react';
import { api } from '../api/client';
import { human } from '../labels';

interface Props {
  onToast: (msg: string) => void;
}

interface CoverageRow {
  symbol: string;
  bars: number;
  from_utc: string;
  to_utc: string;
  duration_days: number;
  pct_of_window: number;
}

interface CalibrationRow {
  lower: number;
  upper: number;
  predicted: number;
  actual: number;
  gap: number;
  count: number;
}

interface GroupMetricRow {
  group: string;
  total_trades: number;
  win_rate: number | null;
  profit_factor: number | null;
  expectancy_r: number | null;
  max_drawdown_pct: number | null;
}

interface AuditPayload {
  generated_utc: string;
  window_bars: number;
  server_offset_sec: number;
  dry_run: boolean;
  force: boolean;
  mode_label: string;
  coverage: CoverageRow[];
  dl: {
    warm_start: boolean;
    train_loss: number | null;
    val_loss: number | null;
    val_acc: number | null;
    candidate_oos: { loss?: number; acc?: number; n?: number } | null;
    champion_oos: { loss?: number; acc?: number; n?: number } | null;
    adopted: boolean;
    status: string;
    reason: string;
    save_action: string;
    saved: boolean;
  };
  ml: {
    train_setups: number;
    oos_setups: number;
    train_brier: number | null;
    candidate_brier: number | null;
    champion_brier: number | null;
    candidate_acc: number | null;
    adopted: boolean;
    status: string;
    reason: string;
    save_action: string;
    saved: boolean;
    calibration_curve: CalibrationRow[];
    ece: number | null;
    feature_importances: { name: string; importance: number }[];
  };
  oos_backtest: {
    total_trades: number;
    win_rate: number;
    profit_factor: number;
    sharpe_ratio: number;
    expectancy_r: number;
    max_drawdown_pct: number;
    net_return_pct: number;
    mc_p95_dd_pct: number;
    mc_ruin_prob: number;
    by_symbol: GroupMetricRow[];
    by_strategy: GroupMetricRow[];
  };
  warnings: string[];
  markdown_report?: string;
}

interface ResearchStatus {
  running: boolean;
  window_bars_default: number;
  recency_half_life_days: number;
  dl_min_oos_gain: number;
  ml_min_oos_gain: number;
  server_offset_sec: number;
  coverage: CoverageRow[];
  dl_champion: { trained: boolean; val_acc: number };
  ml_champion: { trained: boolean; samples: number; brier_score: number; oos_accuracy: number };
  last_audit: AuditPayload | null;
}

const fmt = (v: number | null | undefined, d = 4) =>
  v == null || !Number.isFinite(v) ? '—' : v.toFixed(d);

const fmtPct = (v: number | null | undefined, d = 1) =>
  v == null || !Number.isFinite(v) ? '—' : `${(v * 100).toFixed(d)}%`;

export const ResearchView: React.FC<Props> = ({ onToast }) => {
  const [status, setStatus] = useState<ResearchStatus | null>(null);
  const [windowBars, setWindowBars] = useState<number>(20160);
  const [dryRun, setDryRun] = useState<boolean>(false);
  const [forceSave, setForceSave] = useState<boolean>(false);
  const [busy, setBusy] = useState<boolean>(false);
  const [uploading, setUploading] = useState<boolean>(false);
  const [showMarkdown, setShowMarkdown] = useState<boolean>(false);
  const fileRef = useRef<HTMLInputElement | null>(null);

  const load = async () => {
    try {
      const res = (await api.getResearch()) as ResearchStatus;
      setStatus(res);
    } catch {
      // ignore transient poll errors
    }
  };

  useEffect(() => {
    load();
    const timer = window.setInterval(load, 6000);
    return () => window.clearInterval(timer);
  }, []);

  const runTraining = async () => {
    setBusy(true);
    try {
      const res = await api.runResearch({
        window_bars: windowBars,
        dry_run: dryRun,
        force: forceSave,
      });
      if (res.status === 'completed') {
        const dlSt = res.audit?.dl?.status || 'DONE';
        const mlSt = res.audit?.ml?.status || 'DONE';
        onToast(`Training Ground complete — DL: ${dlSt}, ML: ${mlSt}`);
      } else {
        onToast(res.message || `Training Ground: ${res.status}`);
      }
      await load();
    } catch (err) {
      onToast(`Training failed: ${(err as Error).message}`);
    } finally {
      setBusy(false);
    }
  };

  const handleFileImport = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    setUploading(true);
    let totalBars = 0;
    const importedSyms: string[] = [];
    try {
      for (const file of Array.from(files)) {
        const text = await file.text();
        const res = await api.importResearchCsv(file.name, text);
        totalBars += Number(res.imported || 0);
        if (res.symbol && !importedSyms.includes(res.symbol)) {
          importedSyms.push(res.symbol);
        }
      }
      onToast(`Imported ${totalBars.toLocaleString()} M1 bars (${importedSyms.join(', ')})`);
      await load();
    } catch (err) {
      onToast(`CSV import failed: ${(err as Error).message}`);
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = '';
    }
  };

  const copyMarkdown = async () => {
    const md = status?.last_audit?.markdown_report;
    if (!md) return;
    try {
      await navigator.clipboard.writeText(md);
      onToast('Audit report copied to clipboard');
    } catch {
      onToast('Could not copy report to clipboard');
    }
  };

  const audit = status?.last_audit ?? null;
  const coverage = status?.coverage ?? [];
  const totalBars = coverage.reduce((acc, r) => acc + r.bars, 0);
  const running = busy || Boolean(status?.running);

  return (
    <div className="space-y-3">
      {/* Header & Training Controls */}
      <div className="cockpit-card rounded-lg p-3.5 space-y-3">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="space-y-0.5">
            <div className="flex items-center gap-2">
              <FlaskConical className="w-4 h-4 text-[#06B6D4]" />
              <h2 className="text-sm font-bold">Training Ground — Backtest Pre-Training &amp; Scientist OOS Audit</h2>
            </div>
            <p className="text-xs text-[#8A9BB5]">
              Train Pillar 1 (1D-CNN) and Pillar 2 (Calibrated ML Meta-Labeler) on real broker M1 history with exponential
              time-decay ({status?.recency_half_life_days ?? 5}d half-life) and strict purged OOS gating. RL learners &amp; paper books stay untouched.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <select
              value={windowBars}
              onChange={(e) => setWindowBars(Number(e.target.value))}
              disabled={running}
              className="mono text-xs font-semibold px-2.5 py-1.5 rounded-md cockpit-sub border border-[#1F2D45] cursor-pointer focus:outline-none"
              title="Maximum M1 bars per symbol used for training + purged OOS evaluation"
            >
              <option value={3000}>3,000 bars (~2d)</option>
              <option value={10080}>10,080 bars (7d)</option>
              <option value={20160}>20,160 bars (14d)</option>
              <option value={40320}>40,320 bars (28d)</option>
            </select>

            <label className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-md cockpit-sub text-xs cursor-pointer" title="Run full OOS audit without overwriting champion snapshots">
              <input
                type="checkbox"
                checked={dryRun}
                onChange={(e) => {
                  setDryRun(e.target.checked);
                  if (e.target.checked) setForceSave(false);
                }}
                disabled={running}
                className="accent-[#06B6D4]"
              />
              <span className="font-semibold">Dry Run</span>
            </label>

            <label className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-md cockpit-sub text-xs cursor-pointer" title="Overwrite champion snapshots even if OOS gain is below minimum margin">
              <input
                type="checkbox"
                checked={forceSave}
                onChange={(e) => {
                  setForceSave(e.target.checked);
                  if (e.target.checked) setDryRun(false);
                }}
                disabled={running || dryRun}
                className="accent-[#F59E0B]"
              />
              <span className="font-semibold">Force Save</span>
            </label>

            <input
              ref={fileRef}
              type="file"
              accept=".csv,.tsv,.txt"
              multiple
              onChange={handleFileImport}
              className="hidden"
            />
            <button
              onClick={() => fileRef.current?.click()}
              disabled={uploading || running}
              className="px-2.5 py-1.5 rounded-md cockpit-sub hover:bg-[#162236] border border-[#1F2D45] text-xs font-semibold flex items-center gap-1.5 cursor-pointer disabled:opacity-50"
              title="Import M1 bar CSV/TSV exported from MT5 History Center (e.g. XAUUSD_M1.csv)"
            >
              <Upload className="w-3.5 h-3.5 text-[#06B6D4]" />
              <span>{uploading ? 'Importing...' : 'Import MT5 CSV'}</span>
            </button>

            <button
              onClick={runTraining}
              disabled={running}
              className="px-3 py-1.5 rounded-md bg-[#3B82F6] hover:bg-[#2563EB] text-white text-xs font-bold flex items-center gap-1.5 cursor-pointer disabled:opacity-50"
            >
              {running ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5" />}
              <span>{running ? 'Training & Auditing...' : 'Run Training & OOS Audit'}</span>
            </button>
          </div>
        </div>

        {/* Live Data Coverage in SQLite */}
        <div className="pt-2 border-t border-[#1F2D45] space-y-2">
          <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
            <div className="flex items-center gap-2 font-bold uppercase tracking-wider text-[#8A9BB5]">
              <Database className="w-3.5 h-3.5 text-[#06B6D4]" />
              <span>Local Broker M1 Data Coverage ({totalBars.toLocaleString()} total bars)</span>
            </div>
            <div className="flex items-center gap-3 mono text-[#8A9BB5]">
              <span>
                Active DL:{' '}
                <b className={status?.dl_champion?.trained ? 'text-[#10B981]' : 'text-[#F59E0B]'}>
                  {status?.dl_champion?.trained ? `TRAINED (val ${fmtPct(status.dl_champion.val_acc)})` : 'UNTRAINED'}
                </b>
              </span>
              <span>
                Active ML:{' '}
                <b className={status?.ml_champion?.trained ? 'text-[#10B981]' : 'text-[#F59E0B]'}>
                  {status?.ml_champion?.trained
                    ? `TRAINED (n=${status.ml_champion.samples}, Brier ${fmt(status.ml_champion.brier_score)})`
                    : 'UNTRAINED'}
                </b>
              </span>
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-xs text-left border-collapse mono">
              <thead>
                <tr className="border-b border-[#1F2D45] text-[#8A9BB5] font-sans">
                  <th className="py-1.5 px-2">Symbol</th>
                  <th className="py-1.5 px-2 text-right">M1 Bars</th>
                  <th className="py-1.5 px-2">From (UTC)</th>
                  <th className="py-1.5 px-2">To (UTC)</th>
                  <th className="py-1.5 px-2 text-right">Span (Days)</th>
                  <th className="py-1.5 px-2 w-44">
                    {Math.max(1, Math.round(windowBars / 1440))}-Day Window Fill
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#1F2D45]/70">
                {coverage.length === 0 ? (
                  <tr>
                    <td colSpan={6} className="py-4 text-center text-[#8A9BB5] font-sans">
                      No M1 bars in local DB yet. Connect MT5 EA v2.21 (auto-backfills up to 20,160 bars/symbol) or click Import MT5 CSV.
                    </td>
                  </tr>
                ) : (
                  coverage.map((row) => {
                    const fillPct = Math.min(100, (row.bars / Math.max(windowBars, 1)) * 100);
                    const barColor =
                      fillPct >= 95
                        ? 'bg-[#10B981]'
                        : fillPct >= 50
                          ? 'bg-[#06B6D4]'
                          : 'bg-[#F59E0B]';
                    return (
                      <tr key={row.symbol}>
                        <td className="py-1.5 px-2 font-bold text-white">{row.symbol}</td>
                        <td className="py-1.5 px-2 text-right">{row.bars.toLocaleString()}</td>
                        <td className="py-1.5 px-2 text-[#8A9BB5]">{row.from_utc}</td>
                        <td className="py-1.5 px-2 text-[#8A9BB5]">{row.to_utc}</td>
                        <td className={`py-1.5 px-2 text-right font-bold ${row.duration_days >= 7 ? 'text-[#10B981]' : 'text-[#F59E0B]'}`}>
                          {row.duration_days.toFixed(2)}d
                        </td>
                        <td className="py-1.5 px-2">
                          <div className="flex items-center gap-2">
                            <div className="flex-1 h-1.5 rounded-full bg-[#070B12] overflow-hidden">
                              <div className={`h-full ${barColor}`} style={{ width: `${fillPct}%` }} />
                            </div>
                            <span className="w-10 text-right text-[#8A9BB5]">{fillPct.toFixed(0)}%</span>
                          </div>
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Audit Results */}
      {!audit ? (
        <div className="cockpit-card rounded-lg p-8 text-center space-y-2">
          <FlaskConical className="w-8 h-8 text-[#06B6D4] mx-auto opacity-80" />
          <div className="text-sm font-bold">No Training Ground Audit Yet</div>
          <p className="text-xs text-[#8A9BB5] max-w-lg mx-auto">
            Click <b>Run Training &amp; OOS Audit</b> above to train the 1D-CNN and Calibrated ML Meta-Labeler on your local
            broker M1 history and inspect calibration, feature importances, OOS backtest metrics, and overfitting warnings.
          </p>
        </div>
      ) : (
        <>
          {/* Overfitting Risk Diagnosis Banner */}
          <div
            className={`cockpit-card rounded-lg p-3.5 border ${
              audit.warnings.length > 0
                ? 'border-[#F59E0B]/50 bg-[#F59E0B]/5'
                : 'border-[#10B981]/50 bg-[#10B981]/5'
            }`}
          >
            <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
              <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider">
                {audit.warnings.length > 0 ? (
                  <AlertTriangle className="w-4 h-4 text-[#F59E0B]" />
                ) : (
                  <CheckCircle2 className="w-4 h-4 text-[#10B981]" />
                )}
                <span>Scientist Overfitting Diagnosis</span>
                <span className="mono px-2 py-0.5 rounded bg-[#070B12] text-[#8A9BB5] font-semibold normal-case">
                  {audit.mode_label} · {audit.generated_utc}
                </span>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setShowMarkdown((v) => !v)}
                  className="px-2 py-1 rounded cockpit-sub text-xs font-semibold flex items-center gap-1 cursor-pointer hover:bg-[#162236]"
                >
                  <FileCode2 className="w-3.5 h-3.5 text-[#06B6D4]" />
                  <span>{showMarkdown ? 'Hide Raw Report' : 'Raw Markdown'}</span>
                </button>
                <button
                  onClick={copyMarkdown}
                  className="px-2 py-1 rounded cockpit-sub text-xs font-semibold flex items-center gap-1 cursor-pointer hover:bg-[#162236]"
                >
                  <Copy className="w-3.5 h-3.5 text-[#8A9BB5]" />
                  <span>Copy</span>
                </button>
              </div>
            </div>

            {audit.warnings.length === 0 ? (
              <div className="text-xs text-[#6EE7B7]">
                No automated overfitting warnings triggered at current audit thresholds (span ≥ 7d, OOS setups ≥ 100, healthy DL/ML generalization gap, ECE ≤ 0.08).
              </div>
            ) : (
              <div className="flex flex-wrap gap-2">
                {audit.warnings.map((w, idx) => (
                  <span
                    key={idx}
                    className="px-2.5 py-1 rounded-md bg-[#F59E0B]/15 border border-[#F59E0B]/35 text-[#FCD34D] text-xs font-medium"
                  >
                    {w}
                  </span>
                ))}
              </div>
            )}
            <p className="text-[11px] text-[#8A9BB5] mt-2">
              Note: The purged OOS split also acts as the DL/ML adoption gate — validate final edge on live Shadow/Paper forward trades.
            </p>
          </div>

          {/* Pillar 1 (DL) & Pillar 2 (ML) Cards */}
          <div className="grid grid-cols-1 xl:grid-cols-12 gap-3">
            {/* Pillar 1: 1D-CNN */}
            <div className="xl:col-span-5 cockpit-card rounded-lg p-3.5 space-y-3">
              <div className="flex items-center justify-between gap-2">
                <div>
                  <h3 className="text-xs font-bold uppercase tracking-wider">Pillar 1 — 1D-CNN Sequence Model</h3>
                  <div className="text-[11px] text-[#8A9BB5]">
                    Warm-start: <b className="text-white">{audit.dl.warm_start ? 'Yes' : 'No (fresh)'}</b> · {audit.dl.save_action}
                  </div>
                </div>
                <span
                  className={`mono text-xs font-bold px-2 py-0.5 rounded border ${
                    audit.dl.adopted
                      ? 'bg-[#10B981]/15 text-[#10B981] border-[#10B981]/40'
                      : 'bg-[#F59E0B]/15 text-[#F59E0B] border-[#F59E0B]/40'
                  }`}
                >
                  {audit.dl.status}
                </span>
              </div>

              <div className="grid grid-cols-3 gap-2 text-xs">
                <div className="p-2.5 rounded cockpit-sub">
                  <div className="text-[#8A9BB5]">Train Loss</div>
                  <div className="mono font-bold text-sm mt-0.5">{fmt(audit.dl.train_loss)}</div>
                </div>
                <div className="p-2.5 rounded cockpit-sub">
                  <div className="text-[#8A9BB5]">Val Loss</div>
                  <div className="mono font-bold text-sm mt-0.5">{fmt(audit.dl.val_loss)}</div>
                </div>
                <div className="p-2.5 rounded cockpit-sub">
                  <div className="text-[#8A9BB5]">Val Acc</div>
                  <div className="mono font-bold text-sm mt-0.5 text-[#06B6D4]">{fmtPct(audit.dl.val_acc)}</div>
                </div>
              </div>

              <table className="w-full text-xs text-left border-collapse mono">
                <thead>
                  <tr className="border-b border-[#1F2D45] text-[#8A9BB5] font-sans">
                    <th className="py-1.5">Unseen OOS</th>
                    <th className="py-1.5 text-right">Loss</th>
                    <th className="py-1.5 text-right">Direction Acc</th>
                    <th className="py-1.5 text-right">Samples</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#1F2D45]/70">
                  <tr>
                    <td className="py-1.5 font-sans font-semibold text-white">New Candidate</td>
                    <td className="py-1.5 text-right">{fmt(audit.dl.candidate_oos?.loss)}</td>
                    <td className="py-1.5 text-right">{fmtPct(audit.dl.candidate_oos?.acc)}</td>
                    <td className="py-1.5 text-right">{audit.dl.candidate_oos?.n ? Math.round(audit.dl.candidate_oos.n).toLocaleString() : '—'}</td>
                  </tr>
                  <tr>
                    <td className="py-1.5 font-sans text-[#8A9BB5]">Previous Champion</td>
                    <td className="py-1.5 text-right text-[#8A9BB5]">{fmt(audit.dl.champion_oos?.loss)}</td>
                    <td className="py-1.5 text-right text-[#8A9BB5]">{fmtPct(audit.dl.champion_oos?.acc)}</td>
                    <td className="py-1.5 text-right text-[#8A9BB5]">{audit.dl.champion_oos?.n ? Math.round(audit.dl.champion_oos.n).toLocaleString() : '—'}</td>
                  </tr>
                </tbody>
              </table>

              <div className="p-2 rounded cockpit-sub text-[11px] text-[#8A9BB5] mono">
                Gate: {audit.dl.reason}
              </div>
            </div>

            {/* Pillar 2: Calibrated ML Meta-Labeler */}
            <div className="xl:col-span-7 cockpit-card rounded-lg p-3.5 space-y-3">
              <div className="flex items-center justify-between gap-2">
                <div>
                  <h3 className="text-xs font-bold uppercase tracking-wider">Pillar 2 — Calibrated ML Meta-Labeler</h3>
                  <div className="text-[11px] text-[#8A9BB5]">
                    Setups: <b className="text-white">{audit.ml.train_setups}</b> train / <b className="text-white">{audit.ml.oos_setups}</b> OOS · {audit.ml.save_action}
                  </div>
                </div>
                <span
                  className={`mono text-xs font-bold px-2 py-0.5 rounded border ${
                    audit.ml.adopted
                      ? 'bg-[#10B981]/15 text-[#10B981] border-[#10B981]/40'
                      : 'bg-[#F59E0B]/15 text-[#F59E0B] border-[#F59E0B]/40'
                  }`}
                >
                  {audit.ml.status}
                </span>
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
                <div className="p-2 rounded cockpit-sub">
                  <div className="text-[#8A9BB5]">OOS Brier (New)</div>
                  <div className="mono font-bold text-sm mt-0.5">{fmt(audit.ml.candidate_brier)}</div>
                </div>
                <div className="p-2 rounded cockpit-sub">
                  <div className="text-[#8A9BB5]">OOS Brier (Champ)</div>
                  <div className="mono font-bold text-sm mt-0.5 text-[#8A9BB5]">{fmt(audit.ml.champion_brier)}</div>
                </div>
                <div className="p-2 rounded cockpit-sub">
                  <div className="text-[#8A9BB5]">OOS Accuracy</div>
                  <div className="mono font-bold text-sm mt-0.5 text-[#06B6D4]">{fmtPct(audit.ml.candidate_acc)}</div>
                </div>
                <div className="p-2 rounded cockpit-sub">
                  <div className="text-[#8A9BB5]">Calibration ECE</div>
                  <div className={`mono font-bold text-sm mt-0.5 ${(audit.ml.ece ?? 1) <= 0.08 ? 'text-[#10B981]' : 'text-[#F59E0B]'}`}>
                    {fmt(audit.ml.ece)}
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {/* Calibration Curve */}
                <div className="space-y-1.5">
                  <div className="text-[11px] font-bold uppercase tracking-wider text-[#8A9BB5]">OOS Calibration Curve</div>
                  <table className="w-full text-xs text-left border-collapse mono">
                    <thead>
                      <tr className="border-b border-[#1F2D45] text-[#8A9BB5] font-sans">
                        <th className="py-1">Pred P(win)</th>
                        <th className="py-1 text-right">Actual WR</th>
                        <th className="py-1 text-right">Gap</th>
                        <th className="py-1 text-right">n</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-[#1F2D45]/70">
                      {audit.ml.calibration_curve.length === 0 ? (
                        <tr><td colSpan={4} className="py-3 text-center text-[#8A9BB5] font-sans">No OOS bins</td></tr>
                      ) : (
                        audit.ml.calibration_curve.map((c, idx) => (
                          <tr key={idx}>
                            <td className="py-1">{(c.predicted * 100).toFixed(1)}%</td>
                            <td className="py-1 text-right">{(c.actual * 100).toFixed(1)}%</td>
                            <td className={`py-1 text-right ${Math.abs(c.gap) <= 0.06 ? 'text-[#10B981]' : 'text-[#F59E0B]'}`}>
                              {c.gap >= 0 ? '+' : ''}{(c.gap * 100).toFixed(1)}%
                            </td>
                            <td className="py-1 text-right text-[#8A9BB5]">{c.count}</td>
                          </tr>
                        ))
                      )}
                    </tbody>
                  </table>
                </div>

                {/* Top 8 Permutation Feature Importances */}
                <div className="space-y-1.5">
                  <div className="text-[11px] font-bold uppercase tracking-wider text-[#8A9BB5]">Top OOS Feature Importances</div>
                  <div className="space-y-1.5">
                    {audit.ml.feature_importances.length === 0 ? (
                      <div className="py-3 text-xs text-[#8A9BB5]">No OOS permutation estimate</div>
                    ) : (
                      audit.ml.feature_importances.slice(0, 8).map((fi) => {
                        const maxImp = Math.max(...audit.ml.feature_importances.map((x) => x.importance), 0.01);
                        const pct = Math.min(100, (fi.importance / maxImp) * 100);
                        return (
                          <div key={fi.name} className="space-y-0.5 text-xs">
                            <div className="flex justify-between mono">
                              <span className="text-[#CBD5E1] truncate pr-2">{fi.name}</span>
                              <span className="text-[#06B6D4] font-bold">{(fi.importance * 100).toFixed(1)}%</span>
                            </div>
                            <div className="h-1.5 rounded-full bg-[#070B12] overflow-hidden">
                              <div className="h-full bg-[#06B6D4]" style={{ width: `${pct}%` }} />
                            </div>
                          </div>
                        );
                      })
                    )}
                  </div>
                </div>
              </div>

              <div className="p-2 rounded cockpit-sub text-[11px] text-[#8A9BB5] mono">
                Gate: {audit.ml.reason}
              </div>
            </div>
          </div>

          {/* Unseen OOS Backtest Audit */}
          <div className="cockpit-card rounded-lg p-3.5 space-y-3">
            <div className="flex items-center justify-between flex-wrap gap-2">
              <h3 className="text-xs font-bold uppercase tracking-wider">Unseen OOS Backtest Audit (Causal M1 + Commission + 1,000× Monte Carlo)</h3>
              <span className={`mono text-xs font-bold px-2 py-0.5 rounded ${
                audit.oos_backtest.profit_factor >= 1.15
                  ? 'bg-[#10B981]/15 text-[#10B981]'
                  : audit.oos_backtest.profit_factor >= 1.0
                    ? 'bg-[#F59E0B]/15 text-[#F59E0B]'
                    : 'bg-[#EF4444]/15 text-[#EF4444]'
              }`}>
                PF {audit.oos_backtest.profit_factor.toFixed(2)} · Exp {audit.oos_backtest.expectancy_r >= 0 ? '+' : ''}{audit.oos_backtest.expectancy_r.toFixed(3)}R
              </span>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-2 text-xs">
              <div className="p-2.5 rounded cockpit-sub">
                <div className="text-[#8A9BB5]">OOS Trades</div>
                <div className="mono font-bold text-sm mt-0.5">{audit.oos_backtest.total_trades}</div>
              </div>
              <div className="p-2.5 rounded cockpit-sub">
                <div className="text-[#8A9BB5]">Win Rate</div>
                <div className="mono font-bold text-sm mt-0.5">{fmtPct(audit.oos_backtest.win_rate)}</div>
              </div>
              <div className="p-2.5 rounded cockpit-sub">
                <div className="text-[#8A9BB5]">Profit Factor</div>
                <div className={`mono font-bold text-sm mt-0.5 ${audit.oos_backtest.profit_factor >= 1.0 ? 'text-[#10B981]' : 'text-[#EF4444]'}`}>
                  {audit.oos_backtest.profit_factor.toFixed(2)}
                </div>
              </div>
              <div className="p-2.5 rounded cockpit-sub">
                <div className="text-[#8A9BB5]">Sharpe</div>
                <div className={`mono font-bold text-sm mt-0.5 ${audit.oos_backtest.sharpe_ratio >= 0 ? 'text-[#10B981]' : 'text-[#EF4444]'}`}>
                  {audit.oos_backtest.sharpe_ratio.toFixed(2)}
                </div>
              </div>
              <div className="p-2.5 rounded cockpit-sub">
                <div className="text-[#8A9BB5]">Expectancy</div>
                <div className={`mono font-bold text-sm mt-0.5 ${audit.oos_backtest.expectancy_r >= 0 ? 'text-[#10B981]' : 'text-[#EF4444]'}`}>
                  {audit.oos_backtest.expectancy_r >= 0 ? '+' : ''}{audit.oos_backtest.expectancy_r.toFixed(3)}R
                </div>
              </div>
              <div className="p-2.5 rounded cockpit-sub">
                <div className="text-[#8A9BB5]">Max DD</div>
                <div className="mono font-bold text-sm mt-0.5 text-[#F59E0B]">{audit.oos_backtest.max_drawdown_pct.toFixed(2)}%</div>
              </div>
              <div className="p-2.5 rounded cockpit-sub">
                <div className="text-[#8A9BB5]">MC P95 DD</div>
                <div className="mono font-bold text-sm mt-0.5 text-[#EF4444]">{audit.oos_backtest.mc_p95_dd_pct.toFixed(2)}%</div>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
              {/* Per Symbol */}
              <div className="p-2.5 rounded cockpit-sub space-y-2">
                <div className="text-[11px] font-bold uppercase tracking-wider text-[#8A9BB5]">OOS Breakdown by Symbol</div>
                <div className="overflow-x-auto">
                  <table className="w-full text-xs text-left border-collapse mono">
                    <thead>
                      <tr className="border-b border-[#1F2D45] text-[#8A9BB5] font-sans">
                        <th className="py-1">Symbol</th>
                        <th className="py-1 text-right">Trades</th>
                        <th className="py-1 text-right">Win %</th>
                        <th className="py-1 text-right">PF</th>
                        <th className="py-1 text-right">Exp (R)</th>
                        <th className="py-1 text-right">Max DD</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-[#1F2D45]/70">
                      {audit.oos_backtest.by_symbol.map((row) => (
                        <tr key={row.group}>
                          <td className="py-1.5 font-bold text-white">{row.group}</td>
                          <td className="py-1.5 text-right">{row.total_trades}</td>
                          <td className="py-1.5 text-right">{fmtPct(row.win_rate)}</td>
                          <td className={`py-1.5 text-right font-bold ${row.profit_factor == null ? 'text-[#8A9BB5]' : row.profit_factor >= 1.0 ? 'text-[#10B981]' : 'text-[#EF4444]'}`}>
                            {fmt(row.profit_factor, 2)}
                          </td>
                          <td className={`py-1.5 text-right ${row.expectancy_r == null ? 'text-[#8A9BB5]' : row.expectancy_r >= 0 ? 'text-[#10B981]' : 'text-[#EF4444]'}`}>
                            {row.expectancy_r == null ? '—' : `${row.expectancy_r >= 0 ? '+' : ''}${row.expectancy_r.toFixed(3)}`}
                          </td>
                          <td className="py-1.5 text-right text-[#8A9BB5]">{row.max_drawdown_pct == null ? '—' : `${row.max_drawdown_pct.toFixed(2)}%`}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Per Strategy */}
              <div className="p-2.5 rounded cockpit-sub space-y-2">
                <div className="text-[11px] font-bold uppercase tracking-wider text-[#8A9BB5]">OOS Breakdown by Strategy</div>
                <div className="overflow-x-auto">
                  <table className="w-full text-xs text-left border-collapse mono">
                    <thead>
                      <tr className="border-b border-[#1F2D45] text-[#8A9BB5] font-sans">
                        <th className="py-1">Strategy</th>
                        <th className="py-1 text-right">Trades</th>
                        <th className="py-1 text-right">Win %</th>
                        <th className="py-1 text-right">PF</th>
                        <th className="py-1 text-right">Exp (R)</th>
                        <th className="py-1 text-right">Max DD</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-[#1F2D45]/70">
                      {audit.oos_backtest.by_strategy.length === 0 ? (
                        <tr><td colSpan={6} className="py-3 text-center text-[#8A9BB5] font-sans">No OOS trades triggered</td></tr>
                      ) : (
                        audit.oos_backtest.by_strategy.map((row) => (
                          <tr key={row.group}>
                            <td className="py-1.5 font-sans font-semibold text-white">{human(row.group)}</td>
                            <td className="py-1.5 text-right">{row.total_trades}</td>
                            <td className="py-1.5 text-right">{fmtPct(row.win_rate)}</td>
                            <td className={`py-1.5 text-right font-bold ${row.profit_factor == null ? 'text-[#8A9BB5]' : row.profit_factor >= 1.0 ? 'text-[#10B981]' : 'text-[#EF4444]'}`}>
                              {fmt(row.profit_factor, 2)}
                            </td>
                            <td className={`py-1.5 text-right ${row.expectancy_r == null ? 'text-[#8A9BB5]' : row.expectancy_r >= 0 ? 'text-[#10B981]' : 'text-[#EF4444]'}`}>
                              {row.expectancy_r == null ? '—' : `${row.expectancy_r >= 0 ? '+' : ''}${row.expectancy_r.toFixed(3)}`}
                            </td>
                            <td className="py-1.5 text-right text-[#8A9BB5]">{row.max_drawdown_pct == null ? '—' : `${row.max_drawdown_pct.toFixed(2)}%`}</td>
                          </tr>
                        ))
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>

          {/* Collapsible Raw Markdown Report */}
          {showMarkdown && audit.markdown_report && (
            <div className="cockpit-card rounded-lg p-3.5 space-y-2">
              <div className="flex items-center justify-between text-xs font-bold uppercase tracking-wider text-[#8A9BB5]">
                <span>Raw Scientist Audit Report (Markdown)</span>
                <button
                  onClick={copyMarkdown}
                  className="px-2 py-0.5 rounded cockpit-sub text-xs font-semibold cursor-pointer hover:bg-[#162236]"
                >
                  Copy
                </button>
              </div>
              <pre className="p-3 rounded bg-[#070B12] border border-[#1F2D45] text-[11px] mono text-[#CBD5E1] overflow-x-auto whitespace-pre-wrap max-h-96 overflow-y-auto">
                {audit.markdown_report}
              </pre>
            </div>
          )}
        </>
      )}
    </div>
  );
};
