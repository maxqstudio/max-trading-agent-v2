import React, { useState } from 'react';
import { OctagonAlert, X } from 'lucide-react';

interface Props {
  open: boolean;
  onClose: () => void;
  onSoftPause: () => Promise<void>;
  onHardKill: (code: string) => Promise<void>;
}

export const KillSwitchModal: React.FC<Props> = ({ open, onClose, onSoftPause, onHardKill }) => {
  const [confirmText, setConfirmText] = useState('');
  const [busy, setBusy] = useState(false);

  if (!open) return null;
  const armed = confirmText.trim().toUpperCase() === 'KILL';

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75">
      <div className="w-full max-w-md rounded-xl cockpit-card border-2 border-[#EF4444] p-5 space-y-4 shadow-2xl">
        <div className="flex items-start justify-between gap-3">
          <div className="flex items-center gap-2 text-[#EF4444]">
            <OctagonAlert className="w-5 h-5 shrink-0" />
            <h3 className="text-sm font-bold uppercase tracking-wider">Emergency Kill-Switch</h3>
          </div>
          <button onClick={onClose} className="p-1 rounded cockpit-sub text-[#8A9BB5] hover:text-white cursor-pointer">
            <X className="w-4 h-4" />
          </button>
        </div>

        <p className="text-xs text-[#8A9BB5]">
          Choose a stop protocol. The command goes to the MQL5 EA on the next sync.
        </p>

        <div className="space-y-3 text-xs">
          <div className="p-3 rounded-lg cockpit-sub flex items-center justify-between gap-3">
            <div>
              <div className="font-bold text-[#E6EDF7]">Level 1 — Soft Entry Pause</div>
              <div className="text-[#8A9BB5] mt-0.5">Stop new entries; open positions stay managed by SL/TP and the RL exit.</div>
            </div>
            <button
              disabled={busy}
              onClick={async () => {
                setBusy(true);
                await onSoftPause();
                setBusy(false);
                onClose();
              }}
              className="px-3 py-1.5 rounded font-bold bg-[#F59E0B]/20 text-[#F59E0B] border border-[#F59E0B]/40 cursor-pointer shrink-0"
            >
              PAUSE
            </button>
          </div>

          <div className="p-3 rounded-lg cockpit-sub border border-[#EF4444]/50 space-y-2.5">
            <div>
              <div className="font-bold text-[#EF4444]">Level 2 — FLATTEN ALL & HARD LOCK</div>
              <div className="text-[#8A9BB5] mt-0.5">Force close every MT5 position and lock the system until resumed by hand.</div>
            </div>
            <div className="flex gap-2">
              <input
                type="text"
                value={confirmText}
                onChange={(e) => setConfirmText(e.target.value)}
                placeholder="Type KILL to unlock"
                className="flex-1 px-2.5 py-1.5 rounded cockpit-card mono text-xs focus:outline-none focus:border-[#EF4444]"
              />
              <button
                disabled={!armed || busy}
                onClick={async () => {
                  setBusy(true);
                  await onHardKill(confirmText);
                  setBusy(false);
                  setConfirmText('');
                  onClose();
                }}
                className={`px-3 py-1.5 rounded font-bold text-white ${
                  armed ? 'bg-[#EF4444] cursor-pointer hover:bg-[#DC2626]' : 'bg-[#EF4444]/35 cursor-not-allowed'
                }`}
              >
                FLATTEN ALL
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
