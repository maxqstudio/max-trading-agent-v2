import { useEffect, useRef, useState } from 'react';
import type { LiveStateSnapshot } from '../types/trading';

async function req<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    headers: { 'Content-Type': 'application/json', ...(init?.headers || {}) },
    ...init,
  });
  if (!res.ok) {
    const txt = await res.text();
    throw new Error(txt || `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  getState: () => req<LiveStateSnapshot>('/api/state'),
  getBrain: (symbol: string) => req<any>(`/api/brain?symbol=${encodeURIComponent(symbol)}`),
  getEvolution: () => req<any>('/api/evolution'),
  runEvolution: () => req<any>('/api/evolution/run', { method: 'POST' }),
  approveChallenger: () => req<any>('/api/evolution/approve', { method: 'POST' }),
  rejectChallenger: () => req<any>('/api/evolution/reject', { method: 'POST' }),
  rollbackChampion: (version_id: string) =>
    req<any>('/api/evolution/rollback', { method: 'POST', body: JSON.stringify({ version_id }) }),
  getJournal: (mode = 'ALL', symbol = 'ALL') =>
    req<any>(`/api/journal?mode=${encodeURIComponent(mode)}&symbol=${encodeURIComponent(symbol)}&limit=100`),
  getConfig: () => req<any>('/api/config'),
  updateRisk: (body: Record<string, unknown>) =>
    req<any>('/api/config/risk', { method: 'PUT', body: JSON.stringify(body) }),
  setMode: (trading_mode: string) =>
    req<any>('/api/control/mode', { method: 'POST', body: JSON.stringify({ trading_mode }) }),
  togglePause: () => req<any>('/api/control/pause', { method: 'POST' }),
  toggleShield: () => req<any>('/api/control/shield-toggle', { method: 'POST' }),
  killSwitch: (mode: 'SOFT_PAUSE' | 'FLATTEN_KILL', confirm_code = '') =>
    req<any>('/api/control/kill', { method: 'POST', body: JSON.stringify({ mode, confirm_code }) }),
  closePosition: (ticket: number) => req<any>(`/api/control/close/${ticket}`, { method: 'POST' }),
  triggerDemoSignal: (symbol: string) =>
    req<any>(`/api/demo/signal?symbol=${encodeURIComponent(symbol)}`, { method: 'POST' }),
};

export function useLiveTelemetry() {
  const [state, setState] = useState<LiveStateSnapshot | null>(null);
  const [wsConnected, setWsConnected] = useState(false);
  const lastFrameRef = useRef<number>(0);

  useEffect(() => {
    let alive = true;
    let ws: WebSocket | null = null;
    let retryMs = 800;
    let retryTimer: number | undefined;

    const loadInitial = async () => {
      try {
        const snap = await api.getState();
        if (alive) setState(snap);
      } catch {
        // retry via WS/polling
      }
    };
    loadInitial();

    const connect = () => {
      if (!alive) return;
      const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
      const url = `${proto}://${window.location.host}/ws/live`;
      ws = new WebSocket(url);

      ws.onopen = () => {
        if (!alive) return;
        setWsConnected(true);
        retryMs = 800;
      };
      ws.onmessage = (ev) => {
        if (!alive) return;
        const now = performance.now();
        // Throttle UI frames to ~250ms so 60fps stays smooth under fast tick bursts
        if (now - lastFrameRef.current < 240) return;
        lastFrameRef.current = now;
        try {
          const parsed = JSON.parse(ev.data) as LiveStateSnapshot;
          setState(parsed);
        } catch {
          // ignore malformed frame
        }
      };
      ws.onclose = () => {
        if (!alive) return;
        setWsConnected(false);
        retryTimer = window.setTimeout(connect, retryMs);
        retryMs = Math.min(5000, retryMs * 1.5);
      };
      ws.onerror = () => {
        ws?.close();
      };
    };

    connect();
    const pollFallback = window.setInterval(async () => {
      if (!wsConnected) {
        try {
          const snap = await api.getState();
          if (alive) setState(snap);
        } catch {
          // offline
        }
      }
    }, 2000);

    return () => {
      alive = false;
      window.clearInterval(pollFallback);
      if (retryTimer) window.clearTimeout(retryTimer);
      ws?.close();
    };
  }, []);

  return { state, wsConnected };
}
