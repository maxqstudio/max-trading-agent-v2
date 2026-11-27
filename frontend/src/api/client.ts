import { useEffect, useRef, useState } from 'react';
import type { LiveStateSnapshot } from '../types/trading';

const TOKEN_KEY = 'mta2_control_token';

// start_all.bat opens the dashboard as http://127.0.0.1:8765/#token=... so nobody has to paste it.
// The fragment never reaches the server; it is saved here and removed from the address bar.
(function takeTokenFromUrl() {
  try {
    const m = window.location.hash.match(/token=([^&]+)/);
    if (m) {
      localStorage.setItem(TOKEN_KEY, decodeURIComponent(m[1]));
      window.history.replaceState(null, '', window.location.pathname + window.location.search);
    }
  } catch {
    // ignore: falls back to the prompt
  }
})();

let promptOpen = false;
let lastPromptMs = 0;

// CONTROL_API_TOKEN from your .env. Kept only in this browser's localStorage.
export function getControlToken(): string {
  try {
    return localStorage.getItem(TOKEN_KEY) || '';
  } catch {
    return '';
  }
}

function askForControlToken(): string {
  // Ask at most once every 20 s so background polling never spams prompts.
  if (promptOpen || Date.now() - lastPromptMs < 20000) return getControlToken();
  promptOpen = true;
  const t = (
    window.prompt(
      'The dashboard needs CONTROL_API_TOKEN from the .env file.\n' +
        'start_all.bat copied it to the clipboard: press Ctrl+V, then OK.',
    ) || ''
  ).trim();
  promptOpen = false;
  lastPromptMs = Date.now();
  try {
    if (t) localStorage.setItem(TOKEN_KEY, t);
  } catch {
    // storage unavailable: token is used for this request only
  }
  return t;
}

async function req<T>(url: string, init?: RequestInit, retried = false): Promise<T> {
  let token = getControlToken();
  if (!token) token = askForControlToken();
  const res = await fetch(url, {
    ...init,
    headers: { 'Content-Type': 'application/json', 'X-Control-Token': token, ...(init?.headers || {}) },
  });
  if (res.status === 401 && !retried) {
    try {
      localStorage.removeItem(TOKEN_KEY);
    } catch {
      // ignore
    }
    askForControlToken();
    return req<T>(url, init, true);
  }
  if (!res.ok) {
    const txt = await res.text();
    let msg = txt;
    try {
      const j = JSON.parse(txt);
      msg = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail ?? j);
    } catch {
      // plain text
    }
    throw new Error(msg || `HTTP ${res.status}`);
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
  setMode: (trading_mode: string, confirm_code = '') =>
    req<any>('/api/control/mode', { method: 'POST', body: JSON.stringify({ trading_mode, confirm_code }) }),
  releaseKill: (confirm_code: string) =>
    req<any>('/api/control/kill/release', { method: 'POST', body: JSON.stringify({ confirm_code }) }),
  togglePause: () => req<any>('/api/control/pause', { method: 'POST' }),
  toggleShield: () => req<any>('/api/control/shield-toggle', { method: 'POST' }),
  killSwitch: (mode: 'SOFT_PAUSE' | 'FLATTEN_KILL', confirm_code = '') =>
    req<any>('/api/control/kill', { method: 'POST', body: JSON.stringify({ mode, confirm_code }) }),
  closePosition: (ticket: number) => req<any>(`/api/control/close/${ticket}`, { method: 'POST' }),
  getSetup: () => req<any>('/api/setup'),
  getPaperCompare: () => req<any>('/api/paper/compare'),
  resetPaper: (reset_learners = true) =>
    req<any>('/api/paper/reset', { method: 'POST', body: JSON.stringify({ reset_learners }) }),
  saveCosts: (body: Record<string, unknown>) => req<any>('/api/setup/costs', { method: 'PUT', body: JSON.stringify(body) }),
  saveFilters: (body: Record<string, unknown>) => req<any>('/api/setup/filters', { method: 'PUT', body: JSON.stringify(body) }),
  saveExits: (body: Record<string, unknown>) => req<any>('/api/setup/exits', { method: 'PUT', body: JSON.stringify(body) }),
  saveLlmSetup: (body: Record<string, unknown>) => req<any>('/api/setup/llm', { method: 'PUT', body: JSON.stringify(body) }),
  listModels: (provider: string, api_key = '', base_url = '') =>
    req<any>('/api/setup/llm/models', { method: 'POST', body: JSON.stringify({ provider, api_key, base_url }) }),
  testLlm: (provider?: string) => req<any>('/api/setup/llm/test', { method: 'POST', body: JSON.stringify({ provider }) }),
  saveSupervisorSetup: (body: Record<string, unknown>) =>
    req<any>('/api/setup/supervisor', { method: 'PUT', body: JSON.stringify(body) }),
  refreshSupervisorBrief: () => req<any>('/api/setup/supervisor/brief', { method: 'POST' }),
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
      const url = `${proto}://${window.location.host}/ws/live?token=${encodeURIComponent(getControlToken())}`;
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
