// Human-friendly names for the engine's internal identifiers (SHADOW_ONLY, TRENDING_BULL,
// liquidity_sweep, ...). The backend keeps its identifiers; only what people read changes.

const LABELS: Record<string, string> = {
  // trading modes
  SHADOW_ONLY: 'Paper (no orders)',
  SIMULATOR: 'Simulator',
  MT5_DEMO: 'MT5 Demo',
  MT5_LIVE: 'MT5 Live (real money)',
  // market regimes
  TRENDING_BULL: 'Trending up',
  TRENDING_BEAR: 'Trending down',
  RANGING: 'Sideways',
  HIGH_VOL_SPIKE: 'High volatility',
  LOW_VOL_CHOP: 'Quiet / choppy',
  // strategies
  liquidity_sweep: 'Liquidity Sweep',
  micro_breakout: 'Micro Breakout',
  ema_vwap_pullback: 'EMA VWAP Pullback',
  tick_imbalance: 'Tick Imbalance',
  // exit actions
  HOLD: 'Hold',
  MOVE_BE: 'Move to breakeven',
  MOVE_SL: 'Move SL',
  TIGHT_TRAIL: 'Tight trail',
  PARTIAL_50: 'Close 50%',
  PARTIAL_CLOSE: 'Partial close',
  EXIT_NOW: 'Exit now',
  PROFIT_LOCK: 'Profit lock',
  HIT_RUN_TAKE: 'Hit & Run take',
  HIT_RUN_TIME_STOP: 'Hit & Run time stop',
  // exit reasons
  TP: 'Take profit',
  SL: 'Stop loss',
  TRAIL_BE: 'Trailing / BE',
  TIME_EXPIRY: 'Time expired',
  TIME_STOP: 'Time stop',
  TAKE: 'Take profit (net)',
  LOCK: 'Profit locked',
  // gates and risk codes
  KILL_SWITCH_ACTIVE: 'Kill switch on',
  SHADOW_ONLY_PAPER_MODE: 'Paper (no orders sent)',
  LIVE_TRADING_NOT_ENABLED: 'Live locked (.env)',
  REAL_ACCOUNT_NOT_AUTHORIZED: 'Real account not allowed',
  DAILY_DD_LOCKOUT: 'Daily loss limit hit',
  ENGINE_PAUSED: 'Engine paused',
  LOSS_COOLDOWN: 'Cooldown after losses',
  LOT_TOO_SMALL: 'Lot too small',
  MAX_POSITIONS: 'Max positions reached',
  SPREAD_ATR_GATE: 'Spread too wide',
  SPREAD_SPIKE: 'Spread spike',
  NEWS_BLACKOUT: 'News blackout',
  OFF_SESSION: 'Outside session',
  SESSION_FILTER: 'Session filter',
  HTF_AGAINST: 'Against M15/H1 direction',
  ROLLOVER_BLACKOUT: 'Rollover blackout',
  SHOCK_COOLDOWN: 'Cooldown after spike',
  SOFT_PAUSE: 'Soft pause',
  MANUAL_SHIELD: 'Manual shield',
  MANUAL_KILL: 'Manual kill',
  DASHBOARD_KILL_SWITCH: 'Kill from dashboard',
  TELEGRAM_KILL: 'Kill from Telegram',
  FRIDAY_CLOSE_FLATTEN: 'Flatten before weekend',
  FLATTEN_ALL: 'Flatten all',
  // evolution statuses
  ACTIVE: 'Active',
  SHADOWING: 'Testing',
  IN_SHADOW: 'Testing',
  PENDING_APPROVAL: 'Waiting for approval',
  PASSED_OOS: 'Passed test',
  REJECTED: 'Rejected',
  SHADOW_REJECTED: 'Failed challenger test',
  AUTO_PROMOTED: 'Auto promoted',
  AUTO_ROLLED_BACK: 'Auto rolled back',
  ROLLED_BACK: 'Rolled back',
  PROMOTED: 'Promoted',
  // supervisor
  APPROVE: 'Approve',
  REDUCE: 'Reduce',
  REJECT: 'Reject',
};

const ID_TOKEN = /\b[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+\b/g;

/** One identifier -> label. Unknown ones lose their underscores: "MAX_SPREAD" -> "Max spread". */
export function human(id: string | null | undefined): string {
  if (!id) return '';
  if (LABELS[id]) return LABELS[id];
  if (!id.includes('_')) return id;
  const words = id.toLowerCase().split('_').filter(Boolean).join(' ');
  return words.charAt(0).toUpperCase() + words.slice(1);
}

/** Free text from the engine (tags, messages, traces): replace every snake_case token inside it. */
export function humanText(text: string | null | undefined): string {
  if (!text) return '';
  return text.replace(ID_TOKEN, (tok) => human(tok)).replace(/ -> /g, ' → ');
}
