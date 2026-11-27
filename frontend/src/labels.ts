// Human-friendly names for the engine's internal identifiers (SHADOW_ONLY, TRENDING_BULL,
// liquidity_sweep, ...). The backend keeps its identifiers; only what people read changes.

const LABELS: Record<string, string> = {
  // trading modes
  SHADOW_ONLY: 'Paper (tanpa order)',
  SIMULATOR: 'Simulator',
  MT5_DEMO: 'MT5 Demo',
  MT5_LIVE: 'MT5 Live (uang asli)',
  // market regimes
  TRENDING_BULL: 'Tren naik',
  TRENDING_BEAR: 'Tren turun',
  RANGING: 'Sideways',
  HIGH_VOL_SPIKE: 'Volatil tinggi',
  LOW_VOL_CHOP: 'Sepi / choppy',
  // strategies
  liquidity_sweep: 'Liquidity Sweep',
  micro_breakout: 'Micro Breakout',
  ema_vwap_pullback: 'EMA VWAP Pullback',
  tick_imbalance: 'Tick Imbalance',
  // exit actions
  HOLD: 'Tahan',
  MOVE_BE: 'Geser ke BE',
  MOVE_SL: 'Geser SL',
  TIGHT_TRAIL: 'Trailing ketat',
  PARTIAL_50: 'Tutup 50%',
  PARTIAL_CLOSE: 'Tutup sebagian',
  EXIT_NOW: 'Keluar sekarang',
  PROFIT_LOCK: 'Kunci profit',
  HIT_RUN_TAKE: 'Hit & Run ambil untung',
  // exit reasons
  TP: 'Take profit',
  SL: 'Stop loss',
  TRAIL_BE: 'Trailing / BE',
  TIME_EXPIRY: 'Waktu habis',
  TAKE: 'Ambil untung',
  LOCK: 'Profit terkunci',
  // gates and risk codes
  KILL_SWITCH_ACTIVE: 'Kill switch aktif',
  SHADOW_ONLY_PAPER_MODE: 'Paper (tidak kirim order)',
  LIVE_TRADING_NOT_ENABLED: 'Live terkunci (.env)',
  REAL_ACCOUNT_NOT_AUTHORIZED: 'Akun real belum diizinkan',
  DAILY_DD_LOCKOUT: 'Batas rugi harian tercapai',
  ENGINE_PAUSED: 'Mesin dijeda',
  LOSS_COOLDOWN: 'Jeda setelah rugi beruntun',
  LOT_TOO_SMALL: 'Lot terlalu kecil',
  MAX_POSITIONS: 'Posisi sudah maksimal',
  SPREAD_ATR_GATE: 'Spread terlalu lebar',
  SPREAD_SPIKE: 'Spread melonjak',
  NEWS_BLACKOUT: 'Jeda berita',
  OFF_SESSION: 'Di luar sesi',
  ROLLOVER_BLACKOUT: 'Jeda rollover',
  SHOCK_COOLDOWN: 'Jeda setelah lonjakan',
  SOFT_PAUSE: 'Jeda sementara',
  MANUAL_SHIELD: 'Perisai manual',
  MANUAL_KILL: 'Kill manual',
  DASHBOARD_KILL_SWITCH: 'Kill dari dashboard',
  TELEGRAM_KILL: 'Kill dari Telegram',
  FRIDAY_CLOSE_FLATTEN: 'Tutup semua sebelum weekend',
  FLATTEN_ALL: 'Tutup semua',
  // evolution statuses
  ACTIVE: 'Aktif',
  SHADOWING: 'Sedang diuji',
  IN_SHADOW: 'Sedang diuji',
  PENDING_APPROVAL: 'Menunggu persetujuan',
  PASSED_OOS: 'Lolos uji',
  REJECTED: 'Ditolak',
  SHADOW_REJECTED: 'Gagal di uji kandidat',
  AUTO_PROMOTED: 'Dipromosikan otomatis',
  AUTO_ROLLED_BACK: 'Dikembalikan otomatis',
  ROLLED_BACK: 'Dikembalikan',
  PROMOTED: 'Dipromosikan',
  // supervisor
  APPROVE: 'Setuju',
  REDUCE: 'Kecilkan',
  REJECT: 'Tolak',
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
