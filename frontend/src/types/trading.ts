export interface Bar {
  symbol: string;
  timeframe: string;
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  spread_points: number;
}

export interface DLPrediction {
  prob_down: number;
  prob_flat: number;
  prob_up: number;
  vol_expansion: number;
  trained: boolean;
}

export interface RadarSymbol {
  symbol: string;
  digits: number;
  point: number;
  bid: number;
  ask: number;
  spread_points: number;
  spread_atr_pct: number;
  regime: string;
  dl: DLPrediction;
  readiness: number;
  buy_pressure: number;
  shield: string;
  bars: Bar[];
}

export interface AccountState {
  balance: number;
  equity: number;
  margin: number;
  free_margin: number;
  day_start_equity: number;
  daily_pnl: number;
  daily_dd_pct: number;
  currency: string;
  is_demo: boolean;
}

export interface LivePosition {
  ticket: number;
  client_order_id: string;
  symbol: string;
  side: 'BUY' | 'SELL';
  volume: number;
  open_price: number;
  current_price: number;
  sl: number;
  tp: number;
  profit: number;
  strategy_id: string;
  regime: string;
  ml_confidence: number;
  dl_confidence: number;
  rl_exit_action: string;
  mae_r: number;
  mfe_r: number;
  initial_risk_price: number;
  bars_held: number;
}

export interface DecisionEvent {
  id: string;
  timestamp_ms: number;
  symbol: string;
  category: 'EXEC' | 'AI' | 'BLOCK' | 'EVOLUTION' | 'RISK';
  tag: string;
  strategy_id: string;
  side: string;
  message: string;
  latency_ms: number;
  metadata?: { pnl?: number; pnl_r?: number } & Record<string, unknown>;
}

export interface BacktestMetrics {
  total_trades: number;
  win_rate: number;
  profit_factor: number;
  sharpe_ratio: number;
  expectancy_r: number;
  max_drawdown_pct: number;
  net_return_pct: number;
  avg_mae_r: number;
  avg_mfe_r: number;
  mc_p95_dd_pct: number;
  mc_ruin_prob: number;
}

export interface ParameterPatch {
  path: string;
  old_value: number;
  new_value: number;
  rationale: string;
}

export interface ChallengerCandidate {
  version_id: string;
  parent_version_id: string;
  source: string;
  created_at_ms: number;
  status: string;
  patches: ParameterPatch[];
  oos_metrics: BacktestMetrics;
  champion_oos_metrics: BacktestMetrics;
  shadow_trades: number;
  shadow_pnl_r: number;
  shadow_pf: number;
  shadow_max_dd_pct: number;
  rejection_reason: string;
}

export interface ChampionSnapshot {
  version_id: string;
  created_at_ms: number;
  status: string;
  oos_metrics: BacktestMetrics;
  note: string;
  live_trades_since_promo: number;
  live_max_dd_pct: number;
}

export interface ReflectionReport {
  report_id: string;
  created_at_ms: number;
  provider: string;
  trades_analyzed: number;
  finding: string;
  evidence: string;
  patches: ParameterPatch[];
  gate_status: string;
}

export interface EquityPoint {
  timestamp_ms: number;
  champion_equity: number;
  shadow_equity: number;
  daily_dd_pct: number;
  event_label: string;
}

export interface LiveStateSnapshot {
  server_time_ms: number;
  trading_mode: string;
  champion_version: string;
  auto_promote: boolean;
  paused: boolean;
  killed: boolean;
  evolution_running: boolean;
  bridge_connected: boolean;
  bridge_stale_sec: number;
  bridge_latency_ms: number;
  ticks_per_sec: number;
  scans_count: number;
  active_thought: string;
  account: AccountState;
  shadow_equity: number;
  radar: RadarSymbol[];
  positions: LivePosition[];
  decisions: DecisionEvent[];
  active_challenger: ChallengerCandidate | null;
  manual_shield_override: boolean;
  equity_curve?: EquityPoint[];
  kill_reason?: string;
  live_trading_enabled?: boolean;
  entry_block_reason?: string | null;
  supervisor?: SupervisorStatus;
  llm?: LLMBrief;
  paper?: PaperBook;
}

export interface PaperPosition {
  book?: 'normal' | 'hitrun' | 'raw';
  label?: string;
  symbol: string;
  side: 'BUY' | 'SELL';
  volume: number;
  entry_price: number;
  current_price: number;
  sl: number;
  tp: number;
  strategy_id: string;
  profit: number;
  open_time: number;
}

export interface PaperBook {
  equity: number;
  positions: PaperPosition[];
  books?: Record<string, { label: string; equity: number; positions: PaperPosition[] }>;
}

export interface SupervisorLast {
  decision?: 'APPROVE' | 'REDUCE' | 'REJECT';
  size_multiplier?: number;
  reason?: string;
  provider?: string;
  latency_ms?: number;
  error?: string;
  ts_ms?: number;
  symbol?: string;
}

export interface SupervisorStatus {
  active: boolean;
  enabled: boolean;
  timeout_s: number;
  daily_budget_usd: number;
  spent_today_usd: number;
  calls: number;
  approve: number;
  reduce: number;
  reject: number;
  unavailable: number;
  last: SupervisorLast;
}

export interface LLMBrief {
  enabled: boolean;
  provider: string;
  label: string;
  model: string;
  fast_model: string;
  fallbacks: string[];
  total_cost_usd: number;
}

export interface TradeRecord {
  trade_id: string;
  ticket: number;
  is_shadow: boolean;
  version_id: string;
  symbol: string;
  strategy_id: string;
  regime: string;
  side: 'BUY' | 'SELL';
  volume: number;
  entry_price: number;
  exit_price: number;
  pnl_money: number;
  pnl_r: number;
  mae_r: number;
  mfe_r: number;
  spread_at_entry: number;
  slippage_points: number;
  commission: number;
  ml_confidence: number;
  dl_confidence: number;
  exit_reason: string;
  rl_exit_trace: string;
  open_time: number;
  close_time: number;
}
