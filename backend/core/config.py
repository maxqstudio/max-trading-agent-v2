from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from backend.core.models import TradingMode


class AppConfig(BaseModel):
    name: str = "Max Trading Agent v2"
    version: str = "v1.0"
    # Paper trading by default: validated setups are logged and tracked in the
    # Shadow Arena, but nothing is sent to MT5 until the operator switches mode.
    trading_mode: TradingMode = TradingMode.SHADOW_ONLY
    auto_promote_challenger: bool = False
    database_path: str = "data/max_agent.db"
    kill_switch_path: str = "data/kill_switch.json"
    snapshots_dir: str = "data/snapshots"
    host: str = "127.0.0.1"
    port: int = 8765


class InstrumentsConfig(BaseModel):
    symbols: list[str] = Field(default_factory=lambda: ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY"])
    symbol_suffix: str = ""
    primary_timeframe: str = "M1"
    confirm_timeframe: str = "M5"
    order_ttl_ms: int = 1500
    magic_number: int = 202602


class RiskConfig(BaseModel):
    risk_per_trade_pct: float = 0.50
    min_risk_pct: float = 0.20
    max_risk_pct: float = 1.00
    half_kelly_enabled: bool = True
    max_daily_drawdown_pct: float = 3.00
    max_concurrent_positions: int = 3
    max_usd_correlated_positions: int = 2
    max_spread_atr_ratio: float = 0.30
    consecutive_loss_limit: int = 3
    consecutive_loss_cooldown_min: int = 15
    min_rr_ratio: float = 1.25


class ShieldsConfig(BaseModel):
    rollover_blackout_enabled: bool = True
    rollover_start_utc: str = "21:50"
    rollover_end_utc: str = "23:10"
    news_blackout_enabled: bool = True
    news_file: str = "config/news_blackout.json"
    news_buffer_minutes: int = 15
    spread_anomaly_multiplier: float = 2.5
    shock_bar_atr_multiplier: float = 3.0
    shock_cooldown_minutes: int = 10
    friday_flatten_enabled: bool = True
    friday_flatten_utc: str = "20:45"
    preferred_sessions_only: bool = False


class AIConfig(BaseModel):
    dl_seq_len: int = 32
    dl_min_train_samples: int = 120
    dl_epochs: int = 18
    dl_lr: float = 0.008
    ml_min_probability: float = 0.56
    ml_min_train_samples: int = 80
    bandit_forgetting_factor: float = 0.985
    rl_exit_enabled: bool = True
    rl_exit_be_trigger_r: float = 0.80
    rl_exit_trail_trigger_r: float = 1.25
    rl_exit_partial_trigger_r: float = 1.50


class EvolutionConfig(BaseModel):
    schedule_interval_minutes: int = 240
    min_bars_for_wfo: int = 220
    oos_min_trades: int = 20
    oos_min_profit_factor: float = 1.20
    oos_max_drawdown_pct: float = 3.00
    monte_carlo_iterations: int = 1000
    monte_carlo_p95_max_dd_pct: float = 4.50
    shadow_min_trades: int = 15
    rollback_live_dd_pct: float = 1.50
    rollback_min_trades: int = 8
    wfo_trials: int = 12
    # The DL/ML models learn from this many recent M1 bars per symbol, read from the database
    # (4320 = 3 days). Strategy parameters are searched on the most recent param_window_bars.
    train_window_bars: int = 4320
    param_window_bars: int = 600
    # The CNN continues from the champion's weights at this fraction of the normal learning rate.
    dl_finetune_lr_mult: float = 0.5


class SupervisorConfig(BaseModel):
    """LLM (Claude first) veto layer between the risk manager and MT5. See backend/engine/supervisor.py."""

    enabled: bool = True
    timeout_s: float = 6.0
    use_fast_model: bool = True
    effort: str = "low"
    on_unavailable: str = "block"  # block = skip the trade if the LLM is slow/down; allow = trade without veto
    max_calls_per_hour: int = 120
    daily_budget_usd: float = 5.0
    max_price_drift_atr: float = 0.25
    review_in_shadow: bool = True  # also review paper trades so you can judge the supervisor before going live


class ExitConfig(BaseModel):
    """Profit protection for the existing system (MT5 positions + 'existing' paper book)."""

    be_at_r: float = 0.5  # move SL to breakeven once the trade has been +0.5R
    be_offset_r: float = 0.05
    trail_start_r: float = 1.0
    trail_dist_r: float = 0.5
    # Learn per symbol+regime from the two paper books whether Hit & Run beats Normal,
    # and use the winner for real MT5 positions once there is enough evidence.
    adaptive: bool = True
    adaptive_min_samples: int = 20


class HitRunConfig(BaseModel):
    take_r: float = 0.5  # minimum profit before Hit & Run takes it
    min_spread_mult: float = 2.0  # and at least 2x spread
    trail_dist_r: float = 0.15  # 0 = close at once when take_r is reached
    be_offset_r: float = 0.05
    # Learn the take level (0.3/0.5/0.7/1.0R) per symbol+regime; take_r stays the default
    # until a group has adaptive_min_samples trades and another level is clearly better.
    adaptive: bool = True
    adaptive_min_samples: int = 20


class PaperConfig(BaseModel):
    """Three virtual books run side by side on the same setups, in every trading mode."""

    enabled: bool = True
    start_equity: float = 10000.0
    max_hold_min: int = 240
    hit_run: HitRunConfig = Field(default_factory=HitRunConfig)


class TradingConfig(BaseModel):
    app: AppConfig = Field(default_factory=AppConfig)
    instruments: InstrumentsConfig = Field(default_factory=InstrumentsConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    shields: ShieldsConfig = Field(default_factory=ShieldsConfig)
    ai: AIConfig = Field(default_factory=AIConfig)
    strategies: dict[str, dict[str, Any]] = Field(default_factory=dict)
    evolution: EvolutionConfig = Field(default_factory=EvolutionConfig)
    supervisor: SupervisorConfig = Field(default_factory=SupervisorConfig)
    exits: ExitConfig = Field(default_factory=ExitConfig)
    paper: PaperConfig = Field(default_factory=PaperConfig)


class EnvSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Shared secret between this server and the MT5 EA. Empty / placeholder = bridge refuses all syncs.
    bridge_auth_token: str = ""
    # Secret required by the dashboard & any API call that changes state (mode, risk, kill...).
    control_api_token: str = ""
    # Real-money switch. Only settable in .env, never from the dashboard, Telegram or the LLM.
    allow_live_trading: bool = False
    # LLM providers are resolved by backend/llm/settings.py straight from .env + environment
    # (LLM_PROVIDER_CHAIN, ANTHROPIC_API_KEY, GEMINI_MODEL, ...). Kept here for legacy .env files only.
    llm_provider: str = ""
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    api_host: str = "127.0.0.1"
    api_port: int = 8765


LOCAL_CONFIG_PATH = Path("config/local.yaml")  # written by the dashboard Setup page, git-ignored


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_trading_config(
    config_path: str | Path = "config/default.yaml", local_path: str | Path | None = LOCAL_CONFIG_PATH
) -> TradingConfig:
    path = Path(config_path)
    raw: dict[str, Any] = {}
    if path.exists():
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if local_path and Path(local_path).exists():
        local = yaml.safe_load(Path(local_path).read_text(encoding="utf-8")) or {}
        # The trading mode is never taken from the local file: every start begins in the safe default.
        if isinstance(local.get("app"), dict):
            local["app"].pop("trading_mode", None)
        raw = _deep_merge(raw, local)
    return TradingConfig.model_validate(raw)


def save_local_config(section: str, values: dict[str, Any], local_path: str | Path = LOCAL_CONFIG_PATH) -> None:
    """Persist one config section (e.g. supervisor) to config/local.yaml, atomically."""
    p = Path(local_path)
    data = (yaml.safe_load(p.read_text(encoding="utf-8")) or {}) if p.exists() else {}
    data[section] = values
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    tmp.replace(p)


def get_nested_value(data: dict[str, Any], dotted_path: str) -> Any:
    cur: Any = data
    for key in dotted_path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            raise KeyError(f"Path '{dotted_path}' not found at segment '{key}'")
        cur = cur[key]
    return cur


def set_nested_value(data: dict[str, Any], dotted_path: str, value: Any) -> dict[str, Any]:
    out = copy.deepcopy(data)
    parts = dotted_path.split(".")
    cur: Any = out
    for key in parts[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[parts[-1]] = value
    return out
