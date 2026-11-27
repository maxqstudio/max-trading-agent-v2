# Max Trading Agent v2

**Battle-Ready Autonomous Self-Improving MT5 Scalping Agent** (`XAUUSD`, `EURUSD`, `GBPUSD`, `USDJPY` — M1/M5) featuring a **4-Pillar Full Hybrid AI Brain** (**Lightweight NumPy 1D-CNN DL** + **Calibrated HistGradientBoosting ML** + **Dual RL [Contextual Thompson Bandit + Q-Learning Dynamic Exit]** + **LLM Post-Mortem Reflection**), guarded by a **4-Stage Anti-Overfit Champion/Challenger Gate**, a **Zero-DLL Idempotent MQL5 Execution Bridge**, an **Institutional Quant Cockpit Web Dashboard**, and a **Telegram Control Bot**.

## One-Click Setup & Launch (Windows)

After cloning on any Windows PC (or VPS):

```powershell
git clone https://github.com/maxqstudio/max-trading-agent-v2.git
cd max-trading-agent-v2
```

1. **Double-click `setup_and_test.bat`** (runs `scripts/setup_all.py`):
   - Automatically creates `.env` with a strong random `BRIDGE_AUTH_TOKEN`
   - Creates `.venv` and installs all Python dependencies (`fastapi`, `numpy`, `scikit-learn`, `aiosqlite`, `pytest`, etc.)
   - Installs & builds the React **Institutional Quant Cockpit** (`frontend/dist`)
   - Auto-detects MetaTrader 5, deploys `MaxTradingBridge.mq5` with your matching `BRIDGE_AUTH_TOKEN` pre-filled, and compiles `MaxTradingBridge.ex5` via `metaeditor64.exe`
   - Runs the full automated `pytest` verification suite (16 tests)

2. **Test immediately in Built-in Simulator Mode (No MT5 needed)**:
   - Double-click **`start_simulator.bat`** → opens **`http://127.0.0.1:8765`**

3. **Run Live with MetaTrader 5 (Demo Account First!)**:
   - In MT5: **Tools → Options → Expert Advisors** → check *Allow Algorithmic Trading* + *Allow WebRequest for listed URL* → add `http://127.0.0.1:8765`
   - Double-click **`start_mt5_live.bat`** → opens **`http://127.0.0.1:8765`**
   - In MT5 Navigator, drag **`MaxTradingBridge`** onto 1 chart (`XAUUSD` M1). Token is already pre-configured! If your broker uses a suffix (e.g., `XAUUSDm`), set `SymbolSuffix = m`.

## AI Providers (Claude first)

Double-click **`llm_setup.bat`** (or `python scripts/llm_setup.py`): pick providers in order, paste keys, accept or change the models, and it tests each connection. Everything is written to your local `.env` only.

| Provider | Key in `.env` | Notes |
|---|---|---|
| Claude (default, 1st) | `ANTHROPIC_API_KEY` | Official SDK, structured outputs, tool use, prompt caching, refusal fallback. `ANTHROPIC_MODEL=claude-opus-5-5` (reflection), `ANTHROPIC_FAST_MODEL=claude-haiku-5-5` (scalping supervisor) |
| Gemini (2nd) | `GEMINI_API_KEY` | Set `GEMINI_MODEL` to the exact id you use |
| OpenAI (3rd) | `OPENAI_API_KEY` | |
| OpenRouter, Groq, DeepSeek, Mistral, xAI, Together | `<NAME>_API_KEY` | OpenAI-compatible presets |
| Ollama, llama.cpp | none | Local and free; `OLLAMA_BASE_URL` / `LLAMACPP_BASE_URL` |
| Custom | `CUSTOM_BASE_URL`, `CUSTOM_MODEL`, `CUSTOM_API_KEY` | Any OpenAI-compatible server |

`LLM_PROVIDER_CHAIN=anthropic,gemini,openai` sets the order; the next provider is tried when one fails or times out, and the deterministic quant reflector is the last resort. `GET /api/llm/status` shows the chain, masked keys, latency and spend; `POST /api/llm/test` pings each provider.

LLM suggestions can never raise risk: patches are whitelisted, clamped, capped at a 20% step, risk-increasing directions are discarded, and every change still has to pass walk-forward + Monte Carlo + shadow trading before a human approves it.

## Claude Trade Supervisor (veto layer)

Every trade the quant engine wants to take (strategy → bandit → ML filter → risk-sized) is sent to the supervisor before it reaches MT5. It answers **APPROVE**, **REDUCE** (0.25–0.9× size) or **REJECT** with a one-line reason that shows up in the dashboard decision log. It can never enlarge a trade, move SL/TP or open anything by itself.

Tuned for M1 scalping (`supervisor:` in `config/default.yaml`):
- one call with a compact (~400 token) context, `ANTHROPIC_FAST_MODEL` (default `claude-haiku-5-5`) at `effort: low`, 6 s timeout; it runs off the bridge loop so the EA never waits
- `on_unavailable: block` (default) skips the trade if the model is slow, down, invalid or over budget; `allow` trades without a veto instead
- after the answer, kill switch / pause / open position / price drift (`max_price_drift_atr`) are re-checked
- `max_calls_per_hour` and `daily_budget_usd` cap spend; `GET /api/llm/status` shows approve/reduce/reject counts and spend
- `review_in_shadow: true` lets you judge the supervisor on paper trades before going live

## Safety Rails (read before trading real money)

- **Paper trading by default.** `trading_mode: SHADOW_ONLY` logs and shadow-tracks every validated setup but sends nothing to MT5.
- **Real money is opt-in in `.env` only.** New entries on a real account are refused unless `ALLOW_LIVE_TRADING=true` **and** the mode is `MT5_LIVE` (switching to it also needs the confirmation code `LIVE`). The dashboard, Telegram and the LLM cannot flip this flag.
- **Persistent kill switch.** `FLATTEN_KILL` is written to `data/kill_switch.json` and survives restarts. Pause/resume cannot clear it; release it explicitly (dashboard → type `RELEASE`, or Telegram `/unkill confirm`). The engine stays paused after release.
- **Authenticated control API.** Everything under `/api` except `/api/health` needs `X-Control-Token` (`CONTROL_API_TOKEN` in `.env`, generated by setup). CORS only allows the local dashboard, so other websites cannot drive the bot.
- **No secrets in git.** `.env`, compiled `.ex5` files and key files are git-ignored; empty or placeholder tokens are rejected; CI runs gitleaks on every push/PR. Optional local hook: `pip install pre-commit && pre-commit install`.

See [mql5/README_MT5_SETUP.md](mql5/README_MT5_SETUP.md) for detailed MT5 configuration and [config/default.yaml](config/default.yaml) for risk, shields, and evolution settings.
