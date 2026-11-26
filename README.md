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

See [mql5/README_MT5_SETUP.md](mql5/README_MT5_SETUP.md) for detailed MT5 configuration and [config/default.yaml](config/default.yaml) for risk, shields, and evolution settings.
