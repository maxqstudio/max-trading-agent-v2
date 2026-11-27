@echo off
cd /d "%~dp0"
title Max Trading Agent v2 - LIVE MT5 BRIDGE (DEMO/REAL)
echo Starting Max Trading Agent v2 in LIVE MT5 BRIDGE Mode...
echo Make sure MetaTrader 5 EA 'MaxTradingBridge' is attached on XAUUSD M1!
".venv\Scripts\python.exe" scripts\mt5_ea.py
echo Opening Institutional Quant Cockpit at http://127.0.0.1:8765
start "" cmd /c "timeout /t 3 /nobreak >nul && start http://127.0.0.1:8765"
".venv\Scripts\python.exe" -m backend.main
pause
