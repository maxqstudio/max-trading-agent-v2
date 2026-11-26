@echo off
cd /d "%~dp0"
title Max Trading Agent v2 - SIMULATOR MODE
echo Starting Max Trading Agent v2 in Built-in SIMULATOR Mode...
echo Opening Institutional Quant Cockpit at http://127.0.0.1:8765
start "" cmd /c "timeout /t 3 /nobreak >nul && start http://127.0.0.1:8765"
".venv\Scripts\python.exe" -m backend.main --sim
pause
