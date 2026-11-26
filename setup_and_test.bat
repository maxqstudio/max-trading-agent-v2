@echo off
cd /d "%~dp0"
title Max Trading Agent v2 - Automated Setup & Verification
echo ================================================================
echo   MAX TRADING AGENT v2 - ONE-CLICK SETUP, MT5 DEPLOY ^& TEST
echo ================================================================
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" scripts\setup_all.py
) else (
    python scripts\setup_all.py
)
echo.
pause
