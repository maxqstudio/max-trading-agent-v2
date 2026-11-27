@echo off
cd /d "%~dp0"
title Max Trading Agent v2 - START ALL
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" scripts\start_all.py %*
) else (
    python scripts\start_all.py %*
)
echo.
pause
