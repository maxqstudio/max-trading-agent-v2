@echo off
cd /d "%~dp0"
title Max Trading Agent v2 - Setup LLM (Claude, Gemini, OpenAI, lokal)
".venv\Scripts\python.exe" scripts\llm_setup.py
pause
