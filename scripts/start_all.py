"""One click: check/repair the install, set up the AI key if needed, start the bot, open the dashboard.

Run via start_all.bat (Windows) or `python scripts/start_all.py`.
Safe by default: the bot starts in paper trading (SHADOW_ONLY) or the simulator; real money
stays impossible unless ALLOW_LIVE_TRADING=true is set in .env by hand.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import setup_all  # noqa: E402  (stdlib-only helpers: .env tokens, frontend build, EA deploy)

ENV_FILE = ROOT / ".env"
URL = "http://127.0.0.1:8765"


def venv_python() -> Path:
    win = ROOT / ".venv" / "Scripts" / "python.exe"
    return win if win.exists() or os.name == "nt" else ROOT / ".venv" / "bin" / "python"


def step(n: int, total: int, text: str) -> None:
    print(f"\n[{n}/{total}] {text}")


def ensure_install() -> Path:
    py = venv_python()
    if not py.exists():
        print("  First install: running the full setup (venv, dependencies, dashboard, MT5 EA, tests)...")
        subprocess.run([sys.executable, str(ROOT / "scripts" / "setup_all.py")], cwd=ROOT, check=False)
        py = venv_python()
        if not py.exists():
            raise SystemExit("  [ERROR] Could not create .venv. Make sure Python 3.11+ is installed, then try again.")
        return py
    # Existing install: make sure new dependencies (e.g. the anthropic SDK) are present.
    probe = subprocess.run([str(py), "-c", "import anthropic, fastapi, sklearn"], cwd=ROOT, capture_output=True)
    if probe.returncode != 0:
        print("  Updating Python dependencies...")
        subprocess.run([str(py), "-m", "pip", "install", "-q", "-e", ".[dev]"], cwd=ROOT, check=False)
    else:
        print("  [OK] Python and dependencies ready")
    return py


def ensure_env() -> None:
    setup_all.ensure_env_and_token()
    setup_all.ensure_control_token()


def ensure_mt5_ea() -> None:
    """Copy + compile the repo's EA into every MT5 terminal (no-op when already current)."""
    try:
        import mt5_ea

        mt5_ea.ensure_ea(env_value("BRIDGE_AUTH_TOKEN"))
    except Exception as exc:  # noqa: BLE001 - never block startup on the EA install
        print(f"  [WARN] MT5 EA was not installed automatically ({type(exc).__name__}). Run: python scripts/mt5_ea.py")


def dashboard_is_current() -> bool:
    """True when frontend/dist was built after the last change to the dashboard sources."""
    built = ROOT / "frontend" / "dist" / "index.html"
    if not built.exists():
        return False
    front = ROOT / "frontend"
    sources = [p for p in (front / "src").rglob("*") if p.is_file()] if (front / "src").exists() else []
    sources += [p for p in (front / n for n in ("index.html", "package.json", "vite.config.ts")) if p.exists()]
    newest = max((p.stat().st_mtime for p in sources), default=0.0)
    return built.stat().st_mtime >= newest


def ensure_dashboard() -> None:
    if dashboard_is_current():
        print("  [OK] Dashboard already built")
        return
    npm = shutil.which("npm.cmd") or shutil.which("npm")
    if not npm:
        print("  [WARN] npm not found: the API still runs, but there is no dashboard. Install Node.js and run this again.")
        return
    print("  Building the dashboard...")
    if not (ROOT / "frontend" / "node_modules").exists():
        subprocess.run([npm, "--prefix", str(ROOT / "frontend"), "install"], check=False)
    subprocess.run([npm, "--prefix", str(ROOT / "frontend"), "run", "build"], check=False)


def ensure_llm(py: Path) -> None:
    res = subprocess.run([str(py), str(ROOT / "scripts" / "llm_setup.py"), "--check"], cwd=ROOT, capture_output=True, text=True)
    if res.returncode == 0:
        print(f"  [OK] AI provider active: {res.stdout.strip()}")
        return
    print("  No AI API key yet (Claude/Gemini/OpenAI). Without a key the robot runs in deterministic mode.")
    if input("  Set up now? (y/n) [y]: ").strip().lower() in ("", "y", "ya", "yes"):
        subprocess.run([str(py), str(ROOT / "scripts" / "llm_setup.py")], cwd=ROOT, check=False)


def env_value(key: str) -> str:
    if not ENV_FILE.exists():
        return ""
    m = re.search(rf"^{key}=(.*)$", ENV_FILE.read_text(encoding="utf-8"), flags=re.MULTILINE)
    return m.group(1).strip() if m else ""


def copy_to_clipboard(text: str) -> bool:
    try:
        if os.name == "nt":
            subprocess.run(["clip"], input=text.encode("utf-16le"), check=True)
            return True
        for cmd in (["pbcopy"], ["xclip", "-selection", "clipboard"]):
            if shutil.which(cmd[0]):
                subprocess.run(cmd, input=text.encode(), check=True)
                return True
    except Exception:
        pass
    return False


def choose_mode() -> bool:
    print("  1. SIMULATOR  : built-in simulated market, no MetaTrader (fastest way to try it)")
    print("  2. MT5 BRIDGE : connects to MetaTrader 5. Default PAPER (SHADOW_ONLY), sends no orders")
    ans = input("  Choose mode (1/2) [1]: ").strip()
    return ans != "2"


def set_env_value(key: str, value: str) -> None:
    content = ENV_FILE.read_text(encoding="utf-8") if ENV_FILE.exists() else ""
    if re.search(rf"^{key}=.*$", content, flags=re.MULTILINE):
        content = re.sub(rf"^{key}=.*$", f"{key}={value}", content, flags=re.MULTILINE)
    else:
        content += ("" if content.endswith("\n") or not content else "\n") + f"{key}={value}\n"
    ENV_FILE.write_text(content, encoding="utf-8")


def already_set_up() -> bool:
    """True once a full run finished: venv, both tokens, dashboard build and a saved start mode."""
    return (
        venv_python().exists()
        and len(env_value("BRIDGE_AUTH_TOKEN")) >= 24
        and len(env_value("CONTROL_API_TOKEN")) >= 24
        and (ROOT / "frontend" / "dist" / "index.html").exists()
        and env_value("START_MODE") in ("sim", "mt5")
    )


def launch(py: Path, sim: bool) -> int:
    live = env_value("ALLOW_LIVE_TRADING").lower() == "true"
    token = env_value("CONTROL_API_TOKEN")
    print("\n" + "-" * 64)
    print(f"  Mode        : {'SIMULATOR' if sim else 'MT5 BRIDGE (paper / SHADOW_ONLY until you change it in the dashboard)'}")
    print(f"  Real money  : {'ALLOWED in .env (careful!)' if live else 'LOCKED (ALLOW_LIVE_TRADING=false)'}")
    if token:
        copied = copy_to_clipboard(token)
        where = "also copied to the clipboard" if copied else "full value in the .env file (CONTROL_API_TOKEN)"
        print(f"  Token login : ...{token[-4:]} — sent to the dashboard automatically, {where}")
    if not sim:
        print("  MT5         : EA 'MaxTradingBridge' installed automatically (token already filled in).")
        print("                In MT5: Tools > Options > Expert Advisors > allow WebRequest to " + URL)
        print("                then drag the EA onto an XAUUSD M1 chart and turn on Algo Trading.")
    print(f"  Dashboard   : {URL}   (close this window or press Ctrl+C to stop)")
    print("  Change mode / redo setup: run  start_all.bat --setup")
    print("-" * 64 + "\n")

    # The token rides in the URL fragment (never sent to the server); the dashboard saves it.
    open_url = f"{URL}/#token={token}" if token else URL
    threading.Timer(3.0, lambda: webbrowser.open(open_url)).start()
    cmd = [str(py), "-m", "backend.main"] + (["--sim"] if sim else [])
    try:
        return subprocess.run(cmd, cwd=ROOT).returncode
    except KeyboardInterrupt:
        print("\nStopped. Bye!")
        time.sleep(0.5)
        return 0


def main() -> int:
    os.chdir(ROOT)
    print("=" * 64)
    print("  MAX TRADING AGENT v2 — START ALL (one click)")
    print("=" * 64)

    if already_set_up() and "--setup" not in sys.argv:
        # Fast path: everything is in place, go straight to the dashboard.
        py = venv_python()
        probe = subprocess.run([str(py), "-c", "import anthropic, fastapi, sklearn"], cwd=ROOT, capture_output=True)
        if probe.returncode != 0:
            print("  Updating Python dependencies...")
            subprocess.run([str(py), "-m", "pip", "install", "-q", "-e", ".[dev]"], cwd=ROOT, check=False)
        if not dashboard_is_current():
            print("  Dashboard changed, rebuilding...")
            ensure_dashboard()
        sim = env_value("START_MODE") == "sim"
        if not sim:
            ensure_mt5_ea()
        return launch(py, sim)

    total = 6
    step(1, total, "Check installation")
    py = ensure_install()
    step(2, total, "Check .env and security tokens")
    ensure_env()
    step(3, total, "Check dashboard")
    ensure_dashboard()
    step(4, total, "Check AI provider (Claude first)")
    ensure_llm(py)
    step(5, total, "Choose mode")
    sim = choose_mode()
    set_env_value("START_MODE", "sim" if sim else "mt5")
    step(6, total, "Install the MetaTrader 5 EA")
    if sim:
        print("  Skipped (SIMULATOR mode)")
    else:
        ensure_mt5_ea()
    return launch(py, sim)


if __name__ == "__main__":
    raise SystemExit(main())
