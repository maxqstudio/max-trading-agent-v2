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
        print("  Instalasi pertama: menjalankan setup lengkap (venv, dependency, dashboard, EA MT5, tes)...")
        subprocess.run([sys.executable, str(ROOT / "scripts" / "setup_all.py")], cwd=ROOT, check=False)
        py = venv_python()
        if not py.exists():
            raise SystemExit("  [ERROR] Gagal membuat .venv. Pastikan Python 3.11+ terpasang lalu coba lagi.")
        return py
    # Existing install: make sure new dependencies (e.g. the anthropic SDK) are present.
    probe = subprocess.run([str(py), "-c", "import anthropic, fastapi, sklearn"], cwd=ROOT, capture_output=True)
    if probe.returncode != 0:
        print("  Memperbarui dependency Python...")
        subprocess.run([str(py), "-m", "pip", "install", "-q", "-e", ".[dev]"], cwd=ROOT, check=False)
    else:
        print("  [OK] Python & dependency siap")
    return py


def ensure_env() -> None:
    setup_all.ensure_env_and_token()
    setup_all.ensure_control_token()


def ensure_dashboard() -> None:
    if (ROOT / "frontend" / "dist" / "index.html").exists():
        print("  [OK] Dashboard sudah ter-build")
        return
    npm = shutil.which("npm.cmd") or shutil.which("npm")
    if not npm:
        print("  [WARN] npm tidak ditemukan: API tetap jalan, tapi dashboard belum ada. Install Node.js lalu jalankan lagi.")
        return
    print("  Membangun dashboard...")
    if not (ROOT / "frontend" / "node_modules").exists():
        subprocess.run([npm, "--prefix", str(ROOT / "frontend"), "install"], check=False)
    subprocess.run([npm, "--prefix", str(ROOT / "frontend"), "run", "build"], check=False)


def ensure_llm(py: Path) -> None:
    res = subprocess.run([str(py), str(ROOT / "scripts" / "llm_setup.py"), "--check"], cwd=ROOT, capture_output=True, text=True)
    if res.returncode == 0:
        print(f"  [OK] AI provider aktif: {res.stdout.strip()}")
        return
    print("  Belum ada API key AI (Claude/Gemini/OpenAI). Tanpa key, robot jalan dengan mode deterministik.")
    if input("  Setup sekarang? (y/n) [y]: ").strip().lower() in ("", "y", "ya", "yes"):
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
    print("  1. SIMULATOR  : pasar simulasi bawaan, tanpa MetaTrader (paling cepat untuk mencoba)")
    print("  2. MT5 BRIDGE : terhubung ke MetaTrader 5. Default PAPER (SHADOW_ONLY), tidak kirim order")
    ans = input("  Pilih mode (1/2) [1]: ").strip()
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
    print(f"  Mode        : {'SIMULATOR' if sim else 'MT5 BRIDGE (paper / SHADOW_ONLY sampai kamu ganti di dashboard)'}")
    print(f"  Uang asli   : {'DIIZINKAN di .env (hati-hati!)' if live else 'TERKUNCI (ALLOW_LIVE_TRADING=false)'}")
    if token:
        copied = copy_to_clipboard(token)
        where = "sudah disalin ke clipboard, tinggal Ctrl+V" if copied else "ada di file .env (CONTROL_API_TOKEN)"
        print(f"  Token login : ...{token[-4:]} — {where} saat dashboard meminta")
    if not sim:
        print("  MT5         : pasang EA 'MaxTradingBridge' di chart XAUUSD M1, izinkan WebRequest ke " + URL)
    print(f"  Dashboard   : {URL}   (tutup jendela ini atau Ctrl+C untuk berhenti)")
    print("  Ganti mode / ulang setup: jalankan  start_all.bat --setup")
    print("-" * 64 + "\n")

    threading.Timer(3.0, lambda: webbrowser.open(URL)).start()
    cmd = [str(py), "-m", "backend.main"] + (["--sim"] if sim else [])
    try:
        return subprocess.run(cmd, cwd=ROOT).returncode
    except KeyboardInterrupt:
        print("\nBerhenti. Sampai jumpa!")
        time.sleep(0.5)
        return 0


def main() -> int:
    os.chdir(ROOT)
    print("=" * 64)
    print("  MAX TRADING AGENT v2 — START ALL (1 klik)")
    print("=" * 64)

    if already_set_up() and "--setup" not in sys.argv:
        # Fast path: everything is in place, go straight to the dashboard.
        py = venv_python()
        probe = subprocess.run([str(py), "-c", "import anthropic, fastapi, sklearn"], cwd=ROOT, capture_output=True)
        if probe.returncode != 0:
            print("  Memperbarui dependency Python...")
            subprocess.run([str(py), "-m", "pip", "install", "-q", "-e", ".[dev]"], cwd=ROOT, check=False)
        return launch(py, env_value("START_MODE") == "sim")

    total = 5
    step(1, total, "Cek instalasi")
    py = ensure_install()
    step(2, total, "Cek .env & token keamanan")
    ensure_env()
    step(3, total, "Cek dashboard")
    ensure_dashboard()
    step(4, total, "Cek AI provider (Claude dulu)")
    ensure_llm(py)
    step(5, total, "Pilih mode")
    sim = choose_mode()
    set_env_value("START_MODE", "sim" if sim else "mt5")
    return launch(py, sim)


if __name__ == "__main__":
    raise SystemExit(main())
