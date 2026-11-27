from __future__ import annotations

import os
import re
import secrets
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"
ENV_EXAMPLE = ROOT / ".env.example"
MQL5_SRC = ROOT / "mql5" / "MaxTradingBridge.mq5"


def banner(title: str) -> None:
    print(f"\n{'=' * 64}\n  {title}\n{'=' * 64}")


def ensure_env_and_token() -> str:
    banner("1/5. Checking .env & BRIDGE_AUTH_TOKEN")
    if not ENV_FILE.exists():
        content = ENV_EXAMPLE.read_text(encoding="utf-8")
        token = f"mta2_{secrets.token_hex(20)}"
        content = re.sub(
            r"^BRIDGE_AUTH_TOKEN=.*$",
            f"BRIDGE_AUTH_TOKEN={token}",
            content,
            flags=re.MULTILINE,
        )
        ENV_FILE.write_text(content, encoding="utf-8")
        print(f"[OK] Created .env with auto-generated BRIDGE_AUTH_TOKEN (...{token[-4:]})")
        return token

    content = ENV_FILE.read_text(encoding="utf-8")
    m = re.search(r"^BRIDGE_AUTH_TOKEN=(.*)$", content, flags=re.MULTILINE)
    if m and m.group(1).strip() and "change_me" not in m.group(1):
        token = m.group(1).strip()
        print(f"[OK] Existing .env found with BRIDGE_AUTH_TOKEN (...{token[-4:]})")
        return token

    token = f"mta2_{secrets.token_hex(20)}"
    if m:
        content = re.sub(r"^BRIDGE_AUTH_TOKEN=.*$", f"BRIDGE_AUTH_TOKEN={token}", content, flags=re.MULTILINE)
    else:
        content += f"\nBRIDGE_AUTH_TOKEN={token}\n"
    ENV_FILE.write_text(content, encoding="utf-8")
    print(f"[OK] Updated .env with secure BRIDGE_AUTH_TOKEN (...{token[-4:]})")
    return token


def ensure_control_token() -> str:
    """Generate CONTROL_API_TOKEN (dashboard/API secret) if missing. Never printed in full."""
    content = ENV_FILE.read_text(encoding="utf-8")
    m = re.search(r"^CONTROL_API_TOKEN=(.*)$", content, flags=re.MULTILINE)
    if m and len(m.group(1).strip()) >= 24 and "change_me" not in m.group(1):
        print(f"[OK] Existing CONTROL_API_TOKEN found (...{m.group(1).strip()[-4:]})")
        return m.group(1).strip()
    token = f"mtc2_{secrets.token_urlsafe(24)}"
    if m:
        content = re.sub(r"^CONTROL_API_TOKEN=.*$", f"CONTROL_API_TOKEN={token}", content, flags=re.MULTILINE)
    else:
        content += f"\nCONTROL_API_TOKEN={token}\n"
    if not re.search(r"^ALLOW_LIVE_TRADING=", content, flags=re.MULTILINE):
        content += "ALLOW_LIVE_TRADING=false\n"
    ENV_FILE.write_text(content, encoding="utf-8")
    print("[OK] Generated CONTROL_API_TOKEN in .env — paste it into the dashboard the first time it asks.")
    return token


def check_python_and_frontend() -> Path:
    banner("2/5. Verifying Python venv & Frontend Build")
    venv_py = ROOT / ".venv" / "Scripts" / "python.exe"
    venv_pip = ROOT / ".venv" / "Scripts" / "pip.exe"
    if not venv_py.exists():
        print("[INFO] Creating .venv...")
        subprocess.run([sys.executable, "-m", "venv", str(ROOT / ".venv")], check=True)
        subprocess.run([str(venv_pip), "install", "-e", ".[dev]"], cwd=ROOT, check=True)
    else:
        print(f"[OK] Python venv ready: {venv_py}")

    dist_index = ROOT / "frontend" / "dist" / "index.html"
    if not dist_index.exists():
        print("[INFO] Building frontend production bundle...")
        npm = shutil.which("npm.cmd") or shutil.which("npm")
        if npm:
            if not (ROOT / "frontend" / "node_modules").exists():
                subprocess.run([npm, "--prefix", str(ROOT / "frontend"), "install"], check=True)
            subprocess.run([npm, "--prefix", str(ROOT / "frontend"), "run", "build"], check=True)
        else:
            print("[WARN] npm not found; skipped frontend build.")
    else:
        print(f"[OK] Frontend production bundle ready: {dist_index}")
    return venv_py


def deploy_and_compile_mt5_ea(token: str) -> list[Path]:
    banner("3/5. Deploying & Compiling MQL5 EA into MetaTrader 5")
    import mt5_ea

    return [r.terminal for r in mt5_ea.ensure_ea(token)]


def run_automated_tests(venv_py: Path) -> bool:
    banner("4/5. Running Full Automated Verification Suite (16 Tests)")
    pytest_exe = ROOT / ".venv" / "Scripts" / "pytest.exe"
    cmd = [str(pytest_exe), "tests/", "-q"] if pytest_exe.exists() else [str(venv_py), "-m", "pytest", "tests/", "-q"]
    res = subprocess.run(cmd, cwd=ROOT)
    return res.returncode == 0


def print_ready_summary(token: str, deployed: list[Path], tests_ok: bool) -> None:
    banner("5/5. READY FOR TESTING!")
    print(f"  Tests Status     : {'ALL PASSED' if tests_ok else 'FAILED (check output above)'}")
    print(f"  MT5 EA Deployed  : {len(deployed)} terminal(s)")
    print(f"  Bridge Token     : ...{token[-4:]} (lengkapnya ada di file .env)")
    print("")
    print("  [MODE A — Uji Cepat Simulator (Tanpa MT5)]")
    print("    -> Double-click: start_simulator.bat")
    print("")
    print("  [MODE B — Testing di MetaTrader 5 Akun DEMO]")
    print("    1. Buka MT5 -> Tools -> Options -> Expert Advisors:")
    print("       [x] Allow Algorithmic Trading")
    print("       [x] Allow WebRequest for listed URL -> tambah: http://127.0.0.1:8765")
    print("    2. Double-click: start_mt5_live.bat  (otomatis buka Dashboard)")
    print("    3. Di MT5 Navigator (klik kanan Expert Advisors -> Refresh),")
    print("       drag 'MaxTradingBridge' ke chart XAUUSD M1 (token SUDAH terisi otomatis!).")
    print("       *Jika pair broker punya akhiran (mis. XAUUSDm), isi SymbolSuffix = m.")
    print("")


def main() -> int:
    token = ensure_env_and_token()
    ensure_control_token()
    venv_py = check_python_and_frontend()
    deployed = deploy_and_compile_mt5_ea(token)
    tests_ok = run_automated_tests(venv_py)
    print_ready_summary(token, deployed, tests_ok)
    return 0 if tests_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
