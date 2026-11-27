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


def find_metaeditor() -> Path | None:
    candidates = [
        Path(r"C:\Program Files\MetaTrader 5\metaeditor64.exe"),
        Path(r"C:\Program Files\MetaTrader 5\metaeditor.exe"),
    ]
    pf = Path(r"C:\Program Files")
    if pf.exists():
        for d in pf.glob("*MetaTrader*"):
            candidates.append(d / "metaeditor64.exe")
    for c in candidates:
        if c.exists():
            return c
    return None


def deploy_and_compile_mt5_ea(token: str) -> list[Path]:
    banner("3/5. Deploying & Compiling MQL5 EA into MetaTrader 5")
    appdata = Path(os.environ.get("APPDATA", ""))
    term_root = appdata / "MetaQuotes" / "Terminal"
    deployed: list[Path] = []

    if not term_root.exists():
        print("[WARN] MetaQuotes Terminal folder not found in %APPDATA%. Copy mql5/MaxTradingBridge.mq5 manually.")
        return deployed

    src_code = MQL5_SRC.read_text(encoding="utf-8")
    # Inject user's active token as default input in the deployed copy (repo file stays secret-free!)
    patched_code = re.sub(
        r'input string\s+BridgeAuthToken\s*=\s*"[^"]*";',
        f'input string   BridgeAuthToken        = "{token}";',
        src_code,
    )

    metaeditor = find_metaeditor()
    for term_dir in term_root.iterdir():
        if not term_dir.is_dir():
            continue
        experts_dir = term_dir / "MQL5" / "Experts"
        if not experts_dir.exists():
            continue

        target_mq5 = experts_dir / "MaxTradingBridge.mq5"
        target_mq5.write_text(patched_code, encoding="utf-8")
        print(f"[OK] Deployed EA source -> {target_mq5}")
        deployed.append(target_mq5)

        # Also drop a ready-to-load MT5 Preset (.set)
        presets_dir = term_dir / "MQL5" / "Presets"
        presets_dir.mkdir(parents=True, exist_ok=True)
        preset_file = presets_dir / "MaxTradingBridge_Demo.set"
        preset_file.write_text(
            f"BridgeUrl=http://127.0.0.1:8765/api/bridge/sync\n"
            f"BridgeAuthToken={token}\n"
            f"SyncIntervalMs=350\n"
            f"LogicalSymbols=XAUUSD,EURUSD,GBPUSD,USDJPY\n"
            f"SymbolSuffix=\n"
            f"HardMaxDailyDDPct=3.50\n"
            f"HardMaxLot=0.50\n"
            f"HardMaxPositions=4\n",
            encoding="utf-8",
        )

        if metaeditor is not None:
            log_file = term_dir / "MQL5" / "Experts" / "MaxTradingBridge_compile.log"
            if log_file.exists():
                log_file.unlink()
            print(f"[INFO] Compiling with {metaeditor.name}...")
            subprocess.run(
                [str(metaeditor), f"/compile:{target_mq5}", f"/log:{log_file}"],
                check=False,
            )
            ex5_file = target_mq5.with_suffix(".ex5")
            if log_file.exists():
                # MetaEditor writes UTF-16LE logs on Windows
                raw = log_file.read_bytes()
                try:
                    log_text = raw.decode("utf-16")
                except UnicodeDecodeError:
                    log_text = raw.decode("utf-8", errors="ignore")
                for line in log_text.splitlines():
                    if any(k in line.lower() for k in ("result", "error", "warning")):
                        print("   ", line.strip())
            if ex5_file.exists():
                print(f"[OK] Compiled binary ready -> {ex5_file}")
            else:
                print(f"[WARN] Could not verify .ex5 output. Open {target_mq5} in MetaEditor and press F7.")

    if not deployed:
        print("[WARN] No MT5 MQL5\\Experts directory found. Launch MetaTrader 5 once, then re-run this script.")
    return deployed


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
