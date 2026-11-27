"""Install the MaxTradingBridge EA into every MetaTrader 5 terminal on this PC, and compile it.

Used by setup_all.py and start_all.py, so the EA in MT5 is always the version in this repo.

- Finds terminals under %APPDATA%\\MetaQuotes\\Terminal\\<id>\\MQL5\\Experts.
- Finds MetaEditor through each terminal's origin.txt (works for broker builds such as
  "MONEX MetaTrader 5"), then falls back to Program Files.
- The installed copy has BridgeAuthToken pre-filled from .env. The repo file stays secret-free.
- Does nothing when the installed copy is already current and compiled, so it is cheap to run
  on every start. An older copy is kept as MaxTradingBridge.mq5.bak before it is replaced.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EA_NAME = "MaxTradingBridge"
MQL5_SRC = ROOT / "mql5" / f"{EA_NAME}.mq5"
BRIDGE_URL = "http://127.0.0.1:8765/api/bridge/sync"

_TOKEN_LINE = re.compile(r'input string\s+BridgeAuthToken\s*=\s*"[^"]*";')


@dataclass
class InstallResult:
    terminal: Path
    action: str  # "installed", "updated", "up_to_date"
    compiled: bool | None = None  # None = MetaEditor not found, compile skipped
    compile_summary: str = ""
    notes: list[str] = field(default_factory=list)


def render_ea(source: str, token: str) -> str:
    """Return the EA source with the BridgeAuthToken input default set to ``token``."""
    if not _TOKEN_LINE.search(source):
        raise ValueError("BridgeAuthToken input line not found in the EA source")
    safe = token.replace("\\", "").replace('"', "")
    return _TOKEN_LINE.sub(lambda _m: f'input string   BridgeAuthToken        = "{safe}";', source, count=1)


def terminal_dirs(appdata: Path | None = None) -> list[Path]:
    """All MT5 terminal data folders that have an MQL5\\Experts directory."""
    base = Path(appdata if appdata is not None else os.environ.get("APPDATA", "")) / "MetaQuotes" / "Terminal"
    if not base.is_dir():
        return []
    return sorted(d for d in base.iterdir() if d.is_dir() and (d / "MQL5" / "Experts").is_dir())


def _read_text_any(path: Path) -> str:
    raw = path.read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16", errors="ignore")
    if len(raw) > 1 and raw[1:2] == b"\x00":
        return raw.decode("utf-16-le", errors="ignore")
    return raw.decode("utf-8-sig", errors="ignore")


def install_dir_of(term_dir: Path) -> Path | None:
    """The terminal's install folder, from origin.txt in its data folder."""
    origin = term_dir / "origin.txt"
    if not origin.is_file():
        return None
    text = _read_text_any(origin).strip().strip("\x00").strip()
    return Path(text) if text else None


def find_metaeditor(term_dir: Path | None = None) -> Path | None:
    candidates: list[Path] = []
    if term_dir is not None:
        inst = install_dir_of(term_dir)
        if inst is not None:
            candidates += [inst / "MetaEditor64.exe", inst / "metaeditor64.exe", inst / "MetaEditor.exe"]
    for pf in (os.environ.get("ProgramFiles", r"C:\Program Files"), os.environ.get("ProgramFiles(x86)", "")):
        if pf and Path(pf).is_dir():
            for d in sorted(Path(pf).glob("*MetaTrader*")):
                candidates += [d / "MetaEditor64.exe", d / "metaeditor64.exe"]
    for c in candidates:
        if c.is_file():
            return c
    return None


def write_preset(term_dir: Path, token: str) -> Path:
    presets = term_dir / "MQL5" / "Presets"
    presets.mkdir(parents=True, exist_ok=True)
    preset = presets / f"{EA_NAME}_Demo.set"
    preset.write_text(
        f"BridgeUrl={BRIDGE_URL}\n"
        f"BridgeAuthToken={token}\n"
        "SyncIntervalMs=350\n"
        "LogicalSymbols=XAUUSD,EURUSD,GBPUSD,USDJPY\n"
        "SymbolSuffix=\n"
        "HardMaxDailyDDPct=3.50\n"
        "HardMaxLot=0.50\n"
        "HardMaxPositions=4\n",
        encoding="utf-8",
    )
    return preset


def compile_ea(metaeditor: Path, target: Path, runner=subprocess.run) -> tuple[bool, str]:
    """Compile with MetaEditor. Returns (ok, summary line from the compile log)."""
    log = target.with_name(f"{EA_NAME}_compile.log")
    if log.exists():
        log.unlink()
    ex5 = target.with_suffix(".ex5")
    before = ex5.stat().st_mtime if ex5.exists() else 0.0
    runner([str(metaeditor), f"/compile:{target}", f"/log:{log}"], check=False, timeout=120)
    summary = ""
    errors = None
    if log.exists():
        for line in _read_text_any(log).splitlines():
            m = re.search(r"(\d+)\s+errors?", line, flags=re.IGNORECASE)
            if "result" in line.lower() or m:
                summary = line.strip()
                if m:
                    errors = int(m.group(1))
    fresh = ex5.exists() and ex5.stat().st_mtime > before
    ok = fresh and (errors in (None, 0))
    return ok, summary


def install_into(term_dir: Path, code: str, token: str, metaeditor: Path | None, runner=subprocess.run) -> InstallResult:
    experts = term_dir / "MQL5" / "Experts"
    target = experts / f"{EA_NAME}.mq5"
    ex5 = target.with_suffix(".ex5")
    existing = target.read_text(encoding="utf-8", errors="ignore") if target.exists() else None

    write_preset(term_dir, token)
    if existing == code and ex5.exists() and ex5.stat().st_mtime >= target.stat().st_mtime:
        return InstallResult(term_dir, "up_to_date", compiled=True)

    action = "installed" if existing is None else ("updated" if existing != code else "up_to_date")
    if existing is not None and existing != code:
        shutil.copy2(target, target.with_name(target.name + ".bak"))
        if ex5.exists():
            shutil.copy2(ex5, ex5.with_name(ex5.name + ".bak"))
    if existing != code:
        target.write_text(code, encoding="utf-8")

    res = InstallResult(term_dir, action)
    if metaeditor is None:
        res.notes.append("MetaEditor tidak ditemukan: buka MaxTradingBridge.mq5 di MetaEditor lalu tekan F7.")
        return res
    try:
        res.compiled, res.compile_summary = compile_ea(metaeditor, target, runner=runner)
    except Exception as exc:  # noqa: BLE001 - never block startup on a compile problem
        res.compiled, res.compile_summary = False, f"compile gagal dijalankan: {type(exc).__name__}"
    if not res.compiled:
        res.notes.append(f"Compile belum berhasil. Buka {target} di MetaEditor lalu tekan F7.")
    return res


def ensure_ea(token: str, appdata: Path | None = None, source: Path = MQL5_SRC, runner=subprocess.run, verbose: bool = True) -> list[InstallResult]:
    """Install/refresh the EA in every MT5 terminal. Never raises; prints a short status."""
    def say(msg: str) -> None:
        if verbose:
            print(msg)

    if len(token) < 24:
        say("  [WARN] BRIDGE_AUTH_TOKEN belum ada di .env, EA tidak dipasang.")
        return []
    dirs = terminal_dirs(appdata)
    if not dirs:
        say("  [WARN] Folder MetaTrader 5 tidak ditemukan. Buka MT5 sekali, lalu jalankan lagi.")
        return []
    code = render_ea(source.read_text(encoding="utf-8"), token)
    results = []
    for d in dirs:
        try:
            r = install_into(d, code, token, find_metaeditor(d), runner=runner)
        except OSError as exc:
            say(f"  [WARN] Gagal memasang EA di {d.name[:8]}…: {exc.strerror or exc}")
            continue
        results.append(r)
        label = {"installed": "dipasang", "updated": "diperbarui", "up_to_date": "sudah versi terbaru"}[r.action]
        comp = "" if r.action == "up_to_date" else (" + compiled" if r.compiled else "")
        say(f"  [OK] EA {label}{comp} di terminal {d.name[:8]}…" + (f"  ({r.compile_summary})" if r.compile_summary else ""))
        for n in r.notes:
            say(f"       {n}")
    return results


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import setup_all

    ensure_ea(setup_all.ensure_env_and_token())
