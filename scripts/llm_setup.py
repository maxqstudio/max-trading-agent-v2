"""Interactive LLM provider setup: pick providers, paste keys, choose models, test.

Run:  python scripts/llm_setup.py      (or double-click llm_setup.bat on Windows)

Keys are written ONLY to your local .env (git-ignored). Nothing is printed in full.
"""

from __future__ import annotations

import asyncio
import getpass
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.llm.base import LLMRequest  # noqa: E402
from backend.llm.presets import DEFAULT_CHAIN, PRESETS  # noqa: E402
from backend.llm.router import LLMRouter  # noqa: E402
from backend.llm.settings import load_chain  # noqa: E402

ENV_FILE = ROOT / ".env"
ENV_EXAMPLE = ROOT / ".env.example"


def read_env() -> str:
    if ENV_FILE.exists():
        return ENV_FILE.read_text(encoding="utf-8")
    return ENV_EXAMPLE.read_text(encoding="utf-8") if ENV_EXAMPLE.exists() else ""


def get_value(content: str, key: str) -> str:
    m = re.search(rf"^{re.escape(key)}=(.*)$", content, flags=re.MULTILINE)
    return m.group(1).strip() if m else ""


def set_value(content: str, key: str, value: str) -> str:
    line = f"{key}={value}"
    if re.search(rf"^{re.escape(key)}=.*$", content, flags=re.MULTILINE):
        return re.sub(rf"^{re.escape(key)}=.*$", lambda _: line, content, flags=re.MULTILINE)
    if content and not content.endswith("\n"):
        content += "\n"
    return content + line + "\n"


def ask(prompt: str, default: str = "") -> str:
    shown = f" [{default}]" if default else ""
    ans = input(f"{prompt}{shown}: ").strip()
    return ans or default


def choose_chain(names: list[str]) -> list[str]:
    print("\nProvider yang tersedia:")
    for i, n in enumerate(names, 1):
        p = PRESETS[n]
        print(f"  {i:>2}. {p.label:<45} ({n})")
    default = ",".join(str(names.index(n) + 1) for n in DEFAULT_CHAIN)
    raw = ask("\nPilih urutan provider (nomor dipisah koma, yang pertama dipakai duluan)", default)
    picked: list[str] = []
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit() and 1 <= int(part) <= len(names) and names[int(part) - 1] not in picked:
            picked.append(names[int(part) - 1])
        elif part in PRESETS and part not in picked:
            picked.append(part)
    return picked


async def test_connections(chain: list[str]) -> None:
    resolved = load_chain(ENV_FILE, overrides={"LLM_PROVIDER_CHAIN": ",".join(chain)})
    schema = {"type": "object", "additionalProperties": False, "required": ["ok"], "properties": {"ok": {"type": "boolean"}}}
    for rp in resolved:
        if not rp.configured:
            print(f"  [SKIP] {rp.name}: belum lengkap (key/model/base URL)")
            continue
        req = LLMRequest(task="ping", system="Connectivity check.", user='Reply with {"ok": true}.', schema=schema,
                         timeout_s=30.0, max_tokens=1000, effort="low")
        try:
            res, _ = await LLMRouter(resolved=[rp]).complete(req)
            print(f"  [OK]   {rp.name}: {res.model} menjawab dalam {res.latency_ms:.0f} ms")
        except Exception as exc:
            print(f"  [FAIL] {rp.name}: {str(exc)[:200]}")


def configured_providers() -> list[str]:
    return [r.name for r in load_chain(ENV_FILE) if r.configured]


def main() -> int:
    if "--check" in sys.argv:
        names = configured_providers()
        print(", ".join(names) if names else "none")
        return 0 if names else 1
    print("=" * 64)
    print("  Max Trading Agent v2 — Setup LLM (Claude dulu)")
    print("=" * 64)
    content = read_env()
    names = list(PRESETS)
    chain = choose_chain(names)
    if not chain:
        print("Tidak ada provider dipilih. LLM dimatikan (mode deterministik).")
        content = set_value(content, "LLM_PROVIDER_CHAIN", "off")
        ENV_FILE.write_text(content, encoding="utf-8")
        return 0

    for n in chain:
        p = PRESETS[n]
        up = n.upper()
        print(f"\n--- {p.label} ---")
        if p.note:
            print(f"  Catatan: {p.note}")
        if n == "custom" or p.kind == "openai_compat" and not p.needs_key:
            base = ask("  Base URL", get_value(content, f"{up}_BASE_URL") or p.base_url)
            if base and base != p.base_url:
                content = set_value(content, f"{up}_BASE_URL", base)
        if p.needs_key or n == "custom":
            existing = get_value(content, f"{up}_API_KEY")
            if p.key_url:
                print(f"  Ambil API key di: {p.key_url}")
            hint = f"(Enter = pakai yang lama ...{existing[-4:]})" if existing else "(Enter = lewati)"
            key = getpass.getpass(f"  Tempel {up}_API_KEY {hint}: ").strip()
            if key:
                content = set_value(content, f"{up}_API_KEY", key)
        model = ask("  Model utama (refleksi)", get_value(content, f"{up}_MODEL") or p.default_model)
        fast = ask("  Model cepat (supervisor scalping)", get_value(content, f"{up}_FAST_MODEL") or p.fast_model or model)
        content = set_value(content, f"{up}_MODEL", model)
        content = set_value(content, f"{up}_FAST_MODEL", fast)

    content = set_value(content, "LLM_PROVIDER_CHAIN", ",".join(chain))
    ENV_FILE.write_text(content, encoding="utf-8")
    print(f"\n[OK] Tersimpan di {ENV_FILE.name} (file ini tidak pernah ikut ke git).")

    if ask("\nTes koneksi sekarang? (y/n)", "y").lower().startswith("y"):
        asyncio.run(test_connections(chain))
    print("\nSelesai. Restart server agar setelan baru dipakai.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
