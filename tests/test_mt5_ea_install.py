"""EA auto-install: token injection, terminal discovery, MetaEditor via origin.txt, idempotence."""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import mt5_ea  # noqa: E402

TOKEN = "t" * 45


def make_terminal(appdata: Path, tid: str = "D0E8209F", install: Path | None = None) -> Path:
    term = appdata / "MetaQuotes" / "Terminal" / tid
    (term / "MQL5" / "Experts").mkdir(parents=True)
    if install is not None:
        (term / "origin.txt").write_bytes(("﻿" + str(install)).encode("utf-16-le"))
    return term


def fake_compiler(calls):
    def run(cmd, check=False, timeout=None):
        calls.append(cmd)
        target = Path(cmd[1].split(":", 1)[1])
        log = Path(cmd[2].split(":", 1)[1])
        target.with_suffix(".ex5").write_bytes(b"EX5")
        log.write_bytes("Result: 0 errors, 0 warnings, 512 msec elapsed".encode("utf-16"))
    return run


def test_repo_ea_has_no_token_and_render_injects_it():
    src = mt5_ea.MQL5_SRC.read_text(encoding="utf-8")
    assert 'BridgeAuthToken        = "";' in src  # repo stays secret-free
    out = mt5_ea.render_ea(src, TOKEN)
    assert f'BridgeAuthToken        = "{TOKEN}";' in out
    assert out.count(TOKEN) == 1


def test_render_strips_quotes():
    out = mt5_ea.render_ea('input string BridgeAuthToken = "";', 'ab"c\\d')
    assert '"abcd"' in out


def test_install_finds_metaeditor_from_origin_and_compiles(tmp_path):
    install = tmp_path / "Program Files" / "MONEX MetaTrader 5"
    install.mkdir(parents=True)
    (install / "MetaEditor64.exe").write_bytes(b"")
    term = make_terminal(tmp_path / "AppData", install=install)
    calls = []
    res = mt5_ea.ensure_ea(TOKEN, appdata=tmp_path / "AppData", runner=fake_compiler(calls), verbose=False)
    assert len(res) == 1 and res[0].action == "installed" and res[0].compiled
    assert calls and calls[0][0] == str(install / "MetaEditor64.exe")
    code = (term / "MQL5" / "Experts" / "MaxTradingBridge.mq5").read_text(encoding="utf-8")
    assert TOKEN in code
    assert TOKEN in (term / "MQL5" / "Presets" / "MaxTradingBridge_Demo.set").read_text(encoding="utf-8")


def test_second_run_is_noop_and_old_version_is_backed_up(tmp_path):
    install = tmp_path / "mt5"
    install.mkdir()
    (install / "MetaEditor64.exe").write_bytes(b"")
    term = make_terminal(tmp_path / "AppData", install=install)
    experts = term / "MQL5" / "Experts"
    (experts / "MaxTradingBridge.mq5").write_text("// old version", encoding="utf-8")
    (experts / "MaxTradingBridge.ex5").write_bytes(b"OLD")
    calls = []
    run = fake_compiler(calls)
    first = mt5_ea.ensure_ea(TOKEN, appdata=tmp_path / "AppData", runner=run, verbose=False)
    assert first[0].action == "updated"
    assert (experts / "MaxTradingBridge.mq5.bak").read_text(encoding="utf-8") == "// old version"
    assert (experts / "MaxTradingBridge.ex5.bak").read_bytes() == b"OLD"
    # make sure the ex5 is not older than the source (filesystem timestamp granularity)
    src = experts / "MaxTradingBridge.mq5"
    os.utime(experts / "MaxTradingBridge.ex5", (src.stat().st_mtime + 1, src.stat().st_mtime + 1))
    second = mt5_ea.ensure_ea(TOKEN, appdata=tmp_path / "AppData", runner=run, verbose=False)
    assert second[0].action == "up_to_date"
    assert len(calls) == 1


def test_token_change_triggers_reinstall(tmp_path):
    term = make_terminal(tmp_path / "AppData")
    mt5_ea.ensure_ea(TOKEN, appdata=tmp_path / "AppData", runner=fake_compiler([]), verbose=False)
    res = mt5_ea.ensure_ea("u" * 30, appdata=tmp_path / "AppData", runner=fake_compiler([]), verbose=False)
    assert res[0].action == "updated"
    assert "u" * 30 in (term / "MQL5" / "Experts" / "MaxTradingBridge.mq5").read_text(encoding="utf-8")


def test_without_metaeditor_still_copies_and_explains(tmp_path, monkeypatch):
    monkeypatch.setattr(mt5_ea, "find_metaeditor", lambda term_dir=None: None)
    term = make_terminal(tmp_path / "AppData")
    res = mt5_ea.ensure_ea(TOKEN, appdata=tmp_path / "AppData", verbose=False)
    assert res[0].compiled is None and "Compile" in res[0].notes[0]
    assert (term / "MQL5" / "Experts" / "MaxTradingBridge.mq5").exists()


@pytest.mark.parametrize("token", ["", "short"])
def test_weak_token_is_refused(tmp_path, token):
    make_terminal(tmp_path / "AppData")
    assert mt5_ea.ensure_ea(token, appdata=tmp_path / "AppData", verbose=False) == []


def test_no_mt5_installed(tmp_path):
    assert mt5_ea.ensure_ea(TOKEN, appdata=tmp_path, verbose=False) == []


def test_compile_errors_are_reported(tmp_path):
    term = make_terminal(tmp_path / "AppData")
    target = term / "MQL5" / "Experts" / "MaxTradingBridge.mq5"
    target.write_text("x", encoding="utf-8")

    def bad(cmd, check=False, timeout=None):
        Path(cmd[2].split(":", 1)[1]).write_bytes("Result: 2 errors, 0 warnings".encode("utf-16"))

    ok, summary = mt5_ea.compile_ea(Path("MetaEditor64.exe"), target, runner=bad)
    assert not ok and "2 errors" in summary
