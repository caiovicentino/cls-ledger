"""Tests for obsidian_ledger: store semantics, install idempotency,
MCP server handshake (subprocess, real protocol)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

from clsledger.ledger import Card, Ledger  # noqa: E402
from obsidian_ledger import store  # noqa: E402


def _lg():
    lg = Ledger()
    lg.add(Card("Paper X", "doi", "10.5281/zenodo.111", 739000, "note_a"))
    lg.add(Card("media rule", "value", "always attach media v1", 739001,
                "note_b"))
    lg.add(Card("media rule", "value", "media from source tweet v2",
                739010, "note_c"))
    lg.add(Card("fish folder", "count", "189", 739000, "r1"))
    lg.add(Card("fish folder", "count", "345", 739060, "r2"))
    return lg


def test_query_returns_provenance_or_sentinel():
    lg = _lg()
    hit = store.query(lg, "what is the DOI of Paper X?")[0]
    assert hit["value"] == "10.5281/zenodo.111" and hit["source"] == "note_a"
    miss = store.query(lg, "zzz qqq totally unrelated")[0]
    assert "NOT IN LEDGER" in miss["answer"]


def test_verify_supported_stale_unknown():
    lg = _lg()
    assert store.verify(lg, "Paper X doi is 10.5281/zenodo.111")["verdict"] == \
        "SUPPORTED"
    stale = store.verify(lg, "media rule says always attach media v1")
    assert stale["verdict"] == "CONTRADICTED_STALE"
    assert stale["current_value"] == "media from source tweet v2"
    assert store.verify(lg, "flying pigs doi 999")["verdict"] == \
        "NOT_IN_LEDGER"
    # bare-token false positive guard (third-party vault finding)
    assert store.verify(lg, "note_a rated something 10")["verdict"] != \
        "SUPPORTED"
    wrong = store.verify(lg, "Paper X doi is 10.5281/zenodo.999")
    assert wrong["verdict"] == "CONTRADICTED_CURRENT"
    assert wrong["actual_value"] == "10.5281/zenodo.111"


def test_induce_counts_and_trend():
    lg = _lg()
    out = store.induce(lg, "fish folder", "count")
    assert out["n_changes"] == 1 and out["trend"] == "increased"
    assert store.induce(lg, "nobody", "nothing")["verdict"] == \
        "NOT_IN_LEDGER"


def test_save_load_roundtrip_and_snapshot():
    with tempfile.TemporaryDirectory() as tmp:
        store.save(tmp, _lg())
        lg2 = store.load(tmp)
        assert len(lg2.cards) == 5 and len(lg2.current_cards()) == 3
        snap = open(os.path.join(tmp, "_ledger",
                                 store.SNAPSHOT_FILE)).read()
        assert "DO NOT EXIST" in snap and "Superseded" in snap


def test_install_idempotent_and_repo_path():
    from obsidian_ledger.instructions import install, START
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "AGENTS.md"), "w") as f:
            f.write("# My vault rules\nkeep this line\n")
        install(tmp)
        install(tmp)  # twice: must not duplicate
        txt = open(os.path.join(tmp, "AGENTS.md")).read()
        assert txt.count(START) == 1
        assert "keep this line" in txt
        assert REPO in txt  # agents must know where the CLI lives
        assert os.path.exists(os.path.join(tmp, ".cursorrules"))
        assert os.path.exists(os.path.join(tmp, "GEMINI.md"))


def test_mcp_server_protocol():
    with tempfile.TemporaryDirectory() as tmp:
        store.save(tmp, _lg())
        reqs = "\n".join([
            json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                        "params": {"protocolVersion": "2025-03-26"}}),
            json.dumps({"jsonrpc": "2.0", "method":
                        "notifications/initialized"}),
            json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
            json.dumps({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                        "params": {"name": "ledger_verify", "arguments":
                                   {"claim": "media rule: always attach media v1"}}}),
        ]) + "\n"
        r = subprocess.run(
            [sys.executable, "-m", "obsidian_ledger.mcp_server",
             "--vault", tmp],
            input=reqs, capture_output=True, text=True,
            env={**os.environ, "PYTHONPATH": REPO}, timeout=30)
        lines = [json.loads(l) for l in r.stdout.strip().split("\n")]
        by_id = {l["id"]: l["result"] for l in lines}
        assert by_id[1]["serverInfo"]["name"] == "vault-ledger"
        assert len(by_id[2]["tools"]) == 4
        verdict = json.loads(by_id[3]["content"][0]["text"])["verdict"]
        assert verdict == "CONTRADICTED_STALE"


if __name__ == "__main__":
    mod = sys.modules[__name__]
    failures = 0
    for name in sorted(dir(mod)):
        if name.startswith("test_"):
            try:
                getattr(mod, name)()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    sys.exit(1 if failures else 0)
