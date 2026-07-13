"""Offline tests for the v1 hybrid router (no model, no network)."""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from agentlife.harness.bm25 import BM25Index
from clsledger.ledger import Card, Ledger
from clsledger.system import CLSLedgerSystem


def _mk_system():
    sys_ = CLSLedgerSystem.__new__(CLSLedgerSystem)  # skip heavy __init__
    lg = Ledger()
    cards = [
        Card("Sofia Almeida", "employer", "Vantalux", 10, "e1"),
        Card("user", "parking_spot", "B47", 25, "e2"),
        Card("Project Falcon", "lead", "Ravi Kumar", 5, "e3"),
    ]
    for c in cards:
        lg.add(c)
    sys_.ledger = lg
    sys_.all_index = BM25Index()
    sys_.cards_by_id = {}
    for c in lg.current_cards():
        sys_.all_index.add(c.card_id, c.text())
        sys_.cards_by_id[c.card_id] = c
    # parking (volatile) deliberately NOT consolidated
    sys_.consolidated_ids = {c.card_id for c in lg.current_cards()
                             if c.attribute != "parking_spot"}
    return sys_


def test_temporal_goes_episodic():
    s = _mk_system()
    assert s._route("As of day 30, what was Sofia Almeida's employer?") \
        == "episodic"


def test_consolidated_entity_goes_weights():
    s = _mk_system()
    assert s._route("What is Sofia Almeida's current employer?") == "weights"


def test_unconsolidated_volatile_goes_episodic():
    s = _mk_system()
    assert s._route("What is the user's current parking spot?") == "episodic"


def test_entity_not_in_question_goes_episodic():
    s = _mk_system()
    # best-matching card is about Sofia, but she is not the stated subject
    assert s._route("Which employer pays the best salaries?") == "episodic"


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
