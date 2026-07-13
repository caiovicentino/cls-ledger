"""Offline tests for the Ledger (supersedence, churn gating, provenance)."""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from clsledger.ledger import Card, Ledger


def _card(entity, attr, value, day, ep):
    return Card(entity, attr, value, day, ep)


def test_supersede_keeps_latest():
    lg = Ledger()
    lg.add(_card("Sofia Almeida", "employer", "Ferrowind", 1, "e1"))
    lg.add(_card("Sofia Almeida", "employer", "Vantalux", 30, "e2"))
    cur = lg.current_cards()
    assert len(cur) == 1 and cur[0].value == "Vantalux"
    assert lg.churn("sofia_almeida.employer") == 1


def test_out_of_order_statement_does_not_regress():
    lg = Ledger()
    lg.add(_card("user", "gym_day", "Friday", 20, "e2"))
    lg.add(_card("user", "gym_day", "Monday", 5, "e1"))  # older re-statement
    cur = lg.current_cards()
    assert len(cur) == 1 and cur[0].value == "Friday"


def test_usage_follows_fact_across_updates():
    lg = Ledger()
    lg.add(_card("user", "parking_spot", "B47", 1, "e1"))
    lg.cards[0].usage = 5
    lg.add(_card("user", "parking_spot", "C13", 8, "e2"))
    assert lg.current_cards()[0].usage == 5


def test_churn_gating_excludes_volatile():
    lg = Ledger()
    for i, (v, d) in enumerate([("A1", 1), ("B2", 8), ("C3", 15),
                                ("D4", 22)]):
        lg.add(_card("user", "parking_spot", v, d, f"e{i}"))
    lg.add(_card("Sofia Almeida", "birthday", "March 8", 3, "e9"))
    sel = lg.select_for_consolidation(today=25, policy="stable")
    keys = {c.key for c in sel}
    assert "sofia_almeida.birthday" in keys
    assert "user.parking_spot" not in keys  # 3 changes, last 3 days ago
    # but a fact that churned long ago and settled is consolidatable
    sel_late = lg.select_for_consolidation(today=60, policy="stable")
    assert "user.parking_spot" in {c.key for c in sel_late}


def test_policy_hot_selects_only_used():
    lg = Ledger()
    lg.add(_card("a", "x", "1", 1, "e1"))
    lg.add(_card("b", "y", "2", 1, "e2"))
    lg.cards[0].usage = 3
    sel = lg.select_for_consolidation(today=10, policy="hot")
    assert [c.entity for c in sel] == ["a"]


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
