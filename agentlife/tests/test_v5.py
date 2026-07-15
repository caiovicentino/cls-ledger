"""v5 self-validation: induction ground truth and disposition machinery."""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from agentlife import banks
from agentlife.generator import LifeGenerator
from agentlife.harness.scorer import score, check_disposition
from agentlife.schema import make_config


def _gen():
    return LifeGenerator(make_config("M", 9)).run()


def test_induction_counts_match_world():
    gen = _gen()
    for q in gen.final_queries:
        if q.qtype != "induction_count":
            continue
        n = len(gen.world.history(q.fact_keys[0])) - 1
        assert q.answer_display == str(n), (q.query_id, q.answer_display, n)


def test_induction_habit_matches_mode():
    gen = _gen()
    for q in gen.final_queries:
        if q.qtype != "induction_habit":
            continue
        act = q.fact_keys[0].split("activity_")[-1].replace("_", " ")
        days = gen.activity_days[act]
        counts = [0] * 7
        for d in days:
            counts[(d - 1) % 7] += 1
        mode = max(range(7), key=lambda i: counts[i])
        assert q.answer_display == banks.WEEKDAYS[mode]


def test_disposition_tagging_respects_day():
    gen = _gen()
    dd = gen.disposition_day
    for q in gen.final_queries:
        attr = q.fact_keys[-1].split(".")[-1] if q.fact_keys else ""
        if "date_first" in q.dispositions:
            assert attr in ("birthday", "deadline")
            assert q.day_asked > dd["date_first"]
        if attr in ("hometown", "lives_in") and q.qtype != "absent":
            assert ("city_upper" in q.dispositions) == (
                q.day_asked > dd["city_upper"])


def test_adherent_oracle_scores_100_adherence():
    gen = _gen()
    answers = [q.private() for q in gen.final_queries]

    def formatted(q):
        v = q["answer_display"]
        if "date_first" in q["dispositions"] and " " in v:
            month, day = v.rsplit(" ", 1)
            return f"{day} {month}"
        if "city_upper" in q["dispositions"]:
            return v.upper()
        return v

    rep = score({q["query_id"]: formatted(q) for q in answers}, answers)
    assert rep["accuracy_final"] == 1.0  # formatting never breaks content
    for rule, a in rep["disposition_adherence"].items():
        assert a["rate"] == 1.0, (rule, a)
    # plain oracle: 0% adherence on date_first (month-first canonical)
    rep2 = score({q["query_id"]: q["answer_display"] for q in answers},
                 answers)
    d = rep2["disposition_adherence"].get("date_first")
    assert d and d["rate"] == 0.0


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
