"""Benchmark self-validation.

The whole point of AgentLife is that its answer key cannot be wrong the way
LoCoMo's was. These tests are the enforcement:
- a world-state oracle must score exactly 100%;
- a deliberately stale oracle must fail every freshness query it can fail;
- every canonical value must be extractable from episode text;
- generation must be deterministic per seed.
"""
from __future__ import annotations

import dataclasses
import json

from agentlife import banks
from agentlife.generator import LifeGenerator, boundary_found
from agentlife.harness.scorer import score
from agentlife.harness.systems import NullSystem
from agentlife.schema import (make_config, QT_ABSENT, QT_CURRENT_UPDATED,
                              QT_MULTIHOP_2, QT_MULTIHOP_3, QT_ONLINE)


def _gen(preset="S", seed=7):
    return LifeGenerator(make_config(preset, seed)).run()


def _fingerprint(gen):
    payload = {
        "episodes": [dataclasses.asdict(e) for e in gen.episodes],
        "online": [q.private() for q in gen.online_queries],
        "final": [q.private() for q in gen.final_queries],
        "world": gen.world.dump(),
    }
    return json.dumps(payload, sort_keys=True)


def _score_with(gen, answer_fn):
    answers = [q.private() for q in gen.online_queries + gen.final_queries]
    responses = {q["query_id"]: answer_fn(q) for q in answers}
    return score(responses, answers), answers


def test_determinism():
    assert _fingerprint(_gen()) == _fingerprint(_gen())


def test_seeds_differ():
    assert _fingerprint(_gen(seed=7)) != _fingerprint(_gen(seed=8))


def test_cross_process_determinism():
    """Generation must not depend on the interpreter's string-hash seed."""
    import hashlib
    import os
    import subprocess
    import sys
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    code = ("from agentlife.generator import LifeGenerator\n"
            "from agentlife.schema import make_config\n"
            "from agentlife.tests.test_agentlife import _fingerprint\n"
            "import hashlib, sys\n"
            "gen = LifeGenerator(make_config('S', 7)).run()\n"
            "sys.stdout.write(hashlib.sha256("
            "_fingerprint(gen).encode()).hexdigest())\n")
    digests = set()
    for hash_seed in ("0", "1", "2"):
        env = dict(os.environ, PYTHONHASHSEED=hash_seed,
                   PYTHONPATH=root)
        out = subprocess.run([sys.executable, "-c", code], env=env,
                             capture_output=True, text=True, check=True)
        digests.add(out.stdout.strip())
    assert len(digests) == 1, digests


def test_oracle_is_perfect():
    for preset in ("S", "M"):
        gen = _gen(preset=preset)
        report, _ = _score_with(gen, lambda q: q["answer_display"])
        assert report["accuracy_final"] == 1.0, report["by_qtype"]
        assert report["accuracy_online"] == 1.0


def test_stale_oracle_fails_freshness():
    gen = _gen(preset="M")
    first = {f["key"]: f["versions"][0]["value"]
             for f in gen.world.dump()}

    def stale(q):
        if q["fact_keys"] and q["fact_keys"][0] in first:
            return first[q["fact_keys"][0]]
        return q["answer_display"]

    report, answers = _score_with(gen, stale)
    updated = [r for r in report["rows"]
               if r["qtype"] == QT_CURRENT_UPDATED]
    assert updated, "no current_updated queries generated"
    assert all(r["class"] == "stale" for r in updated), [
        (r["query_id"], r["class"]) for r in updated if r["class"] != "stale"]
    assert report["accuracy_final"] < 1.0


def test_null_system_abstains():
    gen = _gen(preset="S")
    report, _ = _score_with(gen, lambda q: "unknown")
    for r in report["rows"]:
        if r["qtype"] == QT_ABSENT:
            assert r["class"] == "correct"
        else:
            assert r["class"] == "abstain"


def test_every_value_is_verbalized():
    gen = _gen(preset="M")
    texts = {e.episode_id: e.text for e in gen.episodes}
    for f in gen.world.dump():
        for v in f["versions"]:
            if v["revoked"]:
                continue
            assert boundary_found(v["value"], texts[v["episode_id"]]), (
                f["key"], v)


def test_absent_facts_never_exist():
    gen = _gen(preset="M")
    for q in gen.final_queries:
        if q.qtype == QT_ABSENT:
            assert q.fact_keys[0] not in gen.world.facts


def test_multihop_sources_are_separated():
    gen = _gen(preset="M")
    cfg = gen.cfg
    for q in gen.final_queries:
        if q.qtype not in (QT_MULTIHOP_2, QT_MULTIHOP_3):
            continue
        versions = [gen.world.current(k, cfg.n_days) for k in q.fact_keys]
        eps = {v.episode_id for v in versions}
        days = [v.day for v in versions]
        assert len(eps) == len(q.fact_keys), q.query_id
        assert max(days) - min(days) >= cfg.multihop_min_gap, q.query_id


def test_accepted_rejected_disjoint():
    gen = _gen(preset="M")
    for q in gen.online_queries + gen.final_queries:
        acc = {a.lower() for a in q.accepted}
        rej = {r.lower() for r in q.rejected}
        assert not acc & rej, (q.query_id, acc & rej)


def test_bank_values_have_no_boundary_collisions():
    domains = [banks.CITIES, banks.COMPANIES, banks.RESTAURANTS,
               banks.PROJECTS, [p[0] for p in banks.PERSONS],
               banks.COFFEES, banks.WIFI_PASSWORDS, banks.DIETARY]
    for bank in domains:
        for a in bank:
            for b in bank:
                if a != b:
                    assert not boundary_found(a, b), (a, b)


def test_online_queries_track_usage():
    gen = _gen(preset="M")
    assert gen.online_queries, "no online queries generated"
    assert sum(gen.usage.values()) == len(gen.online_queries)
    for key in gen.usage:
        assert key in gen.hot_set


if __name__ == "__main__":
    import sys
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
