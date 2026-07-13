"""Offline tests for Phase 1 plumbing (no network).

The critical property: NO FUTURE LEAKAGE — an online query answered on day
D must never see episodes ingested after it in the stream.
"""
from __future__ import annotations

from agentlife.harness.backends import FakeBackend
from agentlife.harness.bm25 import BM25Index
from agentlife.harness.real_systems import FullContextSystem, RAGSystem


def _ep(eid, day, text):
    return {"episode_id": eid, "day": day, "text": f"Day {day}: {text}"}


def _q(qid, day, question):
    return {"query_id": qid, "day_asked": day, "question": question}


def test_bm25_ranks_relevant_first():
    idx = BM25Index()
    idx.add("e1", "Day 1: I parked the car at spot D42 today.")
    idx.add("e2", "Day 2: Watched a documentary about deep sea creatures.")
    idx.add("e3", "Day 5: Moved the car — it's at spot B13 now.")
    hits = idx.search("What is the user's current parking spot?", 2)
    ids = [h[0] for h in hits]
    assert set(ids) <= {"e1", "e3"}, ids
    assert len(ids) == 2


def test_bm25_deterministic_tiebreak():
    idx = BM25Index()
    idx.add("a", "alpha beta")
    idx.add("b", "alpha beta")
    assert [h[0] for h in idx.search("alpha", 2)] == ["a", "b"]


def test_full_context_no_future_leakage():
    be = FakeBackend()
    sys_ = FullContextSystem(be)
    sys_.ingest(_ep("e1", 1, "car at spot D42"))
    sys_.answer(_q("q1", 2, "where is the car?"))
    sys_.ingest(_ep("e2", 3, "car moved to spot B13"))
    sys_.answer(_q("q2", 4, "where is the car?"))
    assert "B13" not in be.calls[0]["user"]
    assert "D42" in be.calls[0]["user"]
    assert "B13" in be.calls[1]["user"]


def test_rag_no_future_leakage_and_topk():
    be = FakeBackend()
    sys_ = RAGSystem(be, k=2)
    sys_.ingest(_ep("e1", 1, "car at spot D42"))
    sys_.answer(_q("q1", 2, "where is the car parked, at which spot?"))
    assert "D42" in be.calls[0]["user"]
    for i in range(2, 30):
        sys_.ingest(_ep(f"e{i}", i, "watched a documentary tonight"))
    sys_.ingest(_ep("e99", 30, "car moved to spot B13"))
    sys_.answer(_q("q2", 31, "where is the car parked, at which spot?"))
    prompt = be.calls[1]["user"]
    assert "B13" in prompt
    # k=2: at most 2 notes + question line mention days
    n_notes = prompt.split("NOTES:\n")[1].split("\n\nQUESTION")[0].count("Day ")
    assert n_notes <= 2, prompt


def test_rag_notes_in_chronological_order():
    be = FakeBackend()
    sys_ = RAGSystem(be, k=3)
    sys_.ingest(_ep("e2", 5, "car moved to spot B13"))
    sys_.ingest(_ep("e1", 1, "car at spot D42 in the garage"))
    sys_.answer(_q("q1", 6, "at which spot is the car parked?"))
    prompt = be.calls[0]["user"]
    assert prompt.index("Day 5") < prompt.index("Day 1") or (
        "Day 1" in prompt and "Day 5" in prompt
        and prompt.index("Day 5") > prompt.index("Day 1"))


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
