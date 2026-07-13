"""Real memory-system baselines (Phase 1).

- full: entire episode stream in context. The upper bound of
  context-as-memory — and the thing any consolidation method must beat on
  cost while matching on accuracy.
- rag : BM25 top-k episodes. The industry-default workaround; our research
  predicts it fails on multi-hop and freshness (redundancy collapse).

Both only ever see episodes already ingested (no future leakage: online
queries are answered mid-stream).
"""
from __future__ import annotations

from typing import List

from .backends import LLMBackend
from .bm25 import BM25Index
from .systems import MemorySystem

SYSTEM_PROMPT = (
    "You are a personal assistant with memory of the user's life, recorded "
    "as dated notes ('Day N: ...'). Answer the question using only the "
    "notes. Be concise: answer with the value only (a name, a date, a city, "
    "a code), no explanations. If the notes never provide the information, "
    "answer 'unknown'."
)


def build_prompt(notes: List[str], query: dict) -> str:
    joined = "\n".join(notes) if notes else "(no notes)"
    return (f"NOTES:\n{joined}\n\n"
            f"QUESTION (asked on day {query['day_asked']}): "
            f"{query['question']}\nAnswer:")


class FullContextSystem(MemorySystem):
    name = "full"

    def __init__(self, backend: LLMBackend):
        self.backend = backend
        self.notes: List[str] = []

    def ingest(self, episode: dict) -> None:
        self.notes.append(episode["text"])

    def answer(self, query: dict) -> str:
        return self.backend.complete(SYSTEM_PROMPT,
                                     build_prompt(self.notes, query))


class RAGSystem(MemorySystem):
    name = "rag"

    def __init__(self, backend: LLMBackend, k: int = 8):
        self.backend = backend
        self.k = k
        self.index = BM25Index()
        self.texts = {}
        self.order = {}
        self._n = 0

    def ingest(self, episode: dict) -> None:
        eid = episode["episode_id"]
        self.index.add(eid, episode["text"])
        self.texts[eid] = episode["text"]
        self.order[eid] = self._n
        self._n += 1

    def answer(self, query: dict) -> str:
        hits = self.index.search(query["question"], self.k)
        ids = sorted((eid for eid, _ in hits), key=lambda e: self.order[e])
        notes = [self.texts[eid] for eid in ids]
        return self.backend.complete(SYSTEM_PROMPT,
                                     build_prompt(notes, query))
