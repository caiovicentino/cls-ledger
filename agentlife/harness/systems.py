"""Reference systems used to validate the benchmark itself.

- oracle: reads the private answer key; must score 100% (validates the
  generator + scorer pipeline end to end — the test LoCoMo failed).
- stale : always answers the FIRST version of the fact; must fail freshness
  queries (validates that rejected values catch superseded answers).
- null  : always answers 'unknown'; floor, and sanity check for the
  absent/abstain machinery.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional

from ..schema import load_jsonl
from .scorer import FRESHNESS_QTYPES


class MemorySystem:
    name = "base"

    def ingest(self, episode: dict) -> None:
        pass

    def answer(self, query: dict) -> str:
        raise NotImplementedError


class NullSystem(MemorySystem):
    name = "null"

    def answer(self, query: dict) -> str:
        return "unknown"


class OracleSystem(MemorySystem):
    """Privileged: reads private answers. Exists to validate the pipeline."""

    name = "oracle"

    def __init__(self, data_dir: str):
        rows = load_jsonl(os.path.join(data_dir, "private", "answers.jsonl"))
        self._answers: Dict[str, str] = {
            r["query_id"]: r["answer_display"] for r in rows}

    def answer(self, query: dict) -> str:
        return self._answers[query["query_id"]]


class StaleOracleSystem(MemorySystem):
    """Privileged: answers the first-ever version of the queried fact."""

    name = "stale"

    def __init__(self, data_dir: str):
        rows = load_jsonl(os.path.join(data_dir, "private", "answers.jsonl"))
        self._meta = {r["query_id"]: r for r in rows}
        facts = load_jsonl(os.path.join(data_dir, "private",
                                        "world_facts.jsonl"))
        self._first: Dict[str, str] = {
            f["key"]: f["versions"][0]["value"] for f in facts}

    def answer(self, query: dict) -> str:
        meta = self._meta[query["query_id"]]
        if meta["qtype"] in FRESHNESS_QTYPES and meta["fact_keys"]:
            first = self._first.get(meta["fact_keys"][0])
            if first:
                return first
        return meta["answer_display"]


def make_system(name: str, data_dir: str) -> MemorySystem:
    if name == "oracle":
        return OracleSystem(data_dir)
    if name == "stale":
        return StaleOracleSystem(data_dir)
    if name == "null":
        return NullSystem()
    raise ValueError(f"unknown system {name!r}")
