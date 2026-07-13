"""Deterministic BM25 over episode texts. stdlib only.

Ties break by insertion order (earlier document wins) so retrieval is
reproducible across processes.
"""
from __future__ import annotations

import math
import re
from typing import Dict, List, Tuple


def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs: List[List[str]] = []
        self.doc_ids: List[str] = []
        self.df: Dict[str, int] = {}
        self.total_len = 0

    def add(self, doc_id: str, text: str) -> None:
        tokens = tokenize(text)
        self.docs.append(tokens)
        self.doc_ids.append(doc_id)
        self.total_len += len(tokens)
        for term in set(tokens):
            self.df[term] = self.df.get(term, 0) + 1

    def search(self, query: str, k: int) -> List[Tuple[str, float]]:
        n = len(self.docs)
        if n == 0:
            return []
        avg_len = self.total_len / n
        q_terms = tokenize(query)
        scores = []
        for i, doc in enumerate(self.docs):
            tf: Dict[str, int] = {}
            for t in doc:
                tf[t] = tf.get(t, 0) + 1
            s = 0.0
            for t in q_terms:
                if t not in tf:
                    continue
                df = self.df.get(t, 0)
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                denom = tf[t] + self.k1 * (
                    1 - self.b + self.b * len(doc) / avg_len)
                s += idf * tf[t] * (self.k1 + 1) / denom
            scores.append((s, i))
        scores.sort(key=lambda x: (-x[0], x[1]))
        return [(self.doc_ids[i], s) for s, i in scores[:k] if s > 0]
