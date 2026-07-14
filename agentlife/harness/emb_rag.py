"""Embeddings-RAG baseline: OpenAI text-embedding-3-small + cosine top-k,
same local reader as every other system. The strong retrieval baseline the
BM25 straw man was rightly criticized for not being.

Embeddings are disk-cached (one vector per unique text), so reruns are
free and deterministic.
"""
from __future__ import annotations

import json
import math
import os
import urllib.request
from typing import Dict, List, Optional

from .backends import DiskCache, LLMBackend
from .real_systems import SYSTEM_PROMPT, build_prompt
from .systems import MemorySystem

EMB_MODEL = "text-embedding-3-small"


class EmbeddingClient:
    def __init__(self, cache_dir: Optional[str] = None):
        self.api_key = os.environ["OPENAI_API_KEY"]
        self.cache = DiskCache(cache_dir) if cache_dir else None

    def embed(self, text: str) -> List[float]:
        if self.cache:
            key = self.cache.key(f"emb:{EMB_MODEL}", text, "")
            hit = self.cache.get(key)
            if hit is not None:
                return json.loads(hit)
        body = json.dumps({"model": EMB_MODEL, "input": text}).encode()
        req = urllib.request.Request(
            "https://api.openai.com/v1/embeddings", data=body,
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            vec = json.load(r)["data"][0]["embedding"]
        if self.cache:
            self.cache.put(key, EMB_MODEL, json.dumps(vec))
        return vec


def cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


class EmbRAGSystem(MemorySystem):
    name = "embrag"

    def __init__(self, backend: LLMBackend, k: int = 8,
                 cache_dir: Optional[str] = None):
        self.backend = backend
        self.k = k
        self.client = EmbeddingClient(cache_dir)
        self.episodes: List[dict] = []
        self.vectors: List[List[float]] = []

    def ingest(self, episode: dict) -> None:
        self.episodes.append(episode)
        self.vectors.append(self.client.embed(episode["text"]))

    def answer(self, query: dict) -> str:
        qv = self.client.embed(query["question"])
        scored = sorted(
            range(len(self.episodes)),
            key=lambda i: (-cosine(qv, self.vectors[i]), i))[: self.k]
        ordered = sorted(scored)  # chronological presentation
        notes = [self.episodes[i]["text"] for i in ordered]
        return self.backend.complete(SYSTEM_PROMPT,
                                     build_prompt(notes, query))
