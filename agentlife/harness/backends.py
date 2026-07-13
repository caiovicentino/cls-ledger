"""LLM backends. stdlib-only (urllib), disk-cached, temperature 0.

The cache makes reruns free and scoring reproducible: identical
(model, system, user) triples never hit the network twice.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from typing import Optional


class LLMBackend:
    name = "base"

    def complete(self, system: str, user: str, max_tokens: int = 64) -> str:
        raise NotImplementedError


class FakeBackend(LLMBackend):
    """For tests: returns a canned answer and records every prompt."""

    name = "fake"

    def __init__(self, reply: str = "unknown"):
        self.reply = reply
        self.calls = []

    def complete(self, system: str, user: str, max_tokens: int = 64) -> str:
        self.calls.append({"system": system, "user": user})
        return self.reply


class DiskCache:
    def __init__(self, cache_dir: str):
        self.dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _path(self, key: str) -> str:
        return os.path.join(self.dir, key + ".json")

    def key(self, model: str, system: str, user: str) -> str:
        payload = json.dumps([model, system, user], ensure_ascii=False)
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, key: str) -> Optional[str]:
        path = self._path(key)
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)["response"]
        return None

    def put(self, key: str, model: str, response: str) -> None:
        with open(self._path(key), "w") as f:
            json.dump({"model": model, "response": response}, f,
                      ensure_ascii=False)


class OpenAIBackend(LLMBackend):
    name = "openai"

    def __init__(self, model: str, cache_dir: Optional[str] = None,
                 api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.environ["OPENAI_API_KEY"]
        self.cache = DiskCache(cache_dir) if cache_dir else None
        self.n_calls = 0
        self.n_cached = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def complete(self, system: str, user: str, max_tokens: int = 64) -> str:
        if self.cache:
            key = self.cache.key(self.model, system, user)
            hit = self.cache.get(key)
            if hit is not None:
                self.n_cached += 1
                return hit
        body = json.dumps({
            "model": self.model,
            "temperature": 0,
            "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
        }).encode()
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions", data=body,
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"})
        last_err = None
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=90) as r:
                    data = json.load(r)
                out = data["choices"][0]["message"]["content"].strip()
                usage = data.get("usage", {})
                self.prompt_tokens += usage.get("prompt_tokens", 0)
                self.completion_tokens += usage.get("completion_tokens", 0)
                self.n_calls += 1
                if self.cache:
                    self.cache.put(key, self.model, out)
                return out
            except urllib.error.HTTPError as e:
                last_err = e
                if e.code in (429, 500, 502, 503):
                    time.sleep(2 ** attempt)
                    continue
                raise
            except (urllib.error.URLError, TimeoutError) as e:
                last_err = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"backend failed after retries: {last_err}")

    def stats(self) -> dict:
        return {"model": self.model, "api_calls": self.n_calls,
                "cache_hits": self.n_cached,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens}


def make_backend(spec: str, cache_dir: Optional[str]) -> LLMBackend:
    """spec format: 'openai:gpt-4.1-mini' (extensible: 'mlx:...', etc.)."""
    provider, _, model = spec.partition(":")
    if provider == "openai" and model:
        return OpenAIBackend(model, cache_dir=cache_dir)
    if provider == "fake":
        return FakeBackend(model or "unknown")
    raise ValueError(f"unknown backend spec {spec!r}")
