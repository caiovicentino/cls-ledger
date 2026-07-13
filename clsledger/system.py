"""CLS-Ledger v0: complementary learning system with a provenance ledger.

Hippocampus: episodic store (raw episodes + extracted fact cards).
Cortex    : LoRA adapter written by distillation of SELECTED cards.
Ledger    : maps every training example back to cards and episodes.

v0 differences vs. the SEAL-lite baseline (the scientific contrast):
1. state-level, not episode-level: only CURRENT (non-superseded) values are
   consolidated — SEAL memorizes every mention, including stale ones;
2. churn-gated selection: facts observed to change frequently stay
   episodic (freshness in weights is a losing game);
3. provenance: reversible by construction — drop an entity's cards from
   the ledger and re-distill (unlearning eval, H3).

Online (mid-life) queries are answered from the episodic store (BM25 over
episodes, local model) and their usage is charged to matching cards — the
usage signal for the H1 selection ablations.
"""
from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional

from agentlife.harness.backends import MLXBackend, OpenAIBackend
from agentlife.harness.bm25 import BM25Index
from agentlife.harness.parametric_systems import (
    PARAMETRIC_SYSTEM_PROMPT, _run_lora_training, _write_dataset)
from agentlife.harness.real_systems import SYSTEM_PROMPT, build_prompt
from agentlife.harness.systems import MemorySystem

from .ledger import Card, Ledger

EXTRACT_PROMPT = (
    "Extract the atomic facts from this dated personal note.\n"
    "Note (day {day}): {text}\n\n"
    "Return ONLY a JSON array like "
    '[{{"entity": "...", "attribute": "...", "value": "..."}}]. Rules:\n'
    "- entity: who/what the fact is about (a person's full name, a project "
    "name, or 'user');\n"
    "- attribute: snake_case (employer, lives_in, birthday, parking_spot, "
    "deadline, lead, status, ...);\n"
    "- value: the short canonical value only; use 'none' when something is "
    "revoked and 'cancelled' when a project is cancelled;\n"
    "- if the note contains no durable fact (small talk), return [].")


class CLSLedgerSystem(MemorySystem):
    name = "cls"

    def __init__(self, model_id: str, workdir: str, iters: int = 300,
                 cache_dir: Optional[str] = None,
                 extractor_model: str = "gpt-4.1-mini",
                 policy: str = "stable", online_k: int = 8,
                 mode: str = "hybrid", replay: bool = True,
                 sleep_every: int = 0):
        self.model_id = model_id
        self.workdir = workdir
        self.iters = iters
        self.cache_dir = cache_dir
        self.policy = policy
        self.online_k = online_k
        self.mode = mode          # 'hybrid' (v1) or 'weights' (v0)
        self.replay = replay
        self.extractor = OpenAIBackend(extractor_model, cache_dir=cache_dir)
        self.ledger = Ledger()
        self.episodes: List[dict] = []
        self.ep_index = BM25Index()
        self.cards_by_id: Dict[str, Card] = {}
        self.all_index: Optional[BM25Index] = None
        self.consolidated_ids: set = set()
        self.base_backend: Optional[MLXBackend] = None
        self.backend: Optional[MLXBackend] = None
        self.today = 0
        self.routes: Dict[str, str] = {}
        # periodic consolidation ("sleep"): 0 = only at end of life (v1)
        self.sleep_every = sleep_every
        self.next_sleep = sleep_every if sleep_every else None
        self.n_sleeps = 0

    # ------------------------------------------------------------ ingestion

    def _extract_cards(self, episode: dict) -> List[Card]:
        m = re.match(r"Day (\d+):", episode["text"])
        day = int(m.group(1)) if m else 0
        raw = self.extractor.complete(
            "You extract facts from notes. Output only JSON.",
            EXTRACT_PROMPT.format(day=day, text=episode["text"]),
            max_tokens=300)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.M).strip()
        try:
            items = json.loads(raw)
        except json.JSONDecodeError:
            return []
        cards = []
        for it in items:
            if (isinstance(it, dict) and it.get("entity")
                    and it.get("attribute") and it.get("value") is not None):
                cards.append(Card(str(it["entity"]), str(it["attribute"]),
                                  str(it["value"]), day,
                                  episode["episode_id"]))
        return cards

    def ingest(self, episode: dict) -> None:
        self.episodes.append(episode)
        self.ep_index.add(episode["episode_id"], episode["text"])
        m = re.match(r"Day (\d+):", episode["text"])
        if m:
            self.today = max(self.today, int(m.group(1)))
        if self.next_sleep is not None and self.today > self.next_sleep:
            self._consolidate(self.next_sleep)
            while self.next_sleep < self.today:
                self.next_sleep += self.sleep_every
        for card in self._extract_cards(episode):
            self.ledger.add(card)

    # --------------------------------------------------- online (episodic)

    def _ensure_base(self) -> MLXBackend:
        if self.base_backend is None:
            self.base_backend = MLXBackend(self.model_id,
                                           cache_dir=self.cache_dir)
        return self.base_backend

    def _charge_usage(self, question: str) -> None:
        idx = BM25Index()
        for c in self.ledger.current_cards():
            idx.add(c.card_id, c.text())
        for cid, _score in idx.search(question, 3):
            for c in self.ledger.cards:
                if c.card_id == cid:
                    c.usage += 1
                    break

    def _fact_timelines(self) -> Dict[str, str]:
        """One retrieval document per FACT: its full dated value timeline.
        Gives the reader explicit temporal structure — the v1 episodic path
        failed point-in-time (20%) and revocations (0%) because raw
        episodes bury the timeline."""
        docs: Dict[str, str] = {}
        keys = {c.key for c in self.ledger.cards}
        for key in keys:
            hist = self.ledger.history(key)
            if not hist:
                continue
            ent = hist[0].entity
            attr = hist[0].attribute.replace("_", " ")
            spans = []
            for i, c in enumerate(hist):
                end = (f"to day {hist[i + 1].day - 1}"
                       if i + 1 < len(hist) else "current")
                label = "none — revoked" if c.value in ("none", "") else \
                    c.value
                spans.append(f"{label} (day {c.day} {end})")
            docs[key] = f"{ent} | {attr}: " + "; ".join(spans)
        return docs

    def _episodic_answer(self, query: dict) -> str:
        docs = self._fact_timelines()
        if len(docs) >= 5:
            idx = BM25Index()
            for key in sorted(docs):
                idx.add(key, docs[key])
            hits = idx.search(query["question"], self.online_k)
            notes = [docs[key] for key, _ in hits]
        else:  # cold start: ledger too small, use raw episodes
            texts = {e["episode_id"]: e["text"] for e in self.episodes}
            order = {e["episode_id"]: i for i, e in enumerate(self.episodes)}
            hits = self.ep_index.search(query["question"], self.online_k)
            ids = sorted((eid for eid, _ in hits), key=lambda e: order[e])
            notes = [texts[eid] for eid in ids]
        return self._ensure_base().complete(SYSTEM_PROMPT,
                                            build_prompt(notes, query))

    def _route(self, question: str) -> str:
        """Ledger-based routing, no benchmark-specific phrasing tricks:
        - temporal questions go episodic (weights hold current state only);
        - otherwise: if the single most relevant card for the question was
          consolidated AND its entity is explicitly the subject of the
          question, answer from weights; anything else stays episodic."""
        if "As of day" in question:
            return "episodic"
        if self.all_index is None or not self.consolidated_ids:
            return "episodic"
        top = self.all_index.search(question, 1)
        if not top:
            return "episodic"
        cid = top[0][0]
        if cid not in self.consolidated_ids:
            return "episodic"  # e.g. volatile fact kept episodic on purpose
        card = self.cards_by_id[cid]
        ent = card.entity.lower()
        subject = "user" if ent == "user" else ent
        if subject not in question.lower():
            return "episodic"
        return "weights"

    def answer(self, query: dict) -> str:
        if query.get("qtype_public") == "online":
            self._charge_usage(query["question"])
        if self.backend is None:
            return self._episodic_answer(query)
        route = ("weights" if self.mode == "weights"
                 else self._route(query["question"]))
        self.routes[query.get("query_id", "?")] = route
        if route == "episodic":
            return self._episodic_answer(query)
        user = (f"QUESTION (asked on day {query['day_asked']}): "
                f"{query['question']}\nAnswer:")
        return self.backend.complete(PARAMETRIC_SYSTEM_PROMPT, user)

    def stats(self) -> dict:
        counts: Dict[str, int] = {}
        for r in self.routes.values():
            counts[r] = counts.get(r, 0) + 1
        return {"mode": self.mode, "policy": self.policy,
                "routes": counts,
                "routes_by_query": dict(self.routes),
                "cards_current": len(self.ledger.current_cards()),
                "cards_consolidated": len(self.consolidated_ids)}

    # -------------------------------------------------------- consolidation

    def _twin_entities(self) -> set:
        """Entities sharing a first name token with another entity — the
        binding-confusion risk group observed in v1 (Auriga vs Aurora,
        the two Sofias)."""
        firsts: Dict[str, set] = {}
        for c in self.ledger.current_cards():
            token = c.entity.split()[-1] if c.entity.startswith("Project ") \
                else c.entity.split()[0]
            firsts.setdefault(token.lower()[:4], set()).add(c.entity)
        return {e for group in firsts.values() if len(group) > 1
                for e in group}

    def _distill_rows(self, cards: List[Card], day: int) -> List[dict]:
        twins = self._twin_entities()
        rows = []
        for c in cards:
            attr = c.attribute.replace("_", " ")
            entity = "the user" if c.entity.lower() == "user" else c.entity
            questions = [
                f"What is {entity}'s {attr}?",
                f"As of day {day}, what is {entity}'s {attr}?",
                f"Tell me the current {attr} of {entity}.",
            ]
            answer = c.value
            if c.value == "cancelled":
                questions.append(f"Is {entity} still active?")
                rows.append({"messages": [
                    {"role": "system", "content": PARAMETRIC_SYSTEM_PROMPT},
                    {"role": "user", "content": questions.pop()},
                    {"role": "assistant",
                     "content": "no, it was cancelled"},
                ], "_card_id": c.card_id})
            elif c.value in ("none", ""):
                answer = "none"
                rows.append({"messages": [
                    {"role": "system", "content": PARAMETRIC_SYSTEM_PROMPT},
                    {"role": "user",
                     "content": f"Does {entity} currently have any {attr}?"},
                    {"role": "assistant", "content": "no"},
                ], "_card_id": c.card_id})
            reps = 2 if c.entity in twins else 1
            for _ in range(reps):
                for q in questions:
                    rows.append({"messages": [
                        {"role": "system",
                         "content": PARAMETRIC_SYSTEM_PROMPT},
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": answer},
                    ], "_card_id": c.card_id})
        return rows

    def _consolidate(self, day: int) -> None:
        """One sleep cycle: select, distill, retrain from scratch, swap the
        adapter. Each sleep gets its own adapter dir so response caches
        from different sleeps never collide."""
        self.n_sleeps += 1
        sleep_dir = os.path.join(self.workdir, f"sleep-{self.n_sleeps:02d}")
        os.makedirs(sleep_dir, exist_ok=True)
        selected = self.ledger.select_for_consolidation(day,
                                                        policy=self.policy)
        rows = self._distill_rows(selected, day)
        provenance = [{"card_id": r.pop("_card_id")} for r in rows]
        if self.replay:
            from .replay import replay_rows
            extra = replay_rows(PARAMETRIC_SYSTEM_PROMPT)
            provenance += [{"card_id": "__replay__"} for _ in extra]
            rows += extra
            import random as _random
            order = list(range(len(rows)))
            _random.Random(0).shuffle(order)
            rows = [rows[i] for i in order]
            provenance = [provenance[i] for i in order]
        data_dir = os.path.join(sleep_dir, "data")
        adapter = os.path.join(sleep_dir, "adapter")
        n_valid = max(2, len(rows) // 10)
        _write_dataset(data_dir, rows, rows[-n_valid:])
        with open(os.path.join(sleep_dir, "provenance.jsonl"), "w") as f:
            for i, p in enumerate(provenance):
                f.write(json.dumps({"train_row": i, **p}) + "\n")
        self.ledger.dump(os.path.join(sleep_dir, "ledger.jsonl"))
        stats = {
            "sleep": self.n_sleeps, "day": day,
            "cards_total": len(self.ledger.cards),
            "cards_current": len(self.ledger.current_cards()),
            "cards_selected": len(selected),
            "train_rows": len(rows),
            "policy": self.policy, "mode": self.mode,
            "replay": self.replay, "iters": self.iters,
        }
        with open(os.path.join(sleep_dir, "consolidation.json"), "w") as f:
            json.dump(stats, f, indent=2)
        # routing structures: consolidated = the exact card versions the
        # weights were trained on; a fact updated after this sleep gets a
        # new card_id, drops out of this set, and routes episodic — the
        # ledger knows what the weights know.
        self.consolidated_ids = {c.card_id for c in selected}
        self.all_index = BM25Index()
        self.cards_by_id = {}
        for c in self.ledger.current_cards():
            self.all_index.add(c.card_id, c.text())
            self.cards_by_id[c.card_id] = c
        _run_lora_training(self.model_id, data_dir, adapter, self.iters,
                           mask_prompt=True)
        self.backend = MLXBackend(self.model_id, adapter_path=adapter,
                                  cache_dir=self.cache_dir)
        # keep top-level artifacts pointing at the latest sleep
        for name in ("consolidation.json", "provenance.jsonl",
                     "ledger.jsonl"):
            src = os.path.join(sleep_dir, name)
            dst = os.path.join(self.workdir, name)
            with open(src) as fsrc, open(dst, "w") as fdst:
                fdst.write(fsrc.read())

    def finalize(self) -> None:
        os.makedirs(self.workdir, exist_ok=True)
        self._consolidate(self.today)
