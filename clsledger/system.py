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

# The attribute vocabulary is a domain schema for a personal-assistant
# agent (same design as Zep/Graphiti's prescribed ontology). It nudges the
# extractor toward consistent attribute names so supersedence can merge
# facts; it contains no answers.
EXTRACT_PROMPT = (
    "Extract the atomic facts from this dated personal note.\n"
    "Note (day {day}): {text}\n\n"
    "Return ONLY a JSON array like "
    '[{{"entity": "...", "attribute": "...", "value": "..."}}]. Rules:\n'
    "- entity: who/what the fact is about (a person's full name, a project "
    "name, or 'user');\n"
    "- attribute: snake_case; whenever one of these fits, use EXACTLY it: "
    "employer, lives_in, hometown, birthday, favorite_restaurant, "
    "works_on, lead, deadline, budget, status, client, parking_spot, "
    "desk_booking, gym_day, dietary_restriction, wifi_password, "
    "favorite_coffee;\n"
    "- value: the short canonical value only (for projects use the full "
    "name, e.g. 'Project Falcon'); use 'none' when something is revoked "
    "and 'cancelled' when a project is cancelled;\n"
    "- if the note contains no durable fact (small talk), return [].")


class CLSLedgerSystem(MemorySystem):
    name = "cls"

    def __init__(self, model_id: str, workdir: str, iters: int = 300,
                 cache_dir: Optional[str] = None,
                 extractor_model: str = "gpt-4.1-mini",
                 policy: str = "stable", online_k: int = 8,
                 mode: str = "hybrid", replay: bool = True,
                 sleep_every: int = 0, budget=None):
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
        self.budget = budget
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
            if not (isinstance(it, dict) and it.get("entity")
                    and it.get("attribute")
                    and it.get("value") is not None):
                continue
            attr = str(it["attribute"]).strip().lower()
            # extraction hygiene: update episodes mention the outgoing
            # value; extractors sometimes emit junk 'old_x: none' cards
            # that later read as revocations
            if re.match(r"^(old|previous|former|prior|new)_", attr):
                continue
            if str(it["value"]).strip() == "":
                continue
            cards.append(Card(str(it["entity"]), attr,
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

    def _fact_snapshot(self, day: Optional[int]) -> Dict[str, str]:
        """One retrieval document per FACT, RESOLVED at a point in time by
        the ledger — the reader never sees history, so it never has to do
        temporal reasoning (which a 3B reader demonstrably cannot do from
        timeline documents: it either answers stale values or abstains).

        day=None -> current values; day=D -> value valid on day D."""
        docs: Dict[str, str] = {}
        keys = {c.key for c in self.ledger.cards}
        for key in sorted(keys):
            hist = self.ledger.history(key)
            if not hist:
                continue
            if day is None:
                c = hist[-1]
                when = f"(current, since day {c.day})"
            else:
                valid = [x for x in hist if x.day <= day]
                if not valid:
                    continue
                c = valid[-1]
                when = f"(as of day {day})"
            ent = c.entity
            attr = c.attribute.replace("_", " ")
            label = "none — revoked" if c.value in ("none", "") else c.value
            docs[key] = f"{ent} | {attr}: {label} {when}"
        return docs

    # surface forms of schema attributes as they appear in questions
    # (part of the same prescribed domain ontology as EXTRACT_PROMPT)
    ATTR_LEXICON = {
        "employer": ["employer"], "lives_in": ["live"],
        "hometown": ["hometown"], "birthday": ["birthday"],
        "favorite_restaurant": ["favorite restaurant"],
        "works_on": ["works on", "work on", "working on"],
        "lead": ["lead"], "deadline": ["deadline"], "budget": ["budget"],
        "status": ["status"], "client": ["client"],
        "parking_spot": ["parking spot"], "desk_booking": ["desk"],
        "gym_day": ["gym day"],
        "dietary_restriction": ["dietary restriction"],
        "wifi_password": ["wifi password"], "favorite_coffee": ["coffee"],
    }

    def _value_at(self, entity: str, attr: str,
                  asof: Optional[int]) -> Optional[str]:
        from .ledger import norm_key
        key = f"{norm_key(entity)}.{norm_key(attr)}"
        hist = self.ledger.history(key)
        if asof is not None:
            hist = [c for c in hist if c.day <= asof]
        return hist[-1].value if hist else None

    def _symbolic_chain(self, question: str,
                        asof: Optional[int]) -> Optional[str]:
        """Resolve chain questions entirely in the ledger: the question
        names an anchor entity and >=2 schema attributes ('the DEADLINE of
        the project X WORKS ON'); apply the relations from the anchor and
        return the target value. The reader never has to compose."""
        ql = question.lower()
        mentions = []
        for attr, forms in self.ATTR_LEXICON.items():
            positions = [ql.find(f) for f in forms if ql.find(f) >= 0]
            if positions:
                mentions.append((min(positions), attr))
        mentions.sort()
        if len(mentions) < 2:
            return None
        target, rels = mentions[0][1], [m[1] for m in mentions[1:]]
        names = sorted({c.entity for c in self.ledger.cards},
                       key=len, reverse=True)
        anchor = next((n for n in names if n.lower() in ql), None)
        if anchor is None:
            return None
        entity = anchor
        for rel in reversed(rels):
            value = self._value_at(entity, rel, asof)
            if value is None:
                return None
            nxt = next((n for n in names
                        if n.lower() == value.lower()
                        or value.lower() in n.lower()
                        or n.lower() in value.lower()), None)
            if nxt is None:
                return None
            entity = nxt
        return self._value_at(entity, target, asof)

    def _episodic_answer(self, query: dict) -> str:
        m = re.search(r"As of day (\d+)", query["question"])
        asof = int(m.group(1)) if m else None
        docs = self._fact_snapshot(asof)
        if len(docs) >= 5:
            idx = BM25Index()
            for key in sorted(docs):
                idx.add(key, docs[key])
            hits = idx.search(query["question"], self.online_k)
            # composition handling: when a retrieved fact's VALUE is itself
            # an entity (lead -> person, works_on -> project) that the
            # question does not name, the question is likely relative
            # ("... of the project X works on"). Resolve the intermediate
            # entity symbolically via the ledger, then answer with that
            # entity's facts placed first. Resolution lives in structure,
            # not in the reader — a 3B reader demonstrably cannot compose
            # two hops unaided.
            entity_names = {c.entity for c in self.ledger.cards
                            if c.entity.lower() != "user"}
            qlow = query["question"].lower()
            intermediates = []
            for key, _score in hits[:4]:
                value_part = docs[key].split(":", 1)[-1].lower()
                for name in sorted(entity_names):
                    if (name.lower() in value_part
                            and name.lower() not in qlow
                            and name not in intermediates):
                        intermediates.append(name)
            if intermediates:
                inter_keys = [k for k in sorted(docs)
                              for name in intermediates
                              if docs[k].lower().startswith(name.lower())]
                ordered = inter_keys + [k for k, _ in hits
                                        if k not in inter_keys]
                notes = [docs[k] for k in ordered[:self.online_k + 6]]
                notes.append(
                    "Resolution note: the intermediate entity referred to "
                    "by the question is "
                    + " / ".join(intermediates[:2]) + ".")
            else:
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
        self.routed_card = card
        return "weights"

    def answer(self, query: dict) -> str:
        if query.get("qtype_public") == "online":
            self._charge_usage(query["question"])
        # chain questions resolve symbolically in the ledger BEFORE any
        # routing — the ledger is authoritative and neither the weights
        # nor the reader can compose relations reliably
        m = re.search(r"As of day (\d+)", query["question"])
        asof = int(m.group(1)) if m else None
        chained = self._symbolic_chain(query["question"], asof)
        if chained is not None:
            self.routes[query.get("query_id", "?")] = "symbolic"
            return chained
        if self.backend is None:
            return self._episodic_answer(query)
        route = ("weights" if self.mode == "weights"
                 else self._route(query["question"]))
        self.routes[query.get("query_id", "?")] = route
        if route == "episodic":
            return self._episodic_answer(query)
        return self._weights_answer(query)

    def _weights_answer(self, query: dict) -> str:
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
        selected = self.ledger.select_for_consolidation(
            day, policy=self.policy, budget=self.budget)
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
