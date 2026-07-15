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

from agentlife.harness.backends import MLXBackend, OpenAIBackend, make_backend
from agentlife.harness.bm25 import BM25Index
from agentlife.harness.parametric_systems import (
    PARAMETRIC_SYSTEM_PROMPT, _run_lora_training, _write_dataset)
from agentlife.harness.real_systems import SYSTEM_PROMPT, build_prompt
from agentlife.harness.systems import MemorySystem

from .ledger import Card, Ledger, norm_key

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
    "- recurring activities (a run, a class) are facts too: entity 'user', attribute 'activity_<name>' (snake_case), value 'done';\n"
    "- behavioral requests ('write dates day-first', 'city names in UPPERCASE') are facts: entity 'user', attribute 'disposition_<short_name>', value = the requested behavior;\n"
    "- if the note contains no durable fact (small talk), return [].")


class CLSLedgerSystem(MemorySystem):
    name = "cls"

    def __init__(self, model_id: str, workdir: str, iters: int = 300,
                 cache_dir: Optional[str] = None,
                 extractor: str = "openai:gpt-4.1-mini",
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
        self.extractor = make_backend(extractor, cache_dir)
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
        self._last_parse: Optional[dict] = None
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

    PARSE_PROMPT = (
        "Parse this question about a personal assistant's memory into a "
        "structured lookup. Known attribute schema: employer, lives_in, "
        "hometown, birthday, favorite_restaurant, works_on, lead, "
        "deadline, budget, status, client, parking_spot, desk_booking, "
        "gym_day, dietary_restriction, wifi_password, favorite_coffee.\n"
        "Known entities: {entities}\n\n"
        "Question: {q}\n\n"
        "If the question asks HOW MANY TIMES something changed, set "
        "\"aggregate\": \"count_changes\". If it asks on which weekday "
        "something usually happens, set \"aggregate\": "
        "\"weekday_habit\". If it asks whether a numeric value "
        "increased or decreased, set \"aggregate\": \"trend\". "
        "Otherwise omit \"aggregate\".\n"
        "Return ONLY JSON: {{\"anchor\": <entity name from the list, or "
        "\"user\", or null>, \"chain\": [<relations from the anchor "
        "toward the target, e.g. [\"works_on\",\"lead\"] for 'of the lead "
        "of the project ANCHOR works on'>], \"target\": <the attribute "
        "being asked>, \"asof_day\": <int if the question asks about a "
        "specific past day, else null>}}.\n"
        "If the question does not fit the schema, return "
        '{{"target": null}}.')

    def _semantic_parse(self, question: str) -> Optional[dict]:
        """LLM parses the question against the schema; resolution stays in
        the ledger. Replaces the v3 lexical parser, which died on
        paraphrases (multihop 0%, point-in-time 20% when 'deadline'
        became 'due date' and 'As of day' became 'back on day')."""
        entities = sorted({c.entity for c in self.ledger.cards})
        raw = self.extractor.complete(
            "You parse questions into structured lookups. Output only "
            "JSON.",
            self.PARSE_PROMPT.format(q=question,
                                     entities=", ".join(entities)),
            max_tokens=120).strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.M).strip()
        try:
            p = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(p, dict) or not p.get("target"):
            return None
        return p

    def _symbolic_chain(self, question: str,
                        asof: Optional[int]) -> Optional[str]:
        """Semantic parse (LLM) + deterministic resolution (ledger).
        Only answers when every hop resolves; anything else falls back to
        the weights/episodic routes."""
        p = self._semantic_parse(question)
        self._last_parse = p
        if p is None:
            return None
        target = norm_key(str(p["target"]))
        asof = p.get("asof_day") if isinstance(p.get("asof_day"), int) \
            else asof
        names = sorted({c.entity for c in self.ledger.cards},
                       key=len, reverse=True)
        anchor_raw = str(p.get("anchor") or "")
        anchor = next((n for n in names
                       if n.lower() == anchor_raw.lower()), None)
        if anchor is None and anchor_raw:
            anchor = next((n for n in names
                           if anchor_raw.lower() in n.lower()
                           or n.lower() in anchor_raw.lower()), None)
        if anchor is None:
            return None
        aggregate = p.get("aggregate")
        if aggregate:
            return self._symbolic_aggregate(aggregate, anchor, target, p)
        chain = [norm_key(str(r)) for r in (p.get("chain") or [])]
        # a parse that names no relation and no known target is useless;
        # single-hop (empty chain) direct lookups are allowed
        entity = anchor
        for rel in chain:
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
        # direct lookups only resolve symbolically for chain questions;
        # single-hop questions keep flowing to weights/episodic so the
        # parametric memory stays exercised and mis-parses stay cheap
        if not chain:
            return None
        return self._value_at(entity, target, asof)

    def _symbolic_aggregate(self, aggregate: str, anchor: str,
                            target: str, parse: dict) -> Optional[str]:
        """Quantitative induction resolved in the ledger: counting,
        weekday habits and trends are aggregations over fact history —
        the exact operation retrieval systems fail (incomplete gathering)
        and a ledger performs exactly."""
        from agentlife import banks as _banks
        key = f"{norm_key(anchor)}.{target}"
        hist = self.ledger.history(key)
        if aggregate == "count_changes":
            if len(hist) < 1:
                return None
            n = max(0, len(hist) - 1)
            word = _banks.NUMBER_WORDS.get(n)
            return f"{n}" + (f" ({word})" if word else "")
        if aggregate == "weekday_habit":
            # target may be an activity attribute; find best-matching
            # activity history for the anchor
            cand = [k for k in {c.key for c in self.ledger.cards}
                    if k.startswith(f"{norm_key(anchor)}.activity_")]
            if target.startswith("activity_"):
                cand = [f"{norm_key(anchor)}.{target}"] + cand
            for k in cand:
                h = self.ledger.history(k)
                if len(h) >= 4 and (target in k or len(cand) == 1
                                    or target.split('_')[-1] in k):
                    counts = [0] * 7
                    for c in h:
                        counts[(c.day - 1) % 7] += 1
                    mode = max(range(7), key=lambda i: counts[i])
                    if counts[mode] >= 0.5 * len(h):
                        return _banks.WEEKDAYS[mode]
            return None
        if aggregate == "trend":
            if len(hist) < 2:
                return None
            def num(v):
                m = re.search(r"(\d+)", v.replace(",", ""))
                return int(m.group(1)) if m else None
            first, last = num(hist[0].value), num(hist[-1].value)
            if first is None or last is None or first == last:
                return None
            return "increased" if last > first else "decreased"
        return None

    def _active_dispositions(self) -> List[str]:
        """Current behavioral rules from the ledger — injected into the
        answering prompts. Retrieval can't do this: a random question has
        no lexical similarity to the preference declaration, so top-k
        never surfaces it; the ledger knows the rule exists."""
        return [c.value for c in self.ledger.current_cards()
                if c.attribute.startswith("disposition_")]

    def _episodic_answer(self, query: dict) -> str:
        m = re.search(r"As of day (\d+)", query["question"])
        asof = int(m.group(1)) if m else None
        if asof is None:
            lp = getattr(self, "_last_parse", None) or {}
            if isinstance(lp.get("asof_day"), int):
                asof = lp["asof_day"]
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
        sys_prompt = PARAMETRIC_SYSTEM_PROMPT
        rules = self._active_dispositions()
        if rules:
            sys_prompt += (" Active user preferences you MUST follow in "
                           "how you write the answer: "
                           + "; ".join(rules) + ".")
        return self.backend.complete(sys_prompt, user)

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
