"""Life generator: simulates an assistant-visible "life" and derives
ground-truth queries from the resulting temporal world state.

Anti-LoCoMo principles:
- answers derive from a programmatic world state, never hand annotation;
- every canonical value is asserted to appear verbatim in episode text;
- freshness queries carry explicit rejected values (superseded versions);
- an oracle reading the world state must score 100% by construction.
"""
from __future__ import annotations

import random
import re
from typing import Dict, List, Optional, Tuple

from . import banks
from . import templates as T
from .schema import (Entity, Episode, GenConfig, Query,
                     STABLE, UPDATABLE, VOLATILE, REVOCABLE,
                     QT_CURRENT_STABLE, QT_CURRENT_UPDATED,
                     QT_CURRENT_VOLATILE, QT_POINT_IN_TIME, QT_MULTIHOP_2,
                     QT_MULTIHOP_3, QT_REVOKED, QT_ABSENT, QT_ONLINE)
from .world import World

# tuple, not set: iterated during generation, so order must not depend on
# the interpreter's hash seed
VOLATILE_ATTRS = ("parking_spot", "desk_booking")
STABLE_ATTRS = {"birthday", "hometown", "client"}
TERMINAL_STATUS = {"completed", "cancelled"}
PERSON_STATE_ATTRS = ["birthday", "hometown", "favorite_restaurant"]
USER_FORCED_ATTRS = ["parking_spot", "desk_booking", "dietary_restriction",
                     "gym_day", "wifi_password", "favorite_coffee"]

Q_CURRENT = {
    ("person", "employer"): "What is {name}'s current employer?",
    ("person", "lives_in"): "In which city does {name} currently live?",
    ("person", "birthday"): "What is {name}'s birthday?",
    ("person", "hometown"): "What is {name}'s hometown?",
    ("person", "favorite_restaurant"): "What is {name}'s favorite restaurant?",
    ("person", "works_on"): "Which project does {name} currently work on?",
    ("project", "lead"): "Who is the current lead of {name}?",
    ("project", "deadline"): "What is the current deadline of {name}?",
    ("project", "budget"): "What is the current budget of {name}?",
    ("project", "status"): "What is the current status of {name}?",
    ("project", "client"): "Who is the client of {name}?",
    ("user", "parking_spot"): "What is the user's current parking spot?",
    ("user", "desk_booking"): "Which desk does the user currently have booked?",
    ("user", "gym_day"): "What is the user's current gym day?",
    ("user", "dietary_restriction"):
        "What is the user's current dietary restriction, if any?",
    ("user", "wifi_password"): "What is the current home wifi password?",
    ("user", "favorite_coffee"): "What is the user's usual coffee order?",
}

Q_AT_DAY = {
    ("person", "employer"): "As of day {d}, what was {name}'s employer?",
    ("person", "lives_in"): "As of day {d}, in which city did {name} live?",
    ("person", "favorite_restaurant"):
        "As of day {d}, what was {name}'s favorite restaurant?",
    ("person", "works_on"):
        "As of day {d}, which project was {name} working on?",
    ("project", "lead"): "As of day {d}, who was the lead of {name}?",
    ("project", "deadline"): "As of day {d}, what was the deadline of {name}?",
    ("project", "budget"): "As of day {d}, what was the budget of {name}?",
    ("project", "status"): "As of day {d}, what was the status of {name}?",
    ("user", "parking_spot"):
        "As of day {d}, at which spot was the user's car parked?",
    ("user", "desk_booking"):
        "As of day {d}, which desk did the user have booked?",
    ("user", "gym_day"): "As of day {d}, what was the user's gym day?",
    ("user", "wifi_password"):
        "As of day {d}, what was the home wifi password?",
    ("user", "favorite_coffee"):
        "As of day {d}, what was the user's usual coffee order?",
}


def boundary_found(value: str, text: str) -> bool:
    return re.search(r"(?<!\w)" + re.escape(value.lower()) + r"(?!\w)",
                     text.lower()) is not None


class LifeGenerator:
    def __init__(self, cfg: GenConfig):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self.world = World()
        self.stream: List[dict] = []
        self.episodes: List[Episode] = []
        self.online_queries: List[Query] = []
        self.final_queries: List[Query] = []
        self.usage: Dict[str, int] = {}
        self.warnings: List[str] = []
        self.seq = 0
        self.ep_counter = 0
        self.q_counter = 0
        self.hot_set: set = set()
        self.persons_live: List[Entity] = []
        self.projects_live: List[Entity] = []
        self.pending_states: List[Tuple[str, str]] = []
        self.pending_forced: List[Tuple[str, str]] = []
        self.never_assign: set = set()
        self.forced: Dict[int, List[tuple]] = {}
        self.volatile_next: Dict[str, int] = {}
        self._init_entities_and_plans()

    # ------------------------------------------------------------------ setup

    def _init_entities_and_plans(self) -> None:
        cfg, rng = self.cfg, self.rng
        self.world.add_entity(Entity(id="user", kind="user", name="the user"))

        persons = banks.PERSONS[: cfg.n_persons]
        self.persons = []
        for i, (name, gender) in enumerate(persons):
            e = Entity(id=f"p{i:02d}", kind="person", name=name, gender=gender)
            self.world.add_entity(e)
            self.persons.append(e)

        self.projects = []
        for i, name in enumerate(banks.PROJECTS[: cfg.n_projects]):
            e = Entity(id=f"j{i:02d}", kind="project", name=name)
            self.world.add_entity(e)
            self.projects.append(e)

        # value pools (shuffled copies; popped without replacement)
        deadline_bank = [f"{m} {d}" for m in banks.MONTHS
                         for d in (5, 9, 14, 18, 22, 26)]
        birthday_bank = [f"{m} {d}" for m in banks.MONTHS
                         for d in (3, 8, 13, 17, 21, 27)]
        budget_bank = [f"${n}k" for n in range(40, 400, 5)]
        spot_bank = [f"{c}{n}" for c in "ABCDEF" for n in range(10, 100)]
        desk_bank = [f"Desk {n}" for n in range(10, 80)]
        self._banks = {
            "employer": banks.COMPANIES, "client": banks.COMPANIES,
            "lives_in": banks.CITIES, "hometown": banks.CITIES,
            "favorite_restaurant": banks.RESTAURANTS,
            "birthday": birthday_bank,
            "deadline": deadline_bank, "budget": budget_bank,
            "parking_spot": spot_bank, "desk_booking": desk_bank,
            "gym_day": banks.WEEKDAYS, "dietary_restriction": banks.DIETARY,
            "wifi_password": banks.WIFI_PASSWORDS,
            "favorite_coffee": banks.COFFEES,
        }
        self.pools = {}
        for attr, bank in self._banks.items():
            pool = list(bank)
            rng.shuffle(pool)
            self.pools[attr] = pool

        # facts deliberately never assigned (targets of absent queries)
        na = []
        pp = list(self.persons)
        rng.shuffle(pp)
        for p in pp[:3]:
            na.append((p.id, "birthday"))
        for p in pp[3:5]:
            na.append((p.id, "hometown"))
        for p in pp[5:7]:
            na.append((p.id, "favorite_restaurant"))
        jj = list(self.projects)
        rng.shuffle(jj)
        for j in jj[: max(1, cfg.n_projects // 4)]:
            na.append((j.id, "budget"))
        self.never_assign = set(na)

        # introduction queue: two persons first (projects need a lead)
        rest = self.persons[2:] + self.projects
        rng.shuffle(rest)
        self.intro_queue = self.persons[:2] + rest

        # person state facts (skipping never_assign)
        for p in self.persons:
            for attr in PERSON_STATE_ATTRS:
                if (p.id, attr) not in self.never_assign and rng.random() < 0.85:
                    self.pending_states.append((p.id, attr))
        # project state facts (deadline, budget)
        for j in self.projects:
            self.pending_states.append((j.id, "deadline"))
            if (j.id, "budget") not in self.never_assign:
                self.pending_states.append((j.id, "budget"))
        rng.shuffle(self.pending_states)

        # forced user facts early in the life
        self.forced.setdefault(1, []).append(("state", "user", "parking_spot"))
        self.forced.setdefault(2, []).append(("state", "user", "desk_booking"))
        for attr, lo, hi in [("dietary_restriction", 3, 6), ("gym_day", 3, 8),
                             ("wifi_password", 4, 9), ("favorite_coffee", 5, 10)]:
            self.forced.setdefault(rng.randint(lo, hi), []).append(
                ("state", "user", attr))

        # revocations & completion in the second half
        self.forced.setdefault(
            int(cfg.n_days * rng.uniform(0.55, 0.75)), []).append(
            ("revoke_diet",))
        cancel_count = 1 if cfg.n_projects <= 4 else 2
        cancel_targets = rng.sample(self.projects, cancel_count)
        for j in cancel_targets:
            d = int(cfg.n_days * rng.uniform(0.65, 0.85))
            self.forced.setdefault(d, []).append(("cancel", j.id))
        if cfg.n_projects >= 5:
            remaining = [j for j in self.projects if j not in cancel_targets]
            self.forced.setdefault(int(cfg.n_days * 0.9), []).append(
                ("complete", rng.choice(remaining).id))

    # ------------------------------------------------------------ small utils

    def _pron(self, ent: Entity) -> Tuple[str, str]:
        return T.PRONOUNS.get(ent.gender, T.PRONOUNS[""])

    def _kwargs(self, ent: Entity, value: str = "", old: str = "",
                **extra) -> dict:
        sub, poss = self._pron(ent)
        kw = dict(name=ent.name, value=value, old=old, sub=sub,
                  sub_cap=sub.capitalize(), poss=poss)
        kw.update(extra)
        return kw

    def _new_value(self, attr: str, avoid: set) -> Optional[str]:
        pool = self.pools[attr]
        for i in range(len(pool) - 1, -1, -1):
            if pool[i] not in avoid:
                return pool.pop(i)
        candidates = [v for v in self._banks[attr] if v not in avoid]
        if not candidates:
            return None
        return self.rng.choice(candidates)

    def _emit_episode(self, day: int, kind: str, text: str,
                      fact_keys: List[str]) -> Episode:
        ep = Episode(episode_id=f"e{self.ep_counter:05d}", seq=self.seq,
                     day=day, text=f"Day {day}: {text}", kind=kind,
                     fact_keys=list(fact_keys))
        self.ep_counter += 1
        self.seq += 1
        self.episodes.append(ep)
        row = {"type": "episode"}
        row.update(ep.__dict__)
        self.stream.append(row)
        return ep

    def _next_ep_id(self) -> str:
        return f"e{self.ep_counter:05d}"

    def _assert_verbalized(self, ep: Episode, values: List[str]) -> None:
        for v in values:
            if not boundary_found(v, ep.text):
                raise AssertionError(
                    f"value {v!r} not verbalized in {ep.episode_id}: {ep.text!r}")

    def _entity(self, subject: str) -> Entity:
        return self.world.entities[subject]

    def _project_terminal(self, project: Entity) -> bool:
        cur = self.world.current(f"{project.id}.status", self.cfg.n_days)
        return cur is not None and cur.value in TERMINAL_STATUS

    def _project_terminal_at(self, project_id: str, day: int) -> bool:
        cur = self.world.value_at(f"{project_id}.status", day)
        return cur is not None and cur.value in TERMINAL_STATUS

    def _active_projects(self, day: int) -> List[Entity]:
        return [j for j in self.projects_live
                if not self._project_terminal_at(j.id, day)]

    def _variants(self, attr: str, value: str) -> List[str]:
        if attr == "works_on" and value.startswith("Project "):
            return [value, value[len("Project "):]]
        return banks.accepted_variants(attr, value)

    def _expand(self, attr: str, values: List[str]) -> List[str]:
        out = []
        for v in values:
            for x in self._variants(attr, v):
                if x.lower() not in [o.lower() for o in out]:
                    out.append(x)
        return out

    # ------------------------------------------------------------- life loop

    def run(self) -> "LifeGenerator":
        for day in range(1, self.cfg.n_days + 1):
            self._do_day(day)
        self._build_final_queries()
        return self

    def _do_day(self, day: int) -> None:
        rng = self.rng
        # forced reassignments queued by cancellations
        while self.pending_forced:
            subject, attr = self.pending_forced.pop(0)
            self._do_update(day, force_key=(subject, attr))
        for item in self.forced.get(day, []):
            if item[0] == "state":
                self._do_state(day, force=(item[1], item[2]))
            elif item[0] == "revoke_diet":
                self._do_revoke_diet(day)
            elif item[0] == "cancel":
                self._do_cancel(day, item[1])
            elif item[0] == "complete":
                self._do_complete(day, item[1])
        # volatile timers
        for attr in VOLATILE_ATTRS:
            key = f"user.{attr}"
            if key in self.world.facts and day >= self.volatile_next.get(key, 0):
                self._do_update(day, force_key=("user", attr))
                self.volatile_next[key] = day + rng.randint(5, 9)

        n_events = rng.randint(self.cfg.events_min, self.cfg.events_max)
        for _ in range(n_events):
            actions, weights = [], []
            if self.intro_queue:
                actions.append("introduce"); weights.append(0.5)
            if self.pending_states:
                actions.append("state"); weights.append(0.3)
            actions.append("update"); weights.append(0.25)
            actions.append("mention"); weights.append(0.12)
            actions.append("noise"); weights.append(0.22)
            if self.hot_set and day > self.cfg.hot_pick_day:
                actions.append("online"); weights.append(0.20)
            action = rng.choices(actions, weights=weights, k=1)[0]
            if action == "introduce":
                self._do_introduce(day)
            elif action == "state":
                self._do_state(day)
            elif action == "update":
                self._do_update(day)
            elif action == "mention":
                self._do_mention(day)
            elif action == "online":
                self._do_online_query(day)
            else:
                self._do_noise(day)

        if day == self.cfg.hot_pick_day:
            self._pick_hot_set()

    def _pick_hot_set(self) -> None:
        keys = sorted(self.world.facts)
        k = max(3, int(len(keys) * self.cfg.hot_fraction))
        chosen = set(self.rng.sample(keys, min(k, len(keys))))
        for attr in VOLATILE_ATTRS:
            if f"user.{attr}" in self.world.facts:
                chosen.add(f"user.{attr}")
        self.hot_set = chosen

    # ----------------------------------------------------------- event kinds

    def _do_introduce(self, day: int) -> None:
        ent = self.intro_queue[0]
        if ent.kind == "project" and not self.persons_live:
            return  # a project needs an introduced person as lead
        self.intro_queue.pop(0)
        ep_id = self._next_ep_id()
        if ent.kind == "person":
            employer = self._new_value("employer", set())
            lives_in = self._new_value("lives_in", set())
            self.world.set_value(ent.id, "employer", employer, day, ep_id,
                                 UPDATABLE)
            self.world.set_value(ent.id, "lives_in", lives_in, day, ep_id,
                                 UPDATABLE)
            text = self.rng.choice(T.INTRODUCE_PERSON).format(
                **self._kwargs(ent, employer=employer, lives_in=lives_in))
            ep = self._emit_episode(day, "introduce", text,
                                    [f"{ent.id}.employer",
                                     f"{ent.id}.lives_in"])
            self._assert_verbalized(ep, [employer, lives_in])
            self.persons_live.append(ent)
            if self.projects_live and self.rng.random() < 0.7:
                self.pending_states.append((ent.id, "works_on"))
        else:
            client = self._new_value("client", set())
            lead = self.rng.choice(self.persons_live)
            self.world.set_value(ent.id, "client", client, day, ep_id, STABLE)
            self.world.set_value(ent.id, "lead", lead.name, day, ep_id,
                                 UPDATABLE)
            self.world.set_value(ent.id, "status", "active", day, ep_id,
                                 UPDATABLE)
            text = self.rng.choice(T.INTRODUCE_PROJECT).format(
                name=ent.name, client=client, lead=lead.name)
            ep = self._emit_episode(day, "introduce", text,
                                    [f"{ent.id}.client", f"{ent.id}.lead",
                                     f"{ent.id}.status"])
            self._assert_verbalized(ep, [client, lead.name, "active"])
            self.projects_live.append(ent)
            for p in self.persons_live:
                if (not self.world.has_fact(p.id, "works_on")
                        and (p.id, "works_on") not in self.pending_states
                        and self.rng.random() < 0.7):
                    self.pending_states.append((p.id, "works_on"))

    def _do_state(self, day: int,
                  force: Optional[Tuple[str, str]] = None) -> None:
        rng = self.rng
        if force is not None:
            subject, attr = force
        else:
            if not self.pending_states:
                return
            idx = rng.randrange(len(self.pending_states))
            subject, attr = self.pending_states.pop(idx)
        ent = self._entity(subject)
        if self.world.has_fact(subject, attr):
            return
        if attr == "works_on":
            active = self._active_projects(day)
            if not active:
                self.pending_states.append((subject, attr))
                return
            value = rng.choice(active).name
        else:
            value = self._new_value(attr, set())
            if value is None:
                self.warnings.append(f"pool exhausted for {attr}")
                return
        if attr in VOLATILE_ATTRS:
            dynamic = VOLATILE
            self.volatile_next[f"{subject}.{attr}"] = day + rng.randint(5, 9)
        elif attr in STABLE_ATTRS:
            dynamic = STABLE
        elif attr == "dietary_restriction":
            dynamic = REVOCABLE
        else:
            dynamic = UPDATABLE
        ep_id = self._next_ep_id()
        self.world.set_value(subject, attr, value, day, ep_id, dynamic)
        text = rng.choice(T.STATE[(ent.kind, attr)]).format(
            **self._kwargs(ent, value=value))
        ep = self._emit_episode(day, "state", text, [f"{subject}.{attr}"])
        self._assert_verbalized(ep, [value])

    def _update_candidates(self, day: int) -> List[str]:
        out = []
        for key, fact in self.world.facts.items():
            if fact.dynamic not in (UPDATABLE,):
                continue
            if fact.attribute in STABLE_ATTRS:
                continue
            last = fact.versions[-1]
            if last.revoked or last.day > day - 2:
                continue
            if len(fact.versions) >= 6:
                continue
            ent = self._entity(fact.subject)
            if ent.kind == "project" and self._project_terminal_at(ent.id, day):
                continue
            if fact.attribute == "status" and last.value in TERMINAL_STATUS:
                continue
            if (ent.kind, fact.attribute) not in T.UPDATE:
                continue
            out.append(key)
        return sorted(out)

    def _do_update(self, day: int,
                   force_key: Optional[Tuple[str, str]] = None) -> None:
        rng = self.rng
        if force_key is not None:
            subject, attr = force_key
            key = f"{subject}.{attr}"
            if key not in self.world.facts:
                return
        else:
            candidates = self._update_candidates(day)
            if not candidates:
                return
            key = rng.choice(candidates)
            subject, attr = key.split(".", 1)
        fact = self.world.facts[key]
        old = fact.versions[-1].value
        ent = self._entity(subject)
        if attr == "works_on":
            options = [j.name for j in self._active_projects(day)
                       if j.name != old]
            if not options:
                self.warnings.append(
                    f"no active project to reassign {ent.name}")
                return
            value = rng.choice(options)
        elif attr == "lead":
            options = [p.name for p in self.persons_live if p.name != old]
            if not options:
                return
            value = rng.choice(options)
        elif attr == "status":
            value = "paused" if old == "active" else "active"
        else:
            avoid = {v.value for v in fact.versions}
            value = self._new_value(attr, avoid)
            if value is None:
                self.warnings.append(f"pool exhausted for {attr}")
                return
        ep_id = self._next_ep_id()
        self.world.set_value(subject, attr, value, day, ep_id, fact.dynamic)
        text = rng.choice(T.UPDATE[(ent.kind, attr)]).format(
            **self._kwargs(ent, value=value, old=old))
        ep = self._emit_episode(day, "update", text, [key])
        self._assert_verbalized(ep, [value])

    def _do_mention(self, day: int) -> None:
        rng = self.rng
        candidates = []
        for key, fact in self.world.facts.items():
            ent = self._entity(fact.subject)
            if (ent.kind, fact.attribute) not in T.MENTION:
                continue
            last = fact.versions[-1]
            if last.revoked:
                continue
            if ent.kind == "project" and self._project_terminal_at(ent.id, day):
                continue
            candidates.append(key)
        if not candidates:
            return
        key = rng.choice(sorted(candidates))
        fact = self.world.facts[key]
        ent = self._entity(fact.subject)
        value = fact.versions[-1].value
        text = rng.choice(T.MENTION[(ent.kind, fact.attribute)]).format(
            **self._kwargs(ent, value=value, lead=value))
        self._emit_episode(day, "mention", text, [key])

    def _do_noise(self, day: int) -> None:
        text = self.rng.choice(banks.NOISE_SENTENCES)
        self._emit_episode(day, "noise", text, [])

    def _do_revoke_diet(self, day: int) -> None:
        key = "user.dietary_restriction"
        if key not in self.world.facts:
            self.warnings.append("dietary never set; revoke skipped")
            return
        fact = self.world.facts[key]
        if fact.versions[-1].revoked:
            return
        old = fact.versions[-1].value
        ep_id = self._next_ep_id()
        self.world.set_value("user", "dietary_restriction", "none", day,
                             ep_id, REVOCABLE, revoked=True)
        text = self.rng.choice(T.REVOKE[("user", "dietary_restriction")]).format(
            old=old)
        self._emit_episode(day, "revoke", text, [key])

    def _terminalize(self, day: int, project_id: str, status: str) -> None:
        ent = self._entity(project_id)
        key = f"{project_id}.status"
        if key not in self.world.facts:
            return
        old = self.world.facts[key].versions[-1].value
        if old in TERMINAL_STATUS:
            return
        ep_id = self._next_ep_id()
        revoked = status == "cancelled"
        self.world.set_value(project_id, "status", status, day, ep_id,
                             UPDATABLE, revoked=revoked)
        if revoked:
            text = self.rng.choice(T.REVOKE[("project", "status")]).format(
                name=ent.name)
        else:
            text = self.rng.choice(T.UPDATE[("project", "status")]).format(
                name=ent.name, value=status, old=old)
        ep = self._emit_episode(day, "revoke" if revoked else "update",
                                text, [key])
        self._assert_verbalized(ep, [status])
        # people on this project get reassigned over the next days
        for p in self.persons_live:
            cur = self.world.value_at(f"{p.id}.works_on", day)
            if cur is not None and cur.value == ent.name:
                self.pending_forced.append((p.id, "works_on"))

    def _do_cancel(self, day: int, project_id: str) -> None:
        self._terminalize(day, project_id, "cancelled")

    def _do_complete(self, day: int, project_id: str) -> None:
        self._terminalize(day, project_id, "completed")

    # -------------------------------------------------------- online queries

    def _mk_query(self, qtype: str, question_base: str, attr: str,
                  fact_keys: List[str], day_asked: int, answer_value: str,
                  rejected_values: List[str], hops: int = 1,
                  n_updates: int = 0, hint_attr: Optional[str] = None,
                  accepted_override: Optional[List[str]] = None,
                  hint_override: Optional[str] = None) -> Query:
        hint = hint_override or T.ANSWER_HINTS[hint_attr or attr]
        question = f"{question_base} {hint} {T.UNKNOWN_SUFFIX}"
        if accepted_override is not None:
            accepted = list(accepted_override)
        else:
            accepted = self._expand(attr, [answer_value])
        acc_lower = {a.lower() for a in accepted}
        rejected = [r for r in self._expand(attr, rejected_values)
                    if r.lower() not in acc_lower]
        heat = "hot" if all(k in self.hot_set for k in fact_keys) else "cold"
        q = Query(query_id=f"q{self.q_counter:04d}", question=question,
                  qtype=qtype, day_asked=day_asked, hops=hops,
                  fact_keys=list(fact_keys), answer_display=answer_value,
                  accepted=accepted, rejected=rejected, heat=heat,
                  n_updates_before=n_updates)
        self.q_counter += 1
        return q

    def _do_online_query(self, day: int) -> None:
        rng = self.rng
        for _ in range(5):
            key = rng.choice(sorted(self.hot_set))
            fact = self.world.facts.get(key)
            if fact is None:
                continue
            ent = self._entity(fact.subject)
            if (ent.kind, fact.attribute) not in Q_CURRENT:
                continue
            version = self.world.value_at(key, day)
            if version is None or version.revoked:
                continue
            if ent.kind == "project" and self._project_terminal_at(ent.id, day):
                continue
            base = Q_CURRENT[(ent.kind, fact.attribute)].format(name=ent.name)
            history = [v.value for v in fact.versions if v.day <= day]
            n_updates = len(history) - 1
            q = self._mk_query(QT_ONLINE, base, fact.attribute, [key], day,
                               version.value, history[:-1],
                               n_updates=n_updates)
            q.seq = self.seq
            self.seq += 1
            self.online_queries.append(q)
            self.usage[key] = self.usage.get(key, 0) + 1
            row = {"type": "query"}
            row.update(q.public())
            self.stream.append(row)
            return

    # --------------------------------------------------------- final queries

    def _twin_value(self, ent: Entity, attr: str) -> List[str]:
        """Current value of the same attribute on the same-first-name twin."""
        first = ent.name.split()[0]
        out = []
        for p in self.persons:
            if p.id != ent.id and p.name.split()[0] == first:
                cur = self.world.current(f"{p.id}.{attr}", self.cfg.n_days)
                if cur is not None and not cur.revoked:
                    out.append(cur.value)
        return out

    def _take(self, qtype: str, candidates: List[Query]) -> None:
        quota = self.cfg.quotas.get(qtype, 0)
        self.rng.shuffle(candidates)
        got = candidates[:quota]
        if len(got) < quota:
            self.warnings.append(
                f"quota {qtype}: wanted {quota}, got {len(got)}")
        self.final_queries.extend(got)

    def _build_final_queries(self) -> None:
        F = self.cfg.n_days + 1
        w = self.world

        # -- current_stable
        cands = []
        for key in sorted(w.facts):
            fact = w.facts[key]
            if fact.attribute not in STABLE_ATTRS:
                continue
            ent = self._entity(fact.subject)
            value = fact.versions[-1].value
            if ent.kind == "person":
                rejected = self._twin_value(ent, fact.attribute)
            else:  # client
                rejected = [w.facts[k].versions[-1].value
                            for k in sorted(w.facts)
                            if k.endswith(".client") and k != key]
            base = Q_CURRENT[(ent.kind, fact.attribute)].format(name=ent.name)
            cands.append(self._mk_query(QT_CURRENT_STABLE, base,
                                        fact.attribute, [key], F, value,
                                        rejected))
        self._take(QT_CURRENT_STABLE, cands)

        # -- current_updated
        cands = []
        for key in sorted(w.facts):
            fact = w.facts[key]
            ent = self._entity(fact.subject)
            if (fact.dynamic != UPDATABLE or fact.attribute in STABLE_ATTRS
                    or len(fact.versions) < 2 or fact.versions[-1].revoked):
                continue
            if ent.kind == "project" and self._project_terminal(ent):
                continue
            if (ent.kind, fact.attribute) not in Q_CURRENT:
                continue
            value = fact.versions[-1].value
            if value in [v.value for v in fact.versions[:-1]]:
                # cyclic value (e.g. active->paused->active): a stale answer
                # would be indistinguishable from a fresh one — not a valid
                # freshness probe
                continue
            rejected = [v.value for v in fact.versions[:-1]]
            if ent.kind == "person":
                rejected += self._twin_value(ent, fact.attribute)
            base = Q_CURRENT[(ent.kind, fact.attribute)].format(name=ent.name)
            cands.append(self._mk_query(QT_CURRENT_UPDATED, base,
                                        fact.attribute, [key], F, value,
                                        rejected,
                                        n_updates=len(fact.versions) - 1))
        self._take(QT_CURRENT_UPDATED, cands)

        # -- current_volatile
        cands = []
        for attr in sorted(VOLATILE_ATTRS):
            key = f"user.{attr}"
            fact = w.facts.get(key)
            if fact is None:
                continue
            value = fact.versions[-1].value
            rejected = [v.value for v in fact.versions[:-1]]
            base = Q_CURRENT[("user", attr)]
            cands.append(self._mk_query(QT_CURRENT_VOLATILE, base, attr,
                                        [key], F, value, rejected,
                                        n_updates=len(fact.versions) - 1))
        self._take(QT_CURRENT_VOLATILE, cands)

        # -- point_in_time (only versions that are no longer current)
        cands = []
        for key in sorted(w.facts):
            fact = w.facts[key]
            ent = self._entity(fact.subject)
            if len(fact.versions) < 2:
                continue
            if (ent.kind, fact.attribute) not in Q_AT_DAY:
                continue
            per_fact = 0
            for i in range(len(fact.versions) - 1):
                if per_fact >= 2:
                    break
                v, nxt = fact.versions[i], fact.versions[i + 1]
                if v.revoked:
                    continue
                hi = min(nxt.day - 1, self.cfg.n_days)
                if hi < v.day:
                    continue
                d = self.rng.randint(v.day, hi)
                rejected = [x.value for j, x in enumerate(fact.versions)
                            if j != i and not x.revoked]
                base = Q_AT_DAY[(ent.kind, fact.attribute)].format(
                    d=d, name=ent.name)
                cands.append(self._mk_query(QT_POINT_IN_TIME, base,
                                            fact.attribute, [key], F, v.value,
                                            rejected, n_updates=i))
                per_fact += 1
        self._take(QT_POINT_IN_TIME, cands)

        # -- multihop
        mh2, mh3 = [], []
        gap = self.cfg.multihop_min_gap
        for j in self.projects_live:
            if self._project_terminal(j):
                continue
            lead_v = w.current(f"{j.id}.lead", self.cfg.n_days)
            if lead_v is None:
                continue
            lead = next((p for p in self.persons if p.name == lead_v.value),
                        None)
            if lead is None:
                continue
            # 2-hop: lead -> lives_in
            city_v = w.current(f"{lead.id}.lives_in", self.cfg.n_days)
            if city_v is not None:
                days = [lead_v.day, city_v.day]
                eps = {lead_v.episode_id, city_v.episode_id}
                if len(eps) == 2 and max(days) - min(days) >= gap:
                    rejected = [x.value for x in
                                w.history(f"{lead.id}.lives_in")[:-1]]
                    for old in w.history(f"{j.id}.lead")[:-1]:
                        oldp = next((p for p in self.persons
                                     if p.name == old.value), None)
                        if oldp:
                            c = w.current(f"{oldp.id}.lives_in",
                                          self.cfg.n_days)
                            if c:
                                rejected.append(c.value)
                    base = (f"In which city does the current lead of "
                            f"{j.name} live?")
                    mh2.append(self._mk_query(
                        QT_MULTIHOP_2, base, "lives_in",
                        [f"{j.id}.lead", f"{lead.id}.lives_in"], F,
                        city_v.value, rejected, hops=2))
            # 3-hop: person -> works_on -> lead -> hometown
            home_v = w.current(f"{lead.id}.hometown", self.cfg.n_days)
            if home_v is not None:
                for p in self.persons_live:
                    won = w.current(f"{p.id}.works_on", self.cfg.n_days)
                    if won is None or won.value != j.name or p.id == lead.id:
                        continue
                    days = [won.day, lead_v.day, home_v.day]
                    eps = {won.episode_id, lead_v.episode_id,
                           home_v.episode_id}
                    if len(eps) == 3 and max(days) - min(days) >= gap:
                        rejected = []
                        own = w.current(f"{p.id}.hometown", self.cfg.n_days)
                        if own:
                            rejected.append(own.value)
                        for old in w.history(f"{j.id}.lead")[:-1]:
                            oldp = next((x for x in self.persons
                                         if x.name == old.value), None)
                            if oldp:
                                c = w.current(f"{oldp.id}.hometown",
                                              self.cfg.n_days)
                                if c:
                                    rejected.append(c.value)
                        base = (f"What is the hometown of the current lead "
                                f"of the project that {p.name} currently "
                                f"works on?")
                        mh3.append(self._mk_query(
                            QT_MULTIHOP_3, base, "hometown",
                            [f"{p.id}.works_on", f"{j.id}.lead",
                             f"{lead.id}.hometown"], F, home_v.value,
                            rejected, hops=3))
        # 2-hop: person -> works_on -> deadline
        for p in self.persons_live:
            won = w.current(f"{p.id}.works_on", self.cfg.n_days)
            if won is None:
                continue
            j = next((x for x in self.projects if x.name == won.value), None)
            if j is None or self._project_terminal(j):
                continue
            dl = w.current(f"{j.id}.deadline", self.cfg.n_days)
            if dl is None:
                continue
            days = [won.day, dl.day]
            eps = {won.episode_id, dl.episode_id}
            if len(eps) == 2 and max(days) - min(days) >= gap:
                rejected = [x.value for x in
                            w.history(f"{j.id}.deadline")[:-1]]
                for old in w.history(f"{p.id}.works_on")[:-1]:
                    oldj = next((x for x in self.projects
                                 if x.name == old.value), None)
                    if oldj:
                        c = w.current(f"{oldj.id}.deadline", self.cfg.n_days)
                        if c:
                            rejected.append(c.value)
                base = (f"What is the deadline of the project {p.name} "
                        f"currently works on?")
                mh2.append(self._mk_query(
                    QT_MULTIHOP_2, base, "deadline",
                    [f"{p.id}.works_on", f"{j.id}.deadline"], F, dl.value,
                    rejected, hops=2))
        self._take(QT_MULTIHOP_2, mh2)
        self._take(QT_MULTIHOP_3, mh3)

        # -- revoked
        cands = []
        for j in self.projects:
            cur = w.current(f"{j.id}.status", self.cfg.n_days)
            if cur is not None and cur.value == "cancelled":
                base = Q_CURRENT[("project", "status")].format(name=j.name)
                rejected = [x.value for x in
                            w.history(f"{j.id}.status")[:-1]]
                cands.append(self._mk_query(QT_REVOKED, base, "status",
                                            [f"{j.id}.status"], F,
                                            "cancelled", rejected))
        diet = w.facts.get("user.dietary_restriction")
        if diet is not None and diet.versions[-1].revoked:
            old = diet.versions[0].value
            cands.append(self._mk_query(
                QT_REVOKED, Q_CURRENT[("user", "dietary_restriction")],
                "dietary_restriction", ["user.dietary_restriction"], F,
                "none", [old],
                accepted_override=["none", "no restriction",
                                   "no dietary restriction",
                                   "no restrictions"]))
            cands.append(self._mk_query(
                QT_REVOKED,
                "Does the user currently have any dietary restriction?",
                "dietary_restriction", ["user.dietary_restriction"], F,
                "no", ["yes", old],
                accepted_override=["no"],
                hint_override="Answer 'yes' or 'no'."))
        self._take(QT_REVOKED, cands)

        # -- absent
        cands = []
        for subject, attr in sorted(self.never_assign):
            ent = self._entity(subject)
            key = f"{subject}.{attr}"
            if key in w.facts:
                raise AssertionError(f"never_assign fact {key} was assigned")
            rejected = []
            for other in (self.persons if ent.kind == "person"
                          else self.projects):
                if other.id == subject:
                    continue
                cur = w.current(f"{other.id}.{attr}", self.cfg.n_days)
                if cur is not None and not cur.revoked:
                    rejected.append(cur.value)
            base = Q_CURRENT[(ent.kind, attr)].format(name=ent.name)
            cands.append(self._mk_query(
                QT_ABSENT, base, attr, [key], F, "unknown", rejected,
                accepted_override=list(banks.UNKNOWN_MARKERS)))
        self._take(QT_ABSENT, cands)

    # -------------------------------------------------------------- manifest

    def manifest(self) -> dict:
        by_type: Dict[str, int] = {}
        for q in self.final_queries:
            by_type[q.qtype] = by_type.get(q.qtype, 0) + 1
        words = sum(len(e.text.split()) for e in self.episodes)
        return {
            "config": self.cfg.__dict__,
            "counts": {
                "episodes": len(self.episodes),
                "online_queries": len(self.online_queries),
                "final_queries": len(self.final_queries),
                "final_by_type": by_type,
                "facts": len(self.world.facts),
            },
            "est_tokens": int(words * 1.3),
            "hot_set": sorted(self.hot_set),
            "warnings": self.warnings,
        }
