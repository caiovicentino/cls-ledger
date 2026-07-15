"""Data model for AgentLife: entities, versioned facts, episodes, queries."""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import List, Optional


# Dynamics classes for facts
STABLE = "stable"        # never changes
UPDATABLE = "updatable"  # changes 1..k times over a life
VOLATILE = "volatile"    # changes on its own recurring timer
REVOCABLE = "revocable"  # can be explicitly revoked

# Query types
QT_CURRENT_STABLE = "current_stable"
QT_CURRENT_UPDATED = "current_updated"
QT_CURRENT_VOLATILE = "current_volatile"
QT_POINT_IN_TIME = "point_in_time"
QT_MULTIHOP_2 = "multihop_2"
QT_MULTIHOP_3 = "multihop_3"
QT_REVOKED = "revoked"
QT_ABSENT = "absent"
QT_ONLINE = "online"
QT_INDUCTION_COUNT = "induction_count"
QT_INDUCTION_HABIT = "induction_habit"
QT_INDUCTION_TREND = "induction_trend"


@dataclass
class Entity:
    id: str
    kind: str           # person | project | user
    name: str
    gender: str = ""    # 'm' | 'f' | '' (projects/user)


@dataclass
class FactVersion:
    value: str          # canonical display value ("" when revoked)
    day: int            # day this version became true
    episode_id: str     # episode that stated it
    revoked: bool = False


@dataclass
class Fact:
    subject: str        # entity id
    attribute: str
    dynamic: str        # STABLE | UPDATABLE | VOLATILE | REVOCABLE
    versions: List[FactVersion] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.subject}.{self.attribute}"


@dataclass
class Episode:
    episode_id: str
    seq: int            # global order in the stream
    day: int
    text: str
    kind: str           # introduce | state | update | revoke | noise
    fact_keys: List[str] = field(default_factory=list)


@dataclass
class Query:
    query_id: str
    question: str       # full question shown to systems, incl. answer hint
    qtype: str
    day_asked: int
    seq: int = -1       # position in stream (online queries only)
    hops: int = 1
    fact_keys: List[str] = field(default_factory=list)
    answer_display: str = ""            # canonical answer (private)
    accepted: List[str] = field(default_factory=list)   # private
    rejected: List[str] = field(default_factory=list)   # private
    heat: str = "cold"                  # hot | cold (usage during the life)
    n_updates_before: int = 0           # versions before day_asked, minus 1
    dispositions: List[str] = field(default_factory=list)  # active rules

    def public(self) -> dict:
        return {
            "query_id": self.query_id,
            "question": self.question,
            "qtype_public": "online" if self.qtype == QT_ONLINE else "final",
            "day_asked": self.day_asked,
            "seq": self.seq,
        }

    def private(self) -> dict:
        return dataclasses.asdict(self)


def dump_jsonl(path, items) -> None:
    with open(path, "w") as f:
        for it in items:
            row = dataclasses.asdict(it) if dataclasses.is_dataclass(it) else it
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path) -> List[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


@dataclass
class GenConfig:
    preset: str
    n_days: int
    n_persons: int
    n_projects: int
    events_min: int = 2
    events_max: int = 5
    hot_fraction: float = 0.3
    hot_pick_day: int = 15          # day the hot set is sampled
    multihop_min_gap: int = 5       # min days between hop source episodes
    quotas: dict = field(default_factory=dict)
    seed: int = 0


PRESETS = {
    "S": dict(n_days=30, n_persons=8, n_projects=4, hot_pick_day=8,
              quotas=dict(current_stable=8, current_updated=8,
                          current_volatile=2, point_in_time=10,
                          multihop_2=6, multihop_3=3, revoked=4, absent=6,
                          induction_count=4, induction_habit=2,
                          induction_trend=2)),
    "M": dict(n_days=90, n_persons=12, n_projects=6,
              quotas=dict(current_stable=16, current_updated=16,
                          current_volatile=2, point_in_time=20,
                          multihop_2=12, multihop_3=6, revoked=6, absent=12,
                          induction_count=8, induction_habit=3,
                          induction_trend=4)),
    "L": dict(n_days=365, n_persons=16, n_projects=8,
              quotas=dict(current_stable=24, current_updated=24,
                          current_volatile=2, point_in_time=30,
                          multihop_2=18, multihop_3=9, revoked=8, absent=16)),
    "XL": dict(n_days=1000, n_persons=16, n_projects=8,
               quotas=dict(current_stable=24, current_updated=32,
                           current_volatile=2, point_in_time=40,
                           multihop_2=18, multihop_3=9, revoked=8, absent=16)),
}


def make_config(preset: str, seed: int) -> GenConfig:
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choose from {sorted(PRESETS)}")
    return GenConfig(preset=preset, seed=seed, **PRESETS[preset])
