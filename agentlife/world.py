"""Temporal world state: the single source of truth answers derive from."""
from __future__ import annotations

from typing import Dict, List, Optional

from .schema import Entity, Fact, FactVersion


class World:
    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.facts: Dict[str, Fact] = {}  # key = "subject.attribute"

    def add_entity(self, entity: Entity) -> None:
        if entity.id in self.entities:
            raise ValueError(f"duplicate entity id {entity.id}")
        self.entities[entity.id] = entity

    def set_value(self, subject: str, attribute: str, value: str, day: int,
                  episode_id: str, dynamic: str, revoked: bool = False) -> Fact:
        key = f"{subject}.{attribute}"
        fact = self.facts.get(key)
        if fact is None:
            fact = Fact(subject=subject, attribute=attribute, dynamic=dynamic)
            self.facts[key] = fact
        if fact.versions and day < fact.versions[-1].day:
            raise ValueError(f"non-monotonic update for {key} at day {day}")
        fact.versions.append(
            FactVersion(value=value, day=day, episode_id=episode_id,
                        revoked=revoked))
        return fact

    def value_at(self, key: str, day: int) -> Optional[FactVersion]:
        """Latest version whose start day <= day (may be a revoked version)."""
        fact = self.facts.get(key)
        if fact is None:
            return None
        best = None
        for v in fact.versions:
            if v.day <= day:
                best = v
            else:
                break
        return best

    def current(self, key: str, n_days: int) -> Optional[FactVersion]:
        return self.value_at(key, n_days + 1)

    def history(self, key: str) -> List[FactVersion]:
        fact = self.facts.get(key)
        return list(fact.versions) if fact else []

    def has_fact(self, subject: str, attribute: str) -> bool:
        return f"{subject}.{attribute}" in self.facts

    def dump(self) -> List[dict]:
        rows = []
        for key in sorted(self.facts):
            f = self.facts[key]
            rows.append({
                "key": key,
                "subject": f.subject,
                "attribute": f.attribute,
                "dynamic": f.dynamic,
                "versions": [
                    {"value": v.value, "day": v.day,
                     "episode_id": v.episode_id, "revoked": v.revoked}
                    for v in f.versions
                ],
            })
        return rows
