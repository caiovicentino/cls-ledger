"""The Ledger: fact cards with provenance, supersedence and usage.

This is the bookkeeping heart of CLS-Ledger. Every consolidated weight
update must be traceable back to cards, and every card to the episode that
stated it — that is what makes parametric memory auditable and reversible.
"""
from __future__ import annotations

import json
import re
from typing import Dict, List, Optional


def norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


class Card:
    def __init__(self, entity: str, attribute: str, value: str, day: int,
                 episode_id: str):
        self.entity = entity
        self.attribute = attribute
        self.value = value
        self.day = day
        self.episode_id = episode_id
        self.usage = 0
        self.superseded_by: Optional[str] = None

    @property
    def key(self) -> str:
        return f"{norm_key(self.entity)}.{norm_key(self.attribute)}"

    @property
    def card_id(self) -> str:
        return f"{self.key}@{self.day}.{self.episode_id}"

    def text(self) -> str:
        return (f"{self.entity} | {self.attribute.replace('_', ' ')} | "
                f"{self.value} (day {self.day})")

    def to_dict(self) -> dict:
        return {"card_id": self.card_id, "entity": self.entity,
                "attribute": self.attribute, "value": self.value,
                "day": self.day, "episode_id": self.episode_id,
                "usage": self.usage, "superseded_by": self.superseded_by}


class Ledger:
    def __init__(self):
        self.cards: List[Card] = []
        self._latest: Dict[str, Card] = {}  # key -> current card

    def add(self, card: Card) -> None:
        prev = self._latest.get(card.key)
        if prev is not None and card.day >= prev.day:
            prev.superseded_by = card.card_id
            card.usage += prev.usage  # usage follows the fact, not the copy
            self._latest[card.key] = card
        elif prev is None:
            self._latest[card.key] = card
        else:  # out-of-order older statement of a known fact
            card.superseded_by = prev.card_id
        self.cards.append(card)

    def current_cards(self) -> List[Card]:
        return [c for c in self.cards if c.superseded_by is None]

    def history(self, key: str) -> List[Card]:
        return sorted((c for c in self.cards if c.key == key),
                      key=lambda c: c.day)

    def churn(self, key: str) -> int:
        """Number of observed value changes for this fact."""
        return max(0, len(self.history(key)) - 1)

    def days_since_change(self, key: str, today: int) -> int:
        hist = self.history(key)
        return today - hist[-1].day if hist else 10 ** 9

    # ----------------------------------------------------------- selection

    def select_for_consolidation(self, today: int, policy: str = "stable",
                                 volatile_churn: int = 3,
                                 volatile_recent_days: int = 7,
                                 budget: Optional[int] = None) -> List[Card]:
        """Which current facts deserve to live in the weights?

        policy 'stable' (default): every current card except those that
        look volatile (changed >= volatile_churn times AND changed again
        recently) — high churn facts stay in episodic memory, where
        freshness is cheap.
        policy 'all': every current card (ablation).
        policy 'hot': only cards with observed usage > 0 (ablation for H1).

        budget: hard cap on consolidated cards — the capacity constraint
        that makes selection policies actually compete (H1 under budget).
        Ranking under budget: hot -> by usage; stable -> by settledness
        (days since change desc, churn asc); all -> deterministic random.
        """
        out = []
        for c in self.current_cards():
            if policy == "all":
                out.append(c)
                continue
            if policy == "hot":
                if c.usage > 0:
                    out.append(c)
                continue
            churn = self.churn(c.key)
            recent = self.days_since_change(c.key, today)
            if churn >= volatile_churn and recent < volatile_recent_days:
                continue  # volatile: keep episodic
            out.append(c)
        if budget is not None and len(out) > budget:
            if policy == "hot":
                out.sort(key=lambda c: (-c.usage,
                                        -self.days_since_change(c.key,
                                                                today),
                                        c.card_id))
            elif policy == "stable":
                out.sort(key=lambda c: (-self.days_since_change(c.key,
                                                                today),
                                        self.churn(c.key), c.card_id))
            else:
                import random
                rng = random.Random(0)
                out = sorted(out, key=lambda c: c.card_id)
                rng.shuffle(out)
            out = out[:budget]
        return out

    def dump(self, path: str) -> None:
        with open(path, "w") as f:
            for c in self.cards:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
