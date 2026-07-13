"""Anti-forgetting replay data mixed into consolidation training.

Three deliberate ingredients:
- arithmetic + world facts: keeps general capability alive (the v0 failure:
  general probe fell 93.8% -> 68.8% without replay);
- abstention rows: teaches calibrated 'unknown' for never-stated facts —
  v0/SEAL lost the ability to abstain (0% on absent queries);
- all DISJOINT from the general-capability probe questions, so the probe
  stays a fair test (never train on the test).
"""
from __future__ import annotations

import random
from typing import List

SYSTEM = None  # filled by caller (parametric system prompt)

WORLD_FACTS = [
    ("What is the capital of Brazil?", "Brasília"),
    ("What is the capital of Germany?", "Berlin"),
    ("What is the capital of Canada?", "Ottawa"),
    ("What is the capital of Australia?", "Canberra"),
    ("What is the capital of Italy?", "Rome"),
    ("What is the capital of Spain?", "Madrid"),
    ("What is the capital of Russia?", "Moscow"),
    ("What is the capital of India?", "New Delhi"),
    ("How many minutes are there in an hour?", "60"),
    ("How many sides does a triangle have?", "3"),
    ("What planet do humans live on?", "Earth"),
    ("What is the freezing point of water in Celsius?", "0"),
    ("How many letters are there in the English alphabet?", "26"),
    ("What language is spoken in Mexico?", "Spanish"),
    ("Which animal is known as man's best friend?", "dog"),
]

ABSTAIN_SUBJECTS = [
    "Zorblax Quinterra", "Mirovena Tal", "Kexut Ombra", "Fandor Yll",
    "Prixel Vantome", "Qorvath Ilun", "Zemira Kotal", "Ulvane Dross",
    "Yentil Marroc", "Ovix Trellane",
]
ABSTAIN_ATTRS = ["favorite color", "birthday", "employer", "hometown",
                 "phone number"]


def anchor_rows(system_prompt: str) -> List[dict]:
    """Tiny per-slot anchor: 6 general facts + 4 abstention rows. Without
    it, a slot trained on ~25 uniform QA rows collapses to answering its
    most frequent value for everything and forgets basic world knowledge
    (measured: 'capital of France' -> 'Toronto'). With it, slots
    discriminate attributes, keep general knowledge, and answer 'unknown'
    for entities outside their scope — safe failure under misrouting."""
    rows = []

    def add(q, a):
        rows.append({"messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ]})

    for q, a in WORLD_FACTS[:4]:
        add(q, a)
    add("What is 12 plus 30?", "42")
    add("How many sides does a triangle have?", "3")
    for subj in ABSTAIN_SUBJECTS[:4]:
        add(f"What is {subj}'s employer?", "unknown")
    return rows


def replay_rows(system_prompt: str, n_arith: int = 25,
                seed: int = 0) -> List[dict]:
    rng = random.Random(seed)
    rows = []

    def add(q, a):
        rows.append({"messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ]})

    for _ in range(n_arith):
        a, b = rng.randint(11, 97), rng.randint(11, 97)
        if rng.random() < 0.5:
            add(f"What is {a} plus {b}?", str(a + b))
        else:
            add(f"What is {a} times {b}?", str(a * b))
    for q, a in WORLD_FACTS:
        add(q, a)
    for subj in ABSTAIN_SUBJECTS:
        attr = rng.choice(ABSTAIN_ATTRS)
        add(f"What is {subj}'s {attr}?", "unknown")
    return rows
