"""Ledger store for Obsidian vaults: load/save, query, verify, induce.

Wraps clsledger.Ledger with vault-friendly I/O (ISO dates, _ledger/ dir)
and agent-friendly answers: every answer carries provenance, and a fact
that is not in the ledger is answered as NOT_IN_LEDGER — never guessed.
"""
from __future__ import annotations

import json
import os
import re
from datetime import date
from typing import List, Optional

try:
    from clsledger.ledger import Card, Ledger, norm_key
except ImportError:  # installed package: vendored core
    from obsidian_ledger._ledger_core import Card, Ledger, norm_key

LEDGER_DIR = "_ledger"
LEDGER_FILE = "vault_ledger.jsonl"
SNAPSHOT_FILE = "VAULT_LEDGER.md"
NOT_IN_LEDGER = ("NOT IN LEDGER — do not assert; check the source note "
                 "or omit.")


def iso_to_day(iso: str) -> int:
    return date.fromisoformat(iso).toordinal()


def day_to_iso(day: int) -> str:
    return date.fromordinal(day).isoformat()


def load(vault: str) -> Ledger:
    path = os.path.join(vault, LEDGER_DIR, LEDGER_FILE)
    lg = Ledger()
    if not os.path.exists(path):
        return lg
    for line in open(path):
        c = json.loads(line)
        lg.add(Card(c["entity"], c["attribute"], c["value"], c["day"],
                    c["episode_id"]))
    return lg


def save(vault: str, lg: Ledger) -> str:
    out = os.path.join(vault, LEDGER_DIR)
    os.makedirs(out, exist_ok=True)
    lg.dump(os.path.join(out, LEDGER_FILE))
    _write_snapshot(vault, lg)
    return out


def _write_snapshot(vault: str, lg: Ledger) -> None:
    lines = ["# Vault Ledger — symbolic snapshot", "",
             f"{len(lg.cards)} cards, {len(lg.current_cards())} current "
             "facts. Every fact carries provenance. Facts not listed here "
             "DO NOT EXIST in the ledger — agents must not assert them.",
             "", "## Current facts", ""]
    for c in sorted(lg.current_cards(), key=lambda c: (c.entity.lower(),
                                                       c.attribute)):
        lines.append(f"- **{c.entity}** · {c.attribute} = {c.value} "
                     f"({day_to_iso(c.day)}; source: {c.episode_id})")
    sup = [c for c in lg.cards if c.superseded_by]
    if sup:
        lines += ["", "## Superseded (history preserved)", ""]
        for c in sup:
            lines.append(f"- ~~{c.entity} · {c.attribute} = {c.value}~~ "
                         f"({day_to_iso(c.day)})")
    with open(os.path.join(vault, LEDGER_DIR, SNAPSHOT_FILE), "w") as f:
        f.write("\n".join(lines) + "\n")


def _tokens(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def query(lg: Ledger, question: str, k: int = 3) -> List[dict]:
    """Token-overlap lookup over current facts; returns best matches with
    provenance, or an explicit not-found sentinel."""
    qt = _tokens(question)
    scored = []
    for c in lg.current_cards():
        overlap = len(qt & (_tokens(c.entity) | _tokens(c.attribute)
                            | _tokens(c.value)))
        if overlap:
            scored.append((overlap, c))
    scored.sort(key=lambda x: (-x[0], x[1].card_id))
    if not scored:
        return [{"answer": NOT_IN_LEDGER, "question": question}]
    out = []
    for _s, c in scored[:k]:
        out.append({"entity": c.entity, "attribute": c.attribute,
                    "value": c.value, "as_of": day_to_iso(c.day),
                    "source": c.episode_id,
                    "changes": max(0, len(lg.history(c.key)) - 1)})
    return out


def _stems(text: str) -> set:
    return {t[:4] for t in _tokens(text)}


def verify(lg: Ledger, claim: str) -> dict:
    """Check a claim against the ledger. Verdicts:
    SUPPORTED            - value present AND the entity is named
    CONTRADICTED_STALE   - claim matches a superseded value
    CONTRADICTED_CURRENT - entity+attribute match but the value differs
    NOT_IN_LEDGER        - do not assert
    Entity overlap is required everywhere: a bare token matching some
    unrelated value must never count as support (measured false positive
    on a third-party vault: 'kepano rated X 10' vs 'twitter = kepano')."""
    ct = _tokens(claim)
    cs = _stems(claim)

    def entity_named(c):
        return bool(_tokens(c.entity) & ct)

    for c in lg.current_cards():
        if entity_named(c) and _tokens(c.value) and _tokens(c.value) <= ct:
            return {"verdict": "SUPPORTED",
                    "fact": f"{c.entity} · {c.attribute} = {c.value}",
                    "as_of": day_to_iso(c.day), "source": c.episode_id}
    for c in lg.cards:
        if (c.superseded_by and entity_named(c) and _tokens(c.value)
                and _tokens(c.value) <= ct):
            cur = [x for x in lg.history(c.key) if not x.superseded_by]
            return {"verdict": "CONTRADICTED_STALE",
                    "claim_matches": f"superseded value of {c.key}",
                    "current_value": cur[-1].value if cur else None,
                    "source": c.episode_id}
    for c in lg.current_cards():
        if (entity_named(c) and _stems(c.attribute) & cs
                and _tokens(c.value) and not (_tokens(c.value) <= ct)):
            return {"verdict": "CONTRADICTED_CURRENT",
                    "claimed_about": f"{c.entity} · {c.attribute}",
                    "actual_value": c.value,
                    "as_of": day_to_iso(c.day), "source": c.episode_id}
    return {"verdict": "NOT_IN_LEDGER",
            "guidance": "do not assert; verify at source or omit"}


def induce(lg: Ledger, entity: str, attribute: str) -> dict:
    key = f"{norm_key(entity)}.{norm_key(attribute)}"
    hist = lg.history(key)
    if not hist:
        return {"verdict": "NOT_IN_LEDGER", "key": key}
    vals = [(h.value, day_to_iso(h.day)) for h in hist]
    out = {"key": key, "n_changes": len(hist) - 1, "history": vals,
           "current": vals[-1][0]}
    nums = []
    for v, _d in vals:
        m = re.search(r"(\d+(?:\.\d+)?)\s*k\b", v.replace(",", "").lower())
        n = (float(m.group(1)) * 1000 if m
             else (float(re.search(r"\d+(?:\.\d+)?", v).group())
                   if re.search(r"\d+(?:\.\d+)?", v) else None))
        nums.append(n)
    if len([n for n in nums if n is not None]) >= 2:
        first = next(n for n in nums if n is not None)
        last = next(n for n in reversed(nums) if n is not None)
        if first != last:
            out["trend"] = "increased" if last > first else "decreased"
    return out
