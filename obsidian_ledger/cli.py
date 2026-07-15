"""obsidian-ledger CLI — works from any coding agent with a shell.

  python3 -m obsidian_ledger.cli build   --vault V [--extractor SPEC | --curated cards.json]
  python3 -m obsidian_ledger.cli query   --vault V "question"
  python3 -m obsidian_ledger.cli verify  --vault V "claim"
  python3 -m obsidian_ledger.cli induce  --vault V --entity E --attribute A
  python3 -m obsidian_ledger.cli install --vault V     # write agent files

build modes:
  --curated cards.json : import cards curated by a human/agent
                         [{entity, attribute, value, date(ISO), source}]
  --extractor SPEC     : LLM extraction over vault notes (SPEC grammar:
                         'openai:model[@base_url][#KEY_ENV]' or 'mlx:m')
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

from clsledger.ledger import Card, Ledger  # noqa: E402
from obsidian_ledger import store  # noqa: E402

EXTRACT_PROMPT = (
    "Extract durable atomic facts from this note as JSON:\n"
    '[{{"entity": "...", "attribute": "snake_case", "value": "...", '
    '"date": "YYYY-MM-DD"}}]\n'
    "Rules: only facts explicitly stated; use the note's date ({date}) "
    "when no other date applies; prefer stable entity names; skip "
    "opinions and prose. If nothing durable, return [].\n"
    "Note ({name}):\n{text}")


def note_date(path: str, text: str) -> str:
    m = re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(path))
    if m:
        return m.group(1)
    m = re.search(r"^(?:date|created|updated):\s*[\"']?"
                  r"(\d{4}-\d{2}-\d{2})", text, re.M)
    return m.group(1) if m else "2026-01-01"


def build(args) -> None:
    lg = Ledger()
    if args.curated:
        for c in json.load(open(args.curated)):
            lg.add(Card(c["entity"], c["attribute"], c["value"],
                        store.iso_to_day(c["date"]), c["source"]))
    else:
        from agentlife.harness.backends import make_backend
        be = make_backend(args.extractor, args.cache_dir)
        skip = re.compile(r"token|secret|cred|password", re.I)
        n_notes = 0
        for root, dirs, files in os.walk(args.vault):
            dirs[:] = [d for d in dirs if not d.startswith((".", "_backup"))
                       and d != store.LEDGER_DIR]
            for name in sorted(files):
                if not name.endswith(".md") or skip.search(name):
                    continue
                path = os.path.join(root, name)
                text = open(path, errors="ignore").read()[:6000]
                iso = note_date(path, text)
                rel = os.path.relpath(path, args.vault)
                raw = be.complete(
                    "You extract facts. Output only JSON.",
                    EXTRACT_PROMPT.format(date=iso, name=rel, text=text),
                    max_tokens=500).strip()
                raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.M)
                try:
                    items = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                for it in items:
                    if not (isinstance(it, dict) and it.get("entity")
                            and it.get("attribute") and it.get("value")):
                        continue
                    try:
                        day = store.iso_to_day(it.get("date", iso))
                    except ValueError:
                        day = store.iso_to_day(iso)
                    lg.add(Card(str(it["entity"]), str(it["attribute"]),
                                str(it["value"]), day, rel))
                n_notes += 1
        print(f"extracted from {n_notes} notes")
    out = store.save(args.vault, lg)
    print(f"{len(lg.cards)} cards ({len(lg.current_cards())} current) "
          f"-> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="obsidian-ledger")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("build", "query", "verify", "induce", "install"):
        p = sub.add_parser(name)
        p.add_argument("--vault", required=True)
        if name == "build":
            p.add_argument("--curated")
            p.add_argument("--extractor", default="openai:gpt-4.1-mini")
            p.add_argument("--cache-dir", default=None)
        if name in ("query", "verify"):
            p.add_argument("text")
        if name == "induce":
            p.add_argument("--entity", required=True)
            p.add_argument("--attribute", required=True)
    args = ap.parse_args()

    if args.cmd == "build":
        build(args)
        return
    if args.cmd == "install":
        from obsidian_ledger.instructions import install
        for f in install(args.vault):
            print(f"instruction block -> {f}")
        return
    lg = store.load(args.vault)
    if args.cmd == "query":
        print(json.dumps(store.query(lg, args.text), ensure_ascii=False,
                         indent=2))
    elif args.cmd == "verify":
        print(json.dumps(store.verify(lg, args.text), ensure_ascii=False,
                         indent=2))
    elif args.cmd == "induce":
        print(json.dumps(store.induce(lg, args.entity, args.attribute),
                         ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
