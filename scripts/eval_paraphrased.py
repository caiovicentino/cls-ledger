"""Adversarial paraphrase test — evaluation half (GPU).

Reconstructs a finished slots-routed run from its artifacts (ledger.jsonl,
slots.json) — no retraining — and answers the paraphrased final questions,
scored against the ORIGINAL accepted/rejected sets (a paraphrase does not
change the answer). Reports per-route accuracy so we can see exactly which
mechanism (symbolic chain, router, reader) survives rewording.

Usage:
  PYTHONPATH=. python3 scripts/eval_paraphrased.py \
      --run runs/S-1-slots-stable-hybrid-routed --data data/S-1 \
      --backend mlx:mlx-community/Qwen2.5-3B-Instruct-4bit
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.getcwd())

from agentlife.harness.bm25 import BM25Index  # noqa: E402
from agentlife.harness.scorer import score  # noqa: E402
from agentlife.schema import load_jsonl  # noqa: E402
from clsledger.ledger import Card, Ledger  # noqa: E402
from clsledger.slots import SlotRoutedBackend  # noqa: E402
from clsledger.slots import CLSSlotsSystem as _SlotsSys  # noqa: E402


def rebuild(run_dir: str, model_id: str, cache_dir: str,
            data_dir: str):
    sys_ = _SlotsSys.__new__(_SlotsSys)
    lg = Ledger()
    for l in open(os.path.join(run_dir, "ledger.jsonl")):
        c = json.loads(l)
        if re.match(r"^(old|previous|former|prior|new)_", c["attribute"]):
            continue
        lg.add(Card(c["entity"], c["attribute"], c["value"], c["day"],
                    c["episode_id"]))
    sys_.ledger = lg
    sys_.online_k = 8
    sys_.mode = "hybrid"
    sys_.routes = {}
    sys_.cache_dir = cache_dir
    sys_.model_id = model_id
    sys_.base_backend = None
    meta = json.load(open(os.path.join(run_dir, "slots.json")))["slots"]
    slot_cards = {}
    for slot, m in meta.items():
        for cid in m["card_ids"]:
            slot_cards[cid] = slot
    sys_.consolidated_ids = set(slot_cards)
    sys_.all_index = BM25Index()
    sys_.cards_by_id = {}
    sys_.slot_of_entity = {}
    for c in lg.current_cards():
        sys_.all_index.add(c.card_id, c.text())
        sys_.cards_by_id[c.card_id] = c
        if c.card_id in slot_cards:
            sys_.slot_of_entity[c.entity] = slot_cards[c.card_id]
    sys_.backend = SlotRoutedBackend(
        model_id, {s: m["adapter"] for s, m in meta.items()},
        cache_dir=cache_dir)
    # episodes for the cold-start fallback
    sys_.episodes = load_jsonl(os.path.join(data_dir, "public",
                                            "episodes.jsonl"))
    sys_.ep_index = BM25Index()
    for e in sys_.episodes:
        sys_.ep_index.add(e["episode_id"], e["text"])

    sys_.activation = "routed"
    return sys_


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--backend", required=True)
    ap.add_argument("--cache-dir", default="cache")
    args = ap.parse_args()

    model_id = args.backend.partition(":")[2]
    system = rebuild(args.run, model_id, args.cache_dir, args.data)
    queries = load_jsonl(os.path.join(args.data, "public",
                                      "final_queries_paraphrased.jsonl"))
    answers = load_jsonl(os.path.join(args.data, "private",
                                      "answers.jsonl"))
    responses = {}
    for q in queries:
        responses[q["query_id"]] = system.answer(q)
    report = score(responses, answers)
    report["routes"] = dict(system.routes)
    n_para = sum(q.get("paraphrased", False) for q in queries)

    from collections import Counter
    route_counts = Counter(system.routes.values())
    print(f"paraphrased: {n_para}/{len(queries)}")
    print(f"acc final (paraphrased questions): "
          f"{report['accuracy_final']*100:.1f}%")
    print("routes:", dict(route_counts))
    for k, v in report["by_qtype"].items():
        print(f"  {k:<18} n={v['n']:<4} acc="
              f"{'' if v['acc'] is None else round(v['acc']*100,1)}%")
    dataset = os.path.basename(os.path.normpath(args.data))
    out = f"reports/{dataset}-slots-routed-PARAPHRASED.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
