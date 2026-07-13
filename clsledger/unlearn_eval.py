"""H3 v0: reversibility by provenance.

Take a finished CLS run, delete one entity's cards from the ledger,
re-distill, and measure:
  - forgotten : queries about the target entity must flip to unknown/wrong
  - collateral: answers to all other queries must not change

Usage:
  python -m clsledger.unlearn_eval --run runs/S-1-cls-stable \
      --data data/S-1 --entity "Sofia Almeida" \
      --backend mlx:mlx-community/Qwen2.5-3B-Instruct-4bit --iters 600
"""
from __future__ import annotations

import argparse
import json
import os

from agentlife.harness.backends import MLXBackend
from agentlife.harness.parametric_systems import (
    PARAMETRIC_SYSTEM_PROMPT, _run_lora_training, _write_dataset)
from agentlife.harness.scorer import classify
from agentlife.schema import load_jsonl

from .ledger import norm_key


def _answer_all(backend: MLXBackend, finals) -> dict:
    out = {}
    for q in finals:
        user = (f"QUESTION (asked on day {q['day_asked']}): "
                f"{q['question']}\nAnswer:")
        out[q["query_id"]] = backend.complete(PARAMETRIC_SYSTEM_PROMPT, user)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="finished cls workdir")
    ap.add_argument("--data", required=True)
    ap.add_argument("--entity", required=True)
    ap.add_argument("--backend", required=True)
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--cache-dir", default="cache")
    args = ap.parse_args()

    model_id = args.backend.partition(":")[2]
    target = norm_key(args.entity)

    ledger = load_jsonl(os.path.join(args.run, "ledger.jsonl"))
    provenance = load_jsonl(os.path.join(args.run, "provenance.jsonl"))
    train = load_jsonl(os.path.join(args.run, "data", "train.jsonl"))
    drop_rows = {p["train_row"] for p in provenance
                 if p["card_id"].split(".")[0] == target}
    kept = [r for i, r in enumerate(train) if i not in drop_rows]
    n_target_cards = len({p["card_id"] for p in provenance
                          if p["card_id"].split(".")[0] == target})
    if not drop_rows:
        raise SystemExit(f"no cards for entity {args.entity!r} "
                         f"(normalized {target!r}) in the ledger")

    unlearn_dir = os.path.join(args.run, f"unlearn-{target}")
    data_dir = os.path.join(unlearn_dir, "data")
    adapter = os.path.join(unlearn_dir, "adapter")
    _write_dataset(data_dir, kept, kept[-max(2, len(kept) // 10):])
    _run_lora_training(model_id, data_dir, adapter, args.iters,
                       mask_prompt=True)

    finals = load_jsonl(os.path.join(args.data, "public",
                                     "final_queries.jsonl"))
    answers = {a["query_id"]: a for a in load_jsonl(
        os.path.join(args.data, "private", "answers.jsonl"))}

    before = _answer_all(
        MLXBackend(model_id, adapter_path=os.path.join(args.run, "adapter"),
                   cache_dir=args.cache_dir), finals)
    after = _answer_all(
        MLXBackend(model_id, adapter_path=adapter,
                   cache_dir=args.cache_dir), finals)

    name_lower = args.entity.lower()
    target_qs = [q for q in finals if name_lower in q["question"].lower()]
    other_qs = [q for q in finals if name_lower not in q["question"].lower()]

    forgotten = sum(
        1 for q in target_qs
        if classify(after[q["query_id"]], answers[q["query_id"]])
        != "correct")
    was_correct = sum(
        1 for q in target_qs
        if classify(before[q["query_id"]], answers[q["query_id"]])
        == "correct")
    changed = [q["query_id"] for q in other_qs
               if before[q["query_id"]].strip().lower()
               != after[q["query_id"]].strip().lower()]
    regressed = [qid for qid in changed
                 if classify(before[qid], answers[qid]) == "correct"
                 and classify(after[qid], answers[qid]) != "correct"]

    report = {
        "entity": args.entity,
        "cards_dropped": n_target_cards,
        "train_rows_dropped": len(drop_rows),
        "target_queries": len(target_qs),
        "target_correct_before": was_correct,
        "target_not_correct_after": forgotten,
        "other_queries": len(other_qs),
        "other_answers_changed": len(changed),
        "other_regressed": len(regressed),
        "changed_ids": changed,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    with open(os.path.join(unlearn_dir, "unlearn_report.json"), "w") as f:
        json.dump({**report,
                   "before": {q["query_id"]: before[q["query_id"]]
                              for q in finals},
                   "after": {q["query_id"]: after[q["query_id"]]
                             for q in finals}}, f, indent=2,
                  ensure_ascii=False)


if __name__ == "__main__":
    main()
