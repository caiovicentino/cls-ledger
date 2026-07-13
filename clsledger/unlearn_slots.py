"""H3 with real slots: physical unlearning by re-fusion (no retraining).

Drop one entity's slot from the fusion and measure:
  - target facts flip to unknown/wrong;
  - every other answer should be IDENTICAL (the other slots' deltas are
    untouched — contrast with re-distillation, which changed 19/40).

Usage:
  python -m clsledger.unlearn_slots --run runs/S-1-slots-... \
      --data data/S-1 --entity "Sofia Duarte" \
      --backend mlx:mlx-community/Qwen2.5-3B-Instruct-4bit
"""
from __future__ import annotations

import argparse
import json
import os

from agentlife.harness.backends import MLXBackend
from agentlife.harness.parametric_systems import PARAMETRIC_SYSTEM_PROMPT
from agentlife.harness.scorer import classify
from agentlife.schema import load_jsonl

from .slots import fuse_adapters


def _answer_all(backend: MLXBackend, finals) -> dict:
    out = {}
    for q in finals:
        user = (f"QUESTION (asked on day {q['day_asked']}): "
                f"{q['question']}\nAnswer:")
        out[q["query_id"]] = backend.complete(PARAMETRIC_SYSTEM_PROMPT, user)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True,
                    help="slots run workdir (contains slots.json)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--entity", required=True)
    ap.add_argument("--backend", required=True)
    ap.add_argument("--cache-dir", default="cache")
    args = ap.parse_args()

    model_id = args.backend.partition(":")[2]
    with open(os.path.join(args.run, "slots.json")) as f:
        meta = json.load(f)["slots"]
    if args.entity not in meta:
        raise SystemExit(f"no slot for {args.entity!r}; slots: "
                         f"{sorted(meta)}")
    kept = [m["adapter"] for slot, m in sorted(meta.items())
            if slot != args.entity]
    all_slots = [m["adapter"] for _slot, m in sorted(meta.items())]

    out_dir = os.path.join(args.run, "unlearn-" +
                           args.entity.replace(" ", "_").lower())
    before_dir = os.path.join(out_dir, "fused-before")
    after_dir = os.path.join(out_dir, "fused-after")
    fuse_adapters(all_slots, before_dir)
    fuse_adapters(kept, after_dir)

    finals = load_jsonl(os.path.join(args.data, "public",
                                     "final_queries.jsonl"))
    answers = {a["query_id"]: a for a in load_jsonl(
        os.path.join(args.data, "private", "answers.jsonl"))}
    before = _answer_all(MLXBackend(model_id, adapter_path=before_dir,
                                    cache_dir=args.cache_dir), finals)
    after = _answer_all(MLXBackend(model_id, adapter_path=after_dir,
                                   cache_dir=args.cache_dir), finals)

    name = args.entity.lower()
    target = [q for q in finals if name in q["question"].lower()]
    others = [q for q in finals if name not in q["question"].lower()]
    forgotten = sum(1 for q in target
                    if classify(after[q["query_id"]],
                                answers[q["query_id"]]) != "correct")
    correct_before = sum(1 for q in target
                         if classify(before[q["query_id"]],
                                     answers[q["query_id"]]) == "correct")
    changed = [q["query_id"] for q in others
               if before[q["query_id"]].strip().lower()
               != after[q["query_id"]].strip().lower()]
    regressed = [qid for qid in changed
                 if classify(before[qid], answers[qid]) == "correct"
                 and classify(after[qid], answers[qid]) != "correct"]

    report = {
        "entity": args.entity,
        "method": "slot re-fusion (no retraining)",
        "target_queries": len(target),
        "target_correct_before": correct_before,
        "target_not_correct_after": forgotten,
        "other_queries": len(others),
        "other_answers_changed": len(changed),
        "other_regressed": len(regressed),
        "changed_ids": changed,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
