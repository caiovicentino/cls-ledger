"""CLI: run a memory system through an AgentLife dataset and score it.

Usage: python -m agentlife.harness.run_eval --data data/M-1 --system oracle
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict

from ..schema import load_jsonl
from .scorer import print_report, score
from .systems import MemorySystem, make_system


def run(system: MemorySystem, data_dir: str) -> dict:
    stream = load_jsonl(os.path.join(data_dir, "public", "stream.jsonl"))
    finals = load_jsonl(os.path.join(data_dir, "public",
                                     "final_queries.jsonl"))
    answers = load_jsonl(os.path.join(data_dir, "private", "answers.jsonl"))

    responses: Dict[str, str] = {}
    for item in sorted(stream, key=lambda r: r["seq"]):
        if item["type"] == "episode":
            system.ingest(item)
        else:
            responses[item["query_id"]] = system.answer(item)
    for q in finals:
        responses[q["query_id"]] = system.answer(q)
    return score(responses, answers)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--system", required=True,
                    choices=["oracle", "stale", "null"])
    ap.add_argument("--out", default=None,
                    help="write full JSON report to this path")
    args = ap.parse_args()

    system = make_system(args.system, args.data)
    report = run(system, args.data)
    print(f"system={args.system} data={args.data}\n")
    print_report(report)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nfull report -> {args.out}")


if __name__ == "__main__":
    main()
