"""CLI: run a memory system through an AgentLife dataset and score it.

Usage:
  python -m agentlife.harness.run_eval --data data/M-1 --system oracle
  python -m agentlife.harness.run_eval --data data/M-1 --system full \
      --backend openai:gpt-4.1-mini
  python -m agentlife.harness.run_eval --data data/M-1 --system rag --k 8 \
      --backend openai:gpt-4.1-mini
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict

from ..schema import load_jsonl
from .backends import make_backend
from .real_systems import FullContextSystem, RAGSystem
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
                    choices=["oracle", "stale", "null", "full", "rag"])
    ap.add_argument("--backend", default="openai:gpt-4.1-mini",
                    help="provider:model, e.g. openai:gpt-4.1-mini")
    ap.add_argument("--k", type=int, default=8, help="RAG top-k episodes")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--out", default=None,
                    help="report path (default: reports/<auto>.json)")
    args = ap.parse_args()

    backend = None
    if args.system in ("full", "rag"):
        backend = make_backend(args.backend, args.cache_dir)
        if args.system == "full":
            system: MemorySystem = FullContextSystem(backend)
        else:
            system = RAGSystem(backend, k=args.k)
    else:
        system = make_system(args.system, args.data)

    report = run(system, args.data)
    label = args.system if backend is None else (
        f"{args.system}-{args.backend.split(':', 1)[1]}"
        + (f"-k{args.k}" if args.system == "rag" else ""))
    print(f"system={label} data={args.data}\n")
    print_report(report)
    if backend is not None and hasattr(backend, "stats"):
        report["backend_stats"] = backend.stats()
        print(f"\nbackend: {json.dumps(report['backend_stats'])}")

    out = args.out
    if out is None:
        os.makedirs("reports", exist_ok=True)
        dataset = os.path.basename(os.path.normpath(args.data))
        out = os.path.join("reports", f"{dataset}-{label}.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"full report -> {out}")


if __name__ == "__main__":
    main()
