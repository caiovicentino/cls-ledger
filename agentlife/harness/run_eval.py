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
    if hasattr(system, "finalize"):
        system.finalize()
    for q in finals:
        responses[q["query_id"]] = system.answer(q)
    return score(responses, answers)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--system", required=True,
                    choices=["oracle", "stale", "null", "full", "rag",
                             "lora", "seal", "cls"])
    ap.add_argument("--policy", default="stable",
                    choices=["stable", "all", "hot"],
                    help="cls consolidation policy")
    ap.add_argument("--mode", default="hybrid",
                    choices=["hybrid", "weights"],
                    help="cls answering mode (v1 hybrid / v0 weights-only)")
    ap.add_argument("--no-replay", action="store_true",
                    help="cls: disable anti-forgetting replay mix-in")
    ap.add_argument("--backend", default="openai:gpt-4.1-mini",
                    help="provider:model, e.g. openai:gpt-4.1-mini or "
                         "mlx:mlx-community/Qwen2.5-3B-Instruct-4bit")
    ap.add_argument("--k", type=int, default=8, help="RAG top-k episodes")
    ap.add_argument("--iters", type=int, default=None,
                    help="LoRA training iterations (lora/seal)")
    ap.add_argument("--workdir", default=None,
                    help="training workdir (lora/seal)")
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
    elif args.system in ("lora", "seal", "cls"):
        from .parametric_systems import LoRANaiveSystem, SEALLiteSystem
        provider, _, model_id = args.backend.partition(":")
        if provider != "mlx":
            raise SystemExit("lora/seal/cls need an mlx backend "
                             "(--backend mlx:<model>)")
        dataset = os.path.basename(os.path.normpath(args.data))
        suffix = (f"-{args.policy}-{args.mode}"
                  if args.system == "cls" else "")
        workdir = args.workdir or os.path.join(
            "runs", f"{dataset}-{args.system}{suffix}")
        kwargs = dict(model_id=model_id, workdir=workdir,
                      cache_dir=args.cache_dir)
        if args.iters:
            kwargs["iters"] = args.iters
        if args.system == "lora":
            system = LoRANaiveSystem(**kwargs)
        elif args.system == "seal":
            system = SEALLiteSystem(**kwargs)
        else:
            import sys as _sys
            _sys.path.insert(0, os.getcwd())
            from clsledger.system import CLSLedgerSystem
            system = CLSLedgerSystem(policy=args.policy, mode=args.mode,
                                     replay=not args.no_replay, **kwargs)
    else:
        system = make_system(args.system, args.data)

    model_tag = args.backend.split(":", 1)[1].rsplit("/", 1)[-1]
    label = args.system if args.system in ("oracle", "stale", "null") else (
        f"{args.system}-{model_tag}"
        + (f"-k{args.k}" if args.system == "rag" else "")
        + (f"-{args.policy}-{args.mode}" if args.system == "cls" else ""))
    report = run(system, args.data)
    print(f"system={label} data={args.data}\n")
    print_report(report)
    stats_src = backend if backend is not None else getattr(
        system, "backend", None)
    if stats_src is not None and hasattr(stats_src, "stats"):
        report["backend_stats"] = stats_src.stats()
        print(f"\nbackend: {json.dumps(report['backend_stats'])}")
    if hasattr(system, "stats"):
        report["system_stats"] = system.stats()
        print(f"system: {json.dumps(report['system_stats'])}")

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
