"""CLI: generate an AgentLife dataset.

Usage: python -m agentlife.generate --preset M --seed 1 --out data/M-1
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os

from .generator import LifeGenerator
from .schema import make_config


def write_dataset(gen: LifeGenerator, out_dir: str) -> None:
    public = os.path.join(out_dir, "public")
    private = os.path.join(out_dir, "private")
    os.makedirs(public, exist_ok=True)
    os.makedirs(private, exist_ok=True)

    def jsonl(path, rows):
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    jsonl(os.path.join(public, "stream.jsonl"), gen.stream)
    jsonl(os.path.join(public, "episodes.jsonl"),
          [dataclasses.asdict(e) for e in gen.episodes])
    jsonl(os.path.join(public, "final_queries.jsonl"),
          [q.public() for q in gen.final_queries])
    jsonl(os.path.join(private, "answers.jsonl"),
          [q.private() for q in gen.online_queries + gen.final_queries])
    jsonl(os.path.join(private, "world_facts.jsonl"), gen.world.dump())
    with open(os.path.join(private, "usage_stats.json"), "w") as f:
        json.dump({"usage": gen.usage, "hot_set": sorted(gen.hot_set)}, f,
                  indent=2)
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(gen.manifest(), f, indent=2, ensure_ascii=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="M", choices=["S", "M", "L", "XL"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gen = LifeGenerator(make_config(args.preset, args.seed)).run()
    write_dataset(gen, args.out)
    m = gen.manifest()
    print(f"preset={args.preset} seed={args.seed} -> {args.out}")
    print(json.dumps(m["counts"], indent=2))
    print(f"est_tokens={m['est_tokens']}")
    if m["warnings"]:
        print("warnings:")
        for wmsg in m["warnings"]:
            print(f"  - {wmsg}")


if __name__ == "__main__":
    main()
