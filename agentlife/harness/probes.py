"""General-capability probe: measures catastrophic forgetting.

Fixed set of world-knowledge/arithmetic questions any instruct model
answers correctly. Run on base model and on model+adapter; the delta is
the forgetting cost of consolidation. Scored with the same boundary
matcher as the benchmark.

Usage:
  python -m agentlife.harness.probes --backend mlx:<model> [--adapter path]
"""
from __future__ import annotations

import argparse
import json

from .backends import make_backend
from .scorer import contains, normalize

PROBES = [
    ("What is the capital of France?", ["paris"]),
    ("What is the capital of Japan?", ["tokyo"]),
    ("Who wrote Romeo and Juliet?", ["shakespeare"]),
    ("What is the largest planet in the solar system?", ["jupiter"]),
    ("What is the chemical symbol for gold?", ["au"]),
    ("At sea level, what is the boiling point of water in Celsius?",
     ["100"]),
    ("How many days are there in a leap year?", ["366"]),
    ("What is 7 times 8?", ["56", "fifty-six", "fifty six"]),
    ("What is 15 plus 27?", ["42", "forty-two", "forty two"]),
    ("What is the square root of 81?", ["9", "nine"]),
    ("What gas do plants absorb from the atmosphere?",
     ["carbon dioxide", "co2"]),
    ("Who was the first president of the United States?", ["washington"]),
    ("Which ocean is the largest?", ["pacific"]),
    ("What is the currency of the United Kingdom?",
     ["pound", "sterling", "gbp"]),
    ("How many legs does a spider have?", ["8", "eight"]),
    ("In which continent is Egypt located?", ["africa"]),
]

SYSTEM = ("Answer the question concisely with the value only, "
          "no explanations.")


def run_probe(backend) -> dict:
    rows = []
    for question, accepted in PROBES:
        resp = backend.complete(SYSTEM, question, max_tokens=32)
        ok = any(contains(a, normalize(resp)) for a in accepted)
        rows.append({"q": question, "response": resp, "correct": int(ok)})
    acc = sum(r["correct"] for r in rows) / len(rows)
    return {"accuracy": acc, "n": len(rows), "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--cache-dir", default="cache")
    args = ap.parse_args()
    backend = make_backend(args.backend, args.cache_dir,
                           adapter_path=args.adapter)
    report = run_probe(backend)
    print(f"general-capability probe: {report['accuracy']*100:.1f}% "
          f"({report['n']} questions) adapter={args.adapter or 'none'}")
    for r in report["rows"]:
        if not r["correct"]:
            print(f"  WRONG: {r['q']} -> {r['response']!r}")


if __name__ == "__main__":
    main()
