"""Aggregate multi-seed results: mean ± sd per condition + paired sign
counts for the headline comparisons.

Usage: PYTHONPATH=. python3 scripts/aggregate.py
"""
from __future__ import annotations

import json
import math
import os
from collections import defaultdict

PROGRESS = "reports/multiseed_progress.jsonl"


def mean_sd(xs):
    if not xs:
        return None, None
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(var)


def main() -> None:
    rows = []
    if os.path.exists(PROGRESS):
        for l in open(PROGRESS):
            r = json.loads(l)
            if r.get("status") == "ok":
                rows.append(r)
    # de-dup (idempotent reruns append twice): last wins
    seen = {}
    for r in rows:
        key = (r["dataset"], r["system"], r["policy"], r["budget"])
        seen[key] = r
    rows = list(seen.values())

    # group: (preset, system, policy, budget) -> accs per seed
    groups = defaultdict(dict)
    for r in rows:
        preset = r["dataset"].split("-")[0]
        cond = (preset, r["system"], r["policy"], r["budget"])
        groups[cond][r["seed"]] = r["acc_final"]

    print(f"{'condition':<38} {'n':>2} {'mean':>7} {'sd':>6}  seeds")
    for cond in sorted(groups, key=str):
        accs = groups[cond]
        m, sd = mean_sd(list(accs.values()))
        label = f"{cond[0]} {cond[1]} {cond[2]}" + (
            f" b{cond[3]}" if cond[3] else "")
        print(f"{label:<38} {len(accs):>2} {m*100:6.1f}% {sd*100:5.1f}%  "
              + ",".join(f"s{s}:{a*100:.0f}" for s, a in sorted(accs.items())))

    # paired comparisons per preset: slots-stable vs rag
    for preset in ("S", "M"):
        slots = groups.get((preset, "slots", "stable", None), {})
        rag = groups.get((preset, "rag", "stable", None), {})
        common = sorted(set(slots) & set(rag))
        if common:
            wins = sum(slots[s] > rag[s] for s in common)
            diffs = [slots[s] - rag[s] for s in common]
            m, sd = mean_sd(diffs)
            print(f"\n{preset}: slots vs rag, paired on {len(common)} seeds: "
                  f"slots wins {wins}/{len(common)}, "
                  f"mean diff {m*100:+.1f}pp (sd {sd*100:.1f})")
    # H1 pairs
    for a, b in (("hot", "stable"), ("hot", "all"), ("stable", "all")):
        ga = groups.get(("M", "slots", a, 40), {})
        gb = groups.get(("M", "slots", b, 40), {})
        common = sorted(set(ga) & set(gb))
        if common:
            wins = sum(ga[s] > gb[s] for s in common)
            diffs = [ga[s] - gb[s] for s in common]
            m, _ = mean_sd(diffs)
            print(f"H1 {a} vs {b} (b40), {len(common)} seeds: "
                  f"{a} wins {wins}/{len(common)}, mean diff {m*100:+.1f}pp")


if __name__ == "__main__":
    main()
