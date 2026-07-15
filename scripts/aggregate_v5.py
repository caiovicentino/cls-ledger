"""Aggregate multi-seed v5 results: per-cell mean ± sd and paired wins
for induction and disposition-adherence claims.

Usage: PYTHONPATH=. python3 scripts/aggregate_v5.py
"""
from __future__ import annotations

import json
import math
import os
from collections import defaultdict

PROGRESS = "reports/multiseed_v5_progress.jsonl"
IND = ("induction_count", "induction_habit", "induction_trend")
RULES = ("date_first", "city_upper")


def mean_sd(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    m = sum(xs) / len(xs)
    sd = (math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
          if len(xs) > 1 else 0.0)
    return m, sd


def main() -> None:
    rows = {}
    if os.path.exists(PROGRESS):
        for l in open(PROGRESS):
            r = json.loads(l)
            if r.get("status") == "ok":
                rows[(r["seed"], r["system"])] = r  # last write wins
    seeds = sorted({s for s, _ in rows})
    systems = ("slots", "rag", "embrag")
    print(f"seeds com dados: {seeds}\n")

    print(f"{'cell':<22}" + "".join(f"{s:>16}" for s in systems))
    cells = ([("final", lambda r: r["acc_final"])]
             + [(q, lambda r, q=q: r["by_qtype"].get(q)) for q in IND]
             + [(f"adh:{rule}", lambda r, rule=rule:
                 r.get("adherence", {}).get(rule)) for rule in RULES])
    for name, get in cells:
        line = f"{name:<22}"
        for system in systems:
            vals = [get(rows[(s, system)]) for s in seeds
                    if (s, system) in rows]
            m, sd = mean_sd(vals)
            line += (f"{100*m:7.1f}±{100*(sd or 0):4.1f}%" if m is not None
                     else f"{'n/a':>16}")
        print(line)

    # paired wins slots vs each retrieval
    for rival in ("rag", "embrag"):
        common = [s for s in seeds
                  if (s, "slots") in rows and (s, rival) in rows]
        wins = sum(rows[(s, "slots")]["acc_final"]
                   > rows[(s, rival)]["acc_final"] for s in common)
        diffs = [rows[(s, "slots")]["acc_final"]
                 - rows[(s, rival)]["acc_final"] for s in common]
        m, sd = mean_sd(diffs)
        if common:
            print(f"\nslots vs {rival}: {wins}/{len(common)} paired wins, "
                  f"mean diff {100*m:+.1f}pp (sd {100*(sd or 0):.1f})")
    # induction pooled: slots vs best retrieval per seed
    pooled = []
    for s in seeds:
        if (s, "slots") not in rows:
            continue
        sl = [rows[(s, "slots")]["by_qtype"].get(q) for q in IND]
        sl = [x for x in sl if x is not None]
        rv = []
        for rival in ("rag", "embrag"):
            if (s, rival) in rows:
                vals = [rows[(s, rival)]["by_qtype"].get(q) for q in IND]
                vals = [x for x in vals if x is not None]
                if vals:
                    rv.append(sum(vals) / len(vals))
        if sl and rv:
            pooled.append((sum(sl) / len(sl), max(rv)))
    if pooled:
        m_s, _ = mean_sd([p[0] for p in pooled])
        m_r, _ = mean_sd([p[1] for p in pooled])
        print(f"induction (pooled): slots {100*m_s:.1f}% vs best-retrieval "
              f"{100*m_r:.1f}%")


if __name__ == "__main__":
    main()
