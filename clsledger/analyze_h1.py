"""H1 ablation analysis: accuracy by route (weights vs episodic) per policy.

Routes come from report.system_stats.routes_by_query when present; for
older runs they are reconstructed from the run artifacts (ledger.jsonl +
provenance.jsonl), which is possible precisely because the ledger keeps
provenance — the same property that makes unlearning auditable.

Usage:
  python -m clsledger.analyze_h1 --data data/M-1 \
      --run reports/M-1-cls-...-stable-hybrid.json:runs/M-1-cls-stable-hybrid \
      --run reports/M-1-cls-...-hot-hybrid.json:runs/M-1-cls-hot-hybrid
"""
from __future__ import annotations

import argparse
import json
import os

from agentlife.harness.bm25 import BM25Index
from agentlife.schema import load_jsonl

from .ledger import Card, Ledger
from .system import CLSLedgerSystem


def reconstruct_routes(run_dir: str, finals) -> dict:
    cards = load_jsonl(os.path.join(run_dir, "ledger.jsonl"))
    prov = load_jsonl(os.path.join(run_dir, "provenance.jsonl"))
    consolidated = {p["card_id"] for p in prov
                    if p["card_id"] != "__replay__"}
    sys_ = CLSLedgerSystem.__new__(CLSLedgerSystem)
    sys_.all_index = BM25Index()
    sys_.cards_by_id = {}
    sys_.consolidated_ids = consolidated
    for c in cards:
        if c["superseded_by"] is None:
            card = Card(c["entity"], c["attribute"], c["value"], c["day"],
                        c["episode_id"])
            sys_.all_index.add(card.card_id, card.text())
            sys_.cards_by_id[card.card_id] = card
    return {q["query_id"]: sys_._route(q["question"]) for q in finals}


def analyze(report_path: str, run_dir: str, finals) -> dict:
    rep = json.load(open(report_path))
    routes = (rep.get("system_stats", {}).get("routes_by_query")
              or reconstruct_routes(run_dir, finals))
    cons = json.load(open(os.path.join(run_dir, "consolidation.json")))
    final_rows = [r for r in rep["rows"] if r["qtype"] != "online"]
    by_route = {"weights": [], "episodic": []}
    for r in final_rows:
        route = routes.get(r["query_id"])
        if route in by_route:
            by_route[route].append(r)

    def acc(rows):
        return (round(100 * sum(x["correct"] for x in rows) / len(rows), 1)
                if rows else None)

    qtype_route = {}
    for route, rows in by_route.items():
        for r in rows:
            key = (r["qtype"], route)
            qtype_route.setdefault(key, []).append(r)

    return {
        "policy": cons["policy"],
        "cards_selected": cons["cards_selected"],
        "train_rows": cons["train_rows"],
        "acc_final": round(100 * rep["accuracy_final"], 1),
        "n_weights": len(by_route["weights"]),
        "n_episodic": len(by_route["episodic"]),
        "acc_weights": acc(by_route["weights"]),
        "acc_episodic": acc(by_route["episodic"]),
        "qtype_route": {f"{q}/{r}": {"n": len(v), "acc": acc(v)}
                        for (q, r), v in sorted(qtype_route.items())},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--run", action="append", required=True,
                    help="report.json:run_dir (repeatable)")
    args = ap.parse_args()
    finals = load_jsonl(os.path.join(args.data, "public",
                                     "final_queries.jsonl"))
    results = []
    for spec in args.run:
        report_path, run_dir = spec.split(":", 1)
        results.append(analyze(report_path, run_dir, finals))
    hdr = (f"{'policy':<8} {'cards':>5} {'rows':>5} {'final':>6} "
           f"{'n_w':>4} {'acc_w':>6} {'n_e':>4} {'acc_e':>6}")
    print(hdr)
    for r in results:
        print(f"{r['policy']:<8} {r['cards_selected']:>5} "
              f"{r['train_rows']:>5} {r['acc_final']:>5}% "
              f"{r['n_weights']:>4} {str(r['acc_weights']):>5}% "
              f"{r['n_episodic']:>4} {str(r['acc_episodic']):>5}%")
    print()
    for r in results:
        print(f"--- {r['policy']}: by qtype/route")
        for k, v in r["qtype_route"].items():
            print(f"  {k:<28} n={v['n']:<3} acc={v['acc']}%")
    out = os.path.join("reports", "h1_analysis.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
