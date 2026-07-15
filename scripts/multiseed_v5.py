"""Multi-seed replication of the v5 claims (induction + dispositions).

Seeds 9 (existing) + 10..12, preset M, v5 generator. Per seed: CLS
slots-routed, BM25 RAG, embeddings RAG. Sequential, idempotent, one
subprocess per run. Appends to reports/multiseed_v5_progress.jsonl.

Usage: PYTHONPATH=. python3 scripts/multiseed_v5.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

MODEL = "mlx:mlx-community/Qwen2.5-3B-Instruct-4bit"
TAG = "Qwen2.5-3B-Instruct-4bit"
PROGRESS = "reports/multiseed_v5_progress.jsonl"
SEEDS = (9, 10, 11, 12)


def log(msg):
    print(msg, flush=True)


def record(row):
    with open(PROGRESS, "a") as f:
        f.write(json.dumps(row) + "\n")
    log(f"RESULT {json.dumps(row)}")


def run(cmd):
    log("RUN " + " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log(f"FAILED ({r.returncode}): {r.stdout[-400:]} {r.stderr[-400:]}")
    return r.returncode == 0


def harvest(report_path, **meta):
    if not os.path.exists(report_path):
        record({**meta, "status": "missing"})
        return
    rep = json.load(open(report_path))
    record({**meta, "status": "ok",
            "acc_final": rep["accuracy_final"],
            "by_qtype": {k: v["acc"] for k, v in rep["by_qtype"].items()},
            "adherence": {k: v["rate"] for k, v in
                          rep.get("disposition_adherence", {}).items()}})


def eval_run(data, system, seed):
    dataset = os.path.basename(data)
    label = {"slots": f"slots-{TAG}-stable-hybrid-routed",
             "rag": f"rag-{TAG}-k8",
             "embrag": "embrag"}[system]
    report = f"reports/{dataset}-{label}.json"
    meta = dict(dataset=dataset, seed=seed, system=system)
    if os.path.exists(report):
        log(f"SKIP (exists): {report}")
        harvest(report, **meta)
        return
    if system == "embrag":
        code = f"""
import sys, json; sys.path.insert(0, '.')
from agentlife.harness.backends import MLXBackend
from agentlife.harness.emb_rag import EmbRAGSystem
from agentlife.harness.scorer import score
from agentlife.schema import load_jsonl
be = MLXBackend('{MODEL.partition(':')[2]}', cache_dir='cache')
sys_ = EmbRAGSystem(be, k=8, cache_dir='cache')
for e in load_jsonl('{data}/public/episodes.jsonl'): sys_.ingest(e)
queries = load_jsonl('{data}/public/final_queries.jsonl')
answers = load_jsonl('{data}/private/answers.jsonl')
rep = score({{q['query_id']: sys_.answer(q) for q in queries}}, answers)
json.dump(rep, open('{report}', 'w'), indent=2)
"""
        ok = run([sys.executable, "-W", "ignore", "-c", code])
    else:
        cmd = [sys.executable, "-W", "ignore", "-m",
               "agentlife.harness.run_eval", "--data", data,
               "--system", system, "--backend", MODEL]
        if system == "slots":
            cmd += ["--policy", "stable", "--mode", "hybrid",
                    "--activation", "routed"]
        else:
            cmd += ["--k", "8"]
        ok = run(cmd)
    if ok:
        harvest(report, **meta)
    else:
        record({**meta, "status": "failed"})


def main():
    os.makedirs("reports", exist_ok=True)
    for seed in SEEDS:
        data = f"data/M-{seed}v5"
        if not os.path.exists(data):
            run([sys.executable, "-m", "agentlife.generate", "--preset",
                 "M", "--seed", str(seed), "--out", data])
        for system in ("slots", "rag", "embrag"):
            eval_run(data, system, seed)
    log("ALL DONE")


if __name__ == "__main__":
    main()
