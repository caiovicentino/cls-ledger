"""Multi-seed replication of the v3 headline claims.

Sequential, idempotent (skips runs whose report exists), one subprocess
per run so memory is fully released between runs. Appends each result to
reports/multiseed_progress.jsonl as it lands.

Usage: PYTHONPATH=. python3 scripts/multiseed.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

MODEL = "mlx:mlx-community/Qwen2.5-3B-Instruct-4bit"
TAG = "Qwen2.5-3B-Instruct-4bit"
PROGRESS = "reports/multiseed_progress.jsonl"


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
        log(f"FAILED ({r.returncode}): {r.stdout[-500:]} {r.stderr[-500:]}")
        return False
    return True


def ensure_dataset(preset, seed):
    path = f"data/{preset}-{seed}"
    if not os.path.exists(path):
        run([sys.executable, "-m", "agentlife.generate", "--preset", preset,
             "--seed", str(seed), "--out", path])
    return path


def harvest(report_path, **meta):
    if not os.path.exists(report_path):
        record({**meta, "status": "missing"})
        return
    rep = json.load(open(report_path))
    record({**meta, "status": "ok",
            "acc_final": rep["accuracy_final"],
            "acc_online": rep["accuracy_online"],
            "by_qtype": {k: v["acc"] for k, v in rep["by_qtype"].items()}})


def eval_run(data, system, seed, policy="stable", budget=None):
    dataset = os.path.basename(data)
    if system == "slots":
        label = (f"slots-{TAG}-{policy}-hybrid"
                 + (f"-b{budget}" if budget else "") + "-routed")
    else:
        label = f"rag-{TAG}-k8"
    report = f"reports/{dataset}-{label}.json"
    meta = dict(dataset=dataset, seed=seed, system=system, policy=policy,
                budget=budget)
    if os.path.exists(report):
        log(f"SKIP (exists): {report}")
        harvest(report, **meta)
        return
    cmd = [sys.executable, "-W", "ignore", "-m",
           "agentlife.harness.run_eval", "--data", data,
           "--system", system, "--backend", MODEL]
    if system == "slots":
        cmd += ["--policy", policy, "--mode", "hybrid",
                "--activation", "routed"]
        if budget:
            cmd += ["--budget", str(budget)]
    else:
        cmd += ["--k", "8"]
    if run(cmd):
        harvest(report, **meta)
    else:
        record({**meta, "status": "failed"})


def main():
    os.makedirs("reports", exist_ok=True)
    # Phase A: S seeds 2-5, slots vs rag
    for seed in (2, 3, 4, 5):
        data = ensure_dataset("S", seed)
        eval_run(data, "slots", seed)
        eval_run(data, "rag", seed)
    # Phase B: M — slots for seeds 1-3 (seed 1 slots-no-budget missing),
    # rag for seeds 2-3 (seed 1 exists)
    for seed in (1, 2, 3):
        data = ensure_dataset("M", seed)
        eval_run(data, "slots", seed)
        if seed != 1:
            eval_run(data, "rag", seed)
    # Phase C: H1 under budget, M seeds 2-3, three policies
    for seed in (2, 3):
        data = f"data/M-{seed}"
        for policy in ("hot", "stable", "all"):
            eval_run(data, "slots", seed, policy=policy, budget=40)
    log("ALL DONE")


if __name__ == "__main__":
    main()
