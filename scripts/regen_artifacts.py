"""Regenerate every number cited in the paper that lacked a JSON artifact
(audit findings B2/B3/M4). Caches make this cheap; every output lands in
reports/ so the reproducibility statement becomes true.

Usage: PYTHONPATH=. python3 scripts/regen_artifacts.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "scripts")

MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"


def save(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f"-> {path}")


def main():
    from agentlife.harness.backends import MLXBackend
    from agentlife.harness.emb_rag import EmbRAGSystem
    from agentlife.harness.probes import run_probe
    from agentlife.harness.scorer import score
    from agentlife.schema import load_jsonl
    from eval_paraphrased import rebuild

    # ---- B2: S-2 EmbRAG (template + paraphrased), saved this time
    be = MLXBackend(MODEL, cache_dir="cache")
    answers2 = load_jsonl("data/S-2/private/answers.jsonl")
    for qfile, tag in (("final_queries.jsonl", "template"),
                       ("final_queries_paraphrased.jsonl", "paraphrased")):
        sys_ = EmbRAGSystem(be, k=8, cache_dir="cache")
        for e in load_jsonl("data/S-2/public/episodes.jsonl"):
            sys_.ingest(e)
        queries = load_jsonl(f"data/S-2/public/{qfile}")
        rep = score({q["query_id"]: sys_.answer(q) for q in queries},
                    answers2)
        print(f"S-2 EmbRAG {tag}: {rep['accuracy_final']*100:.1f}%")
        save(f"reports/S-2-embrag-{tag}.json", rep)

    # ---- B2: S-2 CLS semantic-parser, template questions
    system = rebuild("runs/S-2-slots-stable-hybrid-routed", MODEL, "cache",
                     "data/S-2")
    queries = load_jsonl("data/S-2/public/final_queries.jsonl")
    rep = score({q["query_id"]: system.answer(q) for q in queries}, answers2)
    print(f"S-2 CLS semantic template: {rep['accuracy_final']*100:.1f}%")
    save("reports/S-2-slots-semantic-template.json", rep)

    # ---- M4: S-1 CLS semantic-parser, template questions
    answers1 = load_jsonl("data/S-1/private/answers.jsonl")
    system1 = rebuild("runs/S-1-slots-stable-hybrid-routed", MODEL, "cache",
                      "data/S-1")
    queries1 = load_jsonl("data/S-1/public/final_queries.jsonl")
    rep = score({q["query_id"]: system1.answer(q) for q in queries1},
                answers1)
    print(f"S-1 CLS semantic template: {rep['accuracy_final']*100:.1f}%")
    save("reports/S-1-slots-semantic-template.json", rep)

    # ---- B3: forgetting probes as JSON (existing adapters)
    probes = {"base": None,
              "S-1-lora-naive": "runs/S-1-lora/adapter",
              "S-1-seal": "runs/S-1-seal/adapter",
              "S-1-cls-monolithic-s10":
                  "runs/S-1-cls-stable-hybrid-s10/sleep-03/adapter",
              "M-1-cls-monolithic-s30":
                  "runs/M-1-cls-stable-hybrid-s30/sleep-03/adapter",
              "S-1-slots-fused":
                  "runs/S-1-slots-stable-hybrid/sleep-01/adapter"}
    out = {}
    for name, adapter in probes.items():
        if adapter is not None and not os.path.exists(adapter):
            print(f"SKIP probe {name}: adapter missing ({adapter})")
            continue
        b = MLXBackend(MODEL, adapter_path=adapter, cache_dir="cache")
        r = run_probe(b)
        out[name] = {"accuracy": r["accuracy"], "n": r["n"],
                     "adapter": adapter}
        print(f"probe {name}: {r['accuracy']*100:.1f}%")
    save("reports/forgetting_probes.json", out)

    # ---- B3: fused-slots S-1 eval as JSON (adapter exists; caches warm)
    fused = "runs/S-1-slots-stable-hybrid/sleep-01/adapter"
    if os.path.exists(fused):
        from agentlife.harness.parametric_systems import (
            PARAMETRIC_SYSTEM_PROMPT)
        bf = MLXBackend(MODEL, adapter_path=fused, cache_dir="cache")
        resp = {}
        for q in queries1:
            user = (f"QUESTION (asked on day {q['day_asked']}): "
                    f"{q['question']}\nAnswer:")
            resp[q["query_id"]] = bf.complete(PARAMETRIC_SYSTEM_PROMPT, user)
        # weights-only answering (how the fused run answered final queries
        # via its weights route); full-system fused number remains the
        # historical run — this artifact documents the fused adapter itself
        rep = score(resp, answers1)
        print(f"S-1 fused adapter weights-only: "
              f"{rep['accuracy_final']*100:.1f}%")
        save("reports/S-1-slots-fused-weightsonly.json", rep)


if __name__ == "__main__":
    main()
