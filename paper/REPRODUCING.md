# Reproducing every number in the preprint

Every claim maps to an artifact. Datasets regenerate deterministically
(`python -m agentlife.generate --preset {S,M,L} --seed N --out data/{P}-N`);
all LLM calls are disk-cached; MLX LoRA training is seeded.

| Paper location | Number(s) | Artifact / command |
|---|---|---|
| §5.1 parametric baselines | 15.9 / 20.5 / 31.8, stale 2/10/3 | `reports/S-1-lora-*.json`, `S-1-seal-*.json`, `S-1-cls-Qwen…stable-hybrid.json` (v0 run) |
| §5.2 table, S row | 64.6±6.7 vs 50.1±4.1, 4/4 | `reports/multiseed_progress.jsonl` (seeds 2–5) via `scripts/aggregate.py` |
| §5.2 table, M row | 71.1±1.7 vs 42.8±2.6, 2/2 | same, seeds 2–3 |
| §5.2 table, L row + online + 2+updates | 80.5 / 33.9 / 84.4 / 92.3 | `reports/L-1-slots-…routed.json`, `reports/L-1-rag-…k8.json` |
| §5.2 sign test | 6/6, p=0.031 | recompute: 2·(0.5)⁶ from the paired rows above |
| §5.3 strong retrieval (S-2) | 72.1 / 62.8 / 58.1 / 55.8 | `reports/S-2-slots-semantic-template.json`, `S-2-embrag-{template,paraphrased}.json`, `S-2-slots-routed-PARAPHRASED.json` |
| §5.4 probes | 93.8 base/rest/active; 87.5; 75.0; fused 0.0 | `reports/forgetting_probes.json`, `reports/S-1-slots-routed-h3-probe.json`; 68.8/81.2 log-derived (commits in git history) |
| §5.4 unlearning | 0/40 vs 19/40 | `reports/S-1-slots-routed-h3-probe.json` (slot drop), `reports/S-1-cls-unlearn-sofia_duarte.json` (re-distill) |
| §5.4 fusion collapse | 0.0% probe / 2.3% weights-only / 52.3% system | `reports/forgetting_probes.json`, `reports/S-1-slots-fused-weightsonly.json`; 52.3 log-derived |
| §5.5 H1 under budget | 77.1 / 74.7 / 77.1 | `reports/multiseed_progress.jsonl` (M-2, M-3, b40 rows) |
| §5.6 co-adaptation + paraphrase | 88.6 → 64.6; 56.8; 75.0; 93.2; 52.3 | `reports/S-1-slots-…routed.json`, `S-1-slots-routed-PARAPHRASED.json`, `S-1-slots-semantic-template.json`, `S-1-rag-local-PARAPHRASED.json`; 56.8 pinned to commit da1b072 |
| §5.7 reader ceiling | 58.5 vs 37.8 | `reports/M-1-rag-gpt-4.1-mini-k8.json`, `M-1-rag-Qwen…k8.json` |
| Figures 1–3 | all of the above | `scripts/make_figures.py` |
| Benchmark validity | oracle=100%, stale-oracle fails, determinism | `agentlife/tests/test_agentlife.py` (12 tests) |

Run the full suite (27 tests):

```bash
PYTHONPATH=. python3 -W ignore agentlife/tests/test_agentlife.py
PYTHONPATH=. python3 -W ignore agentlife/tests/test_phase1.py
PYTHONPATH=. python3 agentlife/tests/test_ledger.py
PYTHONPATH=. python3 -W ignore agentlife/tests/test_router.py
```

Environment: Apple Silicon (MLX) for training/reading; `OPENAI_API_KEY`
(or any OpenAI-compatible endpoint via `model[@base_url][#KEY_ENV]`) for
extraction/parsing/embeddings. Total API cost to reproduce from cold
caches: ~US$1.
