# CLS-Ledger

Research project: **selective parametric consolidation with provenance** —
closing the experience→weights loop for LLM agents in a way that is
*selective* (usage-gated), *bounded* (frozen backbone + isolated memory
slots) and *reversible* (a ledger maps facts to parameter locations, so
consolidated knowledge can be audited and deleted).

Phase 0 (this directory): **AgentLife**, the measurement instrument.

## AgentLife benchmark

AgentLife generates synthetic-but-realistic "lives" of a personal-assistant
agent: a stream of dated episodes (facts stated, updated, revoked, plus
noise) followed by a final exam. Answers are *derived from a programmatic
temporal world state*, never hand-annotated.

Design principles (each one is a response to a documented failure of
existing agent-memory benchmarks, e.g. the LoCoMo audit that found 6.4% of
its answer key wrong and an LLM judge accepting ~63% of wrong answers):

1. **Answer key correct by construction.** A privileged oracle that reads
   the world state must score exactly 100%; this is enforced in CI.
2. **Exact, auditable scoring. No LLM judge.** Every query ships with
   `accepted` (all correct surface forms) and `rejected` (superseded
   versions, confusable-entity values) strings, matched with word
   boundaries after normalization.
3. **Freshness is falsifiable.** A deliberately stale oracle (always
   answers the first version of a fact) must score 0% on
   `current_updated` queries — enforced in CI. Facts whose value cycles
   back (A→B→A) are excluded from freshness probes.
4. **Typed error taxonomy.** Responses are classified as
   `correct / stale / hallucination / wrong_value / ambiguous / abstain /
   miss`, so failure modes are measurable, not vibes.
5. **Usage is instrumented.** "Online" queries during the life mark facts
   hot/cold and produce per-fact retrieval statistics — the input for
   consolidation-policy experiments (consolidate hot+stable facts, keep
   volatile facts external).
6. **Deterministic.** Same preset+seed → byte-identical dataset, across
   processes and hash seeds (enforced in CI).

### Query types

| type | probes |
|---|---|
| `current_stable` | retention of facts that never changed |
| `current_updated` | freshness after 1..k updates (old values rejected) |
| `current_volatile` | freshness of high-churn facts (parking spot, desk) |
| `point_in_time` | temporal reasoning: value valid *as of day D* |
| `multihop_2/3` | composition across episodes ≥5 days apart |
| `revoked` | explicit revocations (cancelled project, dropped diet) |
| `absent` | facts never stated — measures hallucination/abstention |
| `online` | mid-life queries; builds the hot/cold usage log |

### Usage

```bash
# generate a dataset (presets S, M, L, XL)
PYTHONPATH=. python3 -m agentlife.generate --preset M --seed 1 --out data/M-1

# evaluate a reference system (oracle must print 100%)
PYTHONPATH=. python3 -m agentlife.harness.run_eval --data data/M-1 --system oracle
PYTHONPATH=. python3 -m agentlife.harness.run_eval --data data/M-1 --system stale
PYTHONPATH=. python3 -m agentlife.harness.run_eval --data data/M-1 --system null

# self-validation suite
PYTHONPATH=. python3 agentlife/tests/test_agentlife.py
```

Dataset layout: `public/` (episode stream + questions) is all a system may
see; `private/` (answer key, world state, usage stats) is for scoring only.

### Reference-system signature (M-1)

| system | final acc | signature |
|---|---|---|
| oracle | 100% | validates answer key + scorer end to end |
| stale | 64.6% | 0% on updated/volatile/revoked, 100% on stable — rejected-sets catch staleness |
| null | 9.8% | 100% on absent only — abstention floor |

### Results so far (S-1 = 30 days; M-1 = 90 days; gpt-4.1-mini for
context systems, Qwen2.5-3B-Instruct-4bit + LoRA on an M4 for parametric)

Context-based (memory = context):

| system | S-1 final | M-1 final | notes |
|---|---|---|---|
| oracle | 100% | 100% | answer-key validation |
| full-context | 86.4% | 76.8% | degrades with life length; multihop_3 0% on M |
| RAG BM25 k=8 | 70.5% | 58.5% | revoked 0% on M, 24 stale errors — freshness fails |

Parametric (memory = weights, answers with NO context; S-1):

| system | final | cur_stable | cur_updated | stale errs | general probe |
|---|---|---|---|---|---|
| base model | — | — | — | — | 93.8% |
| naive LoRA (raw text) | 15.9% | 0% | 0% | 2 | 87.5% |
| SEAL-lite (episode QAs) | 20.5% | 37.5% | 25.0% | 10 | 87.5% |
| **CLS-Ledger v0 (state QAs)** | **31.8%** | **75.0%** | **75.0%** | **3** | 68.8% |

Readings: (1) raw-text finetuning does not produce queryable knowledge;
(2) episode-level self-edits memorize stale mentions — SEAL's stale errors
triple CLS's; (3) state-level consolidation with supersedence is the right
write-unit: 3x SEAL on updated facts; (4) BUT a monolithic adapter pays
heavy interference (general probe 93.8→68.8, "capital of France →
Tampere") — the case for isolated slots + replay in v1.

### Roadmap

- **v1 (Phase 2 cont.)** — hybrid router (volatile/point-in-time queries →
  episodic store; consolidated facts → weights), per-entity slots or
  replay mix-in to bound forgetting, fewer iters; M-1 runs.
- **Phase 3** — H1 ablations (policy: stable vs. hot vs. all — the
  usage-gating experiment), H3 unlearning quantification
  (clsledger/unlearn_eval.py), 7–8B scale check, write-up.
- Pre-registered success criteria (unchanged): ≥90% recall of consolidated
  facts with external memory removed; ≤1% general-probe degradation; slot
  deletion removes the fact with <1% collateral damage.
