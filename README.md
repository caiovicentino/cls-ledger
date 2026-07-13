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

### Roadmap

- **Phase 1** — real baselines: full-context, RAG top-k, naive continual
  LoRA, SEAL-lite. Where does each break, at which life length?
- **Phase 2** — CLS-Ledger mechanism: frozen 3–4B backbone + addressable
  sparse memory slots written by slot-targeted context distillation,
  consolidation gated by usage statistics (frequency × stability), ledger
  for reversible unlearning. Pre-registered success criteria: ≥90% recall
  of consolidated facts with external memory removed; ≤1% general-benchmark
  degradation; slot deletion removes the fact with <1% collateral damage.
- **Phase 3** — ablations (selection policy: usage vs. surprise vs.
  recency), 7–8B scale check, write-up.
