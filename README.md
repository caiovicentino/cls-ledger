# CLS-Ledger

**Paper (v1, July 2026):** *The Ledger Knows What the Weights Know: Entity-Level
Parametric Memory with Provenance Guarantees for LLM Agents* —
[doi:10.5281/zenodo.21375428](https://doi.org/10.5281/zenodo.21375428)
(concept DOI, always latest; this version: 10.5281/zenodo.21375429)

> Vicentino, C. (2026). The Ledger Knows What the Weights Know. Zenodo.
> https://doi.org/10.5281/zenodo.21375429


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

### Backends: any model, any endpoint

Chat backends accept `provider:model[@base_url][#KEY_ENV]`:

```bash
# default: OpenAI (OPENAI_API_KEY; OPENAI_BASE_URL overrides globally)
--extractor "openai:gpt-4.1-mini"
# GLM via z.ai (or any OpenAI-compatible provider)
--extractor "openai:glm-5.2@https://api.z.ai/api/paas/v4#ZAI_API_KEY"
# Ollama / vLLM / LM Studio (no key needed)
--extractor "openai:qwen2.5:3b@http://localhost:11434/v1"
# fully local: extraction+parsing on the same MLX model as the reader
--extractor "mlx:mlx-community/Qwen2.5-3B-Instruct-4bit"
```

Caveat measured on 8 episodes: a 3B local extractor agrees with
gpt-4.1-mini on only 1/11 cards (extracts noise as facts) — extraction
quality is a real dependency, not a formality. The reader/consolidation
side is MLX (Apple Silicon); porting slot training + routed activation to
CUDA (PEFT) is future work.

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

### v1 results (hybrid router + abstention replay; same local 3B for both
systems — fair comparison)

| dataset | RAG local k=8 | CLS v1 hybrid | queries answered from weights | v1 general probe |
|---|---|---|---|---|
| S-1 | 52.3% | 50.0% | 20/44 (45%) | 81.2% (v0: 68.8%) |
| M-1 | 37.8% | 35.4% | 30/82 (37%) | 75.0% |

Readings: (1) CLS v1 matches local-RAG accuracy while answering ~40% of
queries with ZERO context tokens — the cost thesis holds; (2) abstention
replay fixed absent (0%→100% on S-1) and cut forgetting by +12pp, but
binding confusion between twin entities appeared at reduced iters (Project
Auriga's client answered with Project Aurora's); (3) at M-1 scale the
episodic path (BM25 + 3B reader) is the bottleneck — both systems fail
revoked (0%) and point-in-time (20%) locally; (4) the reader model
dominates everything: gpt-4.1-mini RAG scores +20pp over local on the same
data.

### v2.3 results (periodic sleep + ledger-resolved snapshots + symbolic
chain resolution + canonical extraction; local 3B throughout)

| dataset | CLS v1 | CLS v2.3 | RAG local | RAG 4.1-mini | full 4.1-mini |
|---|---|---|---|---|---|
| S-1 | 50.0% | **84.1%** | 52.3% | 70.5% | 86.4% |
| M-1 | 35.4% | **63.4%** | 37.8% | 58.5% | 76.8% |

Online (mid-life) accuracy: S-1 80.0%, M-1 53.7% (v1: 33.3 / 26.9).
Routes on M-1: 63 weights / 68 episodic / 13 symbolic. General probe
after 3 sleeps: 75.0% (base 93.8%) — forgetting still the open front.

What did it: (1) temporal resolution moved from the reader into the
ledger (point-in-time 20%→80-100%: the reader receives facts already
resolved at the asked day); (2) chain questions resolved symbolically in
the ledger (multihop 0%→70%: a 3B reader cannot compose two hops even
with both facts retrieved and highlighted); (3) canonical attribute
vocabulary in extraction (Zep-style prescribed ontology) so supersedence
actually merges facts; (4) periodic sleep gives the weights route mid-life
and the ledger guards weight-staleness by construction (a fact updated
after the last sleep drops out of the consolidated set and routes
episodic).

### v3 results — isolated slots with routed activation

Architecture: untouched base model + one rank-4 LoRA slot per entity;
the ledger's router activates ONLY the queried entity's slot per query
(swap of small tensors, ~ms). Slot recipe that fixed slot collapse:
5 epochs, lr 5e-5, +10 anchor rows (6 general facts, 4 out-of-scope
abstentions) per slot. Exact fusion (rank concat) is also implemented but
routed activation dominates it: summed deltas amplify the slots' common
format component (fused S-1: 52.3%).

| metric | monolithic (v2.3) | slots-routed (v3) |
|---|---|---|
| S-1 final | 84.1% | **88.6%** (beats full-context-4.1-mini 86.4%) |
| general probe | 75.0% | **93.8% = base model exactly (0.0% forgetting)** |
| unlearning collateral | 19/40 answers changed (re-distill) | **0/40 (slot drop, no retraining)** |

Both pre-registered success criteria met: forgetting ≤1% -> 0.0%;
unlearning collateral <1% -> 0/40, target facts -> 'unknown'.

H1 under a capacity budget (M-1, slots-routed, 40 cards max):
**hot 79.3% > stable 76.8% > random 74.4%** — usage-gated selection wins
under budget, in the predicted order (n=82; replicate across seeds for
significance). The hot policy with 11 slots on a local 3B matches
full-context gpt-4.1-mini (76.8%).

### v5 results — capabilities retrieval structurally lacks (4 seeds, M preset)

| cell | CLS-Ledger | BM25 RAG | Embeddings RAG |
|---|---|---|---|
| overall | **74.9 ± 7.2%** | 38.4 ± 2.6% | 48.3 ± 3.3% |
| induction: count | **87.5 ± 10.2%** | 31.2% | 21.9% |
| induction: habit | **100 ± 0%** | 0% | 12.5% |
| induction: trend | **100 ± 0%** | 66.7% | 66.7% |
| adherence: dates day-first | **100 ± 0%** | 0% | 0% |
| adherence: cities uppercase | **77.5 ± 7.9%** | 12.7% | 0% |

4/4 paired wins vs. both retrievals (+36.6pp / +26.7pp). Induction is
ledger aggregation (a query, not retrieval); dispositions are applied by
the ledger to every answer — a preference declaration is never similar
to the questions it governs, so top-k cannot surface it. Paper v2:
`paper/cls-ledger-preprint-v2.pdf`.

### Roadmap (next)

- Fix bindings: more paraphrases per card + ~450-600 iters with replay;
  contrastive rows for confusable-entity pairs.
- Stronger episodic path: better retrieval (per-fact cards instead of raw
  episodes) and/or larger local reader.
- H1 ablations on M-1 (policy stable vs. hot vs. all) once episodic path
  is no longer the confounder.
- Real isolated slots (per-entity LoRA fusion in MLX) for surgical
  unlearning — v0 measured 19/40 collateral answer changes on re-distill.
- Pre-registered success criteria (unchanged): ≥90% recall of consolidated
  facts with external memory removed; ≤1% general-probe degradation; slot
  deletion removes the fact with <1% collateral damage.
