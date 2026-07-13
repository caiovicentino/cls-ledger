"""Adversarial paraphrase test — generation half (API only).

Rewrites each final question with different wording (attribute synonyms,
restructured syntax) while preserving: the entity names, any 'As of day N'
reference, and the answer-format hint. This attacks exactly the lexical
couplings a co-adapted system might have (ATTR_LEXICON, question
templates) without making questions unanswerable.

Validation per paraphrase: entity name still present, day reference still
present (if any), canonical answer NOT leaked into the question.

Usage: PYTHONPATH=. python3 scripts/paraphrase.py --data data/S-1
Writes: <data>/public/final_queries_paraphrased.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.getcwd())

from agentlife.harness.backends import OpenAIBackend  # noqa: E402
from agentlife.harness.scorer import contains, normalize  # noqa: E402
from agentlife.schema import load_jsonl  # noqa: E402

PROMPT = (
    "Rewrite this question using different wording: use synonyms for the "
    "attribute/relation words and restructure the sentence, but KEEP the "
    "exact same meaning and the same answer.\n"
    "Rules:\n"
    "- keep every person/project name EXACTLY as written;\n"
    "- keep any 'As of day N' reference (you may phrase it as 'on day N' "
    "or 'back on day N');\n"
    "- keep the final format instructions (the part starting with "
    "'Answer with' or 'If this was never mentioned') EXACTLY as written;\n"
    "- do not answer the question.\n\n"
    "Question: {q}\n\nRewritten question:"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="gpt-4.1-mini")
    args = ap.parse_args()

    finals = load_jsonl(os.path.join(args.data, "public",
                                     "final_queries.jsonl"))
    answers = {a["query_id"]: a for a in load_jsonl(
        os.path.join(args.data, "private", "answers.jsonl"))}
    backend = OpenAIBackend(args.model, cache_dir="cache")

    out_rows, fallbacks = [], 0
    for q in finals:
        a = answers[q["query_id"]]
        para = backend.complete(
            "You rewrite questions preserving meaning. Output only the "
            "rewritten question.", PROMPT.format(q=q["question"]),
            max_tokens=150).strip().strip('"')
        ok = True
        # entities present?
        for name in re.findall(r"[A-Z][a-z]+ [A-Z][a-z]+|Project [A-Z]\w+",
                               q["question"]):
            if name.lower() not in para.lower():
                ok = False
        # day reference preserved?
        m = re.search(r"As of day (\d+)", q["question"])
        if m and f"day {m.group(1)}" not in para.lower():
            ok = False
        # answer not leaked
        npara = normalize(para)
        if any(contains(v, npara) for v in a["accepted"]
               if v.lower() not in ("unknown", "no", "none")):
            ok = False
        if not ok:
            para = q["question"]  # fall back to original, count it
            fallbacks += 1
        row = dict(q)
        row["question"] = para
        row["paraphrased"] = para != q["question"]
        out_rows.append(row)

    out = os.path.join(args.data, "public",
                       "final_queries_paraphrased.jsonl")
    with open(out, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_para = sum(r["paraphrased"] for r in out_rows)
    print(f"{n_para}/{len(out_rows)} paraphrased "
          f"({fallbacks} fallbacks to original) -> {out}")


if __name__ == "__main__":
    main()
