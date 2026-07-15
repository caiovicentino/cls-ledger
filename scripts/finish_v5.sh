#!/bin/bash
set -uo pipefail
cd "$(dirname "$0")/.."
echo "== [1/3] multiseed v5 (resume) =="
PYTHONPATH=. python3 -u scripts/multiseed_v5.py 2>&1 | grep -E "^(RUN|RESULT|SKIP|FAILED|ALL)"
echo "== [2/3] re-score seed 9 slots (unit-fix) =="
PYTHONPATH=. python3 -W ignore - <<'PYEOF'
import sys, json
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
from eval_paraphrased import rebuild
from agentlife.harness.scorer import score
from agentlife.schema import load_jsonl
system = rebuild('runs/M-9v5-slots-stable-hybrid-routed',
                 'mlx-community/Qwen2.5-3B-Instruct-4bit', 'cache',
                 'data/M-9v5')
finals = load_jsonl('data/M-9v5/public/final_queries.jsonl')
answers = load_jsonl('data/M-9v5/private/answers.jsonl')
rep = score({q['query_id']: system.answer(q) for q in finals}, answers)
out = 'reports/M-9v5-slots-Qwen2.5-3B-Instruct-4bit-stable-hybrid-routed.json'
# preserve original online rows? report covers finals only after re-score;
# keep it explicit
rep['note'] = 'finals re-scored at HEAD (unit-fix); online rows omitted'
json.dump(rep, open(out, 'w'), indent=2, ensure_ascii=False)
print('seed9 re-scored final acc:', round(rep['accuracy_final']*100, 1))
row = {"dataset": "M-9v5", "seed": 9, "system": "slots", "status": "ok",
       "acc_final": rep["accuracy_final"],
       "by_qtype": {k: v["acc"] for k, v in rep["by_qtype"].items()},
       "adherence": {k: v["rate"] for k, v in
                     rep.get("disposition_adherence", {}).items()}}
with open('reports/multiseed_v5_progress.jsonl', 'a') as f:
    f.write(json.dumps(row) + "\n")
PYEOF
echo "== [3/3] aggregate =="
PYTHONPATH=. python3 scripts/aggregate_v5.py
echo "== DONE =="
