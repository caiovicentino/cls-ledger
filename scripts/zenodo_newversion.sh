#!/bin/bash
# Zenodo v2 via actions/newversion on the published v1 deposition.
# Per documented workflow: inherit metadata and modify ONLY changed fields
# (re-specifying from scratch silently drops related_identifiers);
# replace inherited files with the v2 set. Requires ZENODO_TOKEN (ephemeral).
set -euo pipefail
cd "$(dirname "$0")/.."

: "${ZENODO_TOKEN:?export ZENODO_TOKEN first}"
API="https://zenodo.org/api"
V1_DEP=21375429

echo "[1/6] creating new version draft from deposition ${V1_DEP}..."
NV=$(curl -sf -X POST "$API/deposit/depositions/${V1_DEP}/actions/newversion?access_token=${ZENODO_TOKEN}")
DRAFT_URL=$(echo "$NV" | python3 -c "import json,sys; print(json.load(sys.stdin)['links']['latest_draft'])")
DEP_ID=$(basename "$DRAFT_URL")
echo "  draft deposition: $DEP_ID"

echo "[2/6] reading inherited metadata + files..."
DRAFT=$(curl -sf "$API/deposit/depositions/${DEP_ID}?access_token=${ZENODO_TOKEN}")
BUCKET=$(echo "$DRAFT" | python3 -c "import json,sys; print(json.load(sys.stdin)['links']['bucket'])")
echo "$DRAFT" > /tmp/zen_draft.json

echo "[3/6] removing inherited files..."
echo "$DRAFT" | python3 -c "
import json, sys
for f in json.load(sys.stdin).get('files', []):
    print(f['id'])" | while read -r fid; do
  curl -sf -X DELETE "$API/deposit/depositions/${DEP_ID}/files/${fid}?access_token=${ZENODO_TOKEN}" || true
  echo "  removed file $fid"
done

echo "[4/6] uploading v2 files..."
for f in paper/cls-ledger-preprint-v2.pdf paper/draft.md \
         paper/fig1_scale.png paper/fig2_guarantees.png \
         paper/fig3_paraphrase.png paper/fig4_v5.png; do
  name=$(basename "$f")
  curl -sf -X PUT "${BUCKET}/${name}?access_token=${ZENODO_TOKEN}" \
    -H "Content-Type: application/octet-stream" \
    --data-binary @"$f" > /dev/null
  echo "  uploaded $name"
done

echo "[5/6] updating metadata (inherited, modify only changed fields)..."
python3 - <<'PYEOF'
import json
draft = json.load(open('/tmp/zen_draft.json'))
md = draft['metadata']
md['version'] = 'v2'
md['publication_date'] = draft['metadata'].get('publication_date') or None
md['description'] = md['description'].replace(
    '</p>',
    ' <b>v2 additions:</b> quantitative induction resolved as ledger '
    'aggregation (pooled 95.8% vs. 33.7% for the stronger retrieval '
    'baseline over 4 seeds) and behavioral dispositions applied by the '
    'ledger to every answer with no reminder (rule-based adherence '
    '100%/77.5% vs. at most 12.7% for retrieval), with pre-registered '
    'predictions, a marginally missed prediction declared, and '
    'unit-normalization hazards in numeric aggregation reported.</p>', 1)
md = {k: v for k, v in md.items() if v is not None}
json.dump({'metadata': md}, open('/tmp/zen_md.json', 'w'))
print('  metadata prepared (version=v2, description extended; '
      f"related_identifiers preserved: {len(md.get('related_identifiers', []))})")
PYEOF
curl -sf -X PUT "$API/deposit/depositions/${DEP_ID}?access_token=${ZENODO_TOKEN}" \
  -H "Content-Type: application/json" -d @/tmp/zen_md.json > /dev/null

echo "[6/6] publishing (mints v2 DOI — irreversible)..."
PUB=$(curl -sf -X POST "$API/deposit/depositions/${DEP_ID}/actions/publish?access_token=${ZENODO_TOKEN}")
DOI=$(echo "$PUB" | python3 -c "import json,sys; print(json.load(sys.stdin)['doi'])")
RECID=$(echo "$PUB" | python3 -c "import json,sys; print(json.load(sys.stdin)['record_id'])")
echo ""
echo "PUBLISHED v2"
echo "  v2 DOI      : $DOI"
echo "  concept DOI : 10.5281/zenodo.21375428 (now resolves to v2)"
echo "  record      : https://zenodo.org/records/${RECID}"
rm -f /tmp/zen_draft.json /tmp/zen_md.json
