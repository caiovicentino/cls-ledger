#!/bin/bash
# Zenodo publish flow (per established OpenInterpretability workflow):
# create deposition -> upload PDF + MD + 3 figures -> set metadata -> publish.
# Requires ZENODO_TOKEN in env (ephemeral; never logged by this script).
set -euo pipefail
cd "$(dirname "$0")/.."

: "${ZENODO_TOKEN:?export ZENODO_TOKEN first}"
API="https://zenodo.org/api"

echo "[1/4] creating deposition..."
DEP=$(curl -sf -X POST "$API/deposit/depositions?access_token=${ZENODO_TOKEN}" \
  -H "Content-Type: application/json" -d '{}')
DEP_ID=$(echo "$DEP" | python3 -c "import json,sys; print(json.load(sys.stdin)['id'])")
BUCKET=$(echo "$DEP" | python3 -c "import json,sys; print(json.load(sys.stdin)['links']['bucket'])")
echo "  deposition id: $DEP_ID"

echo "[2/4] uploading files..."
for f in paper/cls-ledger-preprint-v1.pdf paper/draft.md \
         paper/fig1_scale.png paper/fig2_guarantees.png \
         paper/fig3_paraphrase.png; do
  name=$(basename "$f")
  curl -sf -X PUT "${BUCKET}/${name}?access_token=${ZENODO_TOKEN}" \
    -H "Content-Type: application/octet-stream" \
    --data-binary @"$f" > /dev/null
  echo "  uploaded $name"
done

echo "[3/4] setting metadata..."
curl -sf -X PUT "$API/deposit/depositions/${DEP_ID}?access_token=${ZENODO_TOKEN}" \
  -H "Content-Type: application/json" \
  -d @paper/zenodo_metadata.json > /dev/null

echo "[4/4] publishing (mints DOI — irreversible)..."
PUB=$(curl -sf -X POST \
  "$API/deposit/depositions/${DEP_ID}/actions/publish?access_token=${ZENODO_TOKEN}")
DOI=$(echo "$PUB" | python3 -c "import json,sys; print(json.load(sys.stdin)['doi'])")
RECID=$(echo "$PUB" | python3 -c "import json,sys; print(json.load(sys.stdin)['record_id'])")
CONCEPT=$(curl -sf "$API/records/${RECID}" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('conceptdoi','?'))")

echo ""
echo "PUBLISHED"
echo "  version DOI : $DOI"
echo "  concept DOI : $CONCEPT   (use this as the living front door)"
echo "  record      : https://zenodo.org/records/${RECID}"
