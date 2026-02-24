#!/usr/bin/env bash
# ============================================================
# NLPMetricLab - Create NEW deposit on Zenodo (v1.0)
# Usage: export ZENODO_TOKEN=<token> && bash zenodo_publish.sh
# ============================================================
set -euo pipefail

if [ -z "${ZENODO_TOKEN:-}" ]; then
  echo "Error: ZENODO_TOKEN environment variable is not set."
  echo "Export it before running: export ZENODO_TOKEN=\$(cat /path/to/zenodo.txt)"
  exit 1
fi
TOKEN="$ZENODO_TOKEN"
BASE="https://zenodo.org/api"

DESCRIPTION="NLPMetricLab: companion evaluation system for the study 'Paradox of Poetic Intent in Back-Translation' (FITEE, Springer, doi:10.1631/FITEE.2500298). Provides interactive computation of BLEU, CHRF, TER, and Semantic Similarity on paired texts with native Chinese support (Han detection, Jieba tokenization). Published as a Streamlit web application."

echo "=== Step 1: Create empty deposit ==="
DEPOSIT=$(curl -s -X POST "$BASE/deposit/depositions" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{}')

DEPOSIT_ID=$(echo "$DEPOSIT" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
echo "Deposit created with ID: $DEPOSIT_ID"

echo ""
echo "=== Step 2: Set metadata ==="
METADATA_JSON=$(python3 << PYEOF
import json
with open('.zenodo.json') as f:
    zj = json.load(f)
data = {
    'metadata': {
        'title': zj['title'],
        'version': '1.0',
        'description': """$DESCRIPTION""",
        'upload_type': zj.get('upload_type', 'software'),
        'publication_date': '$(date +%Y-%m-%d)',
        'creators': zj['creators'],
        'keywords': zj.get('keywords', []),
        'license': 'cc-by-nc-sa-4.0',
        'access_right': 'open',
        'related_identifiers': zj.get('related_identifiers', []),
        'notes': zj.get('references', [''])[0] if zj.get('references') else ''
    }
}
print(json.dumps(data))
PYEOF
)

curl -s -X PUT "$BASE/deposit/depositions/$DEPOSIT_ID" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "$METADATA_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print('Title:', d.get('metadata',{}).get('title','?')); print('PrereserveDOI:', d.get('metadata',{}).get('prereserve_doi',{}).get('doi','N/A'))"

echo ""
echo "=== Step 3: Create zip of repository ==="
cd "$(dirname "$0")"
ZIP_NAME="NLPMetricLab-1.0.zip"
zip -r "/tmp/$ZIP_NAME" . \
  -x ".git/*" \
  -x ".venv/*" \
  -x "zenodo_publish.sh" \
  -x "__pycache__/*" \
  -x "NLPMetricLab.zip" \
  2>/dev/null
echo "Created /tmp/$ZIP_NAME ($(du -h /tmp/$ZIP_NAME | cut -f1))"

echo ""
echo "=== Step 4: Upload zip file ==="
BUCKET_URL=$(echo "$DEPOSIT" | python3 -c "import sys,json; print(json.load(sys.stdin)['links']['bucket'])")
curl -s -X PUT "$BUCKET_URL/$ZIP_NAME" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/octet-stream" \
  --data-binary "@/tmp/$ZIP_NAME" | python3 -c "import sys,json; d=json.load(sys.stdin); print('Uploaded:', d.get('key','?'), '- Size:', d.get('size','?'), 'bytes')"

echo ""
echo "============================================"
echo "=== REVIEW BEFORE PUBLISHING ==="
echo "============================================"
echo ""
echo "Draft URL: https://zenodo.org/deposit/$DEPOSIT_ID"
echo ""
echo "Open the URL above in your browser to review."
echo ""
read -p "Publish this deposit? (y/N) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
  RESULT=$(curl -s -X POST "$BASE/deposit/depositions/$DEPOSIT_ID/actions/publish" \
    -H "Authorization: Bearer $TOKEN")
  NEW_DOI=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['doi'])" 2>/dev/null || echo "check Zenodo")
  CONCEPT_DOI=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['conceptdoi'])" 2>/dev/null || echo "check Zenodo")
  echo ""
  echo "Published!"
  echo "   Version DOI: $NEW_DOI"
  echo "   Concept DOI: $CONCEPT_DOI"
  echo ""
  echo "   IMPORTANT: Update these files with the new DOIs:"
  echo "   - README.md (badge and citation)"
  echo "   - .zenodo.json (if needed)"
else
  echo ""
  echo "Draft saved but NOT published."
  echo "Edit/publish manually at: https://zenodo.org/deposit/$DEPOSIT_ID"
fi

# Cleanup
rm -f "/tmp/$ZIP_NAME"
echo ""
echo "Done."
