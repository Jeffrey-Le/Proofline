#!/usr/bin/env bash
# One-time GCP setup + deploy. Read before running; every step is idempotent-ish.
set -euo pipefail
# Pick up GEMINI_API_KEY / PROJECT / BILLING from .env if present (values already exported win).
if [ -f .env ]; then while IFS='=' read -r k v; do
  [[ "$k" =~ ^[A-Z_]+$ ]] && [ -n "$v" ] && [ -z "${!k:-}" ] && export "$k=$v"; done < .env; fi

PROJECT=${PROJECT:?set PROJECT=your-gcp-project-id}
# What share-link holders see for your resume. Set e.g. OWNER_LABEL="Your Name's resume" when you run
# this if you want your name shown to recruiters; the public page never contains it either way.
REGION=${REGION:-us-west1}
BILLING=${BILLING:?set BILLING=your-billing-account-id}   # gcloud billing accounts list
SERVICE=proofline

command -v gcloud >/dev/null || { echo "Install the gcloud CLI first: https://cloud.google.com/sdk/docs/install"; exit 1; }
gcloud auth application-default print-access-token >/dev/null 2>&1 || {
  echo "Run once first: gcloud auth login && gcloud auth application-default login"; exit 1; }
python -c "import proofline" 2>/dev/null || { echo "Run from the proofline folder with your .venv activated"; exit 1; }

gcloud config set project "$PROJECT"
gcloud auth application-default set-quota-project "$PROJECT" >/dev/null 2>&1 || true
gcloud services enable run.googleapis.com firestore.googleapis.com secretmanager.googleapis.com \
  cloudbuild.googleapis.com artifactregistry.googleapis.com billingbudgets.googleapis.com compute.googleapis.com

# 1) Budget alert at $20/month (emails at $5, $10, $20). Alerts do NOT stop spending.
#    The real cap is DAILY_BUDGET_USD in the app (default $0.50/day of measured model
#    spend for public traffic, so at most ~$15/month), plus max-instances=1.
#    The Gemini key must be on a PAID project: the free tier's terms forbid sending
#    other people's personal information.
gcloud billing budgets create --billing-account="$BILLING" --display-name="$SERVICE-20usd" \
  --budget-amount=20USD --filter-projects="projects/$PROJECT" \
  --threshold-rule=percent=0.25 --threshold-rule=percent=0.5 --threshold-rule=percent=1.0 || true

# 2) Firestore (Native mode).
#    Vector index on (session_id, vector) so every search is pre-filtered to one resume.
#    TTL policies delete visitor data via expire_at (owner rows have no expire_at, so they stay).
gcloud firestore databases create --location="$REGION" || true
# Only "already exists" is OK here; any other error is shown and stops the script.
if ! out=$(gcloud firestore indexes composite create --collection-group=resume_bullets --query-scope=COLLECTION \
     --field-config=field-path=session_id,order=ASCENDING \
     --field-config=field-path=vector,vector-config='{"dimension":"768","flat":"{}"}' --async 2>&1); then
  echo "$out" | grep -qiE 'already exists|ALREADY_EXISTS' || { echo "$out"; echo "Creating the vector index failed (above)."; exit 1; }
fi
gcloud firestore fields ttls update expire_at --collection-group=resume_bullets --enable-ttl --quiet || true
gcloud firestore fields ttls update expire_at --collection-group=sessions --enable-ttl --quiet || true

# 3) Secrets. GEMINI_API_KEY must come from a project WITH billing (paid tier), see step 1.
printf '%s' "${GEMINI_API_KEY:?export GEMINI_API_KEY first}" | \
  gcloud secrets create gemini-api-key --data-file=- 2>/dev/null || \
  printf '%s' "$GEMINI_API_KEY" | gcloud secrets versions add gemini-api-key --data-file=-
API_TOKEN=${API_TOKEN:-$(openssl rand -hex 16)}
printf '%s' "$API_TOKEN" | gcloud secrets create proofline-token --data-file=- 2>/dev/null || true
# Share token: the private link you send recruiters. Rotate it with
#   printf '%s' "$(openssl rand -hex 12)" | gcloud secrets versions add proofline-share --data-file=-
SHARE_TOKEN=${SHARE_TOKEN:-$(openssl rand -hex 12)}
printf '%s' "$SHARE_TOKEN" | gcloud secrets create proofline-share --data-file=- 2>/dev/null || true

SA="$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')-compute@developer.gserviceaccount.com"
for s in gemini-api-key proofline-token proofline-share; do
  gcloud secrets add-iam-policy-binding "$s" --member="serviceAccount:$SA" --role=roles/secretmanager.secretAccessor >/dev/null
done
gcloud projects add-iam-policy-binding "$PROJECT" --member="serviceAccount:$SA" --role=roles/datastore.user >/dev/null
# Newer projects build source deploys with this account, which needs build permissions.
gcloud projects add-iam-policy-binding "$PROJECT" --member="serviceAccount:$SA" --role=roles/run.builder >/dev/null || true

# Wait for the vector index: searches fail until it's READY (usually a few minutes).
echo -n "Waiting for the Firestore vector index"
# Read the index list as JSON and find the one on resume_bullets with a vector field.
index_state() {
  gcloud firestore indexes composite list --format=json 2>/dev/null | python3 -c '
import json, sys
try: idx = json.load(sys.stdin)
except Exception: idx = []
for x in idx:
    vec = any("vectorConfig" in f for f in x.get("fields", []))
    if vec and "resume_bullets" in x.get("name", ""):
        print(x.get("state", "")); break
' || true
}
state=""
for i in $(seq 1 60); do
  state=$(index_state)
  [ "$state" = "READY" ] && { echo " ready."; break; }
  [ -z "$state" ] && [ "$i" -ge 6 ] && { echo; echo "No vector index listed. Paste the output of:"; \
    echo "  gcloud firestore indexes composite list --format=json"; exit 1; }
  echo -n "."; sleep 10
done
[ "$state" = "READY" ] || { echo; echo "Index still building ($state). Re-run this script in a few minutes."; exit 1; }

# 4) Load bullets into Firestore from your machine (uses your gcloud ADC).
VECTOR_BACKEND=firestore GOOGLE_CLOUD_PROJECT="$PROJECT" python -m scripts.load_bullets            # yours (private)
VECTOR_BACKEND=firestore GOOGLE_CLOUD_PROJECT="$PROJECT" python -m scripts.load_bullets --sample   # fictional (public)

# 5) Deploy. Scale-to-zero, one instance max, small box.
gcloud run deploy "$SERVICE" --source . --region "$REGION" --allow-unauthenticated \
  --min-instances=0 --max-instances=1 --concurrency=4 --cpu=1 --memory=1Gi --timeout=120 \
  --set-env-vars="^|^VECTOR_BACKEND=firestore|GOOGLE_CLOUD_PROJECT=$PROJECT|PER_IP_PER_HOUR=8|DAILY_BUDGET_USD=0.50|OWNER_LABEL=${OWNER_LABEL:-Shared resume}|OWNER_NOTE=${OWNER_NOTE:-A resume shared with you through a private link.}" \
  --set-secrets=GEMINI_API_KEY=gemini-api-key:latest,API_TOKEN=proofline-token:latest,SHARE_TOKEN=proofline-share:latest

URL=$(gcloud run services describe "$SERVICE" --region "$REGION" --format='value(status.url)')
echo "Public site: $URL   (8 requests per IP per hour, \$0.50/day public spend cap)"
SHARE=$(gcloud secrets versions access latest --secret=proofline-share)
echo "Recruiter link (checks against your real resume): $URL/?for=$SHARE"
API_TOKEN=$(gcloud secrets versions access latest --secret=proofline-token)
echo "Your owner link, keep private (no limits, shows diagnostics): $URL/#token=$API_TOKEN"
echo
echo "Smoke test: python -m scripts.smoke --url $URL --share $SHARE"
