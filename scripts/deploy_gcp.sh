#!/usr/bin/env bash
# Deploy SpecSnap to Google Cloud Run
set -euo pipefail

PROJECT_ID="${GCP_PROJECT_ID:-project-90550666-d7db-40a5-868}"
REGION="${GCP_REGION:-asia-south1}"
SERVICE="specsnap"

gcloud config set project "$PROJECT_ID"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com

DEPLOY_ARGS=(
  run deploy "$SERVICE"
  --source .
  --region "$REGION"
  --platform managed
  --allow-unauthenticated
  --memory 1Gi
  --cpu 1
  --timeout 120
  --min-instances 0
  --max-instances 3
)

if [[ -n "${GROQ_API_KEY:-}" ]]; then
  DEPLOY_ARGS+=(--set-env-vars "GROQ_API_KEY=${GROQ_API_KEY}")
fi

gcloud "${DEPLOY_ARGS[@]}"

echo "Deployed. URL:"
gcloud run services describe "$SERVICE" --region "$REGION" --format 'value(status.url)'
