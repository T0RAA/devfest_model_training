#!/usr/bin/env bash
# Upload approved dataset manifests, splits, and model artifacts to Google Cloud Storage (§7.2.1 & §7.3)
set -euo pipefail

PROJECT_ID="${GCP_PROJECT_ID:-}"
BUCKET="${GCS_BUCKET:-}"
PREFIX="${GCS_PREFIX:-ucsc-wildlife}"
REGION="${GCP_REGION:-us-west1}"

if [[ -z "$PROJECT_ID" || -z "$BUCKET" ]]; then
  echo "Usage: GCP_PROJECT_ID=<project> GCS_BUCKET=<bucket> ./scripts/upload_to_gcs.sh [--create-bucket]"
  echo "Blocker: GCP_PROJECT_ID and GCS_BUCKET must be set and authenticated via 'gcloud auth login'."
  exit 1
fi

gcloud config set project "$PROJECT_ID"

if [[ "${1:-}" == "--create-bucket" ]]; then
  gcloud storage buckets create "gs://${BUCKET}" --location="${REGION}" || true
fi

echo "Syncing manifests and splits to gs://${BUCKET}/${PREFIX}/datasets/v1/ ..."
gcloud storage cp data/manifests/manifest.parquet "gs://${BUCKET}/${PREFIX}/datasets/v1/manifest.parquet"
gcloud storage cp data/splits/split_v1.csv "gs://${BUCKET}/${PREFIX}/datasets/v1/split_v1.csv"
gcloud storage cp data/splits/classes.json "gs://${BUCKET}/${PREFIX}/datasets/v1/classes.json"
gcloud storage cp data/manifests/photo_attribution.csv "gs://${BUCKET}/${PREFIX}/datasets/v1/attribution.csv"
gcloud storage cp data/manifests/license_audit.json "gs://${BUCKET}/${PREFIX}/datasets/v1/license_audit.json"

if [[ -d "data/processed" ]]; then
  gcloud storage rsync --recursive data/processed "gs://${BUCKET}/${PREFIX}/datasets/v1/images"
fi

if [[ -d "runs/efficientnet_b0_v1" ]]; then
  gcloud storage rsync --recursive runs/efficientnet_b0_v1 "gs://${BUCKET}/${PREFIX}/experiments/efficientnet_b0_v1"
fi

echo "GCS sync complete: gs://${BUCKET}/${PREFIX}/"
