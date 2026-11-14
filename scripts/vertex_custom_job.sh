#!/usr/bin/env bash
# Submit reproducible Vertex AI Custom Training Job (1 NVIDIA T4 GPU) and configure Gemini 4 Argon (§8)
set -euo pipefail

PROJECT_ID="${GCP_PROJECT_ID:-}"
BUCKET="${GCS_BUCKET:-}"
REGION="${GCP_REGION:-us-west1}"
IMAGE_URI="${VERTEX_IMAGE_URI:-us-west1-docker.pkg.dev/${PROJECT_ID}/ucsc-wildlife/trainer:v1}"
RUN_ID="${RUN_ID:-efficientnet_b0_$(date +%Y%m%d_%H%M%S)}"

if [[ -z "$PROJECT_ID" || -z "$BUCKET" ]]; then
  echo "Usage: GCP_PROJECT_ID=<project> GCS_BUCKET=<bucket> ./scripts/vertex_custom_job.sh"
  echo "Note: Ensure Vertex AI, Artifact Registry, and Cloud Storage APIs are enabled first."
  exit 1
fi

gcloud config set project "$PROJECT_ID"

gcloud ai custom-jobs create \
  --region="${REGION}" \
  --display-name="ucsc-wildlife-${RUN_ID}" \
  --worker-pool-spec="machine-type=n1-standard-4,replica-count=1,accelerator-type=NVIDIA_TESLA_T4,accelerator-count=1,container-image-uri=${IMAGE_URI}" \
  --args="-m,wildlife.train,--config,configs/train.yaml,--manifest,/gcs/${BUCKET}/ucsc-wildlife/datasets/v1/manifest.parquet,--splits,/gcs/${BUCKET}/ucsc-wildlife/datasets/v1/split_v1.csv,--output,/gcs/${BUCKET}/ucsc-wildlife/experiments/${RUN_ID}"
