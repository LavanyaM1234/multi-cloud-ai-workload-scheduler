#!/bin/bash
# ── setup_gcs.sh ──────────────────────────────────────────────────
# Run this ONCE to create the GCS bucket and upload the trainer.
#
# Usage:
#   chmod +x setup_gcs.sh
#   ./setup_gcs.sh
# ─────────────────────────────────────────────────────────────────

PROJECT_ID="tensile-method-459009-k2"
REGION="us-central1"
BUCKET="ml-scheduler-jobs-${PROJECT_ID}"

echo "==> Creating GCS bucket: gs://${BUCKET}"
gcloud storage buckets create "gs://${BUCKET}" \
  --project="${PROJECT_ID}" \
  --location="${REGION}" \
  --uniform-bucket-level-access 2>/dev/null || echo "Bucket may already exist, continuing..."

echo "==> Uploading trainer/train.py to GCS"
gcloud storage cp trainer/train.py "gs://${BUCKET}/trainer/train.py"

echo "==> Uploading trainer/startup.sh to GCS"
gcloud storage cp trainer/startup.sh "gs://${BUCKET}/trainer/startup.sh"

echo "==> Enabling required GCP APIs"
gcloud services enable compute.googleapis.com --project="${PROJECT_ID}"
gcloud services enable storage.googleapis.com --project="${PROJECT_ID}"

echo ""
echo "✓ Setup complete!"
echo "  Bucket  : gs://${BUCKET}"
echo "  Trainer : gs://${BUCKET}/trainer/train.py"
echo ""
echo "Set this in your .env file:"
echo "  CHECKPOINT_GCS_BUCKET=${BUCKET}"
