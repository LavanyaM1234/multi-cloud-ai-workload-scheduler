#!/bin/bash
# ── trainer/startup.sh ────────────────────────────────────────────
# GCP VM startup script injected via instance metadata by launcher.py
#
# Reads from instance metadata:
#   JOB_ID          — unique job identifier
#   GCS_BUCKET      — bucket with job_config.json + checkpoints
#   INSTANCE_TYPE   — e.g. e2-standard-4
#   RESUME_STEP     — 0 for fresh start, N to resume from step N
#   PREV_CLOUD      — which cloud we migrated from (for logging)
#
# Exits 0 on success or preemption (VM shuts itself down).
# ─────────────────────────────────────────────────────────────────

set -e
LOG="/var/log/trainer.log"
exec > >(tee -a "$LOG") 2>&1

echo "=========================================="
echo " ML Scheduler — VM Startup"
echo " $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "=========================================="

# ── 1. Read metadata ──────────────────────────────────────────────
META="http://metadata.google.internal/computeMetadata/v1/instance/attributes"
H='Metadata-Flavor: Google'

JOB_ID=$(       curl -sf "${META}/JOB_ID"        -H "$H" || echo "unknown")
GCS_BUCKET=$(   curl -sf "${META}/GCS_BUCKET"    -H "$H" || echo "")
INSTANCE_TYPE=$(curl -sf "${META}/INSTANCE_TYPE" -H "$H" || echo "e2-standard-4")
RESUME_STEP=$(  curl -sf "${META}/RESUME_STEP"   -H "$H" || echo "0")
PREV_CLOUD=$(   curl -sf "${META}/PREV_CLOUD"    -H "$H" || echo "")

export JOB_ID GCS_BUCKET INSTANCE_TYPE RESUME_STEP PREV_CLOUD
export CLOUD="gcp"

echo "JOB_ID       = ${JOB_ID}"
echo "GCS_BUCKET   = ${GCS_BUCKET}"
echo "INSTANCE_TYPE= ${INSTANCE_TYPE}"
echo "RESUME_STEP  = ${RESUME_STEP}"
echo "PREV_CLOUD   = ${PREV_CLOUD:-fresh start}"

if [ "${RESUME_STEP}" != "0" ]; then
  echo ""
  echo ">>> RESUMING from step ${RESUME_STEP}"
  if [ -n "${PREV_CLOUD}" ]; then
    echo ">>> Cross-cloud migration: ${PREV_CLOUD} → gcp"
  fi
fi

# ── 2. Install dependencies ───────────────────────────────────────
echo ""
echo "==> Installing dependencies..."

apt-get update -qq
apt-get install -y -qq python3-pip python3-venv

python3 -m venv /opt/trainer-env
source /opt/trainer-env/bin/activate

pip install --quiet --upgrade pip
pip install --quiet \
    torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu
pip install --quiet \
    google-cloud-storage \
    numpy

echo "==> Dependencies ready"

# ── 3. Download train.py from GCS ────────────────────────────────
echo ""
echo "==> Downloading trainer..."
mkdir -p /opt/trainer
cd /opt/trainer

gsutil cp "gs://${GCS_BUCKET}/trainer/train.py" ./train.py
echo "==> train.py ready"

# ── 4. Run training ───────────────────────────────────────────────
echo ""
echo "==> Starting training (resume_step=${RESUME_STEP})..."
echo ""

python3 train.py
EXIT_CODE=$?

# ── 5. Self-destruct: shutdown VM to stop billing ─────────────────
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "==> Training finished (exit 0). Shutting down in 60s..."
else
    echo "==> Training exited with code ${EXIT_CODE}. Shutting down in 60s..."
fi

echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') — shutdown queued"

# 60s delay so final logs flush to Cloud Logging
sleep 60
shutdown -h now
