#!/bin/bash
# ── trainer/startup.sh ────────────────────────────────────────────
# GCP VM startup script — injected via instance metadata by launcher.py
#
# NOTE: This file is used when a GCP VM is launched directly (e.g. manual
# testing or legacy paths). When launching via scheduler/launcher.py the
# startup script is generated inline by _build_training_script() and this
# file is NOT used. Both scripts do the same thing — this one is kept for
# reference and manual use.
#
# Reads from GCP instance metadata:
#   JOB_ID                — unique job id
#   CHECKPOINT_S3_BUCKET  — S3 bucket (single source of truth)
#   AWS_ACCESS_KEY_ID     — S3 credentials
#   AWS_SECRET_ACCESS_KEY
#   AWS_DEFAULT_REGION
#   CHECKPOINT_GCS_BUCKET — GCS bucket (optional, .pt storage only)
#   INSTANCE_TYPE         — e.g. e2-standard-4
#   RESUME_STEP           — 0 = fresh, N = resume from step N
#   PREV_CLOUD            — cloud we migrated from (empty on fresh start)
#
# File layout expected in S3 bucket:
#   trainer/train.py              ← uploaded by launcher
#   trainer/checkpoint_pkg.tar.gz ← uploaded by launcher (checkpoint/ folder)
#   checkpoints/{job_id}/job_config.json ← uploaded by launcher
# ─────────────────────────────────────────────────────────────────

set -e
LOG="/var/log/trainer.log"
exec > >(tee -a "$LOG") 2>&1

echo "=========================================="
echo " ML Scheduler — VM Startup"
echo " $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "=========================================="

# ── 1. Read instance metadata ─────────────────────────────────────
META="http://metadata.google.internal/computeMetadata/v1/instance/attributes"
H="Metadata-Flavor: Google"

JOB_ID=$(        curl -sf "${META}/JOB_ID"                -H "$H" || echo "unknown")
INSTANCE_TYPE=$( curl -sf "${META}/INSTANCE_TYPE"         -H "$H" || echo "e2-standard-4")
RESUME_STEP=$(   curl -sf "${META}/RESUME_STEP"           -H "$H" || echo "0")
PREV_CLOUD=$(    curl -sf "${META}/PREV_CLOUD"            -H "$H" || echo "")
S3_BUCKET=$(     curl -sf "${META}/CHECKPOINT_S3_BUCKET"  -H "$H" || echo "")
AWS_KEY=$(       curl -sf "${META}/AWS_ACCESS_KEY_ID"     -H "$H" || echo "")
AWS_SECRET=$(    curl -sf "${META}/AWS_SECRET_ACCESS_KEY" -H "$H" || echo "")
AWS_REGION=$(    curl -sf "${META}/AWS_DEFAULT_REGION"    -H "$H" || echo "us-east-1")
GCS_BUCKET=$(    curl -sf "${META}/CHECKPOINT_GCS_BUCKET" -H "$H" || echo "")

export JOB_ID INSTANCE_TYPE RESUME_STEP PREV_CLOUD
export CLOUD="gcp"
export CHECKPOINT_S3_BUCKET="${S3_BUCKET}"
export CHECKPOINT_GCS_BUCKET="${GCS_BUCKET}"
export AWS_ACCESS_KEY_ID="${AWS_KEY}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET}"
export AWS_DEFAULT_REGION="${AWS_REGION}"

echo ""
echo "  JOB_ID                = ${JOB_ID}"
echo "  CLOUD                 = gcp"
echo "  CHECKPOINT_S3_BUCKET  = ${CHECKPOINT_S3_BUCKET}"
echo "  CHECKPOINT_GCS_BUCKET = ${CHECKPOINT_GCS_BUCKET:-none}"
echo "  INSTANCE_TYPE         = ${INSTANCE_TYPE}"
echo "  RESUME_STEP           = ${RESUME_STEP}"
echo "  PREV_CLOUD            = ${PREV_CLOUD:-none (fresh start)}"
echo "  AWS_DEFAULT_REGION    = ${AWS_DEFAULT_REGION}"
echo ""

if [ -z "${CHECKPOINT_S3_BUCKET}" ]; then
    echo "ERROR: CHECKPOINT_S3_BUCKET is empty — metadata injection may have failed"
    exit 1
fi

if [ "${RESUME_STEP}" != "0" ]; then
    echo ">>> RESUMING job from step ${RESUME_STEP}"
    if [ -n "${PREV_CLOUD}" ] && [ "${PREV_CLOUD}" != "gcp" ]; then
        echo ">>> Cross-cloud migration: ${PREV_CLOUD} → gcp"
    fi
    echo ""
fi

# ── 2. Install system packages (including awscli) ─────────────────
echo "==> System packages..."
apt-get update -qq
apt-get install -y -qq python3-pip python3-venv awscli

# ── 3. Create virtualenv ──────────────────────────────────────────
echo "==> Creating virtualenv at /opt/trainer-env..."
python3 -m venv /opt/trainer-env
source /opt/trainer-env/bin/activate
pip install --quiet --upgrade pip

# ── 4. Install Python dependencies ───────────────────────────────
echo "==> Installing Python deps..."

# torch CPU build — e2-standard-4 has no GPU
# torch 2.2.0 is pinned for reproducibility
pip install --quiet \
    torch==2.2.0 \
    --index-url https://download.pytorch.org/whl/cpu

pip install --quiet \
    google-cloud-storage \
    boto3 \
    "numpy<2" \
    pandas

echo "==> Deps installed"

# ── 5. Download trainer files from S3 ────────────────────────────
# S3 is the single source of truth for trainer files on all clouds.
echo ""
echo "==> Downloading trainer files from s3://${CHECKPOINT_S3_BUCKET}/trainer/..."

WORKDIR="/opt/trainer"
mkdir -p "${WORKDIR}"
cd "${WORKDIR}"

aws s3 cp "s3://${CHECKPOINT_S3_BUCKET}/trainer/train.py"              ./train.py
aws s3 cp "s3://${CHECKPOINT_S3_BUCKET}/trainer/checkpoint_pkg.tar.gz" ./checkpoint_pkg.tar.gz

# Extract checkpoint/ package (single tar is atomic — no partial download risk)
tar -xzf checkpoint_pkg.tar.gz
rm  -f   checkpoint_pkg.tar.gz

echo "==> Trainer files ready:"
ls -la "${WORKDIR}/"
echo ""
echo "==> Checkpoint package contents:"
ls -la "${WORKDIR}/checkpoint/"

# ── 6. Run training ───────────────────────────────────────────────
echo ""
echo "==> Starting training..."
echo "    job_id      = ${JOB_ID}"
echo "    resume_step = ${RESUME_STEP}"
echo ""

# Run from WORKDIR so `import checkpoint` resolves to ./checkpoint/
cd "${WORKDIR}"
python3 train.py
EXIT_CODE=$?

# ── 7. Shutdown VM ────────────────────────────────────────────────
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "==> Training exited cleanly (code 0). Shutting down in 60s..."
else
    echo "==> Training exited with code ${EXIT_CODE}. Shutting down in 60s..."
fi

echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') shutdown queued"

# 60s buffer so final logs flush to Cloud Logging before VM dies
sleep 60
shutdown -h now