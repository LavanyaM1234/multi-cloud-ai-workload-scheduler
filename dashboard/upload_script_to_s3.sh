#!/bin/bash
# scripts/upload_trainer.sh

set -euo pipefail

# ── Load .env ─────────────────────────────────────────────
if [ -f ".env" ]; then
    echo "==> Loading .env..."
    set -a
    source .env
    set +a
fi

# ── Validate env vars ────────────────────────────────────
MISSING=0

for VAR in AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY CHECKPOINT_S3_BUCKET; do
    if [ -z "${!VAR:-}" ]; then
        echo "ERROR: $VAR is not set"
        MISSING=1
    fi
done


BUCKET="${CHECKPOINT_S3_BUCKET}"

echo ""
echo "==> Uploading trainer files to s3://${BUCKET}/trainer/"
echo ""

# ── Upload train.py ──────────────────────────────────────
if [ ! -f "trainer/train.py" ]; then
    echo "ERROR: trainer/train.py not found"
    exit 1
fi

aws.exe s3 cp trainer/train.py "s3://${BUCKET}/trainer/train.py"
echo "  train.py uploaded"

# ── Upload startup.sh ────────────────────────────────────
if [ ! -f "trainer/startup.sh" ]; then
    echo "ERROR: trainer/startup.sh not found"
    exit 1
fi

aws.exe s3 cp trainer/startup.sh "s3://${BUCKET}/trainer/startup.sh"
echo "  ✓ startup.sh uploaded"

# ── Package checkpoint/ folder ───────────────────────────
if [ ! -d "checkpoint" ]; then
    echo "ERROR: checkpoint/ directory not found"
    exit 1
fi

TMP_TAR="./checkpoint_pkg.tar.gz"

echo "  Packaging checkpoint/ → ${TMP_TAR}"

tar \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  -czf "${TMP_TAR}" checkpoint/

aws.exe s3 cp \
  "${TMP_TAR}" \
  "s3://${BUCKET}/trainer/checkpoint_pkg.tar.gz"

rm -f "${TMP_TAR}"

echo "  ✓ checkpoint_pkg.tar.gz uploaded"

# ── Summary ──────────────────────────────────────────────
echo ""
echo "==> Upload complete"
echo ""

aws.exe s3 ls "s3://${BUCKET}/trainer/" --human-readable

echo ""
echo "VMs will download:"
echo "  s3://${BUCKET}/trainer/train.py"
echo "  s3://${BUCKET}/trainer/startup.sh"
echo "  s3://${BUCKET}/trainer/checkpoint_pkg.tar.gz"