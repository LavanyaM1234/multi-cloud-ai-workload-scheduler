"""
scheduler/launcher.py
──────────────────────────────────────────────────────────────────────
Multi-cloud launcher — routes VM creation based on selector decision.

Entry points (backward-compatible):
    submit_job(job)   → write config + state to S3, pick best cloud, launch VM
    resume_job(job)   → find checkpoint on S3, relaunch with RESUME_STEP

Pre-launch sequence (happens before any VM boots):
    1. _write_job_config()    → s3://.../checkpoints/{job_id}/job_config.json
    2. _write_initial_state() → s3://.../checkpoints/{job_id}/job_state.json
    3. _launch_with_fallback() → creates VM; startup.sh runs automatically
    4. _write_failed_state()  → called only if launch itself throws

Why S3 for all JSON files:
    train.py reads job_config.json from S3 via load_config().
    CheckpointEngine writes job_state.json to S3 after every save.
    server.py polls job_state.json from S3 to detect preemption.
    All three clouds can read/write S3 — it is the single source of truth.
    GCS is only used by GCP VMs for .pt checkpoint file storage.

Shutdown hook (added to all cloud startup scripts):
    Runs after train.py exits for ANY reason (clean, crash, preemption).
    Logs to /var/log/trainer.log (same file as everything else).
    Updates job_state.json → status=preempted (if not already terminal).
    Updates job_config.json → migration_count++, resume_from_step=last step.
    To view logs on the VM: sudo tail -f /var/log/trainer.log
"""

from __future__ import annotations

import os
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import boto3
from botocore.exceptions import ClientError

from scheduler.selector import pick_best_cloud

logger = logging.getLogger(__name__)

load_dotenv("credentials.env")

# ── Env ───────────────────────────────────────────────────────────
JOB_STATE_PATH = os.getenv("JOB_STATE_PATH",        "")
S3_BUCKET      = os.getenv("CHECKPOINT_S3_BUCKET",  "")
GCS_BUCKET     = os.getenv("CHECKPOINT_GCS_BUCKET", "ml-scheduler-jobs-tensile-method-459009-k2")

_FALLBACK_CHAIN = ["aws", "azure", "gcp"]


# ══════════════════════════════════════════════════════════════════
# S3 helpers
# ══════════════════════════════════════════════════════════════════

def _s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id     = os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name           = os.getenv("AWS_REGION", "us-east-1"),
    )


def _s3_put_json(key: str, data: dict) -> bool:
    if not S3_BUCKET:
        local = Path(key.replace("/", "_"))
        local.write_text(json.dumps(data, indent=2))
        logger.warning(f"[Launcher] S3_BUCKET not set — wrote {key} locally to {local}")
        return True
    try:
        _s3_client().put_object(
            Bucket      = S3_BUCKET,
            Key         = key,
            Body        = json.dumps(data, indent=2).encode(),
            ContentType = "application/json",
        )
        logger.debug(f"[Launcher] S3 put: s3://{S3_BUCKET}/{key}")
        return True
    except Exception as e:
        logger.error(f"[Launcher] S3 put failed for {key}: {e}")
        return False


def _s3_get_json(key: str) -> Optional[dict]:
    if not S3_BUCKET:
        return None
    try:
        obj = _s3_client().get_object(Bucket=S3_BUCKET, Key=key)
        return json.loads(obj["Body"].read())
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None
        logger.warning(f"[Launcher] S3 get failed for {key}: {e}")
        return None
    except Exception as e:
        logger.warning(f"[Launcher] S3 get failed for {key}: {e}")
        return None


# ══════════════════════════════════════════════════════════════════
# Pre-launch writers
# ══════════════════════════════════════════════════════════════════

def _write_job_config(job_id: str, config: dict) -> bool:
    key = f"checkpoints/{job_id}/job_config.json"
    ok  = _s3_put_json(key, config)
    if ok:
        logger.info(f"[Launcher] job_config.json → s3://{S3_BUCKET}/{key}")
    else:
        logger.error(f"[Launcher] FAILED to write job_config.json for {job_id}")
    return ok


def _write_initial_state(job_id: str, config: dict, decision: dict,
                          status: str = "queued"):
    key   = f"checkpoints/{job_id}/job_state.json"
    state = {
        "job_id":          job_id,
        "task_name":       config.get("task_name",        "Untitled"),
        "status":          status,
        "epoch":           0,
        "total_epochs":    config.get("epochs",           50),
        "step":            config.get("resume_from_step", 0),
        "loss":            None,
        "accuracy":        None,
        "best_val_acc":    None,
        "cloud":           decision.get("cloud",          "unknown"),
        "instance":        decision.get("instance_type",  "unknown"),
        "price_usd_hr":    decision.get("price_usd_hr",   0.0),
        "est_cost":        decision.get("est_cost",        0.0),
        "preemption_risk": decision.get("preemption_risk", 0.0),
        "migration_count": config.get("migration_count",  0),
        "resumed_from":    config.get("resume_from_step", 0),
        "updated_at":      datetime.now(timezone.utc).isoformat(),
    }
    _s3_put_json(key, state)
    logger.info(f"[Launcher] Initial state ({status}) written for {job_id}")


def _write_launched_state(job_id: str, config: dict, decision: dict,
                           launch_result: dict):
    key      = f"checkpoints/{job_id}/job_state.json"
    existing = _s3_get_json(key) or {}
    existing.update({
        "status":          "launched",
        "cloud":           launch_result.get("cloud",         decision.get("cloud")),
        "instance":        decision.get("instance_type",      "unknown"),
        "instance_id":     (launch_result.get("instance_id")
                            or launch_result.get("instance_name")
                            or launch_result.get("vm_name", "")),
        "price_usd_hr":    decision.get("price_usd_hr",       0.0),
        "est_cost":        decision.get("est_cost",            0.0),
        "preemption_risk": decision.get("preemption_risk",     0.0),
        "updated_at":      datetime.now(timezone.utc).isoformat(),
    })
    _s3_put_json(key, existing)
    logger.info(f"[Launcher] State → launched for {job_id}")


def _write_failed_state(job_id: str, error: str):
    key      = f"checkpoints/{job_id}/job_state.json"
    existing = _s3_get_json(key) or {}
    existing.update({
        "status":     "launch_failed",
        "error":      error,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    })
    _s3_put_json(key, existing)
    logger.error(f"[Launcher] State → launch_failed for {job_id}: {error}")


# ══════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════

def submit_job(job: dict) -> dict:
    job_id = job.get("job_id", f"job-{int(time.time())}")
    job["job_id"] = job_id

    logger.info(f"[Launcher] ▶ submit_job: {job_id}")

    try:
        decision = pick_best_cloud(job)
        decision["job_id"] = job_id
        logger.info(
            f"[Launcher] Selector → [{decision['cloud']}] "
            f"{decision['instance_type']}  "
            f"est=${decision['est_cost']:.2f}  "
            f"risk={decision['preemption_risk']:.2f}"
        )
    except Exception as e:
        logger.error(f"[Launcher] Selector failed: {e} — using emergency fallback")
        decision = _emergency_decision(job)

    job_config = {
        "job_id":            job_id,
        "task_name":         job.get("task_name",      "Untitled"),
        "lr":                float(job.get("lr",         0.001)),
        "hidden_dim":        int(job.get("hidden_dim",   256)),
        "dropout":           float(job.get("dropout",    0.3)),
        "batch_size":        int(job.get("batch_size",   64)),
        "epochs":            int(job.get("epochs",       50)),
        "ckpt_every":        int(job.get("ckpt_every",   50)),
        "input_dim":         int(job.get("input_dim",    50)),
        "num_classes":       int(job.get("num_classes",  5)),
        "max_budget":        float(job.get("max_budget", 2.0)),
        "dataset_type":      job.get("dataset_type",    "synthetic-500k"),
        "s3_dataset_path":   job.get("s3_dataset_path", ""),
        "price_usd_hr":      decision["price_usd_hr"],
        "instance_type":     decision["instance_type"],
        "cloud":             decision["cloud"],
        "resume_from_step":  int(job.get("resume_step",       0)),
        "migration_count":   int(job.get("migration_count",   0)),
        "gcs_bucket":        GCS_BUCKET,
    }

    if not _write_job_config(job_id, job_config):
        error = "Could not write job_config.json to S3 — aborting launch"
        _write_failed_state(job_id, error)
        raise RuntimeError(f"[Launcher] {error}")

    _write_initial_state(job_id, job_config, decision, status="queued")

    startup_script = _build_training_script(job, decision)

    try:
        launch_result = _launch_with_fallback(decision, startup_script, job)
    except Exception as e:
        _write_failed_state(job_id, str(e))
        raise

    _write_launched_state(job_id, job_config, decision, launch_result)

    logger.info(f"[Launcher] ✓ Job {job_id} launched on [{launch_result.get('cloud')}]")
    return {
        "job_id":        job_id,
        "decision":      decision,
        "launch_result": launch_result,
    }


def resume_job(job: dict) -> dict:
    job_id = job.get("job_id")
    if not job_id:
        raise ValueError("[Launcher] resume_job requires job_id")

    logger.info(f"[Launcher] ↺ resume_job: {job_id}")

    prev_config = _s3_get_json(f"checkpoints/{job_id}/job_config.json") or {}
    if prev_config:
        merged = {**prev_config, **job}
        merged["job_id"] = job_id
    else:
        logger.warning(f"[Launcher] No job_config.json on S3 for {job_id} — using job dict only")
        merged = dict(job)
        merged["job_id"] = job_id

    prev_cloud = prev_config.get("cloud", "")

    checkpoint_path, resume_step = _find_latest_checkpoint(job_id)
    if checkpoint_path:
        merged["resume_step"]     = resume_step
        merged["prev_cloud"]      = prev_cloud
        merged["migration_count"] = int(prev_config.get("migration_count", 0)) + 1
    else:
        merged["resume_step"]     = 0
        merged["prev_cloud"]      = ""
        merged["migration_count"] = 0

    return submit_job(merged)


# ══════════════════════════════════════════════════════════════════
# Launch routing
# ══════════════════════════════════════════════════════════════════

def _decision_for_cloud(original_decision: dict, target_cloud: str) -> dict:
    if original_decision["cloud"] == target_cloud:
        return dict(original_decision)

    pareto_set       = original_decision.get("pareto_set", [])
    cloud_candidates = [p for p in pareto_set if p.get("cloud") == target_cloud]
    if cloud_candidates:
        best     = min(cloud_candidates, key=lambda p: p.get("est_cost", 999))
        fallback = dict(best)
        fallback["job_id"] = original_decision.get("job_id", "")
        logger.info(
            f"[Launcher] Fallback remapped to [{target_cloud}] "
            f"{fallback['instance_type']}  (from Pareto set)"
        )
        return fallback

    _CLOUD_DEFAULTS = {
        "aws": {
            "cloud":           "aws",
            "instance_type":   "t3.medium",
            "region":          os.getenv("AWS_REGION", "us-east-1"),
            "zone":            os.getenv("AWS_REGION", "us-east-1") + "a",
            "price_usd_hr":    0.013,
            "est_hours":       original_decision.get("est_hours", 4.0),
            "est_cost":        original_decision.get("est_cost",  0.05),
            "preemption_risk": 0.03,
            "gpu_model": None, "gpu_mem_gb": 0, "gpu_count": 0,
            "vcpus": 2, "ram_gb": 4,
        },
        "gcp": {
            "cloud":           "gcp",
            "instance_type":   "e2-standard-4",
            "region":          os.getenv("GCP_REGION", "us-central1"),
            "zone":            os.getenv("GCP_ZONE",   "us-central1-a"),
            "price_usd_hr":    0.034,
            "est_hours":       original_decision.get("est_hours", 4.0),
            "est_cost":        original_decision.get("est_cost",  0.14),
            "preemption_risk": 0.05,
            "gpu_model": None, "gpu_mem_gb": 0, "gpu_count": 0,
            "vcpus": 4, "ram_gb": 16,
        },
        "azure": {
            "cloud":           "azure",
            "instance_type":   "Standard_D2as_v4",
            "region":          os.getenv("AZURE_LOCATION", "centralindia"),
            "zone":            os.getenv("AZURE_LOCATION", "centralindia"),
            "price_usd_hr":    0.022,
            "est_hours":       original_decision.get("est_hours", 4.0),
            "est_cost":        original_decision.get("est_cost",  0.09),
            "preemption_risk": 0.04,
            "gpu_model": None, "gpu_mem_gb": 0, "gpu_count": 0,
            "vcpus": 2, "ram_gb": 8,
        },
    }
    fallback = dict(_CLOUD_DEFAULTS.get(target_cloud, _CLOUD_DEFAULTS["gcp"]))
    fallback["job_id"] = original_decision.get("job_id", "")
    logger.info(
        f"[Launcher] Fallback remapped to [{target_cloud}] "
        f"{fallback['instance_type']}  (default)"
    )
    return fallback


def _launch_with_fallback(decision: dict, startup_script: str, job: dict) -> dict:
    job_id       = decision.get("job_id", "unknown")
    chosen_cloud = decision["cloud"]
    attempted    = []
    order        = [chosen_cloud] + [c for c in _FALLBACK_CHAIN if c != chosen_cloud]

    for cloud in order:
        attempted.append(cloud)
        cloud_decision = _decision_for_cloud(decision, cloud)
        cloud_script   = _build_training_script(job, cloud_decision)

        logger.info(
            f"[Launcher] Trying [{cloud}]  "
            f"instance={cloud_decision['instance_type']}  "
            f"region={cloud_decision['region']}  "
            f"${cloud_decision.get('price_usd_hr', 0):.4f}/hr  "
            f"job={job_id}"
        )

        try:
            if cloud == "gcp":
                result = _launch_gcp(cloud_decision, cloud_script)
            elif cloud == "aws":
                result = _launch_aws(cloud_decision, cloud_script)
            elif cloud == "azure":
                result = _launch_azure(cloud_decision, cloud_script)
            else:
                logger.error(f"[Launcher] Unknown cloud: {cloud}")
                continue

            result["cloud"]         = cloud
            result["instance_type"] = cloud_decision["instance_type"]
            logger.info(f"[Launcher] ✓ Launch succeeded on [{cloud}]")
            return result

        except ModuleNotFoundError as e:
            if "azure" in str(e).lower():
                logger.warning(
                    f"[Launcher] ✗ [{cloud}] skipped — Azure SDK not installed. "
                    f"Fix: pip install azure-identity azure-mgmt-compute azure-mgmt-network"
                )
            else:
                logger.error(f"[Launcher] ✗ [{cloud}] missing module: {e}")
            if cloud != order[-1]:
                logger.info("[Launcher] Trying next cloud in fallback chain...")

        except Exception as e:
            logger.error(f"[Launcher] ✗ [{cloud}] failed: {e}  (tried: {attempted})")
            if cloud != order[-1]:
                logger.info("[Launcher] Trying next cloud in fallback chain...")

    # All spot failed → GCP on-demand
    logger.error(f"[Launcher] All spot clouds failed {attempted} → trying GCP on-demand")
    try:
        gcp_decision         = _decision_for_cloud(decision, "gcp")
        gcp_decision["spot"] = False
        gcp_script           = _build_training_script(job, gcp_decision)
        result               = _launch_gcp(gcp_decision, gcp_script, on_demand=True)
        result["cloud"]      = "gcp"
        result["on_demand"]  = True
        return result
    except Exception as e:
        raise RuntimeError(
            f"[Launcher] All launch attempts failed ({attempted} + on-demand): {e}"
        )


def _launch_gcp(decision: dict, startup_script: str, on_demand: bool = False) -> dict:
    from scheduler.launchers.gcp_launcher import create_gcp_vm
    if on_demand:
        os.environ["_GCP_FORCE_ONDEMAND"] = "1"
    try:
        return create_gcp_vm(decision, startup_script)
    finally:
        os.environ.pop("_GCP_FORCE_ONDEMAND", None)


def _launch_aws(decision: dict, startup_script: str) -> dict:
    from scheduler.launchers.aws_launcher import create_aws_vm
    return create_aws_vm(decision, startup_script)


def _launch_azure(decision: dict, startup_script: str) -> dict:
    from scheduler.launchers.azure_launcher import create_azure_vm
    return create_azure_vm(decision, startup_script)


# ══════════════════════════════════════════════════════════════════
# Startup script builder
# ══════════════════════════════════════════════════════════════════

def _build_training_script(job: dict, decision: dict) -> str:
    """
    Build a cloud-specific startup script.

    GCP / Azure  — Ubuntu 22.04, apt-get, awscli via apt
    AWS          — Amazon Linux 2023, dnf, awscli v2 pre-installed

    Shutdown hook (all clouds):
        Runs after train.py exits for any reason.
        Logs to /var/log/trainer.log.
        Updates job_state.json  → status=preempted  (if not already terminal)
        Updates job_config.json → migration_count++, resume_from_step=last step
    """
    job_id      = job.get("job_id",      "unknown")
    resume_step = job.get("resume_step", 0)
    prev_cloud  = job.get("prev_cloud",  "")
    cloud       = decision.get("cloud",  "gcp")
    s3_bucket   = S3_BUCKET
    aws_key     = os.getenv("AWS_ACCESS_KEY_ID",     "")
    aws_secret  = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    aws_region  = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))

    # ── Shared header ─────────────────────────────────────────────
    header = f"""#!/bin/bash
# startup.sh — {cloud} — generated by scheduler/launcher.py
# Job: {job_id}  Resume: step {resume_step}
set -e
LOG="/var/log/trainer.log"
exec > >(tee -a "$LOG") 2>&1

echo "=========================================="
echo " ML Scheduler — VM Startup ({cloud})"
echo " $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "=========================================="

export JOB_ID="{job_id}"
export RESUME_STEP="{resume_step}"
export PREV_CLOUD="{prev_cloud}"
export CLOUD="{cloud}"
export CHECKPOINT_S3_BUCKET="{s3_bucket}"
export AWS_ACCESS_KEY_ID="{aws_key}"
export AWS_SECRET_ACCESS_KEY="{aws_secret}"
export AWS_DEFAULT_REGION="{aws_region}"

echo "  JOB_ID               = ${{JOB_ID}}"
echo "  CLOUD                = ${{CLOUD}}"
echo "  CHECKPOINT_S3_BUCKET = ${{CHECKPOINT_S3_BUCKET}}"
echo "  RESUME_STEP          = ${{RESUME_STEP}}"
echo "  PREV_CLOUD           = ${{PREV_CLOUD:-none}}"
echo ""

if [ "${{RESUME_STEP}}" != "0" ]; then
    echo ">>> RESUMING from step ${{RESUME_STEP}}"
    [ -n "${{PREV_CLOUD}}" ] && echo ">>> Cross-cloud: ${{PREV_CLOUD}} → {cloud}"
    echo ""
fi

if [ -z "${{CHECKPOINT_S3_BUCKET}}" ]; then
  echo "ERROR: CHECKPOINT_S3_BUCKET is empty — cannot download trainer files."
  exit 1
fi
"""

    # ── AWS (Amazon Linux 2023) ───────────────────────────────────
    if cloud == "aws":
        pkg_block = """
echo "==> System packages (Amazon Linux 2023 — dnf)..."
dnf install -y -q python3-pip python3 tar gzip

echo "==> Verifying system awscli (pre-installed, must stay untouched)..."
/usr/local/bin/aws --version || /usr/bin/aws --version

SWAP_FILE="/swapfile"
if [ ! -f "$SWAP_FILE" ]; then
  echo "==> Creating 2GB swap file..."
  dd if=/dev/zero of="$SWAP_FILE" bs=128M count=16 status=progress 2>/dev/null
  chmod 600 "$SWAP_FILE"
  mkswap "$SWAP_FILE"
  swapon "$SWAP_FILE"
  echo "==> Swap enabled: $(free -h | grep Swap)"
fi

echo "==> Creating virtualenv /opt/trainer-env (isolated from system pip)..."
python3 -m venv /opt/trainer-env
source /opt/trainer-env/bin/activate
pip install --quiet --upgrade pip

echo "==> Installing Python deps into virtualenv..."
pip install --quiet --no-cache-dir "numpy<2" pandas
pip install --quiet --no-cache-dir boto3
pip install --quiet --no-cache-dir torch==2.2.0 \
    --index-url https://download.pytorch.org/whl/cpu
echo "==> Deps installed"

echo "==> Confirming awscli intact after venv: $(/usr/local/bin/aws --version 2>/dev/null || /usr/bin/aws --version)"

PYTHON_CMD="/opt/trainer-env/bin/python3"
AWS_CMD="/usr/local/bin/aws"
[ -f "$AWS_CMD" ] || AWS_CMD="/usr/bin/aws"
"""

    # ── GCP / Azure (Ubuntu 22.04) ────────────────────────────────
    else:
        pkg_block = """
echo "==> System packages (Ubuntu 22.04 — apt-get)..."
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    python3-pip python3-venv python3-dev awscli

echo "==> awscli version: $(aws --version)"

SWAP_FILE="/swapfile"
if [ ! -f "$SWAP_FILE" ]; then
  echo "==> Creating 2GB swap file..."
  dd if=/dev/zero of="$SWAP_FILE" bs=128M count=16 status=progress 2>/dev/null
  chmod 600 "$SWAP_FILE"
  mkswap "$SWAP_FILE"
  swapon "$SWAP_FILE"
  echo "==> Swap enabled: $(free -h | grep Swap)"
fi

echo "==> Creating virtualenv /opt/trainer-env..."
python3 -m venv /opt/trainer-env
source /opt/trainer-env/bin/activate
pip install --quiet --upgrade pip

echo "==> Installing Python deps..."
pip install --quiet --no-cache-dir "numpy<2" pandas boto3 google-cloud-storage
pip install --quiet --no-cache-dir torch==2.2.0 \
    --index-url https://download.pytorch.org/whl/cpu
echo "==> Deps installed"

PYTHON_CMD="python3"
"""

    # ── Shared tail: download + run train.py + shutdown hook ──────
    # NOTE: the shutdown hook Python script uses single-brace {{ }} for
    # Python dict literals because this is inside an f-string.
    tail = f"""
AWS_BIN="${{AWS_CMD:-aws}}"

echo ""
echo "==> Downloading trainer/ from s3://${{CHECKPOINT_S3_BUCKET}}/trainer/..."
WORKDIR="/opt/trainer"
mkdir -p "${{WORKDIR}}"
cd "${{WORKDIR}}"

${{AWS_BIN}} s3 cp "s3://${{CHECKPOINT_S3_BUCKET}}/trainer/train.py"              ./train.py
${{AWS_BIN}} s3 cp "s3://${{CHECKPOINT_S3_BUCKET}}/trainer/checkpoint_pkg.tar.gz" ./checkpoint_pkg.tar.gz

tar -xzf checkpoint_pkg.tar.gz
rm  -f   checkpoint_pkg.tar.gz

echo "==> trainer/ ready: $(ls ${{WORKDIR}} | tr '\\n' ' ')"
echo "==> checkpoint/   : $(ls ${{WORKDIR}}/checkpoint | tr '\\n' ' ')"

echo ""
echo "==> Starting train.py  job=${{JOB_ID}}  cloud=${{CLOUD}}  resume=${{RESUME_STEP}}"
echo ""

cd "${{WORKDIR}}"
${{PYTHON_CMD:-python3}} train.py
EXIT_CODE=$?

echo ""
echo "=================================================="
echo " train.py exited with code $EXIT_CODE"
echo " $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "=================================================="
[ $EXIT_CODE -eq 0 ] \
    && echo "==> train.py finished cleanly." \
    || echo "==> train.py exited with error (code $EXIT_CODE)."

# ══════════════════════════════════════════════════════════════════
# SHUTDOWN HOOK — runs after train.py exits for ANY reason
# Logs to /var/log/trainer.log (same file — exec redirect above)
# Updates job_state.json  → status=preempted  (if not already terminal)
# Updates job_config.json → migration_count++, resume_from_step
# To view: sudo tail -f /var/log/trainer.log
# ══════════════════════════════════════════════════════════════════
echo ""
echo "==> [shutdown-hook] Starting S3 state update..."
echo "==> [shutdown-hook] JOB_ID=${{JOB_ID}}  CLOUD=${{CLOUD}}  EXIT_CODE=$EXIT_CODE"
echo "==> [shutdown-hook] CHECKPOINT_S3_BUCKET=${{CHECKPOINT_S3_BUCKET}}"

${{PYTHON_CMD:-python3}} - <<PYEOF
import os, json, sys, boto3
from datetime import datetime, timezone

LOG_PREFIX = "[shutdown-hook]"
bucket     = os.environ.get("CHECKPOINT_S3_BUCKET", "")
job_id     = os.environ.get("JOB_ID", "")
cloud      = os.environ.get("CLOUD", "unknown")
exit_code  = int("$EXIT_CODE") if "$EXIT_CODE".lstrip("-").isdigit() else 1

print(f"{{LOG_PREFIX}} Initialising  bucket={{bucket}}  job_id={{job_id}}  cloud={{cloud}}  exit_code={{exit_code}}")

if not bucket:
    print(f"{{LOG_PREFIX}} ERROR: CHECKPOINT_S3_BUCKET not set — cannot update S3. Skipping.")
    sys.exit(0)

if not job_id:
    print(f"{{LOG_PREFIX}} ERROR: JOB_ID not set — cannot determine S3 paths. Skipping.")
    sys.exit(0)

# ── S3 client ─────────────────────────────────────────────────────
try:
    s3 = boto3.client(
        "s3",
        aws_access_key_id     = os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region_name           = os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    )
    print(f"{{LOG_PREFIX}} boto3 S3 client created OK")
except Exception as e:
    print(f"{{LOG_PREFIX}} ERROR: Failed to create S3 client: {{e}}")
    sys.exit(0)

def s3_read(key):
    try:
        obj  = s3.get_object(Bucket=bucket, Key=key)
        data = json.loads(obj["Body"].read().decode())
        print(f"{{LOG_PREFIX}} Read s3://{{bucket}}/{{key}} OK")
        return data
    except s3.exceptions.NoSuchKey:
        print(f"{{LOG_PREFIX}} WARN: s3://{{bucket}}/{{key}} does not exist — returning empty dict")
        return {{}}
    except Exception as e:
        print(f"{{LOG_PREFIX}} WARN: Failed to read s3://{{bucket}}/{{key}}: {{e}}")
        return {{}}

def s3_write(key, data):
    try:
        s3.put_object(
            Bucket      = bucket,
            Key         = key,
            Body        = json.dumps(data, indent=2).encode(),
            ContentType = "application/json",
        )
        print(f"{{LOG_PREFIX}} Wrote s3://{{bucket}}/{{key}} OK")
        return True
    except Exception as e:
        print(f"{{LOG_PREFIX}} ERROR: Failed to write s3://{{bucket}}/{{key}}: {{e}}")
        return False

now = datetime.now(timezone.utc).isoformat()

# ── Update job_state.json ─────────────────────────────────────────
state_key = f"checkpoints/{{job_id}}/job_state.json"
print(f"{{LOG_PREFIX}} Reading job_state.json from {{state_key}} ...")
state = s3_read(state_key)

TERMINAL_STATUSES = {{"done", "budget_exceeded", "failed", "preempted"}}
current_status    = state.get("status", "unknown")
print(f"{{LOG_PREFIX}} Current job_state status = '{{current_status}}'")

if current_status in TERMINAL_STATUSES:
    print(f"{{LOG_PREFIX}} Status is already terminal ('{{current_status}}') — skipping job_state.json update")
else:
    print(f"{{LOG_PREFIX}} Status '{{current_status}}' is not terminal — updating to 'preempted'")
    state.update({{
        "status":             "preempted",
        "cloud":              cloud,
        "updated_at":         now,
        "shutdown_exit_code": exit_code,
        "shutdown_reason":    "vm_shutdown_hook",
    }})
    if s3_write(state_key, state):
        print(f"{{LOG_PREFIX}} job_state.json → status=preempted  exit_code={{exit_code}}")
    else:
        print(f"{{LOG_PREFIX}} ERROR: job_state.json update FAILED")

# ── Update job_config.json ────────────────────────────────────────
config_key = f"checkpoints/{{job_id}}/job_config.json"
print(f"{{LOG_PREFIX}} Reading job_config.json from {{config_key}} ...")
config = s3_read(config_key)

if not config:
    print(f"{{LOG_PREFIX}} WARN: job_config.json not found on S3 — skipping config update")
else:
    old_migration = config.get("migration_count", 0)
    old_step      = config.get("resume_from_step", 0)
    new_step      = state.get("step", old_step)
    new_migration = old_migration + 1

    config.update({{
        "last_cloud":        cloud,
        "last_shutdown_at":  now,
        "migration_count":   new_migration,
        "resume_from_step":  new_step,
    }})
    print(f"{{LOG_PREFIX}} job_config.json update: migration_count {{old_migration}} → {{new_migration}},  resume_from_step {{old_step}} → {{new_step}}")

    if s3_write(config_key, config):
        print(f"{{LOG_PREFIX}} job_config.json updated OK")
    else:
        print(f"{{LOG_PREFIX}} ERROR: job_config.json update FAILED")

print(f"{{LOG_PREFIX}} Shutdown hook complete. Handing off to OS shutdown.")
PYEOF

HOOK_EXIT=$?
if [ $HOOK_EXIT -eq 0 ]; then
    echo "==> [shutdown-hook] Completed successfully."
else
    echo "==> [shutdown-hook] Exited with code $HOOK_EXIT (non-fatal — VM will still shut down)."
fi

echo ""
echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') — shutdown queued (30s delay)"
sleep 30
shutdown -h now
"""

    return header + pkg_block + tail


# ══════════════════════════════════════════════════════════════════
# Checkpoint finder
# ══════════════════════════════════════════════════════════════════

def _find_latest_checkpoint(job_id: str) -> tuple[Optional[str], int]:
    if not S3_BUCKET:
        logger.warning("[Launcher] CHECKPOINT_S3_BUCKET not set")
        return None, 0

    prefix      = f"checkpoints/{job_id}/"
    resume_step = 0

    state = _s3_get_json(f"{prefix}job_state.json")
    if state:
        resume_step = int(state.get("step", 0))
        logger.info(
            f"[Launcher] S3 job_state: step={resume_step} "
            f"status={state.get('status','?')}"
        )

    s3         = _s3_client()
    latest_key = f"{prefix}checkpoint_latest.pt"
    try:
        s3.head_object(Bucket=S3_BUCKET, Key=latest_key)
        path = f"s3://{S3_BUCKET}/{latest_key}"
        logger.info(f"[Launcher] Checkpoint: {path}  step={resume_step}")
        return path, resume_step
    except ClientError:
        pass

    try:
        resp = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
        pts  = [o for o in resp.get("Contents", []) if o["Key"].endswith(".pt")]
        if pts:
            latest = max(pts, key=lambda o: o["LastModified"])
            path   = f"s3://{S3_BUCKET}/{latest['Key']}"
            logger.info(f"[Launcher] Step checkpoint: {path}  step={resume_step}")
            return path, resume_step
    except Exception as e:
        logger.warning(f"[Launcher] S3 list failed: {e}")

    return None, 0


# ══════════════════════════════════════════════════════════════════
# Misc helpers
# ══════════════════════════════════════════════════════════════════

def _emergency_decision(job: dict) -> dict:
    PRICE    = 0.034
    budget   = float(job.get("max_budget",   2.0))
    deadline = float(job.get("deadline_hrs", 8.0))
    hours    = min(budget / PRICE, deadline)
    return {
        "cloud":           "gcp",
        "instance_type":   "e2-standard-4",
        "region":          os.getenv("GCP_REGION", "us-central1"),
        "zone":            os.getenv("GCP_ZONE",   "us-central1-a"),
        "price_usd_hr":    PRICE,
        "est_hours":       round(hours, 2),
        "est_cost":        round(hours * PRICE, 4),
        "preemption_risk": 0.05,
        "gpu_model":       None, "gpu_mem_gb": 0, "gpu_count": 0,
        "vcpus":           4,    "ram_gb":     16,
        "s3_bucket":       S3_BUCKET,
        "reason":          "Emergency fallback — selector unavailable (GCP e2-standard-4 CPU)",
        "pareto_set":      [],
    }


def _write_job_state(job_id: str, data: dict):
    try:
        path  = Path(JOB_STATE_PATH)
        state = {}
        if path.exists():
            try:
                state = json.loads(path.read_text())
            except json.JSONDecodeError:
                logger.warning("[Launcher] Local job_state.json corrupted — resetting")
        if job_id not in state:
            state[job_id] = {}
        state[job_id].update(data)
        path.write_text(json.dumps(state, indent=2))
    except Exception as e:
        logger.error(f"[Launcher] Local job_state write failed: {e}")


def _read_job_state(job_id: str) -> Optional[dict]:
    try:
        path = Path(JOB_STATE_PATH)
        if not path.exists():
            return None
        return json.loads(path.read_text()).get(job_id)
    except Exception as e:
        logger.warning(f"[Launcher] Local job_state read failed: {e}")
        return None