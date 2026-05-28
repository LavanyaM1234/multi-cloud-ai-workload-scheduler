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

Does the startup script run automatically?
    YES — each cloud launcher injects it differently but it always runs:
        GCP   → passed as instance metadata "startup-script"; GCP runs it
                automatically on first boot via the google-startup-scripts
                service (no SSH required).
        AWS   → passed as base64 UserData; cloud-init runs it at first boot
                automatically before the instance is marked running.
        Azure → passed as base64 customData; cloud-init on the Ubuntu image
                runs it at first boot automatically.
    In all three cases: VM boots → cloud-init / startup service picks up the
    script → runs as root → installs deps, downloads trainer/ from S3,
    runs train.py, shuts down. You never need to SSH in.

Is the startup script generic across all clouds?
    YES — the same bash script body runs on GCP, AWS, and Azure.
    The only difference is how credentials + bucket names reach the VM:
        GCP   → curl metadata.google.internal to read them
        AWS   → they're already exported as env vars by the UserData preamble
                that aws_launcher prepends before injecting
        Azure → same; azure_launcher prepends an env-export preamble

Startup script downloads trainer/ from S3:
    s3://$CHECKPOINT_S3_BUCKET/trainer/train.py
    s3://$CHECKPOINT_S3_BUCKET/trainer/checkpoint_pkg.tar.gz
    You must upload these files to S3 before submitting any job.
    See: scripts/upload_trainer.sh
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

# ── Env ───────────────────────────────────────────────────────────
JOB_STATE_PATH = os.getenv("JOB_STATE_PATH",        "job_state.json")
S3_BUCKET      = os.getenv("CHECKPOINT_S3_BUCKET",  "")
GCS_BUCKET     = os.getenv("CHECKPOINT_GCS_BUCKET", "")   # GCP .pt storage only

# ── Fallback order ────────────────────────────────────────────────
_FALLBACK_CHAIN = ["aws", "azure", "gcp"]


# ══════════════════════════════════════════════════════════════════
# S3 helpers  (all JSON state lives here)
# ══════════════════════════════════════════════════════════════════

def _s3_client():
    """Return a boto3 S3 client using env credentials."""
    return boto3.client(
        "s3",
        aws_access_key_id     = os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name           = os.getenv("AWS_REGION", "us-east-1"),
    )


def _s3_put_json(key: str, data: dict) -> bool:
    """
    Write a dict as JSON to s3://S3_BUCKET/{key}.
    Returns True on success, False on failure.
    Logs to local file as fallback when S3_BUCKET is not configured
    (useful for local testing without real AWS creds).
    """
    if not S3_BUCKET:
        # Local fallback — write alongside job_state.json for testing
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
    """Read JSON from s3://S3_BUCKET/{key}. Returns None on any error."""
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
    """
    Write job_config.json to S3 BEFORE the VM boots.

    This is the file train.py's load_config() reads at startup:
        s3://CHECKPOINT_S3_BUCKET/checkpoints/{job_id}/job_config.json

    It contains all training hyperparameters (epochs, lr, batch_size,
    hidden_dim, etc.) plus job metadata (task_name, max_budget,
    price_usd_hr, migration_count).

    If this isn't written before the VM starts, train.py falls back
    to defaults and loses all your submitted parameters.

    Returns True on success so submit_job() can abort on failure.
    """
    key = f"checkpoints/{job_id}/job_config.json"
    ok  = _s3_put_json(key, config)
    if ok:
        logger.info(f"[Launcher] job_config.json → s3://{S3_BUCKET}/{key}")
    else:
        logger.error(f"[Launcher] FAILED to write job_config.json for {job_id}")
    return ok


def _write_initial_state(job_id: str, config: dict, decision: dict,
                          status: str = "queued"):
    """
    Write job_state.json to S3 immediately after config upload.

    Purpose: the dashboard (server.py) polls job_state.json to show
    job status. Without this write, the dashboard shows nothing while
    the VM is still booting (typically 2–3 minutes).

    After this write the dashboard immediately shows status=queued,
    then launched once the VM is created, then running once train.py
    starts writing its own updates.

    CheckpointEngine in train.py will overwrite this with live progress
    (step, loss, accuracy, cost_usd, etc.) after every checkpoint save.
    """
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
    """
    Update job_state.json to status=launched once the VM is confirmed running.
    Adds the actual instance ID / IP from the launch result.
    """
    key = f"checkpoints/{job_id}/job_state.json"

    # Merge over the initial state rather than replacing it
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
    """
    Write status=launch_failed to S3 when the VM creation itself throws.
    Lets the dashboard show an error instead of hanging on queued forever.
    """
    key = f"checkpoints/{job_id}/job_state.json"
    existing = _s3_get_json(key) or {}
    existing.update({
        "status":     "launch_failed",
        "error":      error,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    })
    _s3_put_json(key, existing)
    logger.error(f"[Launcher] State → launch_failed for {job_id}: {error}")


# ══════════════════════════════════════════════════════════════════
# Public API  (backward-compatible)
# ══════════════════════════════════════════════════════════════════

def submit_job(job: dict) -> dict:
    """
    Main entry point. Full sequence:
        1. Assign job_id
        2. Build job_config dict from job params
        3. Run Pareto selector to pick best cloud + instance
        4. Write job_config.json to S3  ← train.py reads this at boot
        5. Write initial job_state.json (status=queued) to S3
        6. Build startup.sh (stamps JOB_ID / RESUME_STEP / PREV_CLOUD)
        7. Launch VM (with AWS→Azure→GCP fallback chain)
        8. Update job_state.json to status=launched
        9. On any VM launch failure: write status=launch_failed

    Args:
        job: dict — keys your API / server.py passes in:
            job_id          str    (auto-generated if absent)
            task_name       str    human-readable label
            lr              float  learning rate          (default 0.001)
            hidden_dim      int    MLP hidden size        (default 256)
            dropout         float                         (default 0.3)
            batch_size      int                           (default 64)
            epochs          int    total training epochs  (default 50)
            ckpt_every      int    steps between saves    (default 50)
            input_dim       int    feature count          (default 50)
            num_classes     int                           (default 5)
            max_budget      float  $ spend cap            (default 2.0)
            deadline_hrs    float  must finish within N h (default 8.0)
            min_gpu_mem     float  minimum VRAM GB        (default 0)
            dataset_type    str    "synthetic-500k"|"custom"
            s3_dataset_path str    s3://bucket/prefix/ for custom data
            resume_step     int    0 = fresh, N = resume  (default 0)
            prev_cloud      str    cloud migrated from    (default "")
            migration_count int    how many migrations    (default 0)

    Returns:
        {"job_id", "decision", "launch_result"}
    """
    job_id = job.get("job_id", f"job-{int(time.time())}")
    job["job_id"] = job_id

    logger.info(f"[Launcher] ▶ submit_job: {job_id}")

    # ── 1. Select best cloud ──────────────────────────────────────
    try:
        decision = pick_best_cloud(job)
        print(job)
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

    # ── 2. Build job_config — what train.py reads at boot ─────────
    # Every key here has a matching config.setdefault() in train.py's
    # load_config(). If a key is missing here, train.py uses its own
    # default — these values WIN over train.py defaults.
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
        # Runtime metadata read by train.py for cost tracking
        "price_usd_hr":      decision["price_usd_hr"],
        "instance_type":     decision["instance_type"],
        "cloud":             decision["cloud"],
        # Resume / migration metadata
        "resume_from_step":  int(job.get("resume_step",       0)),
        "migration_count":   int(job.get("migration_count",   0)),
        "gcs_bucket":        GCS_BUCKET,
    }

    # ── 3. Write job_config.json to S3 BEFORE VM boots ───────────
    if not _write_job_config(job_id, job_config):
        error = "Could not write job_config.json to S3 — aborting launch"
        _write_failed_state(job_id, error)
        raise RuntimeError(f"[Launcher] {error}")

    # ── 4. Write initial queued state so dashboard shows it now ──
    _write_initial_state(job_id, job_config, decision, status="queued")

    # ── 5. Build startup script (stamps JOB_ID/RESUME_STEP/PREV_CLOUD) ──
    startup_script = _build_training_script(job, decision)

    # ── 6. Launch VM with fallback chain ─────────────────────────
    try:
        launch_result = _launch_with_fallback(decision, startup_script, job)
    except Exception as e:
        _write_failed_state(job_id, str(e))
        raise

    # ── 7. Update state to launched ──────────────────────────────
    _write_launched_state(job_id, job_config, decision, launch_result)

    logger.info(f"[Launcher] ✓ Job {job_id} launched on [{launch_result.get('cloud')}]")
    return {
        "job_id":        job_id,
        "decision":      decision,
        "launch_result": launch_result,
    }


def resume_job(job: dict) -> dict:
    """
    Resume a preempted job.

    End-to-end flow:
        server.py polls job_state.json on S3 → sees status=preempted
        → calls resume_job({"job_id": "job-123"})
        → we read prev step from job_state.json on S3
        → confirm checkpoint_latest.pt exists on S3
        → call submit_job() with resume_step=N and prev_cloud=X
        → submit_job() writes updated job_config (resume_from_step=N)
        → new VM boots, reads RESUME_STEP from env
        → train.py calls engine.load() → downloads checkpoint_latest.pt
        → training continues from step N on potentially a different cloud

    Args:
        job: dict with job_id (required). All other keys are optional
             overrides (e.g. bump max_budget for the retry).
    """
    job_id = job.get("job_id")
    if not job_id:
        raise ValueError("[Launcher] resume_job requires job_id")

    logger.info(f"[Launcher] ↺ resume_job: {job_id}")

    # ── Read previous job_config from S3 to carry params forward ─
    prev_config = _s3_get_json(f"checkpoints/{job_id}/job_config.json") or {}
    if prev_config:
        logger.info(
            f"[Launcher] Loaded previous config: epochs={prev_config.get('epochs')} "
            f"lr={prev_config.get('lr')} cloud={prev_config.get('cloud')}"
        )
        # Merge prev_config under job (job keys win — lets caller override)
        merged = {**prev_config, **job}
        merged["job_id"] = job_id
    else:
        logger.warning(f"[Launcher] No job_config.json on S3 for {job_id} — using job dict only")
        merged = dict(job)
        merged["job_id"] = job_id

    prev_cloud = prev_config.get("cloud", "")

    # ── Find checkpoint on S3 ─────────────────────────────────────
    checkpoint_path, resume_step = _find_latest_checkpoint(job_id)
    if checkpoint_path:
        logger.info(
            f"[Launcher] Checkpoint found: {checkpoint_path}  step={resume_step}"
        )
        merged["resume_step"]    = resume_step
        merged["prev_cloud"]     = prev_cloud
        merged["migration_count"] = int(prev_config.get("migration_count", 0)) + 1
    else:
        logger.warning(
            f"[Launcher] No checkpoint on S3 for {job_id} — starting fresh"
        )
        merged["resume_step"]    = 0
        merged["prev_cloud"]     = ""
        merged["migration_count"] = 0

    return submit_job(merged)


# ══════════════════════════════════════════════════════════════════
# Launch routing
# ══════════════════════════════════════════════════════════════════

def _decision_for_cloud(original_decision: dict, target_cloud: str) -> dict:
    """
    Build a cloud-specific decision dict for the fallback target.

    The root cause of the bug in the logs:
        Azure wins the Pareto selection → decision has:
            instance_type = "Standard_NC4as_T4_v3"
            region        = "centralindia"
            zone          = "centralindia"
        When Azure fails, the SAME dict was passed to AWS and GCP
        launchers unchanged — AWS tried to launch "Standard_NC4as_T4_v3"
        (an Azure SKU name), GCP tried zone "centralindia" (doesn't exist).

    Fix: when falling back, look up the best available instance for
    the target cloud from the Pareto set first, then from the full
    provider catalog. Always use that cloud's own region/zone env vars.
    """
    if original_decision["cloud"] == target_cloud:
        return dict(original_decision)   # no remap needed

    # ── Try to find a Pareto candidate for the target cloud ───────
    pareto_set = original_decision.get("pareto_set", [])
    cloud_candidates = [p for p in pareto_set if p.get("cloud") == target_cloud]
    if cloud_candidates:
        # Pick lowest cost from that cloud's Pareto candidates
        best = min(cloud_candidates, key=lambda p: p.get("est_cost", 999))
        fallback = dict(best)
        fallback["job_id"] = original_decision.get("job_id", "")
        logger.info(
            f"[Launcher] Fallback remapped to [{target_cloud}] "
            f"{fallback['instance_type']}  "
            f"${fallback.get('price_usd_hr', 0):.4f}/hr  "
            f"est_cost=${fallback.get('est_cost', 0):.2f}  "
            f"(from Pareto set)"
        )
        return fallback

    # ── No Pareto candidate — use default approved instance ───────
    # These are the cheapest approved GPU instance per cloud
    _CLOUD_DEFAULTS = {
        "aws": {
            "cloud":           "aws",
            "instance_type":   "t3.medium",
            "region":          os.getenv("AWS_REGION", "us-east-1"),
            "zone":            os.getenv("AWS_REGION", "us-east-1") + "a",
            "price_usd_hr":    0.013,    # t3.medium spot
            "est_hours":       original_decision.get("est_hours", 4.0),
            "est_cost":        original_decision.get("est_cost",  0.05),
            "preemption_risk": 0.03,
            "gpu_model":       None,
            "gpu_mem_gb":      0,
            "gpu_count":       0,
            "vcpus":           2,
            "ram_gb":          4,
        },
        "gcp": {
            "cloud":           "gcp",
            "instance_type":   "e2-standard-4",
            "region":          os.getenv("GCP_REGION", "us-central1"),
            "zone":            os.getenv("GCP_ZONE",   "us-central1-a"),
            "price_usd_hr":    0.034,    # e2-standard-4 spot
            "est_hours":       original_decision.get("est_hours", 4.0),
            "est_cost":        original_decision.get("est_cost",  0.14),
            "preemption_risk": 0.05,
            "gpu_model":       None,
            "gpu_mem_gb":      0,
            "gpu_count":       0,
            "vcpus":           4,
            "ram_gb":          16,
        },
        "azure": {
            "cloud":           "azure",
            "instance_type":   "Standard_D2as_v4",
            "region":          os.getenv("AZURE_LOCATION", "centralindia"),
            "zone":            os.getenv("AZURE_LOCATION", "centralindia"),
            "price_usd_hr":    0.022,    # Standard_D2as_v4 spot
            "est_hours":       original_decision.get("est_hours", 4.0),
            "est_cost":        original_decision.get("est_cost",  0.09),
            "preemption_risk": 0.04,
            "gpu_model":       None,
            "gpu_mem_gb":      0,
            "gpu_count":       0,
            "vcpus":           2,
            "ram_gb":          8,
        },
    }
    fallback = dict(_CLOUD_DEFAULTS.get(target_cloud, _CLOUD_DEFAULTS["gcp"]))
    fallback["job_id"] = original_decision.get("job_id", "")
    logger.info(
        f"[Launcher] Fallback remapped to [{target_cloud}] "
        f"{fallback['instance_type']}  "
        f"${fallback['price_usd_hr']:.4f}/hr  "
        f"(default — no Pareto candidate found for {target_cloud})"
    )
    return fallback


def _launch_with_fallback(decision: dict, startup_script: str, job: dict) -> dict:
    """
    Try the chosen cloud first, then rotate through the fallback chain.

    Key fix vs previous version:
        Each fallback attempt gets a REMAPPED decision dict with the
        correct instance_type / region / zone for THAT cloud.
        Previously the original decision (e.g. Azure's Standard_NC4as_T4_v3
        in centralindia) was passed unchanged to AWS and GCP launchers,
        causing InvalidParameterValue and 403 zone-not-found errors.

    Also detects "No module named 'azure'" and skips Azure gracefully
    with a clear install hint rather than showing a traceback.

    Last resort: GCP on-demand (not spot) if all spot attempts fail.
    """
    job_id       = decision.get("job_id", "unknown")
    chosen_cloud = decision["cloud"]
    attempted    = []
    order        = [chosen_cloud] + [c for c in _FALLBACK_CHAIN if c != chosen_cloud]

    # ── Log the live price snapshot for this job ──────────────────
    logger.info(
        f"[Launcher] ── Live price snapshot for job {job_id} ──────────"
    )
    pareto = decision.get("pareto_set", [])
    if pareto:
        for p in sorted(pareto, key=lambda x: x.get("est_cost", 999)):
            logger.info(
                f"[Launcher]   [{p.get('cloud','?'):5}] "
                f"{p.get('instance_type','?'):28} "
                f"${p.get('price_usd_hr', 0):.4f}/hr  "
                f"est_total=${p.get('est_cost', 0):.3f}  "
                f"risk={p.get('preemption_risk', 0):.2f}  "
                f"gpu={p.get('gpu_model','CPU')}"
            )
    else:
        logger.info(
            f"[Launcher]   [{decision.get('cloud','?'):5}] "
            f"{decision.get('instance_type','?'):28} "
            f"${decision.get('price_usd_hr', 0):.4f}/hr  "
            f"est_total=${decision.get('est_cost', 0):.3f}  "
            f"risk={decision.get('preemption_risk', 0):.2f}"
        )
    logger.info(f"[Launcher] ────────────────────────────────────────────────")

    for cloud in order:
        attempted.append(cloud)

        # Remap instance_type / region / zone for this specific cloud
        cloud_decision = _decision_for_cloud(decision, cloud)

        # Build a cloud-specific startup script.
        # AWS uses Amazon Linux 2023 (dnf, no apt-get, awscli pre-installed).
        # GCP/Azure use Ubuntu 22.04 (apt-get, awscli installed via apt).
        # Stamping CLOUD at generation time here is correct because
        # cloud_decision["cloud"] == cloud at this point.
        cloud_script = _build_training_script(job, cloud_decision)

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
            logger.info(
                f"[Launcher] ✓ Launch succeeded on [{cloud}]  "
                f"instance={cloud_decision['instance_type']}"
            )
            return result
        except ModuleNotFoundError as e:
            # Azure SDK (azure-mgmt-compute etc.) not installed
            if "azure" in str(e).lower():
                logger.warning(
                    f"[Launcher] ✗ [{cloud}] skipped — Azure SDK not installed. "
                    f"Fix: pip install azure-identity azure-mgmt-compute azure-mgmt-network  "
                    f"(tried: {attempted})"
                )
            else:
                logger.error(f"[Launcher] ✗ [{cloud}] missing module: {e}  (tried: {attempted})")
            if cloud != order[-1]:
                logger.info("[Launcher] Trying next cloud in fallback chain...")

        except Exception as e:
            logger.error(
                f"[Launcher] ✗ [{cloud}] failed: {e}  (tried: {attempted})"
            )
            if cloud != order[-1]:
                logger.info("[Launcher] Trying next cloud in fallback chain...")

    # ── All spot attempts failed → GCP on-demand ─────────────────
    logger.error(
        f"[Launcher] All spot clouds failed {attempted} → trying GCP on-demand"
    )
    try:
        gcp_decision         = _decision_for_cloud(decision, "gcp")
        gcp_decision["spot"] = False
        gcp_script           = _build_training_script(job, gcp_decision)
        result = _launch_gcp(gcp_decision, gcp_script, on_demand=True)
        result["cloud"]     = "gcp"
        result["on_demand"] = True
        logger.warning(
            f"[Launcher] ⚠ Running on GCP on-demand: "
            f"{gcp_decision['instance_type']} in {gcp_decision.get('zone','?')}"
        )
        return result
    except Exception as e:
        raise RuntimeError(
            f"[Launcher] All launch attempts failed "
            f"({attempted} + on-demand): {e}"
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
# Startup script
# ══════════════════════════════════════════════════════════════════

def _build_training_script(job: dict, decision: dict) -> str:
    """
    Return a cloud-specific startup script for the VM being launched.

    Why separate scripts per cloud:
        GCP / Azure  — Ubuntu 22.04, uses apt-get, awscli not pre-installed
        AWS          — Amazon Linux 2023, uses dnf, awscli v2 pre-installed,
                       NO apt-get (that's what caused the error)

    The cloud value IS stamped at generation time here because this function
    is called inside _launch_with_fallback() AFTER cloud_decision is resolved
    for the specific cloud being launched — not from the original Pareto winner.
    So "cloud" correctly reflects which VM is actually being created.

    Values stamped at generation time (safe — VM metadata only readable
    by the instance owner):
        JOB_ID, RESUME_STEP, PREV_CLOUD, CLOUD
        CHECKPOINT_S3_BUCKET, AWS_ACCESS_KEY_ID,
        AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
    """
    job_id      = job.get("job_id",      "unknown")
    resume_step = job.get("resume_step", 0)
    prev_cloud  = job.get("prev_cloud",  "")
    cloud       = decision.get("cloud",  "gcp")
    s3_bucket   = S3_BUCKET
    aws_key     = os.getenv("AWS_ACCESS_KEY_ID",     "")
    aws_secret  = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    aws_region  = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))

    # ── Shared header (identical on all clouds) ───────────────────
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

# ── Credentials stamped at launch time ───────────────────────────
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
  echo "       Set CHECKPOINT_S3_BUCKET in your .env before submitting jobs."
  exit 1
fi
"""

    # ── AWS (Amazon Linux 2023) ───────────────────────────────────
    # Uses dnf (NOT apt-get). awscli v2 is pre-installed as a SYSTEM tool.
    # CRITICAL: do NOT pip3 install boto3/dateutil at system level — it
    # overwrites the system python-dateutil that awscli v2 bundles, causing
    # "ModuleNotFoundError: No module named 'dateutil'" when running `aws`.
    # Fix: use a virtualenv for all pip installs, keeping system Python clean.
    # The venv's `aws` binary is NOT used — we use the system awscli (/usr/bin/aws)
    # which has its own bundled Python and is unaffected by the venv.
    if cloud == "aws":
        pkg_block = """
echo "==> System packages (Amazon Linux 2023 — dnf)..."
# python3-venv does NOT exist on AL2023 — venv is built into python3 itself
dnf install -y -q python3-pip python3 tar gzip

echo "==> Verifying system awscli (pre-installed, must stay untouched)..."
/usr/local/bin/aws --version || /usr/bin/aws --version

# ── Swap file — t3.small (2GB RAM) needs swap for torch extraction ──
SWAP_FILE="/swapfile"
if [ ! -f "$SWAP_FILE" ]; then
  echo "==> Creating 2GB swap file..."
  dd if=/dev/zero of="$SWAP_FILE" bs=128M count=16 status=progress 2>/dev/null
  chmod 600 "$SWAP_FILE"
  mkswap "$SWAP_FILE"
  swapon "$SWAP_FILE"
  echo "==> Swap enabled: $(free -h | grep Swap)"
fi

# ── Virtualenv — isolates pip from system Python ──────────────────
# This is REQUIRED on Amazon Linux 2023 to avoid breaking awscli.
# The venv gets boto3/numpy/torch. System Python keeps awscli's deps.
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

# Verify awscli still works after venv activation
echo "==> Confirming awscli intact after venv: $(/usr/local/bin/aws --version 2>/dev/null || /usr/bin/aws --version)"

PYTHON_CMD="/opt/trainer-env/bin/python3"
AWS_CMD="/usr/local/bin/aws"
# Fallback to /usr/bin/aws if v2 not in /usr/local
[ -f "$AWS_CMD" ] || AWS_CMD="/usr/bin/aws"
"""

    # ── GCP / Azure (Ubuntu 22.04) ────────────────────────────────
    # Uses apt-get. awscli NOT pre-installed — install via apt.
    else:
        pkg_block = """
echo "==> System packages (Ubuntu 22.04 — apt-get)..."
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    python3-pip python3-venv python3-dev awscli

echo "==> awscli version: $(aws --version)"

# ── Swap file (safety net for e2-standard-2 with 8GB RAM) ────────
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

    # ── Shared: download trainer + run + shutdown (all clouds) ────
    tail = f"""
if [ -z "${{CHECKPOINT_S3_BUCKET}}" ]; then
  echo "ERROR: CHECKPOINT_S3_BUCKET is empty — cannot download trainer files."
  exit 1
fi

# AWS path sets $AWS_CMD to /usr/local/bin/aws or /usr/bin/aws (system awscli v2)
# GCP/Azure use plain `aws` (installed via apt-get into PATH)
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
[ $EXIT_CODE -eq 0 ] \\
    && echo "==> Done (exit 0). Shutdown in 60s..." \\
    || echo "==> Exited ${{EXIT_CODE}}. Shutdown in 60s..."

echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') shutdown queued"
sleep 60
shutdown -h now
"""

    return header + pkg_block + tail

    return header + pkg_block + tail


def _find_latest_checkpoint(job_id: str) -> tuple[Optional[str], int]:
    """
    Look for the latest checkpoint on S3 for this job.

    Priority:
        1. checkpoint_latest.pt  — always written by CheckpointEngine.save()
        2. step_XXXXXXXX.pt      — milestone saves, use most-recently modified

    Also reads job_state.json on S3 to get the authoritative RESUME_STEP
    (the step counter train.py was at when it last saved).

    Returns:
        (s3_path, resume_step)  — s3_path is None if no checkpoint found
    """
    if not S3_BUCKET:
        logger.warning("[Launcher] CHECKPOINT_S3_BUCKET not set")
        return None, 0

    prefix      = f"checkpoints/{job_id}/"
    resume_step = 0

    # ── Read step from job_state.json ─────────────────────────────
    state = _s3_get_json(f"{prefix}job_state.json")
    if state:
        resume_step = int(state.get("step", 0))
        logger.info(
            f"[Launcher] S3 job_state: step={resume_step} "
            f"status={state.get('status','?')}"
        )

    # ── Look for checkpoint_latest.pt ─────────────────────────────
    s3  = _s3_client()
    latest_key = f"{prefix}checkpoint_latest.pt"
    try:
        s3.head_object(Bucket=S3_BUCKET, Key=latest_key)
        path = f"s3://{S3_BUCKET}/{latest_key}"
        logger.info(f"[Launcher] Checkpoint: {path}  step={resume_step}")
        return path, resume_step
    except ClientError:
        pass

    # ── Fall back: most-recent step_*.pt ─────────────────────────
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
    """GCP e2-standard-4 spot — CPU, works on free/trial accounts."""
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
        "gpu_model":       None,
        "gpu_mem_gb":      0,
        "gpu_count":       0,
        "vcpus":           4,
        "ram_gb":          16,
        "s3_bucket":       S3_BUCKET,
        "reason":          "Emergency fallback — selector unavailable (GCP e2-standard-4 CPU)",
        "pareto_set":      [],
    }


def _write_job_state(job_id: str, data: dict):
    """Local job_state.json write (used by cloud launchers for local tracking)."""
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
    """Read from local job_state.json (fallback when S3 unavailable)."""
    try:
        path = Path(JOB_STATE_PATH)
        if not path.exists():
            return None
        return json.loads(path.read_text()).get(job_id)
    except Exception as e:
        logger.warning(f"[Launcher] Local job_state read failed: {e}")
        return None