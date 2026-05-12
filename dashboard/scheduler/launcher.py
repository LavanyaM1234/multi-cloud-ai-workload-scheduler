"""
scheduler/launcher.py
──────────────────────
Creates GCP spot VMs to run training jobs.

Key change from previous version:
  Before launch, packages and uploads the local checkpoint/ folder
  as checkpoint_pkg.tar.gz so startup.sh can download it in one shot.

Decision: launcher uploads from the local project directory where
  server.py runs. So your project layout must be:
    dashboard/
      api/server.py
      scheduler/launcher.py
      scheduler/selector.py
      trainer/train.py
      checkpoint/engine.py
      checkpoint/storage.py
      checkpoint/trainer.py
      checkpoint/__init__.py

  launcher.py finds trainer/ and checkpoint/ relative to its own location:
    launcher.py is at  scheduler/launcher.py
    trainer/    is at  ../trainer/
    checkpoint/ is at  ../checkpoint/

Called by:
  api/server.py → submit_job()   for new jobs
  api/server.py → PreemptionPoller._migrate()  for resumed jobs
"""

import os, json, time, tarfile, tempfile
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ID = os.getenv("GCP_PROJECT_ID",        "tensile-method-459009-k2")
REGION     = os.getenv("GCP_REGION",            "us-central1")
ZONE       = os.getenv("GCP_ZONE",              "us-central1-a")
GCS_BUCKET    = os.getenv("CHECKPOINT_GCS_BUCKET", "")
S3_BUCKET     = os.getenv("CHECKPOINT_S3_BUCKET",  "")
AWS_KEY_ID    = os.getenv("AWS_ACCESS_KEY_ID",      "")
AWS_SECRET    = os.getenv("AWS_SECRET_ACCESS_KEY",  "")
AWS_REGION    = os.getenv("AWS_DEFAULT_REGION",     "us-east-1")
BASE_IMAGE = "projects/debian-cloud/global/images/family/debian-12"

# Project root = one level above this file (scheduler/../)
_HERE        = Path(__file__).parent
PROJECT_ROOT = _HERE.parent
TRAINER_DIR  = PROJECT_ROOT / "trainer"
CKPT_DIR     = PROJECT_ROOT / "checkpoint"


# ══════════════════════════════════════════════════════════════════
# GCS HELPERS
# ══════════════════════════════════════════════════════════════════

def _gcs():
    from google.cloud import storage
    return storage.Client(project=PROJECT_ID)

def _gcs_write_json(gcs_path: str, data: dict):
    if not GCS_BUCKET:
        return
    _gcs().bucket(GCS_BUCKET).blob(gcs_path).upload_from_string(
        json.dumps(data, indent=2), content_type="application/json"
    )

def _gcs_read_json(gcs_path: str) -> dict:
    return json.loads(
        _gcs().bucket(GCS_BUCKET).blob(gcs_path).download_as_text()
    )

def _gcs_upload_file(local_path: str, gcs_path: str):
    """Upload a local file to GCS bucket."""
    _gcs().bucket(GCS_BUCKET).blob(gcs_path).upload_from_filename(str(local_path))
    size_kb = Path(local_path).stat().st_size // 1024
    print(f"  [gcs] uploaded {Path(local_path).name} ({size_kb}KB)"
          f" → gs://{GCS_BUCKET}/{gcs_path}")


# ══════════════════════════════════════════════════════════════════
# UPLOAD TRAINER FILES TO GCS
# Decision: done at launch time so the VM always gets the latest
# version of train.py and checkpoint/ — no stale code on VM.
# ══════════════════════════════════════════════════════════════════

def _upload_trainer_files():
    """
    Upload trainer/train.py and checkpoint/ package to GCS.
    Creates:
      gs://{bucket}/trainer/train.py
      gs://{bucket}/trainer/checkpoint_pkg.tar.gz
    """
    if not GCS_BUCKET:
        print("[WARN] GCS_BUCKET not set — skipping trainer file upload")
        return

    print("[launcher] Uploading trainer files to GCS...")

    # Upload train.py
    train_py = TRAINER_DIR / "train.py"
    if not train_py.exists():
        raise FileNotFoundError(
            f"train.py not found at {train_py}\n"
            f"Expected project layout:\n"
            f"  {PROJECT_ROOT}/trainer/train.py\n"
            f"  {PROJECT_ROOT}/checkpoint/engine.py"
        )
    _gcs_upload_file(train_py, "trainer/train.py")

    # Package checkpoint/ as tar.gz
    # Decision: include all .py files + __init__.py
    # Exclude __pycache__ and .pyc files
    if not CKPT_DIR.exists():
        raise FileNotFoundError(
            f"checkpoint/ not found at {CKPT_DIR}\n"
            f"Expected: {PROJECT_ROOT}/checkpoint/"
        )

    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f:
        tar_path = f.name

    with tarfile.open(tar_path, "w:gz") as tar:
        for py_file in sorted(CKPT_DIR.glob("*.py")):
            # arcname puts files under checkpoint/ inside the tar
            tar.add(py_file, arcname=f"checkpoint/{py_file.name}")
            print(f"  [tar] added checkpoint/{py_file.name}")

    _gcs_upload_file(tar_path, "trainer/checkpoint_pkg.tar.gz")
    Path(tar_path).unlink(missing_ok=True)

    print("[launcher] Trainer files uploaded ✓")


# ══════════════════════════════════════════════════════════════════
# JOB CONFIG + INITIAL STATE
# ══════════════════════════════════════════════════════════════════

def _write_job_config(job_id: str, config: dict) -> bool:
    """Write job_config.json to GCS. Returns True on success."""
    if not GCS_BUCKET:
        # Local fallback for testing without GCS
        local = Path(f"job_config_{job_id}.json")
        local.write_text(json.dumps(config, indent=2))
        print(f"[WARN] GCS_BUCKET not set — wrote config to {local}")
        return True
    try:
        _gcs_write_json(f"checkpoints/{job_id}/job_config.json", config)
        print(f"[launcher] Config → gs://{GCS_BUCKET}/checkpoints/{job_id}/job_config.json")
        return True
    except Exception as e:
        print(f"[ERROR] _write_job_config: {e}")
        return False


def _write_initial_state(job_id: str, config: dict, status: str = "queued"):
    """
    Write job_state.json with the given status immediately.
    Dashboard shows this while VM is still booting (~2 min).
    """
    if not GCS_BUCKET:
        return
    try:
        state = {
            "job_id":          job_id,
            "task_name":       config.get("task_name", "Untitled"),
            "status":          status,
            "epoch":           0,
            "total_epochs":    config.get("epochs", 50),
            "step":            config.get("resume_from_step", 0),
            "loss":            None,
            "accuracy":        None,
            "cloud":           "gcp",
            "instance":        config.get("instance_type", "e2-standard-4"),
            "migration_count": config.get("migration_count", 0),
            "resumed_from":    config.get("resume_from_step", 0),
            "updated_at":      datetime.now(timezone.utc).isoformat(),
        }
        _gcs_write_json(f"checkpoints/{job_id}/job_state.json", state)
        print(f"[launcher] State({status}) written for {job_id}")
    except Exception as e:
        print(f"[WARN] _write_initial_state: {e}")


def _write_failed_state(job_id: str, error: str):
    if not GCS_BUCKET:
        return
    try:
        _gcs_write_json(f"checkpoints/{job_id}/job_state.json", {
            "job_id":     job_id,
            "status":     "launch_failed",
            "error":      error,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════
# STARTUP SCRIPT
# ══════════════════════════════════════════════════════════════════

def _get_startup_script() -> str:
    """
    Load startup.sh from GCS (uploaded there by setup_gcs.sh).
    Falls back to inline minimal script if not found.
    """
    if GCS_BUCKET:
        try:
            script = _gcs().bucket(GCS_BUCKET).blob(
                "trainer/startup.sh").download_as_text()
            print("[launcher] startup.sh loaded from GCS")
            return script
        except Exception as e:
            print(f"[WARN] startup.sh not in GCS: {e} — using inline fallback")

    # Inline fallback — mirrors startup.sh but minimal
    return r"""#!/bin/bash
set -e; LOG="/var/log/trainer.log"; exec > >(tee -a "$LOG") 2>&1
echo "==> Startup $(date -u)"
META="http://metadata.google.internal/computeMetadata/v1/instance/attributes"
H="Metadata-Flavor: Google"
export JOB_ID=$(curl -sf "${META}/JOB_ID" -H "$H" || echo "unknown")
export GCS_BUCKET=$(curl -sf "${META}/GCS_BUCKET" -H "$H" || echo "")
export INSTANCE_TYPE=$(curl -sf "${META}/INSTANCE_TYPE" -H "$H" || echo "e2-standard-4")
export RESUME_STEP=$(curl -sf "${META}/RESUME_STEP" -H "$H" || echo "0")
export PREV_CLOUD=$(curl -sf "${META}/PREV_CLOUD" -H "$H" || echo "")
export CLOUD="gcp"
apt-get update -qq && apt-get install -y -qq python3-pip python3-venv
python3 -m venv /opt/trainer-env && source /opt/trainer-env/bin/activate
pip install -q torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu
pip install -q google-cloud-storage boto3 numpy
mkdir -p /opt/trainer && cd /opt/trainer
gsutil cp "gs://${GCS_BUCKET}/trainer/train.py" ./train.py
gsutil cp "gs://${GCS_BUCKET}/trainer/checkpoint_pkg.tar.gz" ./checkpoint_pkg.tar.gz
tar -xzf checkpoint_pkg.tar.gz && rm -f checkpoint_pkg.tar.gz
python3 train.py
sleep 60; shutdown -h now
"""


# ══════════════════════════════════════════════════════════════════
# VM CREATION
# ══════════════════════════════════════════════════════════════════

def _create_vm(
    job_id:        str,
    instance_type: str,
    resume_step:   int = 0,
    prev_cloud:    str = "",
) -> dict:
    """
    Create a GCP spot VM.
    Injects job params as instance metadata — startup.sh reads them.
    Returns dict with instance_name, console_url, logs_url.
    """
    try:
        from google.cloud import compute_v1
    except ImportError:
        raise RuntimeError("pip install google-cloud-compute")

    # Decision: instance name must be ≤ 63 chars, lowercase, no underscores
    raw_name      = f"ml-{job_id[:18]}".lower()
    instance_name = "".join(
        c if c.isalnum() or c == "-" else "-" for c in raw_name
    ).strip("-")

    # ── GCP instance type mapping ──────────────────────────────────
    # Phase 1 (GCP only): selector always returns e2-standard-4.
    # The modal instance dropdown has AWS names (g4dn.xlarge etc) —
    # we ignore that and always use what selector decided.
    # TODO Phase 4 (multi-cloud): when selector can return AWS/Azure
    # instances, launcher.py will need to route to the correct cloud's
    # SDK here. For now we only call _create_vm for GCP.
    gcp_instance_type = instance_type  # already e2-standard-4 from selector
    machine_type      = f"zones/{ZONE}/machineTypes/{gcp_instance_type}"
    startup           = _get_startup_script()

    print(f"[launcher] Creating VM: {instance_name}")
    print(f"           machine: {gcp_instance_type}  zone: {ZONE}")
    print(f"           resume_step: {resume_step}  prev_cloud: {prev_cloud or 'none'}")

    # Boot disk — Debian 12, 20GB standard persistent
    inst = compute_v1.Instance()
    inst.name         = instance_name
    inst.machine_type = machine_type

    disk = compute_v1.AttachedDisk()
    disk.boot            = True
    disk.auto_delete     = True
    disk.initialize_params = compute_v1.AttachedDiskInitializeParams(
        source_image = BASE_IMAGE,
        disk_size_gb = 20,
        disk_type    = f"zones/{ZONE}/diskTypes/pd-standard",
    )
    inst.disks = [disk]

    # Network — default VPC, external IP for pip/gsutil access
    nic = compute_v1.NetworkInterface()
    nic.name = "global/networks/default"
    ac = compute_v1.AccessConfig()
    ac.type_ = "ONE_TO_ONE_NAT"
    ac.name  = "External NAT"
    nic.access_configs = [ac]
    inst.network_interfaces = [nic]

    # Spot scheduling
    sched = compute_v1.Scheduling()
    sched.provisioning_model          = "SPOT"
    sched.instance_termination_action = "STOP"
    sched.on_host_maintenance         = "TERMINATE"
    inst.scheduling = sched

    # Service account — default Compute SA with cloud-platform scope
    # Decision: cloud-platform gives GCS + Compute read/write.
    # Narrower scopes (storage-rw only) would also work but
    # cloud-platform is simpler and safe for internal VMs.
    sa = compute_v1.ServiceAccount()
    sa.email  = "default"
    sa.scopes = [
        "https://www.googleapis.com/auth/cloud-platform",
        "https://www.googleapis.com/auth/logging.write",
    ]
    inst.service_accounts = [sa]

    # Metadata — startup script + all job env vars
    inst.metadata = compute_v1.Metadata(items=[
        compute_v1.Items(key="startup-script",        value=startup),
        compute_v1.Items(key="JOB_ID",                value=job_id),
        compute_v1.Items(key="GCS_BUCKET",            value=GCS_BUCKET),
        compute_v1.Items(key="INSTANCE_TYPE",         value=gcp_instance_type),
        compute_v1.Items(key="RESUME_STEP",           value=str(resume_step)),
        compute_v1.Items(key="PREV_CLOUD",            value=prev_cloud),
        # AWS creds — needed by storage.py (checkpoint S3) and
        # train.py (dataset download). Read from server .env.
        compute_v1.Items(key="CHECKPOINT_GCS_BUCKET", value=GCS_BUCKET),
        compute_v1.Items(key="CHECKPOINT_S3_BUCKET",  value=S3_BUCKET),
        compute_v1.Items(key="AWS_ACCESS_KEY_ID",     value=AWS_KEY_ID),
        compute_v1.Items(key="AWS_SECRET_ACCESS_KEY", value=AWS_SECRET),
        compute_v1.Items(key="AWS_DEFAULT_REGION",    value=AWS_REGION),
    ])

    # Create
    client    = compute_v1.InstancesClient()
    operation = client.insert(
        project=PROJECT_ID, zone=ZONE, instance_resource=inst
    )
    print("[launcher] Waiting for VM creation operation...")
    operation.result(timeout=120)
    print(f"[launcher] ✓ VM created: {instance_name}")

    return {
        "instance_name": instance_name,
        "zone":          ZONE,
        "project":       PROJECT_ID,
        "console_url": (
            f"https://console.cloud.google.com/compute/instancesDetail"
            f"/zones/{ZONE}/instances/{instance_name}"
            f"?project={PROJECT_ID}"
        ),
        "logs_url": (
            f"https://console.cloud.google.com/logs/query"
            f";query=resource.type%3D%22gce_instance%22%20"
            f"labels.%22compute.googleapis.com%2Fresource_name%22"
            f"%3D%22{instance_name}%22?project={PROJECT_ID}"
        ),
    }


# ══════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════

def launch_job(job_id: str, job_params: dict, decision: dict) -> dict:
    """
    Full launch sequence for a NEW job:
      1. Upload latest train.py + checkpoint/ package to GCS
      2. Write job_config.json to GCS
      3. Write initial job_state.json (status=queued)
      4. Create spot VM — startup.sh runs automatically

    Args:
        job_id:     unique job identifier
        job_params: all form fields from modal (lr, hidden_dim, etc.)
        decision:   output of selector.pick_best_cloud()

    Returns:
        dict with launched=True + VM info, or error key on failure
    """
    print(f"\n{'='*50}\n  Launching NEW job: {job_id}\n{'='*50}")

    # ── Upload latest trainer code to GCS ─────────────────────────
    # Decision: always upload at launch time so VM gets latest code.
    # Small overhead (~2s) but ensures no stale code on VM.
    try:
        _upload_trainer_files()
    except FileNotFoundError as e:
        return {"error": str(e), "launched": False, "job_id": job_id}

    # ── Build full config ─────────────────────────────────────────
    config = {
        "job_id":            job_id,
        "task_name":         job_params.get("task_name",    "Untitled"),
        "lr":                float(job_params.get("lr",           0.001)),
        "hidden_dim":        int(  job_params.get("hidden_dim",   256)),
        "dropout":           float(job_params.get("dropout",      0.3)),
        "batch_size":        int(  job_params.get("batch_size",   64)),
        "epochs":            int(  job_params.get("epochs",       50)),
        "ckpt_every":        int(  job_params.get("ckpt_every",   50)),
        "input_dim":         int(  job_params.get("input_dim",    50)),
        "num_classes":       int(  job_params.get("num_classes",  5)),
        "max_budget":        float(job_params.get("max_budget",   2.0)),
        "deadline_hrs":      float(job_params.get("deadline_hrs", 8.0)),
        "cloud":             decision["cloud"],
        "instance_type":     decision["instance_type"],
        "region":            decision["region"],
        "zone":              decision["zone"],
        "price_usd_hr":      decision["price_usd_hr"],
        "gcs_bucket":        GCS_BUCKET,
        "dataset":           job_params.get("dataset", "synthetic"),
        "resume_from_step":  0,
        "migration_count":   0,
        "submitted_at":      datetime.now(timezone.utc).isoformat(),
    }

    if not _write_job_config(job_id, config):
        return {"error": "Failed to write job_config.json to GCS",
                "launched": False, "job_id": job_id}

    _write_initial_state(job_id, config, status="queued")

    try:
        vm = _create_vm(job_id, decision["instance_type"],
                        resume_step=0, prev_cloud="")
        print(f"\n[OK] Job launched!\n"
              f"     VM:      {vm['instance_name']}\n"
              f"     Console: {vm['console_url']}\n")
        return {**decision, **vm, "launched": True, "job_id": job_id}
    except Exception as e:
        print(f"[ERROR] VM launch failed: {e}")
        _write_failed_state(job_id, str(e))
        return {**decision, "error": str(e), "launched": False, "job_id": job_id}


def resume_job(
    job_id:      str,
    resume_step: int,
    prev_cloud:  str,
    decision:    dict,
) -> dict:
    """
    Relaunch a preempted job on a (possibly different) cloud.
    Called by server.py PreemptionPoller._migrate().

    Key differences from launch_job:
      - Does NOT re-upload trainer files (already in GCS from launch)
      - Does NOT overwrite job_config.json entirely — patches it
      - Writes status=migrating before VM creation
      - Passes RESUME_STEP + PREV_CLOUD as VM metadata
    """
    print(f"\n{'='*50}")
    print(f"  RESUMING: {job_id}")
    print(f"  step:     {resume_step}")
    print(f"  migration:{prev_cloud} → {decision['cloud']}")
    print(f"{'='*50}")

    # Patch job config with updated cloud + resume info
    try:
        config = _gcs_read_json(f"checkpoints/{job_id}/job_config.json")
    except Exception as e:
        print(f"[WARN] Could not read existing config: {e} — using minimal config")
        config = {"job_id": job_id}

    config.update({
        "cloud":            decision["cloud"],
        "instance_type":    decision["instance_type"],
        "region":           decision["region"],
        "zone":             decision["zone"],
        "price_usd_hr":     decision["price_usd_hr"],
        "resume_from_step": resume_step,
        "migration_count":  config.get("migration_count", 0) + 1,
        "last_migration":   datetime.now(timezone.utc).isoformat(),
        "prev_cloud":       prev_cloud,
    })

    _write_job_config(job_id, config)
    _write_initial_state(job_id, config, status="migrating")

    try:
        vm = _create_vm(job_id, decision["instance_type"],
                        resume_step=resume_step, prev_cloud=prev_cloud)
        print(f"\n[OK] Job resumed!\n"
              f"     VM:      {vm['instance_name']}\n"
              f"     Console: {vm['console_url']}\n")
        return {**decision, **vm, "launched": True,
                "job_id": job_id, "resumed_from_step": resume_step}
    except Exception as e:
        print(f"[ERROR] Resume VM launch failed: {e}")
        _write_failed_state(job_id, str(e))
        return {**decision, "error": str(e), "launched": False, "job_id": job_id}