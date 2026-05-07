"""
scheduler/launcher.py
──────────────────────
Creates GCP spot VMs to run training jobs.
Supports both fresh starts and resume-after-preemption.

Called by:
  api/server.py → submit_job()        for new jobs
  api/server.py → preemption_poller() for migrated/resumed jobs
"""

import os, json, time
from datetime import datetime, timezone

PROJECT_ID = os.getenv("GCP_PROJECT_ID",        "tensile-method-459009-k2")
REGION     = os.getenv("GCP_REGION",            "us-central1")
ZONE       = os.getenv("GCP_ZONE",              "us-central1-a")
GCS_BUCKET = os.getenv("CHECKPOINT_GCS_BUCKET", "")
BASE_IMAGE = "projects/debian-cloud/global/images/family/debian-12"


# ── GCS helpers ───────────────────────────────────────────────────

def _gcs():
    from google.cloud import storage
    return storage.Client(project=PROJECT_ID)

def _write_gcs_json(path, data):
    if not GCS_BUCKET:
        return
    _gcs().bucket(GCS_BUCKET).blob(path).upload_from_string(
        json.dumps(data, indent=2), content_type="application/json"
    )

def _read_gcs_json(path):
    return json.loads(_gcs().bucket(GCS_BUCKET).blob(path).download_as_text())


# ── Startup script ────────────────────────────────────────────────

def _get_startup_script():
    """Load startup.sh from GCS. Falls back to inline minimal script."""
    if GCS_BUCKET:
        try:
            script = _gcs().bucket(GCS_BUCKET).blob("trainer/startup.sh").download_as_text()
            print("[OK] startup.sh loaded from GCS")
            return script
        except Exception as e:
            print(f"[WARN] startup.sh from GCS failed: {e} — using inline fallback")

    # Minimal inline fallback
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
apt-get update -qq && apt-get install -y -qq python3-pip
pip install -q torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu
pip install -q google-cloud-storage numpy
mkdir -p /opt/trainer && cd /opt/trainer
gsutil cp "gs://${GCS_BUCKET}/trainer/train.py" ./train.py
python3 train.py
sleep 60; shutdown -h now
"""


# ══════════════════════════════════════════════════════════════════
# WRITE JOB CONFIG + INITIAL STATE
# ══════════════════════════════════════════════════════════════════

def write_job_config(job_id, config):
    """Write job_config.json to GCS before VM boots."""
    if not GCS_BUCKET:
        with open(f"job_config_{job_id}.json", "w") as f:
            json.dump(config, f, indent=2)
        print(f"[WARN] GCS_BUCKET not set — wrote config locally")
        return True
    try:
        _write_gcs_json(f"checkpoints/{job_id}/job_config.json", config)
        print(f"[OK] Config → gs://{GCS_BUCKET}/checkpoints/{job_id}/job_config.json")
        return True
    except Exception as e:
        print(f"[ERROR] write_job_config: {e}")
        return False


def write_initial_state(job_id, config, status="queued"):
    """
    Write job_state.json with status=queued so dashboard shows the
    job immediately, even before the VM finishes booting (~2 min).
    """
    if not GCS_BUCKET:
        return
    try:
        state = {
            "job_id":       job_id,
            "task_name":    config.get("task_name", "Untitled"),
            "status":       status,
            "epoch":        0,
            "total_epochs": config.get("epochs", 50),
            "step":         config.get("resume_from_step", 0),
            "loss":         None,
            "accuracy":     None,
            "cloud":        "gcp",
            "instance":     config.get("instance_type", "e2-standard-4"),
            "updated_at":   datetime.now(timezone.utc).isoformat(),
        }
        _write_gcs_json(f"checkpoints/{job_id}/job_state.json", state)
        print(f"[OK] State({status}) → gs://{GCS_BUCKET}/checkpoints/{job_id}/job_state.json")
    except Exception as e:
        print(f"[WARN] write_initial_state: {e}")


# ══════════════════════════════════════════════════════════════════
# VM CREATION
# ══════════════════════════════════════════════════════════════════

def _create_vm(job_id, instance_type, resume_step=0, prev_cloud=""):
    """
    Create a GCP spot VM with job metadata injected.
    resume_step=0 → fresh start
    resume_step>0 → VM will download checkpoint and resume
    """
    try:
        from google.cloud import compute_v1
    except ImportError:
        raise RuntimeError("pip install google-cloud-compute")

    instance_name = f"ml-{job_id[:18].lower().replace('_','-').replace('.','-')}"
    machine_type  = f"zones/{ZONE}/machineTypes/{instance_type}"
    startup       = _get_startup_script()

    print(f"[launcher] Creating VM: {instance_name}")
    print(f"           type: {instance_type} | zone: {ZONE}")
    print(f"           resume_step: {resume_step} | prev_cloud: {prev_cloud or 'none'}")

    # ── Instance resource ─────────────────────────────────────────
    inst = compute_v1.Instance()
    inst.name         = instance_name
    inst.machine_type = machine_type

    # Boot disk — 20GB Debian 12
    disk = compute_v1.AttachedDisk()
    disk.boot            = True
    disk.auto_delete     = True
    disk.initialize_params = compute_v1.AttachedDiskInitializeParams(
        source_image = BASE_IMAGE,
        disk_size_gb = 20,
        disk_type    = f"zones/{ZONE}/diskTypes/pd-standard",
    )
    inst.disks = [disk]

    # Network — default VPC with external IP
    nic            = compute_v1.NetworkInterface()
    nic.name       = "global/networks/default"
    ac             = compute_v1.AccessConfig()
    ac.type_       = "ONE_TO_ONE_NAT"
    ac.name        = "External NAT"
    nic.access_configs = [ac]
    inst.network_interfaces = [nic]

    # Spot VM scheduling
    sched = compute_v1.Scheduling()
    sched.provisioning_model          = "SPOT"
    sched.instance_termination_action = "STOP"
    sched.on_host_maintenance         = "TERMINATE"
    inst.scheduling = sched

    # Service account — default SA with full GCS access
    sa        = compute_v1.ServiceAccount()
    sa.email  = "default"
    sa.scopes = ["https://www.googleapis.com/auth/cloud-platform",
                 "https://www.googleapis.com/auth/logging.write"]
    inst.service_accounts = [sa]

    # Metadata — startup script + all job params
    meta_items = [
        compute_v1.Items(key="startup-script",  value=startup),
        compute_v1.Items(key="JOB_ID",          value=job_id),
        compute_v1.Items(key="GCS_BUCKET",      value=GCS_BUCKET),
        compute_v1.Items(key="INSTANCE_TYPE",   value=instance_type),
        compute_v1.Items(key="RESUME_STEP",     value=str(resume_step)),
        compute_v1.Items(key="PREV_CLOUD",      value=prev_cloud),
    ]
    inst.metadata = compute_v1.Metadata(items=meta_items)

    # ── Create ────────────────────────────────────────────────────
    client    = compute_v1.InstancesClient()
    operation = client.insert(project=PROJECT_ID, zone=ZONE,
                               instance_resource=inst)
    print("[launcher] Waiting for VM creation...")
    operation.result(timeout=120)
    print(f"[launcher] ✓ VM created: {instance_name}")

    return {
        "instance_name": instance_name,
        "zone":          ZONE,
        "project":       PROJECT_ID,
        "console_url":   (f"https://console.cloud.google.com/compute/instancesDetail"
                          f"/zones/{ZONE}/instances/{instance_name}"
                          f"?project={PROJECT_ID}"),
        "logs_url":      (f"https://console.cloud.google.com/logs/query"
                          f";query=resource.type%3D%22gce_instance%22%20"
                          f"labels.%22compute.googleapis.com%2Fresource_name%22"
                          f"%3D%22{instance_name}%22?project={PROJECT_ID}"),
    }


# ══════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════

def launch_job(job_id, job_params, decision):
    """
    Full launch for a NEW job (resume_step=0).

    Args:
        job_id:     unique job id string
        job_params: dict from modal (lr, hidden_dim, epochs, etc.)
        decision:   dict from selector.pick_best_cloud()
    """
    print(f"\n{'='*50}\n  Launching NEW job: {job_id}\n{'='*50}")

    config = {
        # Identity
        "job_id":         job_id,
        "task_name":      job_params.get("task_name", "Untitled"),
        # Hyperparams
        "lr":             float(job_params.get("lr",          0.001)),
        "hidden_dim":     int(  job_params.get("hidden_dim",  256)),
        "dropout":        float(job_params.get("dropout",     0.3)),
        "batch_size":     int(  job_params.get("batch_size",  64)),
        "epochs":         int(  job_params.get("epochs",      50)),
        "ckpt_every":     int(  job_params.get("ckpt_every",  50)),
        "input_dim":      int(  job_params.get("input_dim",   50)),
        "num_classes":    int(  job_params.get("num_classes", 5)),
        # Budget
        "max_budget":     float(job_params.get("max_budget",  2.0)),
        "deadline_hrs":   float(job_params.get("deadline_hrs",8.0)),
        # Cloud
        "cloud":          decision["cloud"],
        "instance_type":  decision["instance_type"],
        "region":         decision["region"],
        "zone":           decision["zone"],
        "price_usd_hr":   decision["price_usd_hr"],
        "gcs_bucket":     GCS_BUCKET,
        # Dataset
        "dataset":        job_params.get("dataset", "synthetic"),
        # Migration tracking
        "resume_from_step": 0,
        "migration_count":  0,
        "submitted_at":     datetime.now(timezone.utc).isoformat(),
    }

    if not write_job_config(job_id, config):
        return {**decision, "error": "Failed to write job config to GCS"}

    write_initial_state(job_id, config, status="queued")

    try:
        vm = _create_vm(job_id, decision["instance_type"],
                        resume_step=0, prev_cloud="")
        print(f"\n[OK] Job launched!\n     VM: {vm['instance_name']}\n     Console: {vm['console_url']}\n")
        return {**decision, **vm, "launched": True, "job_id": job_id}
    except Exception as e:
        print(f"[ERROR] VM launch failed: {e}")
        _safe_write_failed_state(job_id, str(e))
        return {**decision, "error": str(e), "launched": False, "job_id": job_id}


def resume_job(job_id, resume_step, prev_cloud, decision):
    """
    Relaunch a preempted/migrated job on a (possibly different) cloud.
    Called by the preemption poller in server.py.

    Args:
        job_id:      same job_id as the original job
        resume_step: step N to resume from (from job_state.json)
        prev_cloud:  cloud the job was running on before preemption
        decision:    new cloud decision from selector.pick_best_cloud()
    """
    print(f"\n{'='*50}")
    print(f"  RESUMING job: {job_id}")
    print(f"  From step:    {resume_step}")
    print(f"  Migration:    {prev_cloud} → {decision['cloud']}")
    print(f"{'='*50}")

    # Update job config with new cloud + resume info
    try:
        config = _read_gcs_json(f"checkpoints/{job_id}/job_config.json")
    except Exception:
        config = {}

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

    write_job_config(job_id, config)
    write_initial_state(job_id, config, status="migrating")

    try:
        vm = _create_vm(job_id, decision["instance_type"],
                        resume_step=resume_step, prev_cloud=prev_cloud)
        print(f"\n[OK] Job resumed!\n     VM: {vm['instance_name']}\n     Console: {vm['console_url']}\n")
        return {**decision, **vm, "launched": True, "job_id": job_id,
                "resumed_from_step": resume_step}
    except Exception as e:
        print(f"[ERROR] Resume launch failed: {e}")
        _safe_write_failed_state(job_id, str(e))
        return {**decision, "error": str(e), "launched": False}


def _safe_write_failed_state(job_id, error_msg):
    if not GCS_BUCKET:
        return
    try:
        _write_gcs_json(f"checkpoints/{job_id}/job_state.json", {
            "job_id":     job_id,
            "status":     "launch_failed",
            "error":      error_msg,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
    except Exception:
        pass
