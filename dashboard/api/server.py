"""
api/server.py
──────────────
Flask backend — serves real BigQuery data + manages GCP training jobs.

New in this version:
  - submit_job()        → calls selector + launcher for real GCP VM
  - get_jobs()          → reads job_state.json files from GCS
  - preemption_poller() → background thread, watches for preempted jobs
                          and auto-relaunches on cheapest available cloud
  - /api/jobs/resume    → manual resume endpoint (for debugging)

Run:
    cd dashboard
    python api/server.py

Endpoints (unchanged):
    GET  /api/prices/history     → 30-point time series per cloud
    GET  /api/prices/summary     → min/max/avg/current per cloud
    GET  /api/prices/latest      → latest price per instance type
    GET  /api/prices/preemptions → recent preempted=TRUE rows
    GET  /api/stats              → total rows, preemption count
    GET  /api/health             → BigQuery reachable?

New endpoints:
    GET  /api/jobs               → active jobs from GCS job_state.json
    POST /api/jobs/submit        → submit new job → create GCP VM
    POST /api/jobs/resume        → manually resume a preempted job
    GET  /api/jobs/<job_id>      → single job state
    GET  /api/risk               → LSTM + XGBoost preemption risk scores
"""

import os
import sys
import json
import time
import logging
import threading
from datetime import datetime, timezone, timedelta
from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv
from flask import render_template

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
load_dotenv()

# ── Logging setup ─────────────────────────────────────────────────
# Print logs go to stdout so you can see them in the terminal where
# you run `python api/server.py`. Level=DEBUG shows all model steps.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("google").setLevel(logging.WARNING)
logging.getLogger("google.cloud").setLevel(logging.WARNING)
logging.getLogger("google.auth").setLevel(logging.WARNING)
log = logging.getLogger("server")

app = Flask(
    __name__,
    template_folder="../templates",
    static_folder="../static"
)
CORS(app)

# ── Config ─────────────────────────────────────────────────────────
PROJECT_ID  = os.getenv("GCP_PROJECT_ID",         "tensile-method-459009-k2")
DATASET     = os.getenv("BIGQUERY_DATASET",        "spot_prices")
TABLE       = os.getenv("BIGQUERY_TABLE",          "price_history")
GCS_BUCKET  = os.getenv("CHECKPOINT_GCS_BUCKET",   "ml-scheduler-jobs-tensile-method-459009-k2")
GCP_ZONE    = os.getenv("GCP_ZONE",                "us-central1-a")

POLLER_INTERVAL = 30
MAX_MIGRATIONS  = 5


# ══════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════════

def get_bq_client():
    from google.cloud import bigquery
    return bigquery.Client(project=PROJECT_ID)


def get_gcs_client():
    from google.cloud import storage
    return storage.Client(project=PROJECT_ID)


def gcs_read_json(path):
    client = get_gcs_client()
    blob   = client.bucket(GCS_BUCKET).blob(path)
    return json.loads(blob.download_as_text())


def gcs_write_json(path, data):
    client = get_gcs_client()
    client.bucket(GCS_BUCKET).blob(path).upload_from_string(
        json.dumps(data, indent=2), content_type="application/json"
    )


def gcs_list_job_states():
    if not GCS_BUCKET:
        return []
    try:
        client = get_gcs_client()
        blobs  = client.bucket(GCS_BUCKET).list_blobs(prefix="checkpoints/")
        states = []
        for blob in blobs:
            if blob.name.endswith("job_state.json"):
                try:
                    state = json.loads(blob.download_as_text())
                    epoch        = state.get("epoch", 0)
                    total_epochs = state.get("total_epochs", 50)
                    state["progress_pct"] = min(
                        99, round((epoch / max(total_epochs, 1)) * 100)
                    )
                    states.append(state)
                    #print(state)
                except Exception:
                    pass
        return states
    except Exception as e:
        print(f"[gcs] list_job_states error: {e}")
        return []


# ══════════════════════════════════════════════════════════════════
# PREEMPTION POLLER
# ══════════════════════════════════════════════════════════════════

class PreemptionPoller(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self._stop_event      = threading.Event()
        self._resuming        = set()
        # Track when each job switched to on-demand {job_id: datetime}
        self._ondemand_since  = {}

    def stop(self):
        self._stop_event.set()

    def run(self):
        print(f"[poller] Preemption poller started — interval={POLLER_INTERVAL}s")
        while not self._stop_event.is_set():
            try:
                self._poll()
            except Exception as e:
                print(f"[poller] Error: {e}")
            self._stop_event.wait(POLLER_INTERVAL)

    def _poll(self):
        if not GCS_BUCKET:
            return
        states = gcs_list_job_states()
        for state in states:
            job_id = state.get("job_id")
            status = state.get("status")

            # ── Preemption → relaunch ─────────────────────────────
            if status == "preempted" and job_id not in self._resuming:
                self._resuming.add(job_id)
                print(f"\n[poller] Detected preempted job: {job_id}")
                t = threading.Thread(
                    target=self._migrate, args=(job_id, state), daemon=True
                )
                t.start()

            # ── On-demand timeout check ───────────────────────────
            # If a job is running on an on-demand instance AND the user
            # set spot_only=False with ondemand_max_hrs, enforce the limit.
            elif status == "running" and state.get("is_spot") is False:
                self._check_ondemand_timeout(job_id, state)

            # ── Cleanup resuming set ──────────────────────────────
            if status in ("running", "done", "budget_exceeded", "launch_failed") \
                    and job_id in self._resuming:
                self._resuming.discard(job_id)

            # Clear on-demand tracker when job finishes or goes back to spot
            if status in ("done", "failed", "budget_exceeded") \
                    and job_id in self._ondemand_since:
                del self._ondemand_since[job_id]

    def _check_ondemand_timeout(self, job_id: str, state: dict):
        """
        If a running job is on on-demand and has exceeded ondemand_max_hrs,
        write a 'migrate' command to GCS so train.py picks it up and
        triggers a checkpoint + exit, after which the poller relaunches
        on a spot instance.
        """
        try:
            config           = gcs_read_json(f"checkpoints/{job_id}/job_config.json")
            spot_only        = config.get("spot_only", True)
            ondemand_max_hrs = float(config.get("ondemand_max_hrs", 1.0))

            # If spot_only=True, on-demand should never have been used — skip
            if spot_only:
                return

            now = datetime.now(timezone.utc)

            # Record when we first noticed this job on on-demand
            if job_id not in self._ondemand_since:
                self._ondemand_since[job_id] = now
                print(f"[poller] {job_id} running on on-demand — "
                      f"max allowed: {ondemand_max_hrs}h")
                return

            elapsed_hrs = (now - self._ondemand_since[job_id]).total_seconds() / 3600

            if elapsed_hrs >= ondemand_max_hrs:
                print(f"[poller] {job_id} on-demand limit reached "
                      f"({elapsed_hrs:.2f}h >= {ondemand_max_hrs}h) — "
                      f"sending migrate command")
                # Write migrate command — train.py checks this every epoch
                gcs_write_json(
                    f"checkpoints/{job_id}/job_command.json",
                    {"command": "migrate", "reason": "ondemand_max_hrs_exceeded"}
                )
                # Remove from tracker — migrate will handle relaunch on spot
                del self._ondemand_since[job_id]

        except Exception as e:
            print(f"[poller] _check_ondemand_timeout error for {job_id}: {e}")

    def _migrate(self, job_id, state):
        try:
            from scheduler.selector import pick_best_cloud
            from scheduler.launcher import resume_job
            try:
                config = gcs_read_json(f"checkpoints/{job_id}/job_config.json")
            except Exception:
                config = {}

            migration_count = config.get("migration_count", 0)
            if migration_count >= MAX_MIGRATIONS:
                print(f"[poller] Job {job_id} hit migration limit — marking failed")
                gcs_write_json(f"checkpoints/{job_id}/job_state.json", {
                    **state,
                    "status":     "failed",
                    "error":      f"Exceeded max migrations ({MAX_MIGRATIONS})",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                })
                self._resuming.discard(job_id)
                return

            resume_step     = state.get("step", 0)
            prev_cloud      = state.get("cloud", "gcp")
            cost_so_far     = float(state.get("cost_usd", 0))
            original_budget = float(config.get("max_budget", 2.0))
            remaining       = max(0.10, original_budget - cost_so_far)

            # ── Pass full user preferences from original config ───
            # Old code only passed job_id, max_budget, deadline_hrs —
            # meaning preferred_clouds, spot_only, carbon_aware etc.
            # were all ignored on every relaunch after preemption.
            decision = pick_best_cloud({
                "job_id":           job_id,
                "max_budget":       remaining,
                "deadline_hrs":     config.get("deadline_hrs",     8.0),
                "priority":         config.get("priority",         "balanced"),
                "spot_only":        config.get("spot_only",        True),
                "ondemand_max_hrs": config.get("ondemand_max_hrs", 1.0),
                "preferred_clouds": config.get("preferred_clouds", ["aws", "gcp", "azure"]),
                "preferred_regions":config.get("preferred_regions",""),
                "gpu_required":     config.get("gpu_required",     False),
                "min_gpu_mem":      config.get("min_gpu_mem",      0),
                "carbon_aware":     config.get("carbon_aware",     False),
                "carbon_weight":    config.get("carbon_weight",    "balanced"),
                # Training fields needed for est_hours calculation
                "epochs":           config.get("epochs",           50),
                "batch_size":       config.get("batch_size",       64),
                "dataset_type":     config.get("dataset_type",     "synthetic-500k"),
                "synthetic_rows":   config.get("synthetic_rows",   500000),
            })

            result = resume_job(job_id=job_id, resume_step=resume_step,
                                prev_cloud=prev_cloud, decision=decision)
            if result.get("launched"):
                print(f"[poller] ✓ {job_id} relaunched on {result['cloud']} "
                      f"(preferred: {config.get('preferred_clouds','any')})")
            else:
                print(f"[poller] ✗ {job_id} relaunch failed: {result.get('error')}")
                self._resuming.discard(job_id)
        except Exception as e:
            print(f"[poller] _migrate error for {job_id}: {e}")
            self._resuming.discard(job_id)


_poller = PreemptionPoller()
_poller.start()


# ══════════════════════════════════════════════════════════════════
# RISK MODEL — load at startup with verbose print logs
# ══════════════════════════════════════════════════════════════════

from risk.predictor import load_models, score_instance_from_api

def _load_risk_models_bg():
    """
    Load LSTM + XGBoost from local disk in a background thread.
    Verbose prints so you can see every step in the terminal.
    """
    print("\n" + "─"*55)
    print("[risk] ── Model load starting ──")
    print(f"[risk]   models/     → dashboard/model/")
    print(f"[risk]   model_data/ → dashboard/model_data/")
    try:
        load_models()
        # load_models() logs its own lines via logging — these appear
        # after it returns to confirm server is ready to score
        print("[risk] ── Model load complete — /api/risk endpoint is live ──")
        print("─"*55 + "\n")
    except FileNotFoundError as e:
        print(f"\n[risk] ✗ MISSING FILES:\n{e}")
        print("[risk]   /api/risk will return 500 until files are present")
        print("─"*55 + "\n")
    except Exception as e:
        print(f"[risk] ✗ Load failed ({type(e).__name__}): {e}")
        print("[risk]   /api/risk will return 500")
        print("─"*55 + "\n")

threading.Thread(target=_load_risk_models_bg, daemon=True).start()


# ══════════════════════════════════════════════════════════════════
# PRICE ENDPOINTS
# ══════════════════════════════════════════════════════════════════

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/prices/history")
def price_history():
    try:
        client = get_bq_client()
        query  = f"""
            WITH buckets AS (
                SELECT cloud,
                       TIMESTAMP_TRUNC(collected_at, MINUTE) AS minute,
                       AVG(price_usd_per_hr) AS avg_price
                FROM `{PROJECT_ID}.{DATASET}.{TABLE}`
                WHERE collected_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 MINUTE)
                  AND preempted = FALSE AND gpu_class != 'none'
                GROUP BY cloud, minute
            )
            SELECT cloud, minute, avg_price FROM buckets ORDER BY minute ASC
        """
        rows = list(client.query(query))
        from collections import defaultdict
        by_cloud    = defaultdict(dict)
        all_minutes = set()
        for r in rows:
            key = r["minute"].strftime("%H:%M")
            by_cloud[r["cloud"]][key] = round(float(r["avg_price"]), 4)
            all_minutes.add(key)
        timestamps = sorted(all_minutes)[-30:]
        def series(cloud):
            d = by_cloud.get(cloud, {})
            return [d.get(t) for t in timestamps]
        return jsonify({"timestamps": timestamps,
                        "aws": series("aws"), "gcp": series("gcp"),
                        "azure": series("azure")})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/prices/summary")
def price_summary():
    try:
        client = get_bq_client()
        query  = f"""
            WITH latest AS (
                SELECT cloud, price_usd_per_hr,
                       ROW_NUMBER() OVER (PARTITION BY cloud ORDER BY collected_at DESC) AS rn
                FROM `{PROJECT_ID}.{DATASET}.{TABLE}`
                WHERE collected_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 3 HOUR)
                  AND preempted = FALSE AND gpu_class != 'none'
            ),
            stats AS (
                SELECT cloud,
                       MIN(price_usd_per_hr) AS min_price,
                       MAX(price_usd_per_hr) AS max_price,
                       AVG(price_usd_per_hr) AS avg_price
                FROM `{PROJECT_ID}.{DATASET}.{TABLE}`
                WHERE collected_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 3 HOUR)
                  AND preempted = FALSE AND gpu_class != 'none'
                GROUP BY cloud
            )
            SELECT s.cloud, s.min_price, s.max_price, s.avg_price,
                   l.price_usd_per_hr AS current_price
            FROM stats s LEFT JOIN latest l ON s.cloud = l.cloud AND l.rn = 1
        """
        rows   = list(client.query(query))
        result = {}
        for r in rows:
            result[r["cloud"]] = {
                "current_price": round(float(r["current_price"] or 0), 4),
                "min_price":     round(float(r["min_price"]     or 0), 4),
                "max_price":     round(float(r["max_price"]     or 0), 4),
                "avg_price":     round(float(r["avg_price"]     or 0), 4),
            }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/prices/latest")
def latest_prices():
    try:
        client = get_bq_client()
        query  = f"""
            SELECT cloud, region, availability_zone, instance_type,
                   gpu_class, price_usd_per_hr, ondemand_price_usd_hr,
                   ROUND((1 - price_usd_per_hr / NULLIF(ondemand_price_usd_hr,0))*100, 1)
                       AS discount_pct
            FROM `{PROJECT_ID}.{DATASET}.{TABLE}`
            WHERE preempted = FALSE
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY cloud, instance_type, region
                ORDER BY collected_at DESC
            ) = 1
            ORDER BY cloud, price_usd_per_hr ASC LIMIT 20
        """
        rows = list(client.query(query))
        return jsonify([dict(r) for r in rows])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/prices/preemptions")
def preemptions():
    try:
        client = get_bq_client()
        query  = f"""
            SELECT cloud, region, availability_zone, instance_type,
                   gpu_class, price_usd_per_hr, preemption_source, collected_at
            FROM `{PROJECT_ID}.{DATASET}.{TABLE}`
            WHERE preempted = TRUE
            ORDER BY collected_at DESC LIMIT 10
        """
        rows   = list(client.query(query))
        result = []
        for r in rows:
            d = dict(r)
            d["collected_at"] = d["collected_at"].isoformat() if d["collected_at"] else None
            result.append(d)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/stats")
def stats():
    try:
        client = get_bq_client()
        query  = f"""
            SELECT COUNT(*) AS total_rows,
                   COUNTIF(preempted = TRUE) AS preemption_count,
                   MIN(collected_at) AS first_poll,
                   MAX(collected_at) AS last_poll,
                   COUNT(DISTINCT cloud) AS clouds_active
            FROM `{PROJECT_ID}.{DATASET}.{TABLE}`
        """
        row = list(client.query(query))[0]
        return jsonify({
            "total_rows":       int(row["total_rows"]),
            "preemption_count": int(row["preemption_count"]),
            "first_poll":       row["first_poll"].isoformat() if row["first_poll"] else None,
            "last_poll":        row["last_poll"].isoformat()  if row["last_poll"]  else None,
            "clouds_active":    int(row["clouds_active"]),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/health")
def health():
    try:
        client = get_bq_client()
        list(client.query(f"SELECT 1 FROM `{PROJECT_ID}.{DATASET}.{TABLE}` LIMIT 1"))
        return jsonify({"status": "ok", "bigquery": "connected"})
    except Exception as e:
        return jsonify({"status": "degraded", "bigquery": str(e)}), 200


# ══════════════════════════════════════════════════════════════════
# JOB ENDPOINTS
# ══════════════════════════════════════════════════════════════════

@app.route("/api/jobs")
def get_jobs():
    if not GCS_BUCKET:
        return jsonify([])
    try:
        states = gcs_list_job_states()
        order  = {
            "running": 0, "migrating": 1, "queued": 2,
            "preempted": 3, "paused": 4,
            "done": 5, "budget_exceeded": 6,
            "failed": 7, "launch_failed": 8,
        }
        states.sort(key=lambda s: (
            order.get(s.get("status", "done"), 9),
            s.get("updated_at", "")
        ))
        return jsonify(states)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/jobs/<job_id>")
def get_job(job_id):
    if not GCS_BUCKET:
        return jsonify({"error": "GCS_BUCKET not configured"}), 500
    try:
        state        = gcs_read_json(f"checkpoints/{job_id}/job_state.json")
        epoch        = state.get("epoch", 0)
        total_epochs = state.get("total_epochs", 50)
        state["progress_pct"] = min(99, round((epoch / max(total_epochs, 1)) * 100))
        return jsonify(state)
    except Exception as e:
        return jsonify({"error": str(e)}), 404


@app.route("/api/jobs/submit", methods=["POST"])
def submit_job():
    try:
        data   = request.json or {}
        job_id = data.get("job_id") or f"job-{int(time.time())}"
        try:
            from scheduler.selector import pick_best_cloud
            from scheduler.launcher import (_write_job_config,
                                            _write_initial_state,
                                            _upload_trainer_files)
        except ImportError as e:
            return jsonify({"error": f"Scheduler module not found: {e}"}), 500

        decision = pick_best_cloud(data)
        try:
            _upload_trainer_files()
        except Exception as e:
            return jsonify({"error": f"Trainer upload failed: {e}",
                            "launched": False, "job_id": job_id}), 500

        dataset_type    = data.get("dataset", "synthetic-500k")
        s3_dataset_path = data.get("s3_dataset_path", "").strip()
        dataset_name    = data.get("dataset_name", "").strip()

        if dataset_type == "custom" and not s3_dataset_path:
            return jsonify({"error": "Custom dataset selected but no S3 path provided.",
                            "launched": False, "job_id": job_id}), 400

        synthetic_rows = {"synthetic-500k": 500_000, "synthetic-100k": 100_000}
        config = {
            # ─────────────────────────────────────────────
            # Identity
            # ─────────────────────────────────────────────
            "job_id":            job_id,
            "task_name":         data.get("task_name", "Untitled"),
            "train_mode":        data.get("train_mode", "manual"),

            # ─────────────────────────────────────────────
            # Budget / scheduling
            # ─────────────────────────────────────────────
            "max_budget":        float(data.get("max_budget", 2.0)),
            "deadline_hrs":      float(data.get("deadline_hrs", 8.0)),
            "priority":          data.get("priority", "balanced"),

            "spot_only":         bool(data.get("spot_only", True)),
            "ondemand_max_hrs": float(data.get("ondemand_max_hrs", 1.0)),

            # ─────────────────────────────────────────────
            # Dataset
            # ─────────────────────────────────────────────
            "dataset_type":      dataset_type,
            "dataset":           dataset_type,

            "s3_dataset_path":   s3_dataset_path,
            "dataset_name":      dataset_name or dataset_type,

            "dataset_size":      int(data.get("dataset_size", 500000)),
            "synthetic_rows":    synthetic_rows.get(dataset_type, 10000),

            "input_dim":         int(data.get("input_dim", 50)),
            "num_classes":       int(data.get("num_classes", 5)),

            # ─────────────────────────────────────────────
            # Model identity
            # ─────────────────────────────────────────────
            "model_arch":        data.get("model_arch", "mlp"),
            "param_count":       data.get("param_count", "<1B"),
            "training_paradigm": data.get("training_paradigm", "fine-tuning"),
            "precision":         data.get("precision", "fp16"),

            "min_gpu_mem":       int(data.get("min_gpu_mem", 0)),
            "gpu_required":      bool(data.get("gpu_required", False)),

            # ─────────────────────────────────────────────
            # Manual training params
            # ─────────────────────────────────────────────
            "lr":                float(data.get("lr", 0.001)),
            "hidden_dim":        int(data.get("hidden_dim", 256)),
            "dropout":           float(data.get("dropout", 0.3)),
            "batch_size":        int(data.get("batch_size", 64)),
            "epochs":            int(data.get("epochs", 50)),
            "ckpt_every":        int(data.get("ckpt_every", 50)),

            # ─────────────────────────────────────────────
            # Sweep mode params
            # ─────────────────────────────────────────────
            "sweep_lr_min":      float(data.get("sweep_lr_min", 0.0001)),
            "sweep_lr_max":      float(data.get("sweep_lr_max", 0.01)),

            "sweep_hidden":      data.get("sweep_hidden", [256]),

            "sweep_trials":      int(data.get("sweep_trials", 5)),
            "sweep_budget":      float(data.get("sweep_budget", 5.0)),

            # ─────────────────────────────────────────────
            # Compute preferences
            # ─────────────────────────────────────────────
            "preferred_clouds":  data.get(
                "preferred_clouds",
                ["aws", "gcp", "azure"]
            ),

            "preferred_regions": data.get("preferred_regions", ""),

            "fallback":          data.get("fallback", "migrate"),

            # ─────────────────────────────────────────────
            # Carbon-aware scheduling
            # ─────────────────────────────────────────────
            "carbon_aware":      bool(data.get("carbon_aware", False)),
            "carbon_weight":     data.get("carbon_weight", "balanced"),

            # ─────────────────────────────────────────────
            # Scheduler decision
            # ─────────────────────────────────────────────
            "cloud":             decision["cloud"],
            "instance_type":     decision["instance_type"],
            "region":            decision["region"],
            "zone":              decision["zone"],
            "price_usd_hr":      decision["price_usd_hr"],

            # ─────────────────────────────────────────────
            # Storage / checkpoints
            # ─────────────────────────────────────────────
            "gcs_bucket":        GCS_BUCKET,

            "aws_access_key_id":
                os.getenv("AWS_ACCESS_KEY_ID", ""),

            "aws_secret_access_key":
                os.getenv("AWS_SECRET_ACCESS_KEY", ""),

            "aws_default_region":
                os.getenv("AWS_DEFAULT_REGION", "us-east-1"),

            "checkpoint_s3_bucket":
                os.getenv("CHECKPOINT_S3_BUCKET", ""),

            # ─────────────────────────────────────────────
            # Migration / resume
            # ─────────────────────────────────────────────
            "resume_from_step": 0,
            "migration_count":  0,
        }

        if not _write_job_config(job_id, config):
            return jsonify({"error": "Failed to write job config to GCS",
                            "launched": False, "job_id": job_id}), 500

        _write_initial_state(job_id, config, status="queued")

        def _bg_launch():
            try:
                from scheduler.launcher import _create_vm, _write_failed_state
                vm = _create_vm(job_id, decision["instance_type"],
                                resume_step=0, prev_cloud="")
                print(f"[submit] ✓ VM created: {vm['instance_name']}")
            except Exception as e:
                print(f"[submit] ✗ VM creation failed: {e}")
                try:
                    from scheduler.launcher import _write_failed_state
                    _write_failed_state(job_id, str(e))
                except Exception:
                    pass

        threading.Thread(target=_bg_launch, daemon=True).start()

        return jsonify({**decision, "job_id": job_id, "launched": True,
                        "status": "queued",
                        "message": "VM creation started (~90s to boot)."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/jobs/resume", methods=["POST"])
def manual_resume():
    try:
        data   = request.json or {}
        job_id = data.get("job_id")
        if not job_id:
            return jsonify({"error": "job_id required"}), 400
        from scheduler.selector import pick_best_cloud
        from scheduler.launcher import resume_job
        try:
            state = gcs_read_json(f"checkpoints/{job_id}/job_state.json")
        except Exception:
            return jsonify({"error": f"job_state.json not found for {job_id}"}), 404
        resume_step = data.get("resume_step") or state.get("step", 0)
        prev_cloud  = state.get("cloud", "gcp")
        try:
            config = gcs_read_json(f"checkpoints/{job_id}/job_config.json")
        except Exception:
            config = {}
        cost_so_far = float(state.get("cost_usd", 0))
        remaining   = max(0.10, float(config.get("max_budget", 2.0)) - cost_so_far)
        decision    = pick_best_cloud({"job_id": job_id, "max_budget": remaining,
                                       "deadline_hrs": config.get("deadline_hrs", 8.0)})
        result      = resume_job(job_id=job_id, resume_step=resume_step,
                                 prev_cloud=prev_cloud, decision=decision)
        return jsonify({**result, "job_id": job_id, "manual_resume": True,
                        "resumed_from_step": resume_step})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/poller/status")
def poller_status():
    return jsonify({
        "running":            _poller.is_alive(),
        "interval_sec":       POLLER_INTERVAL,
        "max_migrations":     MAX_MIGRATIONS,
        "currently_resuming": list(_poller._resuming),
        "gcs_bucket":         GCS_BUCKET or "(not configured)",
    })


# ══════════════════════════════════════════════════════════════════
# RISK ENDPOINT — verbose print logs so you can trace every step
# ══════════════════════════════════════════════════════════════════

@app.route("/api/risk")
def risk_scores():
    sep = "─" * 55
    ts  = datetime.now().strftime("%H:%M:%S")
    print(f"\n{sep}")
    print(f"[risk] /api/risk called — {ts}")

    try:
        # ── Step 1: Read running jobs from S3 ─────────────────────
        # ── Step 1: Read running jobs from S3 ─────────────────────
        import boto3, json as _json
        s3         = boto3.client("s3")
        bucket     = os.getenv("CHECKPOINT_BUCKET", "ml-scheduler-checkpoints")
        paginator  = s3.get_paginator("list_objects_v2")
        pages      = paginator.paginate(Bucket=bucket, Prefix="checkpoints/")

        # Statuses that mean the job is actively running on an instance
        ACTIVE_STATUSES = {"running", "launched", "migrating"}

        running_jobs = []
        for page in pages:
            for obj in page.get("Contents", []):
                if not obj["Key"].endswith("job_state.json"):
                    continue
                try:
                    body  = s3.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read()
                    state = _json.loads(body)
                    if state.get("status") in ACTIVE_STATUSES:
                        running_jobs.append(state)
                except Exception as e:
                    print(f"[risk]   Skipping {obj['Key']}: {e}")

        print(f"[risk] Active jobs found: {len(running_jobs)}")

        if not running_jobs:
            print(f"[risk] No running jobs — returning []")
            print(sep + "\n")
            return jsonify([])

        # ── Step 2: Score each running job ─────────────────────────
        from risk.predictor import score_instance_from_api
        results = []
        for job in running_jobs:
            job_id   = job.get("job_id", "unknown")
            
            # Cloud/region/instance may be at top level OR inside launch_result
            launch   = job.get("launch_result", {})

            cloud         = job.get("cloud")         or launch.get("cloud",         "aws")
            region        = launch.get("region")     or job.get("region",           "")
            az            = launch.get("az")         or job.get("availability_zone", "") or job.get("zone", "")
            instance_type = launch.get("instance_type") or job.get("instance",      "") or job.get("instance_type", "")

            print(f"[risk] ── Job {job_id}: {cloud}/{instance_type}/{region}/{az or 'no-az'}")

            if not region or not instance_type:
                print(f"[risk]   Skipping — missing region or instance_type")
                continue

            try:
                # In your /api/risk route, you already have bq_client somewhere
                # (or initialize it there). Pass it along:

                risk = score_instance_from_api(
                    cloud         = cloud,
                    region        = region,
                    az            = az,
                    instance_type = instance_type,
                    bq_client     = get_bq_client(),   # ← add this
                )
                level = "HIGH" if risk >= 0.6 else "MED" if risk >= 0.3 else "LOW"
                print(f"[risk]   risk={risk:.4f}  ← {level}")

                results.append({
                    "job_id":        job_id,
                    "cloud":         cloud,
                    "instance_type": instance_type,
                    "region":        region,
                    "az":            az,
                    "risk":          risk,
                    "level":         level,
                    "task_name":     job.get("task_name", job_id),
                })
            except Exception as e:
                print(f"[risk]   ✗ Score failed: {e}")

        results.sort(key=lambda x: x["risk"], reverse=True)
        print(f"[risk] Scored {len(results)} running jobs")
        print(sep + "\n")
        return jsonify(results)

    except Exception as e:
        print(f"[risk] ✗ Endpoint error: {type(e).__name__}: {e}")
        print(sep + "\n")
        return jsonify({"error": str(e)}), 500

def _score_with_logs(cloud, region, az, instance_type, bq_client) -> float:
    """
    Verbose wrapper around score_instance().
    All the detailed logging is now inside predictor.py itself.
    """
    from risk.predictor import score_instance as _score
    return _score(
        cloud         = cloud,
        region        = region,
        az            = az,
        instance_type = instance_type,
        bq_client     = bq_client,
    )


# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"\n{'═'*55}")
    print(f"  Multi-Cloud Scheduler — API server")
    print(f"  Project  : {PROJECT_ID}")
    print(f"  BigQuery : {PROJECT_ID}.{DATASET}.{TABLE}")
    print(f"  GCS      : {GCS_BUCKET or '(not set — job features disabled)'}")
    print(f"  Poller   : every {POLLER_INTERVAL}s · max {MAX_MIGRATIONS} migrations")
    print(f"{'═'*55}\n")
    app.run(host="0.0.0.0", port=5050, debug=True, use_reloader=False)
    # use_reloader=False — prevents two poller threads starting