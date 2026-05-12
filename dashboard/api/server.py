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
"""

import os
import sys
import json
import time
import threading
from datetime import datetime, timezone, timedelta
from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv
from flask import render_template

load_dotenv()

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
GCS_BUCKET  = os.getenv("CHECKPOINT_GCS_BUCKET",   "")
GCP_ZONE    = os.getenv("GCP_ZONE",                "us-central1-a")

# How often the poller checks for preempted jobs (seconds)
POLLER_INTERVAL = 30

# How many times we allow a job to be auto-migrated before giving up
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
    """Read a JSON file from GCS bucket."""
    client = get_gcs_client()
    blob   = client.bucket(GCS_BUCKET).blob(path)
    return json.loads(blob.download_as_text())


def gcs_write_json(path, data):
    """Write a dict as JSON to GCS bucket."""
    client = get_gcs_client()
    client.bucket(GCS_BUCKET).blob(path).upload_from_string(
        json.dumps(data, indent=2), content_type="application/json"
    )


def gcs_list_job_states():
    """
    List all job_state.json files in GCS.
    Returns list of parsed state dicts.
    """
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
                    # Compute progress % from epoch/total_epochs
                    epoch        = state.get("epoch", 0)
                    total_epochs = state.get("total_epochs", 50)
                    state["progress_pct"] = min(
                        99, round((epoch / max(total_epochs, 1)) * 100)
                    )
                    states.append(state)
                except Exception:
                    pass
        return states
    except Exception as e:
        print(f"[gcs] list_job_states error: {e}")
        return []


# ══════════════════════════════════════════════════════════════════
# PREEMPTION POLLER  — runs in background thread
# ══════════════════════════════════════════════════════════════════

class PreemptionPoller(threading.Thread):
    """
    Background thread that polls GCS job states every POLLER_INTERVAL
    seconds. When it finds a job with status=preempted it:

      1. Reads migration_count from job_config.json
      2. If under MAX_MIGRATIONS → calls selector + launcher to
         relaunch on cheapest available cloud
      3. If over limit → marks job as failed

    This is what makes migration fully automatic — the dashboard user
    just sees the job go from "preempted" → "migrating" → "running".
    """

    def __init__(self):
        super().__init__(daemon=True)   # daemon=True: dies when Flask dies
        self._stop_event = threading.Event()
        # Track jobs we've already dispatched a resume for, so we don't
        # double-launch if the poller fires before job_state updates
        self._resuming = set()          # set of job_ids currently being resumed

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

            # Only act on preempted jobs we haven't already picked up
            if status == "preempted" and job_id not in self._resuming:
                self._resuming.add(job_id)
                print(f"\n[poller] Detected preempted job: {job_id}")
                # Run migration in its own thread so poller doesn't block
                t = threading.Thread(
                    target=self._migrate,
                    args=(job_id, state),
                    daemon=True
                )
                t.start()

            # Clean up _resuming set once job is back to running/done
            if status in ("running", "done", "budget_exceeded",
                          "launch_failed") and job_id in self._resuming:
                self._resuming.discard(job_id)

    def _migrate(self, job_id, state):
        """Called in its own thread. Relaunches the job on a new cloud."""
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
            from scheduler.selector import pick_best_cloud
            from scheduler.launcher import resume_job

            # Read full config to get migration count + hyperparams
            try:
                config = gcs_read_json(f"checkpoints/{job_id}/job_config.json")
            except Exception:
                config = {}

            migration_count = config.get("migration_count", 0)
            if migration_count >= MAX_MIGRATIONS:
                print(f"[poller] Job {job_id} hit migration limit ({MAX_MIGRATIONS}) — marking failed")
                gcs_write_json(f"checkpoints/{job_id}/job_state.json", {
                    **state,
                    "status":     "failed",
                    "error":      f"Exceeded max migrations ({MAX_MIGRATIONS})",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                })
                self._resuming.discard(job_id)
                return

            resume_step = state.get("step", 0)
            prev_cloud  = state.get("cloud", "gcp")

            print(f"[poller] Migrating {job_id}: step={resume_step} "
                  f"from={prev_cloud} migration={migration_count+1}/{MAX_MIGRATIONS}")

            # Pick best cloud for the resume
            # Pass budget remaining = original_budget - cost_so_far
            cost_so_far     = float(state.get("cost_usd", 0))
            original_budget = float(config.get("max_budget", 2.0))
            remaining       = max(0.10, original_budget - cost_so_far)

            job_spec = {
                "job_id":       job_id,
                "max_budget":   remaining,
                "deadline_hrs": config.get("deadline_hrs", 8.0),
            }
            decision = pick_best_cloud(job_spec)

            result = resume_job(
                job_id      = job_id,
                resume_step = resume_step,
                prev_cloud  = prev_cloud,
                decision    = decision,
            )

            if result.get("launched"):
                print(f"[poller] ✓ Job {job_id} relaunched on "
                      f"{result['cloud']} / {result.get('instance_name','?')}")
            else:
                print(f"[poller] ✗ Job {job_id} relaunch failed: {result.get('error')}")
                self._resuming.discard(job_id)

        except Exception as e:
            print(f"[poller] _migrate error for {job_id}: {e}")
            self._resuming.discard(job_id)


# ── Start poller when module loads ────────────────────────────────
_poller = PreemptionPoller()
_poller.start()


# ══════════════════════════════════════════════════════════════════
# PRICE ENDPOINTS  (unchanged from original)
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
                  AND preempted = FALSE
                  AND gpu_class != 'none'
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
        return jsonify({
            "timestamps": timestamps,
            "aws":        series("aws"),
            "gcp":        series("gcp"),
            "azure":      series("azure"),
        })
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
# JOB ENDPOINTS  (new)
# ══════════════════════════════════════════════════════════════════

@app.route("/api/jobs")
def get_jobs():
    """
    Return all job states from GCS.
    Dashboard polls this every 60s to show live progress.

    Response: list of job state dicts, sorted running-first.
    Each dict has:
        job_id, task_name, status, epoch, total_epochs,
        step, loss, accuracy, elapsed_hrs, cost_usd,
        cloud, instance, progress_pct, updated_at,
        resumed_from, migration_count (if migrated)
    """
    if not GCS_BUCKET:
        return jsonify([])
    try:
        states = gcs_list_job_states()

        # Sort: running/migrating first, then queued, then done/failed
        order = {
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
    """Single job state — used by dashboard to poll a specific job."""
    if not GCS_BUCKET:
        return jsonify({"error": "GCS_BUCKET not configured"}), 500
    try:
        state = gcs_read_json(f"checkpoints/{job_id}/job_state.json")
        epoch        = state.get("epoch", 0)
        total_epochs = state.get("total_epochs", 50)
        state["progress_pct"] = min(99, round((epoch / max(total_epochs, 1)) * 100))
        return jsonify(state)
    except Exception as e:
        return jsonify({"error": str(e)}), 404


@app.route("/api/jobs/submit", methods=["POST"])
def submit_job():
    """
    Non-blocking job submit:
      1. selector.pick_best_cloud() → decision (fast, <1ms)
      2. Write job_config.json + job_state(queued) to GCS (fast, ~1s)
      3. Spin up background thread to create VM (slow, ~90s)
      4. Return immediately with job_id + decision — frontend shows
         job card with status=queued right away

    The dashboard then polls /api/jobs every 60s (and /api/jobs/<id>
    every 10s) — once the VM boots and train.py runs it will update
    job_state.json to status=running with live epoch/loss.

    Why non-blocking: VM creation blocks for ~90-120s via
    operation.result(). Keeping Flask blocked that long causes the
    frontend fetch to hit its timeout and show "signal timed out".
    """
    try:
        data   = request.json or {}
        job_id = data.get("job_id") or f"job-{int(time.time())}"

        # ── Import scheduler modules ───────────────────────────────
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
            from scheduler.selector import pick_best_cloud
            from scheduler.launcher import (launch_job,
                                            _write_job_config,
                                            _write_initial_state,
                                            _upload_trainer_files)
        except ImportError as e:
            return jsonify({"error": f"Scheduler module not found: {e}"}), 500

        # ── Pick cloud (fast) ──────────────────────────────────────
        decision = pick_best_cloud(data)

        # ── Upload trainer files + write GCS state (done inline) ──
        # These are fast (~1-2s) and must complete before VM boots.
        try:
            _upload_trainer_files()
        except Exception as e:
            return jsonify({"error": f"Trainer upload failed: {e}",
                            "launched": False, "job_id": job_id}), 500

        # ── Determine dataset config ───────────────────────────────
        dataset_type    = data.get("dataset", "synthetic-500k")
        s3_dataset_path = data.get("s3_dataset_path", "").strip()
        dataset_name    = data.get("dataset_name", "").strip()

        if dataset_type == "custom" and not s3_dataset_path:
            return jsonify({
                "error": "Custom dataset selected but no S3 path provided.",
                "launched": False, "job_id": job_id
            }), 400

        synthetic_rows = {"synthetic-500k": 500_000, "synthetic-100k": 100_000}

        config = {
            "job_id":           job_id,
            "task_name":        data.get("task_name",    "Untitled"),
            "lr":               float(data.get("lr",           0.001)),
            "hidden_dim":       int(  data.get("hidden_dim",   256)),
            "dropout":          float(data.get("dropout",      0.3)),
            "batch_size":       int(  data.get("batch_size",   64)),
            "epochs":           int(  data.get("epochs",       50)),
            "ckpt_every":       int(  data.get("ckpt_every",   50)),
            "input_dim":        int(  data.get("input_dim",    50)),
            "num_classes":      int(  data.get("num_classes",  5)),
            "max_budget":       float(data.get("max_budget",   2.0)),
            "deadline_hrs":     float(data.get("deadline_hrs", 8.0)),
            "cloud":            decision["cloud"],
            "instance_type":    decision["instance_type"],
            "region":           decision["region"],
            "zone":             decision["zone"],
            "price_usd_hr":     decision["price_usd_hr"],
            "gcs_bucket":       GCS_BUCKET,
            # ── Dataset ───────────────────────────────────────────
            "dataset_type":       dataset_type,
            "s3_dataset_path":    s3_dataset_path,
            "dataset_name":       dataset_name or dataset_type,
            "synthetic_rows":     synthetic_rows.get(dataset_type, 10_000),
            # ── AWS creds injected into VM metadata by launcher ───
            "aws_access_key_id":     os.getenv("AWS_ACCESS_KEY_ID",     ""),
            "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY", ""),
            "aws_default_region":    os.getenv("AWS_DEFAULT_REGION",    "us-east-1"),
            "checkpoint_s3_bucket":  os.getenv("CHECKPOINT_S3_BUCKET",  ""),
            # ── Migration tracking ────────────────────────────────
            "resume_from_step": 0,
            "migration_count":  0,
        }

        if not _write_job_config(job_id, config):
            return jsonify({"error": "Failed to write job config to GCS",
                            "launched": False, "job_id": job_id}), 500

        _write_initial_state(job_id, config, status="queued")

        # ── Launch VM in background thread (slow ~90s) ─────────────
        # We don't wait for it. The poller + /api/jobs polling handles
        # surfacing the result to the dashboard once VM is up.
        def _bg_launch():
            try:
                from scheduler.launcher import _create_vm, _write_failed_state
                vm = _create_vm(job_id, decision["instance_type"],
                                resume_step=0, prev_cloud="")
                print(f"[submit] ✓ VM created in background: "
                      f"{vm['instance_name']}")
            except Exception as e:
                print(f"[submit] ✗ Background VM creation failed: {e}")
                try:
                    from scheduler.launcher import _write_failed_state
                    _write_failed_state(job_id, str(e))
                except Exception:
                    pass

        t = threading.Thread(target=_bg_launch, daemon=True)
        t.start()

        # ── Return immediately ─────────────────────────────────────
        return jsonify({
            **decision,
            "job_id":        job_id,
            "launched":      True,
            "status":        "queued",
            "message":       "VM creation started in background (~90s to boot). "
                             "Job will appear as running once VM is up.",
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/jobs/resume", methods=["POST"])
def manual_resume():
    """
    Manually trigger a resume for a preempted job.
    Useful for debugging or if the auto-poller missed it.

    Request body:
        job_id       — required
        resume_step  — optional, defaults to step from job_state.json
    """
    try:
        data    = request.json or {}
        job_id  = data.get("job_id")
        if not job_id:
            return jsonify({"error": "job_id required"}), 400

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from scheduler.selector import pick_best_cloud
        from scheduler.launcher import resume_job

        # Read current state
        try:
            state = gcs_read_json(f"checkpoints/{job_id}/job_state.json")
        except Exception:
            return jsonify({"error": f"job_state.json not found for {job_id}"}), 404

        resume_step = data.get("resume_step") or state.get("step", 0)
        prev_cloud  = state.get("cloud", "gcp")

        # Read config for budget info
        try:
            config = gcs_read_json(f"checkpoints/{job_id}/job_config.json")
        except Exception:
            config = {}

        cost_so_far = float(state.get("cost_usd", 0))
        remaining   = max(0.10, float(config.get("max_budget", 2.0)) - cost_so_far)

        decision = pick_best_cloud({
            "job_id":       job_id,
            "max_budget":   remaining,
            "deadline_hrs": config.get("deadline_hrs", 8.0),
        })

        result = resume_job(
            job_id      = job_id,
            resume_step = resume_step,
            prev_cloud  = prev_cloud,
            decision    = decision,
        )

        return jsonify({**result, "job_id": job_id,
                        "manual_resume": True,
                        "resumed_from_step": resume_step})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════════
# POLLER STATUS  — useful for debugging
# ══════════════════════════════════════════════════════════════════

@app.route("/api/poller/status")
def poller_status():
    """Shows what jobs the poller is currently tracking."""
    return jsonify({
        "running":        _poller.is_alive(),
        "interval_sec":   POLLER_INTERVAL,
        "max_migrations": MAX_MIGRATIONS,
        "currently_resuming": list(_poller._resuming),
        "gcs_bucket":     GCS_BUCKET or "(not configured)",
    })


# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"Starting API server")
    print(f"  Project  : {PROJECT_ID}")
    print(f"  BigQuery : {PROJECT_ID}.{DATASET}.{TABLE}")
    print(f"  GCS      : {GCS_BUCKET or '(not set — job features disabled)'}")
    print(f"  Poller   : every {POLLER_INTERVAL}s, max {MAX_MIGRATIONS} migrations")
    app.run(host="0.0.0.0", port=5050, debug=True, use_reloader=False)
    # use_reloader=False — reloader would start two poller threads