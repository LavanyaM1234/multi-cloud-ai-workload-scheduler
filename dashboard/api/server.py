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
    level  = logging.DEBUG,
    format = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt= "%H:%M:%S",
)
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
GCS_BUCKET  = os.getenv("CHECKPOINT_GCS_BUCKET",   "")
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
                    print(state)
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
        self._stop_event = threading.Event()
        self._resuming   = set()

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
            if status == "preempted" and job_id not in self._resuming:
                self._resuming.add(job_id)
                print(f"\n[poller] Detected preempted job: {job_id}")
                t = threading.Thread(target=self._migrate, args=(job_id, state), daemon=True)
                t.start()
            if status in ("running", "done", "budget_exceeded", "launch_failed") \
                    and job_id in self._resuming:
                self._resuming.discard(job_id)

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
            decision        = pick_best_cloud({
                "job_id": job_id, "max_budget": remaining,
                "deadline_hrs": config.get("deadline_hrs", 8.0),
            })
            result = resume_job(job_id=job_id, resume_step=resume_step,
                                prev_cloud=prev_cloud, decision=decision)
            if result.get("launched"):
                print(f"[poller] ✓ {job_id} relaunched on {result['cloud']}")
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

from risk.predictor import load_models, score_instance

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
            "dataset_type":       dataset_type,
            "s3_dataset_path":    s3_dataset_path,
            "dataset_name":       dataset_name or dataset_type,
            "synthetic_rows":     synthetic_rows.get(dataset_type, 10_000),
            "aws_access_key_id":     os.getenv("AWS_ACCESS_KEY_ID",     ""),
            "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY", ""),
            "aws_default_region":    os.getenv("AWS_DEFAULT_REGION",    "us-east-1"),
            "checkpoint_s3_bucket":  os.getenv("CHECKPOINT_S3_BUCKET",  ""),
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
    """
    Returns preemption risk score 0–1 for each active instance.
    Prints a detailed log for every instance scored so you can see
    exactly what the model is doing in the server terminal.

    Terminal output per request looks like:
        ─────────────────────────────────────────
        [risk] /api/risk called — 14:22:07
        [risk] BQ query → 5 distinct instances
        [risk] ── Scoring: aws / g4dn.xlarge / us-east-1 / us-east-1a
        [risk]   BQ rows fetched : 10
        [risk]   Feature cols    : ['price_usd_per_hr', 'hour_of_day', ...]
        [risk]   Raw X shape     : (10, 12)
        [risk]   Scaled X shape  : (10, 12)
        [risk]   X_seq shape     : (1, 10, 12)
        [risk]   X_flat shape    : (12,)
        [risk]   LSTM out shape  : (8,)
        [risk]   X_hybrid shape  : (1, 20)
        [risk]   XGB proba       : [0.88  0.12]   ← [preempted, safe]
        [risk]   RISK SCORE      : 0.1200
        [risk] ── Scoring: gcp / g2-standard-4 / us-central1 / us-central1-a
        ...
        [risk] Results (sorted high→low risk):
        [risk]   1. aws/p3.2xlarge    risk=0.8100  ← HIGH
        [risk]   2. aws/g5.xlarge     risk=0.5400  ← MED
        [risk]   3. azure/NC4as_T4_v3 risk=0.4700  ← MED
        [risk]   4. gcp/g2-standard-4 risk=0.2300  ← LOW
        [risk]   5. aws/g4dn.xlarge   risk=0.1200  ← LOW
        ─────────────────────────────────────────
    """
    import numpy as np

    sep = "─" * 55
    ts  = datetime.now().strftime("%H:%M:%S")
    print(f"\n{sep}")
    print(f"[risk] /api/risk called — {ts}")

    try:
        bq = get_bq_client()

        # ── Step 1: fetch distinct active instances ────────────────
        query = f"""
            SELECT DISTINCT cloud, region, availability_zone, instance_type
            FROM `{PROJECT_ID}.{DATASET}.{TABLE}`
            WHERE collected_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 HOUR)
              AND preempted = FALSE
            LIMIT 20
        """
        instances = list(bq.query(query))
        print(f"[risk] BQ query → {len(instances)} distinct instance(s)")

        if not instances:
            print(f"[risk] No instances found in last 1 hour — returning []")
            print(sep + "\n")
            return jsonify([])

        results = []

        for row in instances:
            cloud         = row["cloud"]
            region        = row["region"]
            az            = row["availability_zone"]
            instance_type = row["instance_type"]

            print(f"[risk] ── Scoring: {cloud} / {instance_type} / {region} / {az}")

            try:
                # ── Step 2: call score_instance with extra prints ──
                risk = _score_with_logs(
                    cloud=cloud, region=region, az=az,
                    instance_type=instance_type, bq_client=bq,
                )
                results.append({
                    "cloud":         cloud,
                    "instance_type": instance_type,
                    "region":        region,
                    "az":            az,
                    "risk":          round(risk, 4),
                })

            except Exception as e:
                print(f"[risk]   ✗ FAILED: {type(e).__name__}: {e}")

        # ── Step 3: sort and print summary ────────────────────────
        results.sort(key=lambda x: x["risk"], reverse=True)
        print(f"[risk] Results (sorted high→low risk):")
        for i, r in enumerate(results, 1):
            level = "HIGH" if r["risk"] >= 0.6 else "MED" if r["risk"] >= 0.3 else "LOW"
            print(f"[risk]   {i}. {r['cloud']}/{r['instance_type']:<20} "
                  f"risk={r['risk']:.4f}  ← {level}")

        print(sep + "\n")
        return jsonify(results)

    except Exception as e:
        print(f"[risk] ✗ Endpoint error: {type(e).__name__}: {e}")
        print(sep + "\n")
        return jsonify({"error": str(e)}), 500


"""
Replace the existing _score_with_logs() function in api/server.py
with this version. It uses the same fix as predictor.py:
  - fetch only RAW columns from BQ
  - run _engineer_features() to compute derived features
  - then pass to LSTM + XGBoost

Paste this function BEFORE the /api/risk route in server.py.
"""

def _score_with_logs(cloud, region, az, instance_type, bq_client) -> float:
    """
    Verbose version of score_instance() used by /api/risk endpoint.
    Prints every intermediate tensor shape so you can trace the full pipeline.
    """
    import torch
    import numpy as np

    from risk.predictor import (
        load_models, _fetch_raw_rows, _engineer_features,
        FEATURE_COLS, SEQUENCE_LEN,
        _lstm, _xgb, _scaler, _feat_cols,
    )

    load_models()  # no-op if already loaded

    sep = "─" * 45

    # ── Step 1: fetch RAW rows from BQ ───────────────────────────
    # ONLY selects columns that exist in the table.
    # spot_ratio, price_lag_*, hour etc are NOT in BQ — computed below.
    print(f"[risk]   Fetching RAW rows from BQ...")
    df_raw = _fetch_raw_rows(cloud, region, az, instance_type, bq_client)

    if df_raw.empty or len(df_raw) < SEQUENCE_LEN:
        n = len(df_raw) if not df_raw.empty else 0
        print(f"[risk]   ⚠ Not enough rows ({n}/{SEQUENCE_LEN}) — returning 0.5")
        return 0.5

    print(f"[risk]   Raw rows       : {len(df_raw)}")
    print(f"[risk]   Raw columns    : {list(df_raw.columns)}")

    # ── Step 2: engineer features (mirrors 01_data_prep.py) ───────
    print(f"[risk]   Engineering features...")
    df_eng = _engineer_features(df_raw)

    feat_cols = list(_feat_cols) if _feat_cols is not None else FEATURE_COLS
    missing = [c for c in feat_cols if c not in df_eng.columns]
    if missing:
        print(f"[risk]   ✗ Missing after engineering: {missing} — returning 0.5")
        return 0.5

    print(f"[risk]   Feature cols   : {feat_cols}")
    print(f"[risk]   Engineered rows: {len(df_eng)}")

    # ── Step 3: build feature matrix ─────────────────────────────
    X_raw = df_eng[feat_cols].values.astype(float)
    print(f"[risk]   Raw X shape    : {X_raw.shape}")
    print(f"[risk]   Raw X last row : {X_raw[-1].round(4).tolist()}")

    # ── Step 4: scale ─────────────────────────────────────────────
    X_scaled = _scaler.transform(X_raw)
    print(f"[risk]   Scaled X shape : {X_scaled.shape}")
    print(f"[risk]   Scaled last row: {X_scaled[-1].round(4).tolist()}")

    X_seq  = X_scaled[-SEQUENCE_LEN:]   # (SEQUENCE_LEN, n_features)
    X_flat = X_scaled[-1]               # (n_features,)
    print(f"[risk]   X_seq shape    : {X_seq.shape}")
    print(f"[risk]   X_flat shape   : {X_flat.shape}")

    # ── Step 5: LSTM ──────────────────────────────────────────────
    seq_tensor = torch.tensor(X_seq, dtype=torch.float32).unsqueeze(0)
    print(f"[risk]   seq_tensor     : {tuple(seq_tensor.shape)}")

    with torch.no_grad():
        lstm_feats, lstm_logit = _lstm(seq_tensor)
    lstm_feats = lstm_feats.squeeze(0).numpy()
    print(f"[risk]   LSTM feats     : shape={lstm_feats.shape}  vals={lstm_feats.round(4).tolist()}")
    print(f"[risk]   LSTM logit     : {lstm_logit.item():.4f}")

    # ── Step 6: XGBoost ───────────────────────────────────────────
    X_hybrid = np.hstack([lstm_feats, X_flat]).reshape(1, -1)
    print(f"[risk]   X_hybrid shape : {X_hybrid.shape}  (LSTM_OUT + tabular)")

    proba = _xgb.predict_proba(X_hybrid)[0]
    risk  = float(proba[1])
    print(f"[risk]   XGB proba      : {proba.round(4).tolist()}  [not_preempted, preempted]")
    print(f"[risk]   RISK SCORE     : {risk:.4f}")

    return round(risk, 4)


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