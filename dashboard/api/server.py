"""
api/server.py
──────────────
Flask backend — serves real BigQuery data + manages multi-cloud training jobs.

Storage:
    ALL job state (job_state.json, job_config.json, job_command.json,
    checkpoint .pt files) lives in S3 only.
    BigQuery is used only for price/preemption data (read-only).

Run:
    cd dashboard
    python api/server.py

Endpoints:
    GET  /api/prices/history         → 30-point time series per cloud
    GET  /api/prices/summary         → min/max/avg/current per cloud
    GET  /api/prices/latest          → latest price per instance type
    GET  /api/prices/preemptions     → recent preempted=TRUE rows
    GET  /api/stats                  → total rows, preemption count
    GET  /api/health                 → BigQuery + S3 reachable?
    GET  /api/jobs                   → all jobs from S3 job_state.json files
    GET  /api/jobs/<job_id>          → single job state from S3
    POST /api/jobs/submit            → submit new job → launcher.submit_job()
    POST /api/jobs/resume            → manually resume a preempted job
    POST /api/jobs/<job_id>/command  → send migrate/stop/reduce_lr to VM
    GET  /api/poller/status          → preemption poller debug info
    GET  /api/risk                   → LSTM + XGBoost preemption risk scores
"""

import os
import sys
import json
import time
import logging
import threading
from datetime import datetime, timezone

import boto3
from botocore.exceptions import ClientError
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
load_dotenv()

# ── Logging ────────────────────────────────────────────────────────
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
PROJECT_ID  = os.getenv("GCP_PROJECT_ID",        "tensile-method-459009-k2")
DATASET     = os.getenv("BIGQUERY_DATASET",       "spot_prices")
TABLE       = os.getenv("BIGQUERY_TABLE",         "price_history")
GCS_BUCKET  = os.getenv("CHECKPOINT_GCS_BUCKET",  "")
S3_BUCKET   = os.getenv("CHECKPOINT_S3_BUCKET",   "")
GCP_ZONE    = os.getenv("GCP_ZONE",               "us-central1-a")

POLLER_INTERVAL = 30
MAX_MIGRATIONS  = 5


# ══════════════════════════════════════════════════════════════════
# S3 HELPERS  — all job state lives here
# ══════════════════════════════════════════════════════════════════

def _s3():
    return boto3.client(
        "s3",
        aws_access_key_id     = os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name           = os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    )


def s3_read_json(key: str) -> dict | None:
    """Read a JSON file from S3. Returns None if key missing or S3 not set."""
    if not S3_BUCKET:
        return None
    try:
        obj = _s3().get_object(Bucket=S3_BUCKET, Key=key)
        return json.loads(obj["Body"].read().decode())
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None
        log.debug(f"[s3] read_json({key}): {e}")
        return None
    except Exception as e:
        log.debug(f"[s3] read_json({key}): {e}")
        return None


def s3_write_json(key: str, data: dict):
    """Write a dict as JSON to S3."""
    if not S3_BUCKET:
        log.warning(f"[s3] S3_BUCKET not set — skipping write to {key}")
        return
    try:
        _s3().put_object(
            Bucket      = S3_BUCKET,
            Key         = key,
            Body        = json.dumps(data, indent=2).encode(),
            ContentType = "application/json",
        )
    except Exception as e:
        log.error(f"[s3] write_json({key}): {e}")


def s3_list_job_states() -> list:
    """
    List all checkpoints/*/job_state.json in S3.
    Returns list of parsed state dicts with progress_pct added.
    """
    if not S3_BUCKET:
        return []
    try:
        s3        = _s3()
        paginator = s3.get_paginator("list_objects_v2")
        states    = []
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix="checkpoints/"):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not key.endswith("job_state.json"):
                    continue
                try:
                    body  = s3.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
                    state = json.loads(body.decode())
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
        log.error(f"[s3] list_job_states: {e}")
        return []


# ══════════════════════════════════════════════════════════════════
# BIGQUERY HELPER  — read-only, prices only
# ══════════════════════════════════════════════════════════════════

def get_bq_client():
    from google.cloud import bigquery
    return bigquery.Client(project=PROJECT_ID)


# ══════════════════════════════════════════════════════════════════
# PREEMPTION POLLER
# ══════════════════════════════════════════════════════════════════

class PreemptionPoller(threading.Thread):
    """
    Background daemon thread.
    Polls S3 every POLLER_INTERVAL seconds for preempted jobs,
    then auto-relaunches them via launcher.resume_job().
    """

    def __init__(self):
        super().__init__(daemon=True)
        self._stop_event      = threading.Event()
        self._resuming        = set()
        # Track when each job switched to on-demand {job_id: datetime}
        self._ondemand_since  = {}

    def stop(self):
        self._stop_event.set()

    def run(self):
        log.info(f"[poller] Started — interval={POLLER_INTERVAL}s  "
                 f"S3={S3_BUCKET or '(not configured)'}")
        while not self._stop_event.is_set():
            try:
                self._poll()
            except Exception as e:
                log.error(f"[poller] Error in poll cycle: {e}")
            self._stop_event.wait(POLLER_INTERVAL)

    def _poll(self):
        if not S3_BUCKET:
            return
        for state in s3_list_job_states():
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
        write a 'migrate' command to S3 so train.py picks it up and
        triggers a checkpoint + exit, after which the poller relaunches
        on a spot instance.
        """
        try:
            config           = s3_read_json(f"checkpoints/{job_id}/job_config.json") or {}
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
                s3_write_json(
                    f"checkpoints/{job_id}/job_command.json",
                    {"command": "migrate", "reason": "ondemand_max_hrs_exceeded"}
                )
                del self._ondemand_since[job_id]

        except Exception as e:
            print(f"[poller] _check_ondemand_timeout error for {job_id}: {e}")

    def _migrate(self, job_id, state):
        try:
            from scheduler.launcher import resume_job
            from scheduler.selector import pick_best_cloud
            try:
                config = s3_read_json(f"checkpoints/{job_id}/job_config.json") or {}
            except Exception:
                config = {}

            migration_count = config.get("migration_count", 0)

            if migration_count >= MAX_MIGRATIONS:
                log.error(f"[poller] {job_id} hit migration limit ({MAX_MIGRATIONS}) — marking failed")
                existing = s3_read_json(f"checkpoints/{job_id}/job_state.json") or {}
                existing.update({
                    "status":     "failed",
                    "error":      f"Exceeded max migrations ({MAX_MIGRATIONS})",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                })
                s3_write_json(f"checkpoints/{job_id}/job_state.json", existing)
                self._resuming.discard(job_id)
                return

            cost_so_far     = float(state.get("cost_usd", 0))
            original_budget = float(config.get("max_budget", 2.0))
            remaining       = max(0.10, original_budget - cost_so_far)

            resume_step = state.get("step", 0)
            prev_cloud  = state.get("cloud", "gcp")

            decision = pick_best_cloud({
                "job_id":            job_id,
                "max_budget":        remaining,
                "deadline_hrs":      config.get("deadline_hrs",      8.0),
                "priority":          config.get("priority",          "balanced"),
                "spot_only":         config.get("spot_only",         True),
                "ondemand_max_hrs":  config.get("ondemand_max_hrs",  1.0),
                "preferred_clouds":  config.get("preferred_clouds",  ["aws", "gcp", "azure"]),
                "preferred_regions": config.get("preferred_regions", ""),
                "gpu_required":      config.get("gpu_required",      False),
                "min_gpu_mem":       config.get("min_gpu_mem",       0),
                "carbon_aware":      config.get("carbon_aware",      False),
                "carbon_weight":     config.get("carbon_weight",     "balanced"),
                "epochs":            config.get("epochs",            50),
                "batch_size":        config.get("batch_size",        64),
                "dataset_type":      config.get("dataset_type",      "synthetic-500k"),
                "synthetic_rows":    config.get("synthetic_rows",    500000),
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
            log.error(f"[poller] _migrate error for {job_id}: {e}")
            self._resuming.discard(job_id)


# Start poller immediately when module loads
_poller = PreemptionPoller()
_poller.start()


# ══════════════════════════════════════════════════════════════════
# RISK MODEL — load from local disk at startup
# ══════════════════════════════════════════════════════════════════

_risk_models_ready = False

def _load_risk_models_bg():
    global _risk_models_ready
    print("\n" + "─" * 55)
    print("[risk] ── Model load starting ──")
    try:
        from risk.predictor import load_models
        load_models()
        _risk_models_ready = True
        print("[risk] ── Model load complete — /api/risk is live ──")
    except ImportError as e:
        print(f"[risk] ✗ risk.predictor import failed: {e}")
        print("[risk]   /api/risk will return 503 until fixed")
    except FileNotFoundError as e:
        print(f"[risk] ✗ Missing model files:\n  {e}")
        print("[risk]   /api/risk will return 503 until files are present")
    except Exception as e:
        print(f"[risk] ✗ Load failed ({type(e).__name__}): {e}")
    print("─" * 55 + "\n")


threading.Thread(target=_load_risk_models_bg, daemon=True).start()


# ══════════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════════

@app.route("/")
def home():
    return render_template("index.html")


# ══════════════════════════════════════════════════════════════════
# PRICE ENDPOINTS  (BigQuery read-only)
# ══════════════════════════════════════════════════════════════════

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
    bq_status = "ok"
    try:
        client = get_bq_client()
        list(client.query(f"SELECT 1 FROM `{PROJECT_ID}.{DATASET}.{TABLE}` LIMIT 1"))
    except Exception as e:
        bq_status = str(e)

    s3_status = "ok"
    if S3_BUCKET:
        try:
            _s3().head_bucket(Bucket=S3_BUCKET)
        except Exception as e:
            s3_status = str(e)
    else:
        s3_status = "(not configured)"

    return jsonify({
        "status":    "ok" if bq_status == "ok" else "degraded",
        "bigquery":  bq_status,
        "s3_bucket": S3_BUCKET or "(not configured)",
        "s3":        s3_status,
    })


# ══════════════════════════════════════════════════════════════════
# JOB ENDPOINTS  — all state from S3
# ══════════════════════════════════════════════════════════════════

@app.route("/api/jobs")
def get_jobs():
    """Return all job states from S3, sorted running-first."""
    try:
        states = s3_list_job_states()
        order  = {
            "running": 0, "migrating": 1, "queued": 2, "launched": 3,
            "preempted": 4, "paused": 5,
            "done": 6, "budget_exceeded": 7,
            "failed": 8, "launch_failed": 9,
        }
        states.sort(key=lambda s: (
            order.get(s.get("status", "done"), 10),
            s.get("updated_at", "")
        ))
        return jsonify(states)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/jobs/<job_id>")
def get_job(job_id):
    """Single job state — polled every 10s by the dashboard."""
    try:
        state = s3_read_json(f"checkpoints/{job_id}/job_state.json")
        if not state:
            return jsonify({"error": "job_state.json not found in S3"}), 404
        epoch        = state.get("epoch", 0)
        total_epochs = state.get("total_epochs", 50)
        state["progress_pct"] = min(99, round((epoch / max(total_epochs, 1)) * 100))
        return jsonify(state)
    except Exception as e:
        return jsonify({"error": str(e)}), 404


@app.route("/api/jobs/submit", methods=["POST"])
def submit_job_endpoint():
    """
    Submit a new training job.

    Flow:
        1. Call launcher.submit_job(job_dict) which:
              a. Runs Pareto selector (live spot prices across 3 clouds)
              b. Writes job_config.json to S3  ← train.py reads at boot
              c. Writes queued job_state.json to S3  ← dashboard shows immediately
              d. Builds startup.sh (stamps JOB_ID/RESUME_STEP/PREV_CLOUD)
              e. Launches VM with AWS→Azure→GCP fallback chain
              f. Updates job_state.json to launched
        2. Return immediately with decision + job_id.
           VM creation runs in a background thread (~90s).
           Dashboard polls /api/jobs/<job_id> every 10s for status updates.
    """
    try:
        data   = request.json or {}
        job_id = data.get("job_id") or f"job-{int(time.time())}"
        data["job_id"] = job_id

        if "dataset" in data and "dataset_type" not in data:
            data["dataset_type"] = data.pop("dataset")

        if data.get("dataset_type") == "custom" and not data.get("s3_dataset_path", "").strip():
            return jsonify({
                "error":    "Custom dataset selected but s3_dataset_path is empty.",
                "launched": False,
                "job_id":   job_id,
            }), 400

        def _bg_launch():
            try:
                from scheduler.launcher import submit_job
                result = submit_job(data)
                actual_cloud    = result["launch_result"].get("cloud",
                                  result["decision"].get("cloud", "?"))
                actual_instance = result["launch_result"].get("instance_type",
                                  result["decision"].get("instance_type", "?"))
                log.info(
                    f"[submit] ✓ {job_id} launched on "
                    f"[{actual_cloud}] {actual_instance}"
                )
            except Exception as e:
                log.error(f"[submit] ✗ VM creation failed for {job_id}: {e}")
                s3_write_json(f"checkpoints/{job_id}/job_state.json", {
                    "job_id":     job_id,
                    "status":     "launch_failed",
                    "error":      str(e),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                })

        s3_write_json(f"checkpoints/{job_id}/job_state.json", {
            "job_id":       job_id,
            "task_name":    data.get("task_name", "Untitled"),
            "status":       "queued",
            "epoch":        0,
            "total_epochs": int(data.get("epochs", 50)),
            "step":         0,
            "loss":         None,
            "updated_at":   datetime.now(timezone.utc).isoformat(),
        })

        threading.Thread(target=_bg_launch, daemon=True).start()

        return jsonify({
            "job_id":   job_id,
            "launched": True,
            "status":   "queued",
            "message":  "VM creation started (~90s to boot). "
                        "Poll /api/jobs/{job_id} for status.",
        })

    except Exception as e:
        log.error(f"[submit] Unexpected error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/jobs/resume", methods=["POST"])
def manual_resume():
    """
    Manually resume a preempted job.
    Body: {job_id, max_budget (optional override)}
    """
    try:
        data   = request.json or {}
        job_id = data.get("job_id")
        if not job_id:
            return jsonify({"error": "job_id required"}), 400

        state = s3_read_json(f"checkpoints/{job_id}/job_state.json")
        if not state:
            return jsonify({"error": f"No job_state.json in S3 for {job_id}"}), 404

        config      = s3_read_json(f"checkpoints/{job_id}/job_config.json") or {}
        cost_so_far = float(state.get("cost_usd", 0))
        remaining   = max(0.10, float(config.get("max_budget", 2.0)) - cost_so_far)

        job = {
            "job_id":     job_id,
            "max_budget": data.get("max_budget", remaining),
        }

        def _bg_resume():
            try:
                from scheduler.launcher import resume_job
                result = resume_job(job)
                log.info(
                    f"[resume] ✓ {job_id} relaunched on "
                    f"[{result['launch_result'].get('cloud', '?')}] "
                    f"{result['decision'].get('instance_type', '?')}"
                )
            except Exception as e:
                log.error(f"[resume] ✗ Resume failed for {job_id}: {e}")

        threading.Thread(target=_bg_resume, daemon=True).start()

        return jsonify({
            "job_id":        job_id,
            "manual_resume": True,
            "status":        "queued",
            "message":       "Resume started. Poll /api/jobs/{job_id} for status.",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/jobs/<job_id>/command", methods=["POST"])
def send_command(job_id):
    """
    Write job_command.json to S3.
    train.py polls this at the top of every epoch and consumes it.
    Body: {"command": "migrate" | "stop" | "reduce_lr"}
    """
    try:
        data = request.json or {}
        cmd  = data.get("command")
        if cmd not in ("migrate", "stop", "reduce_lr"):
            return jsonify({"error": "command must be: migrate | stop | reduce_lr"}), 400

        s3_write_json(f"checkpoints/{job_id}/job_command.json", {
            "command":   cmd,
            "issued_at": datetime.now(timezone.utc).isoformat(),
            "issued_by": "dashboard",
        })
        log.info(f"[command] {cmd} → s3://{S3_BUCKET}/checkpoints/{job_id}/job_command.json")
        return jsonify({"ok": True, "job_id": job_id, "command": cmd})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/poller/status")
def poller_status():
    return jsonify({
        "running":            _poller.is_alive(),
        "interval_sec":       POLLER_INTERVAL,
        "max_migrations":     MAX_MIGRATIONS,
        "currently_resuming": list(_poller._resuming),
        "s3_bucket":          S3_BUCKET or "(not configured)",
    })


# ══════════════════════════════════════════════════════════════════
# RISK ENDPOINT
# ══════════════════════════════════════════════════════════════════

@app.route("/api/risk")
def risk_scores():
    if not _risk_models_ready:
        return jsonify({
            "error": "Risk models not yet loaded — check server logs for details. "
                     "All other endpoints are working normally.",
            "retry_after_seconds": 10,
        }), 503

    sep = "─" * 55
    ts  = datetime.now().strftime("%H:%M:%S")
    print(f"\n{sep}")
    print(f"[risk] /api/risk called — {ts}")

    try:
        # ── Step 1: Read running jobs from S3 ─────────────────────
        import boto3, json as _json
        s3         = boto3.client("s3")
        bucket     = os.getenv("CHECKPOINT_S3_BUCKET", S3_BUCKET)
        paginator  = s3.get_paginator("list_objects_v2")
        pages      = paginator.paginate(Bucket=bucket, Prefix="checkpoints/")

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
            job_id = job.get("job_id", "unknown")

            launch        = job.get("launch_result", {})
            cloud         = job.get("cloud")         or launch.get("cloud",          "aws")
            region        = launch.get("region")     or job.get("region",            "")
            az            = launch.get("az")         or job.get("availability_zone", "") or job.get("zone", "")
            instance_type = launch.get("instance_type") or job.get("instance",       "") or job.get("instance_type", "")

            print(f"[risk] ── Job {job_id}: {cloud}/{instance_type}/{region}/{az or 'no-az'}")

            if not region or not instance_type:
                print(f"[risk]   Skipping — missing region or instance_type")
                continue

            try:
                risk = score_instance_from_api(
                    cloud         = cloud,
                    region        = region,
                    az            = az,
                    instance_type = instance_type,
                    bq_client     = get_bq_client(),
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


# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"\n{'═'*55}")
    print(f"  Multi-Cloud Scheduler — API server")
    print(f"  Project  : {PROJECT_ID}")
    print(f"  BigQuery : {PROJECT_ID}.{DATASET}.{TABLE}")
    print(f"  S3 bucket: {S3_BUCKET or '(not set — job features disabled)'}")
    print(f"  GCS      : {GCS_BUCKET or '(not set)'}")
    print(f"  Poller   : every {POLLER_INTERVAL}s · max {MAX_MIGRATIONS} migrations")
    print(f"{'═'*55}\n")
    app.run(host="0.0.0.0", port=5050, debug=True, use_reloader=False)
    # use_reloader=False — prevents two poller threads starting in debug mode