"""
api/server.py
──────────────
Flask backend — serves real BigQuery data to the dashboard.

Run:
    cd dashboard
    python api/server.py

Endpoints:
    GET /api/prices/history     → 30-point time series per cloud
    GET /api/prices/summary     → min/max/avg/current per cloud
    GET /api/prices/latest      → latest price per instance type
    GET /api/prices/preemptions → recent preempted=TRUE rows
    GET /api/jobs               → active jobs from job_state.json
    POST /api/jobs/submit       → submit new job to scheduler

The dashboard calls these every 60s.
If this server is unreachable, the dashboard falls back to mock data.
"""

import os
import json
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
CORS(app)  # allow dashboard HTML to call this from any origin

PROJECT_ID = os.getenv("GCP_PROJECT_ID", "tensile-method-459009-k2")
DATASET    = os.getenv("BIGQUERY_DATASET", "spot_prices")
TABLE      = os.getenv("BIGQUERY_TABLE",   "price_history")
GCS_BUCKET = os.getenv("CHECKPOINT_GCS_BUCKET", "")

@app.route("/")
def home():
    return render_template("index.html")

def get_bq_client():
    from google.cloud import bigquery
    return bigquery.Client(project=PROJECT_ID)


# ── /api/prices/history ────────────────────────────────────────────────────────
@app.route("/api/prices/history")
def price_history():
    """
    Returns 30 data points per cloud for the price chart.
    Each point = avg price across all instances in that poll window.
    Response shape:
        {
          timestamps: ["14:00", "14:01", ...],  // 30 items
          aws:   [0.158, 0.162, ...],
          gcp:   [0.282, 0.282, ...],
          azure: [0.210, 0.208, ...],
        }
    """
    try:
        client = get_bq_client()
        query = f"""
            WITH buckets AS (
                SELECT
                    cloud,
                    TIMESTAMP_TRUNC(collected_at, MINUTE) AS minute,
                    AVG(price_usd_per_hr) AS avg_price
                FROM `{PROJECT_ID}.{DATASET}.{TABLE}`
                WHERE collected_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 MINUTE)
                  AND preempted = FALSE
                  AND gpu_class != 'none'
                GROUP BY cloud, minute
            )
            SELECT cloud, minute, avg_price
            FROM buckets
            ORDER BY minute ASC
        """
        rows = list(client.query(query))

        # Pivot into per-cloud lists, fill missing minutes with None
        from collections import defaultdict
        by_cloud = defaultdict(dict)
        all_minutes = set()
        for r in rows:
            key = r["minute"].strftime("%H:%M")
            by_cloud[r["cloud"]][key] = round(float(r["avg_price"]), 4)
            all_minutes.add(key)

        timestamps = sorted(all_minutes)[-30:]  # last 30 minutes

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


# ── /api/prices/summary ────────────────────────────────────────────────────────
@app.route("/api/prices/summary")
def price_summary():
    """
    Per-cloud summary stats over the last 3 hours.
    Response shape:
        {
          aws:   {current_price, min_price, max_price, avg_price},
          gcp:   {...},
          azure: {...},
        }
    """
    try:
        client = get_bq_client()
        query = f"""
            WITH latest AS (
                SELECT cloud, price_usd_per_hr,
                       ROW_NUMBER() OVER (PARTITION BY cloud ORDER BY collected_at DESC) AS rn
                FROM `{PROJECT_ID}.{DATASET}.{TABLE}`
                WHERE collected_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 3 HOUR)
                  AND preempted = FALSE
                  AND gpu_class != 'none'
            ),
            stats AS (
                SELECT cloud,
                       MIN(price_usd_per_hr) AS min_price,
                       MAX(price_usd_per_hr) AS max_price,
                       AVG(price_usd_per_hr) AS avg_price
                FROM `{PROJECT_ID}.{DATASET}.{TABLE}`
                WHERE collected_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 3 HOUR)
                  AND preempted = FALSE
                  AND gpu_class != 'none'
                GROUP BY cloud
            )
            SELECT s.cloud, s.min_price, s.max_price, s.avg_price,
                   l.price_usd_per_hr AS current_price
            FROM stats s
            LEFT JOIN latest l ON s.cloud = l.cloud AND l.rn = 1
        """
        rows = list(client.query(query))
        result = {}
        for r in rows:
            result[r["cloud"]] = {
                "current_price": round(float(r["current_price"] or 0), 4),
                "min_price":     round(float(r["min_price"] or 0), 4),
                "max_price":     round(float(r["max_price"] or 0), 4),
                "avg_price":     round(float(r["avg_price"] or 0), 4),
            }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── /api/prices/latest ─────────────────────────────────────────────────────────
@app.route("/api/prices/latest")
def latest_prices():
    """
    Latest price row per instance type across all clouds.
    Used to populate the spot prices table.
    Response: list of row dicts.
    """
    try:
        client = get_bq_client()
        query = f"""
            SELECT cloud, region, availability_zone, instance_type,
                   gpu_class, price_usd_per_hr, ondemand_price_usd_hr,
                   ROUND(
                       (1 - price_usd_per_hr / NULLIF(ondemand_price_usd_hr, 0)) * 100, 1
                   ) AS discount_pct
            FROM `{PROJECT_ID}.{DATASET}.{TABLE}`
            WHERE preempted = FALSE
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY cloud, instance_type, region
                ORDER BY collected_at DESC
            ) = 1
            ORDER BY cloud, price_usd_per_hr ASC
            LIMIT 20
        """
        rows = list(client.query(query))
        return jsonify([dict(r) for r in rows])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── /api/prices/preemptions ────────────────────────────────────────────────────
@app.route("/api/prices/preemptions")
def preemptions():
    """
    Recent preemption events — used to populate the event log.
    Response: list of row dicts.
    """
    try:
        client = get_bq_client()
        query = f"""
            SELECT cloud, region, availability_zone, instance_type,
                   gpu_class, price_usd_per_hr, preemption_source,
                   collected_at
            FROM `{PROJECT_ID}.{DATASET}.{TABLE}`
            WHERE preempted = TRUE
            ORDER BY collected_at DESC
            LIMIT 10
        """
        rows = list(client.query(query))
        result = []
        for r in rows:
            d = dict(r)
            d["collected_at"] = d["collected_at"].isoformat() if d["collected_at"] else None
            result.append(d)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── /api/jobs ──────────────────────────────────────────────────────────────────
@app.route("/api/jobs")
def get_jobs():
    """
    Read job_state.json files from GCS to get active jobs.
    Falls back to empty list if GCS not configured.
    """
    if not GCS_BUCKET:
        return jsonify([])
    try:
        from google.cloud import storage
        client  = storage.Client()
        bucket  = client.bucket(GCS_BUCKET)
        blobs   = bucket.list_blobs(prefix="checkpoints/")
        jobs    = []
        for blob in blobs:
            if blob.name.endswith("job_state.json"):
                state = json.loads(blob.download_as_text())
                jobs.append(state)
        return jsonify(jobs)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── /api/jobs/submit ───────────────────────────────────────────────────────────
@app.route("/api/jobs/submit", methods=["POST"])
def submit_job():
    """
    Accept a job from the dashboard and call the scheduler selector.
    Body: {job_id, model_arch, dataset_size, max_budget, deadline_hrs, min_gpu_mem}
    Response: best cloud decision from Pareto optimizer.
    """
    try:
        data    = request.json
        # Import selector if available
        try:
            import sys
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
            from scheduler.selector import pick_best_cloud
            decision = pick_best_cloud(data)
        except ImportError:
            # Scheduler not built yet — return mock decision
            decision = {
                "cloud":         "aws",
                "instance_type": "g4dn.xlarge",
                "region":        "us-east-1",
                "price_usd_hr":  0.158,
                "est_cost":      0.48,
                "est_hours":     3.0,
                "reason":        "Mock decision — scheduler not built yet",
            }
        return jsonify(decision)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── /api/stats ─────────────────────────────────────────────────────────────────
@app.route("/api/stats")
def stats():
    """
    High-level stats for the stat cards at the top of the dashboard.
    Returns total rows, preemption count, and data collection uptime.
    """
    try:
        client = get_bq_client()
        query = f"""
            SELECT
                COUNT(*) AS total_rows,
                COUNTIF(preempted = TRUE) AS preemption_count,
                MIN(collected_at) AS first_poll,
                MAX(collected_at) AS last_poll,
                COUNT(DISTINCT cloud) AS clouds_active
            FROM `{PROJECT_ID}.{DATASET}.{TABLE}`
        """
        row = list(client.query(query))[0]
        return jsonify({
            "total_rows":        int(row["total_rows"]),
            "preemption_count":  int(row["preemption_count"]),
            "first_poll":        row["first_poll"].isoformat() if row["first_poll"] else None,
            "last_poll":         row["last_poll"].isoformat()  if row["last_poll"]  else None,
            "clouds_active":     int(row["clouds_active"]),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Health check ───────────────────────────────────────────────────────────────
@app.route("/api/health")
def health():
    try:
        client = get_bq_client()
        list(client.query(f"SELECT 1 FROM `{PROJECT_ID}.{DATASET}.{TABLE}` LIMIT 1"))
        return jsonify({"status": "ok", "bigquery": "connected"})
    except Exception as e:
        return jsonify({"status": "degraded", "bigquery": str(e)}), 200


if __name__ == "__main__":
    print(f"Starting API server — project: {PROJECT_ID}")
    print(f"BigQuery: {PROJECT_ID}.{DATASET}.{TABLE}")
    app.run(host="0.0.0.0", port=5050, debug=True)
