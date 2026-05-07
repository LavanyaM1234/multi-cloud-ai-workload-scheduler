"""
api/server.py  — updated
─────────────────────────
Key change: /api/jobs/submit now calls launcher.launch_job()
which creates a real GCP VM with the job config baked in.

Also added /api/jobs/status/<job_id> so the dashboard can
poll a specific job's state from GCS.
"""

import os, json, uuid
from datetime import datetime, timezone
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__,
            template_folder="../templates",
            static_folder="../static")
CORS(app)

PROJECT_ID = os.getenv("GCP_PROJECT_ID",        "tensile-method-459009-k2")
DATASET    = os.getenv("BIGQUERY_DATASET",       "spot_prices")
TABLE      = os.getenv("BIGQUERY_TABLE",         "price_history")
GCS_BUCKET = os.getenv("CHECKPOINT_GCS_BUCKET",  "")


@app.route("/")
def home():
    return render_template("index.html")


def get_bq_client():
    from google.cloud import bigquery
    return bigquery.Client(project=PROJECT_ID)


# ── BigQuery routes (unchanged) ───────────────────────────────────────────────

@app.route("/api/prices/history")
def price_history():
    try:
        client = get_bq_client()
        query = f"""
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
        by_cloud, all_minutes = defaultdict(dict), set()
        for r in rows:
            key = r["minute"].strftime("%H:%M")
            by_cloud[r["cloud"]][key] = round(float(r["avg_price"]), 4)
            all_minutes.add(key)
        timestamps = sorted(all_minutes)[-30:]
        return jsonify({
            "timestamps": timestamps,
            "aws":   [by_cloud["aws"].get(t)   for t in timestamps],
            "gcp":   [by_cloud["gcp"].get(t)   for t in timestamps],
            "azure": [by_cloud["azure"].get(t) for t in timestamps],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/prices/summary")
def price_summary():
    try:
        client = get_bq_client()
        query = f"""
            WITH latest AS (
                SELECT cloud, price_usd_per_hr,
                       ROW_NUMBER() OVER (PARTITION BY cloud ORDER BY collected_at DESC) AS rn
                FROM `{PROJECT_ID}.{DATASET}.{TABLE}`
                WHERE collected_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 3 HOUR)
                  AND preempted = FALSE AND gpu_class != 'none'
            ), stats AS (
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
        rows = list(client.query(query))
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
        query = f"""
            SELECT cloud, region, availability_zone, instance_type,
                   gpu_class, price_usd_per_hr, ondemand_price_usd_hr,
                   ROUND((1 - price_usd_per_hr / NULLIF(ondemand_price_usd_hr,0))*100,1) AS discount_pct
            FROM `{PROJECT_ID}.{DATASET}.{TABLE}`
            WHERE preempted = FALSE
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY cloud, instance_type, region ORDER BY collected_at DESC
            ) = 1
            ORDER BY cloud, price_usd_per_hr ASC LIMIT 20
        """
        return jsonify([dict(r) for r in client.query(query)])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/prices/preemptions")
def preemptions():
    try:
        client = get_bq_client()
        query = f"""
            SELECT cloud, region, availability_zone, instance_type,
                   gpu_class, price_usd_per_hr, preemption_source, collected_at
            FROM `{PROJECT_ID}.{DATASET}.{TABLE}`
            WHERE preempted = TRUE ORDER BY collected_at DESC LIMIT 10
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


@app.route("/api/stats")
def stats():
    try:
        client = get_bq_client()
        query = f"""
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


# ── Jobs ──────────────────────────────────────────────────────────────────────

@app.route("/api/jobs")
def get_jobs():
    """Read all job_state.json files from GCS. Dashboard polls this every 60s."""
    if not GCS_BUCKET:
        return jsonify([])
    try:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        jobs   = []
        for blob in bucket.list_blobs(prefix="checkpoints/"):
            if blob.name.endswith("job_state.json"):
                try:
                    state = json.loads(blob.download_as_text())
                    jobs.append(state)
                except Exception:
                    pass
        return jsonify(jobs)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/jobs/status/<job_id>")
def job_status(job_id):
    """Poll a single job's state from GCS. Dashboard calls this every 10s for active jobs."""
    if not GCS_BUCKET:
        return jsonify({"error": "GCS_BUCKET not set"}), 400
    try:
        from google.cloud import storage
        client = storage.Client()
        blob   = client.bucket(GCS_BUCKET).blob(f"checkpoints/{job_id}/job_state.json")
        if not blob.exists():
            return jsonify({"error": "job not found"}), 404
        state = json.loads(blob.download_as_text())
        return jsonify(state)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── KEY ROUTE: submit job → selector → launcher → GCP VM ─────────────────────

@app.route("/api/jobs/submit", methods=["POST"])
def submit_job():
    """
    Full pipeline:
      1. Read form data from modal
      2. Call selector.pick_best_cloud() → gets GCP e2-standard-4
      3. Call launcher.launch_job() → writes config to GCS → creates VM
      4. Return job_id + VM console URL to dashboard

    Body from modal.js:
        task_name, lr, hidden_dim, dropout, batch_size, epochs,
        ckpt_every, input_dim, num_classes, max_budget, deadline_hrs,
        dataset, s3_path (if custom), cloud (auto/aws/gcp/azure),
        instance, fallback, priority
    """
    try:
        data = request.json or {}

        # Generate unique job ID
        ts     = datetime.now(timezone.utc).strftime("%m%d-%H%M")
        suffix = str(uuid.uuid4())[:6]
        job_id = f"job-{ts}-{suffix}"

        # Build job_params matching what launcher.launch_job() expects
        job_params = {
            "task_name":    data.get("task_name",   "Untitled"),
            "lr":           float(data.get("lr",           0.001)),
            "hidden_dim":   int(  data.get("hidden_dim",   256)),
            "dropout":      float(data.get("dropout",      0.3)),
            "batch_size":   int(  data.get("batch_size",   64)),
            "epochs":       int(  data.get("epochs",       50)),
            "ckpt_every":   int(  data.get("ckpt_every",   50)),
            "input_dim":    int(  data.get("input_dim",    50)),
            "num_classes":  int(  data.get("num_classes",  5)),
            "max_budget":   float(data.get("max_budget",   2.0)),
            "deadline_hrs": float(data.get("deadline_hrs", 8.0)),
            "dataset":      data.get("dataset", "synthetic-500k"),
            "s3_dataset_path": data.get("s3_path", ""),
            "priority":     data.get("priority", "balanced"),
        }

        # Pick best cloud (currently always GCP e2-standard-4)
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from scheduler.selector import pick_best_cloud
        decision = pick_best_cloud(job_params)

        # Override cloud if user specified one explicitly
        forced_cloud = data.get("cloud", "auto")
        if forced_cloud != "auto":
            decision["cloud"] = forced_cloud
            # map instance from dropdown
            decision["instance_type"] = data.get("instance", decision["instance_type"])

        # Launch the VM
        from scheduler.launcher import launch_job
        result = launch_job(job_id, job_params, decision)

        if result.get("error"):
            return jsonify({
                "error":   result["error"],
                "job_id":  job_id,
                "launched": False,
            }), 500

        return jsonify({
            "job_id":       job_id,
            "launched":     True,
            "cloud":        decision["cloud"],
            "instance_type": decision["instance_type"],
            "region":       decision["region"],
            "price_usd_hr": decision["price_usd_hr"],
            "est_cost":     decision["est_cost"],
            "est_hours":    decision["est_hours"],
            "console_url":  result.get("console_url", ""),
            "logs_url":     result.get("logs_url", ""),
            "reason":       decision["reason"],
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "launched": False}), 500


if __name__ == "__main__":
    print(f"API server — project: {PROJECT_ID}")
    print(f"BigQuery: {PROJECT_ID}.{DATASET}.{TABLE}")
    print(f"GCS bucket: {GCS_BUCKET or 'NOT SET'}")
    app.run(host="0.0.0.0", port=5050, debug=True)