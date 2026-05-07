"""
scheduler/selector.py
──────────────────────
Cloud selector — picks the best cloud for a job.

Phase 1 (now): Always returns GCP e2-standard-4.
Phase 4 (later): Queries BigQuery for live spot prices,
                 runs Pareto optimizer across all 3 clouds.

Called by api/server.py → submit_job()
"""

import os

# ── GCP config ────────────────────────────────────────────────────
GCP_PROJECT  = os.getenv("GCP_PROJECT_ID",      "tensile-method-459009-k2")
GCP_REGION   = os.getenv("GCP_REGION",          "us-central1")
GCP_ZONE     = os.getenv("GCP_ZONE",            "us-central1-a")
GCS_BUCKET   = os.getenv("CHECKPOINT_GCS_BUCKET", "")

# e2-standard-4 spot price in us-central1 (approx, May 2025)
E2_SPOT_PRICE = 0.067   # $/hr


def pick_best_cloud(job: dict) -> dict:
    """
    Given a job spec, return the best cloud + instance decision.

    Args:
        job: dict with keys:
            job_id, model_arch, dataset_size, max_budget,
            deadline_hrs, min_gpu_mem  (from modal submit)

    Returns:
        dict with keys:
            cloud, instance_type, region, zone,
            price_usd_hr, est_cost, est_hours,
            gcs_bucket, reason
    """
    job_id       = job.get("job_id",       "job-unknown")
    max_budget   = float(job.get("max_budget",   2.0))
    deadline_hrs = float(job.get("deadline_hrs", 8.0))

    # ── Phase 1: always pick GCP e2-standard-4 ────────────────────
    # Estimated training time: budget / price_per_hr
    # (rough — real estimate needs dataset size + model size)
    est_hours = min(max_budget / E2_SPOT_PRICE, deadline_hrs)
    est_cost  = round(est_hours * E2_SPOT_PRICE, 4)

    decision = {
        "cloud":          "gcp",
        "instance_type":  "e2-standard-4",
        "machine_family": "e2",
        "region":         GCP_REGION,
        "zone":           GCP_ZONE,
        "price_usd_hr":   E2_SPOT_PRICE,
        "est_hours":      round(est_hours, 2),
        "est_cost":       est_cost,
        "gcs_bucket":     GCS_BUCKET,
        "reason":         "Phase 1 — GCP e2-standard-4 spot (CPU). "
                          "Pareto optimizer pending (Phase 4).",
    }

    return decision


# ── Manual test ───────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    test_job = {
        "job_id":       "test-001",
        "max_budget":   2.0,
        "deadline_hrs": 6.0,
    }
    result = pick_best_cloud(test_job)
    print(json.dumps(result, indent=2))
