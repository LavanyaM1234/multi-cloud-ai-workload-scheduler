"""
scheduler/providers/gcp_provider.py
────────────────────────────────────────────────────────────────────
Fetches available GCP preemptible (spot) instances with live pricing.

Env vars required:
    GCP_PROJECT_ID   — your GCP project ID
    GCP_ZONE         — primary zone e.g. us-central1-a
    GCP_REGION       — region e.g. us-central1
    GOOGLE_APPLICATION_CREDENTIALS — path to service_account.json

Returns a list of candidate dicts compatible with selector.py.
"""

from __future__ import annotations
import os
import logging

logger = logging.getLogger(__name__)

GCP_PROJECT = os.getenv("GCP_PROJECT_ID", "")
GCP_ZONE    = os.getenv("GCP_ZONE",       "us-central1-a")
GCP_REGION  = os.getenv("GCP_REGION",     "us-central1")

# ── Static GCP spot price table (USD/hr) ────────────────────────
# GCP preemptible prices are fixed (not auction-based like AWS).
# Format: instance_type → (spot_price, vcpus, ram_gb, gpu_model, gpu_mem_gb, gpu_count)
_GCP_CATALOG: dict[str, tuple] = {
    # CPU instances
    "e2-standard-4":      (0.034,  4,  16,  None,  0,  0),
    "e2-standard-8":      (0.068,  8,  32,  None,  0,  0),
    "e2-standard-16":     (0.135, 16,  64,  None,  0,  0),
    "n1-standard-4":      (0.048,  4,  15,  None,  0,  0),
    "n1-standard-8":      (0.096,  8,  30,  None,  0,  0),
    "n1-standard-16":     (0.192, 16,  60,  None,  0,  0),
    # GPU instances
    "n1-standard-4-t4":   (0.138,  4, 15, "T4",   16, 1),
    "n1-standard-8-t4":   (0.186,  8, 30, "T4",   16, 1),
    "n1-standard-8-v100": (0.736,  8, 30, "V100", 16, 1),
    "n1-standard-8-a100": (1.102,  8, 30, "A100", 40, 1),
    "a2-highgpu-1g":      (1.102, 12, 85, "A100", 40, 1),
    "a2-highgpu-2g":      (2.204, 24,170, "A100", 40, 2),
}

_GCP_RISK: dict[str, float] = {
    "e2-standard-4":      0.06,
    "e2-standard-8":      0.07,
    "e2-standard-16":     0.08,
    "n1-standard-4":      0.09,
    "n1-standard-8":      0.10,
    "n1-standard-16":     0.11,
    "n1-standard-4-t4":   0.12,
    "n1-standard-8-t4":   0.13,
    "n1-standard-8-v100": 0.15,
    "n1-standard-8-a100": 0.18,
    "a2-highgpu-1g":      0.18,
    "a2-highgpu-2g":      0.20,
}


def _try_live_prices() -> dict[str, float] | None:
    """
    Attempt to fetch live GCP preemptible prices via the Cloud Billing API.
    GCP preemptible prices are fixed, so catalog values are accurate enough.
    """
    try:
        from google.cloud import billing_v1  # type: ignore
        return None
    except ImportError:
        return None
    except Exception as e:
        logger.warning(f"[GCP] Live pricing unavailable: {e}")
        return None


def _get_risk(cloud, region, zone, instance_type, bq_client=None) -> float:
    try:
        from risk.predictor import score_instance_from_api
        return score_instance_from_api(
            cloud         = cloud,
            region        = region,
            az            = zone,
            instance_type = instance_type,
            bq_client     = bq_client,
        )
    except Exception as e:
        logger.warning(f"[{cloud}] Risk scoring failed for {instance_type}: {e} — using fallback")
        return _GCP_RISK.get(instance_type, 0.10)


def list_instances() -> list[dict]:
    """
    Return list of GCP preemptible instance candidates for selector.py.

    Each dict has these keys (required by selector.py):
        cloud, instance_type, region, zone,
        price_usd_hr, preemption_risk,
        gpu_model, gpu_mem_gb, gpu_count,
        vcpus, ram_gb,
        is_spot,          ← always True for GCP preemptible
        _price_source     ← "live" or "catalog"
    """
    live = _try_live_prices()
    candidates = []

    for instance_type, (catalog_price, vcpus, ram_gb, gpu_model, gpu_mem_gb, gpu_count) in _GCP_CATALOG.items():
        price        = (live or {}).get(instance_type, catalog_price)
        price_source = "live" if (live and instance_type in live) else "catalog"
        risk         = _GCP_RISK.get(instance_type, 0.10)

        candidates.append({
            "cloud":           "gcp",
            "instance_type":   instance_type,
            "region":          GCP_REGION,
            "zone":            GCP_ZONE,
            "price_usd_hr":    price,
            "preemption_risk": risk,
            "gpu_model":       gpu_model,
            "gpu_mem_gb":      gpu_mem_gb,
            "gpu_count":       gpu_count,
            "vcpus":           vcpus,
            "ram_gb":          ram_gb,
            "is_spot":         True,
            "_price_source":   price_source,
        })

    logger.info(f"[GCP] {len(candidates)} preemptible instances loaded ({price_source} pricing)")
    return candidates