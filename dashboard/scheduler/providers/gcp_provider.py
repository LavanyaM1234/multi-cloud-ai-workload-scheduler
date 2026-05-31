"""
scheduler/providers/gcp_provider.py
────────────────────────────────────────────────────────────────────
Fetches available GCP preemptible (spot) instances with live pricing.

Env vars required:
    GCP_PROJECT_ID   — your GCP project ID
    GCP_ZONE         — primary zone e.g. us-central1-a
    GCP_REGION       — region e.g. us-central1
    GOOGLE_APPLICATION_CREDENTIALS — path to service_account.json

GCP machine type naming rules:
    CPU instances  → just the machine type: "e2-standard-4", "n1-standard-8"
    GPU instances  → machine type is still "n1-standard-N"; the GPU is a
                     SEPARATE accelerator attached via guestAccelerators[].
                     "n1-standard-8-t4" does NOT exist — GCP will 400.
    A2 instances   → GPU is built-in: "a2-highgpu-1g" (no separate accel needed)
    G2 instances   → GPU is built-in: "g2-standard-4"

Returns a list of candidate dicts compatible with selector.py.
Each GPU candidate carries gpu_model + gpu_count so gcp_launcher.py
can attach the right accelerator when creating the VM.
"""

from __future__ import annotations
import os
import logging

logger = logging.getLogger(__name__)

GCP_PROJECT = os.getenv("GCP_PROJECT_ID", "")
GCP_ZONE    = os.getenv("GCP_ZONE",       "us-central1-a")
GCP_REGION  = os.getenv("GCP_REGION",     "us-central1")

# ── Static GCP spot price table (USD/hr preemptible) ─────────────
# Format: instance_type → (spot_price, vcpus, ram_gb, gpu_model, gpu_mem_gb, gpu_count)
#
# IMPORTANT — machine type names must match GCP's actual API names:
#   CPU:  e2-standard-4, n1-standard-4, n1-standard-8, etc.
#   GPU:  still "n1-standard-4" or "n1-standard-8" — the GPU (T4/V100/A100)
#         is attached separately via guestAccelerators in gcp_launcher.py.
#         gpu_model + gpu_count in this dict tell the launcher what to attach.
#   A2:   a2-highgpu-1g, a2-highgpu-2g — GPU built-in, no separate accel needed.
#   G2:   g2-standard-4 — L4 GPU built-in.
_GCP_CATALOG: dict[str, tuple] = {
    # ── CPU instances ──────────────────────────────────────────────
    "e2-standard-2":   (0.022,  2,   8, None,  0, 0),
    "e2-standard-4":   (0.034,  4,  16, None,  0, 0),
    "e2-standard-8":   (0.068,  8,  32, None,  0, 0),
    "e2-standard-16":  (0.135, 16,  64, None,  0, 0),
    "n1-standard-4":   (0.048,  4,  15, None,  0, 0),
    "n1-standard-8":   (0.096,  8,  30, None,  0, 0),
    "n1-standard-16":  (0.192, 16,  60, None,  0, 0),

    # ── GPU instances (machine type = n1-standard-N + accelerator) ─
    # gpu_model / gpu_count are read by gcp_launcher.py to attach the GPU.
    # The machine type sent to GCP API is "n1-standard-4" or "n1-standard-8" —
    # NOT "n1-standard-4-t4" (that name does not exist in GCP).
    "n1-standard-4+T4":   (0.138,  4, 15, "T4",   16, 1),
    "n1-standard-8+T4":   (0.186,  8, 30, "T4",   16, 1),
    "n1-standard-8+V100": (0.736,  8, 30, "V100", 16, 1),
    "n1-standard-8+A100": (1.102,  8, 30, "A100", 40, 1),

    # ── A2 instances (A100 built-in, no separate accel) ────────────
    "a2-highgpu-1g":   (1.102, 12,  85, "A100", 40, 1),
    "a2-highgpu-2g":   (2.204, 24, 170, "A100", 40, 2),

    # ── G2 instances (L4 built-in) ────────────────────────────────
    "g2-standard-4":   (0.180,  4,  16, "L4",   24, 1),
    "g2-standard-8":   (0.270,  8,  32, "L4",   24, 1),
}

_GCP_ONDEMAND: dict[str, float] = {
    "e2-standard-2":      0.067,
    "e2-standard-4":      0.134,
    "e2-standard-8":      0.268,
    "e2-standard-16":     0.536,
    "n1-standard-4":      0.190,
    "n1-standard-8":      0.380,
    "n1-standard-16":     0.760,
    "n1-standard-4+T4":   0.556,
    "n1-standard-8+T4":   0.746,
    "n1-standard-8+V100": 2.940,
    "n1-standard-8+A100": 4.408,
    "a2-highgpu-1g":      4.408,
    "a2-highgpu-2g":      8.816,
    "g2-standard-4":      0.720,
    "g2-standard-8":      1.440,
}

# inside candidates.append({...}):


# Preemption risk by instance family
_GCP_RISK: dict[str, float] = {
    "e2-standard-2":      0.05,
    "e2-standard-4":      0.06,
    "e2-standard-8":      0.07,
    "e2-standard-16":     0.08,
    "n1-standard-4":      0.09,
    "n1-standard-8":      0.10,
    "n1-standard-16":     0.11,
    "n1-standard-4+T4":   0.12,
    "n1-standard-8+T4":   0.13,
    "n1-standard-8+V100": 0.15,
    "n1-standard-8+A100": 0.18,
    "a2-highgpu-1g":      0.18,
    "a2-highgpu-2g":      0.20,
    "g2-standard-4":      0.14,
    "g2-standard-8":      0.15,
}


def _gcp_machine_type(instance_type: str) -> str:
    """
    Return the actual GCP machine type string to pass to the Compute API.

    For GPU candidates like "n1-standard-8+T4", the machine type is
    "n1-standard-8" — the GPU is attached separately as an accelerator.
    For A2/G2 instances the name is already correct.
    """
    if "+" in instance_type:
        return instance_type.split("+")[0]
    return instance_type


def _try_live_prices():
    """
    GCP preemptible prices are fixed (not auction-based).
    Catalog values are accurate — live API not needed.
    Returns None to signal catalog should be used.
    """
    return None


def list_instances() -> list[dict]:
    """
    Return list of GCP preemptible instance candidates for selector.py.

    Each dict has these keys (required by selector.py):
        cloud, instance_type, region, zone,
        price_usd_hr, preemption_risk,
        gpu_model, gpu_mem_gb, gpu_count,
        vcpus, ram_gb,
        is_spot,
        _price_source,
        _gcp_machine_type   ← actual machine type sent to GCP API
    """
    candidates = []
    price_source = "catalog"

    for instance_type, (catalog_price, vcpus, ram_gb, gpu_model, gpu_mem_gb, gpu_count) in _GCP_CATALOG.items():
        risk = _GCP_RISK.get(instance_type, 0.10)

        candidates.append({
            "cloud":             "gcp",
            "instance_type":     instance_type,
            "region":            GCP_REGION,
            "zone":              GCP_ZONE,
            "price_usd_hr":      catalog_price,
            "preemption_risk":   risk,
            "gpu_model":         gpu_model,
            "gpu_mem_gb":        gpu_mem_gb,
            "gpu_count":         gpu_count,
            "vcpus":             vcpus,
            "ram_gb":            ram_gb,
            "is_spot":           True,
            "_price_source":     price_source,
            "_gcp_machine_type": _gcp_machine_type(instance_type),
            "ondemand_price": _GCP_ONDEMAND.get(instance_type, 0),

        })

    logger.info(f"[GCP] {len(candidates)} preemptible instances loaded ({price_source} pricing)")
    return candidates