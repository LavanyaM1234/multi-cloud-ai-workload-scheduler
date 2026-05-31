"""
scheduler/providers/aws_provider.py
────────────────────────────────────────────────────────────────────
Fetches AWS spot instance prices using boto3.

Merges three AWS APIs (same approach as your aws_spot_poller.py):
    1. describe_spot_price_history  → live spot price
    2. describe_instance_types      → vcpus, ram, GPU specs
    3. get_spot_placement_scores    → availability signal (used as risk proxy)

Env vars required:
    AWS_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY
    AWS_DEFAULT_REGION   — e.g. us-east-1  (default if not set)
    AWS_REGION           — checked first, falls back to AWS_DEFAULT_REGION
    AWS_ENABLE_GPU       — set to "true" to enable GPU instances (default false)

No extra AWS permissions needed beyond:
    ec2:DescribeSpotPriceHistory
    ec2:DescribeInstanceTypes
"""

from __future__ import annotations
import os
import logging
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

from risk.predictor import score_instance_from_api

AWS_REGION     = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION", "us-east-1")
AWS_ENABLE_GPU = "true"

# ── Instance catalog ─────────────────────────────────────────────
# Format: instance_type → (spot_fallback_price, vcpus, ram_gb, gpu_model, gpu_mem_gb, gpu_count)
# Prices are catalog fallbacks used when live API is unavailable.
# Live prices are fetched via describe_spot_price_history in list_instances().
_AWS_CATALOG: dict[str, tuple] = {
    # CPU — works on free/trial accounts
    # t3.small (2GB RAM): minimum viable for torch workloads
    "t3.small":      (0.006,  2,   2,  None,  0, 0),
    "t3.medium":     (0.014,  2,   4,  None,  0, 0),
    "t3.large":      (0.025,  2,   8,  None,  0, 0),
    "m5.large":      (0.048,  2,   8,  None,  0, 0),
    "m5.xlarge":     (0.096,  4,  16,  None,  0, 0),
    "m5.2xlarge":    (0.192,  8,  32,  None,  0, 0),
    "c5.xlarge":     (0.085,  4,   8,  None,  0, 0),
    "c5.2xlarge":    (0.170,  8,  16,  None,  0, 0),
    # GPU — requires paid account; only included when AWS_ENABLE_GPU=true
    "g4dn.xlarge":   (0.166,  4,  16, "T4",   16, 1),
    "g4dn.2xlarge":  (0.338,  8,  32, "T4",   16, 1),
    "g4dn.4xlarge":  (0.602, 16,  64, "T4",   16, 1),
    "p3.2xlarge":    (0.918,  8,  61, "V100", 16, 1),
    "p3.8xlarge":    (3.672, 32, 244, "V100", 16, 4),
    "p4d.24xlarge": (16.912, 96, 1152, "A100", 40, 8),
}

# On-demand prices — used by get_spot_price() as fallback baseline
# and by estimate_preemption_risk() for spot ratio calculations.
_ONDEMAND_PRICES: dict[str, float] = {
    # CPU
    "t3.small":      0.0208,
    "t3.medium":     0.0416,
    "t3.large":      0.0832,
    "m5.large":      0.096,
    "m5.xlarge":     0.192,
    "m5.2xlarge":    0.384,
    "c5.xlarge":     0.170,
    "c5.2xlarge":    0.340,
    # GPU
    "g4dn.xlarge":   0.526,
    "g4dn.2xlarge":  0.752,
    "g4dn.4xlarge":  1.204,
    "p3.2xlarge":    3.060,
    "p3.8xlarge":    12.240,
    "p4d.24xlarge":  32.770,
}

# Baseline interruption frequency — used by estimate_preemption_risk()
# Source: AWS Spot Advisor historical data (approximated)
_INTERRUPTION_FREQ: dict[str, float] = {
    # CPU
    "t3.small":      0.03,
    "t3.medium":     0.04,
    "t3.large":      0.04,
    "m5.large":      0.07,
    "m5.xlarge":     0.08,
    "m5.2xlarge":    0.09,
    "c5.xlarge":     0.08,
    "c5.2xlarge":    0.09,
    # GPU
    "g4dn.xlarge":   0.14,
    "g4dn.2xlarge":  0.15,
    "g4dn.4xlarge":  0.16,
    "p3.2xlarge":    0.20,
    "p3.8xlarge":    0.22,
    "p4d.24xlarge":  0.25,
}

# ── GPU toggle ────────────────────────────────────────────────────
_GPU_INSTANCE_TYPES = {
    "g4dn.xlarge", "g4dn.2xlarge", "g4dn.4xlarge",
    "p3.2xlarge", "p3.8xlarge", "p4d.24xlarge",
}

def _active_catalog() -> dict[str, tuple]:
    """
    Return the subset of _AWS_CATALOG to use for this run.
    When AWS_ENABLE_GPU=false (default), GPU instances are excluded.
    GPU instances require a paid AWS account — free/trial accounts
    get InvalidParameterCombination from the EC2 API.
    """
    if AWS_ENABLE_GPU:
        return _AWS_CATALOG
    return {
        k: v for k, v in _AWS_CATALOG.items()
        if k not in _GPU_INSTANCE_TYPES
    }


# ══════════════════════════════════════════════════════════════════
# boto3 helpers
# ══════════════════════════════════════════════════════════════════

def _get_boto3_client(service: str):
    """Create boto3 client; returns None if boto3/creds unavailable."""
    try:
        import boto3
        return boto3.client(
            service,
            region_name           = AWS_REGION,
            aws_access_key_id     = os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
    except ImportError:
        logger.warning("[AWS] boto3 not installed — pip install boto3")
        return None
    except Exception as e:
        logger.warning(f"[AWS] boto3 client creation failed: {e}")
        return None


def _fetch_live_spot_prices(ec2, instance_types: list[str]) -> dict[str, float]:
    """
    Bulk-fetch current spot prices via describe_spot_price_history paginator.
    Returns dict of instance_type → spot_price_usd_hr (lowest across AZs).
    """
    prices = {}
    try:
        paginator = ec2.get_paginator("describe_spot_price_history")
        pages = paginator.paginate(
            InstanceTypes=instance_types,
            ProductDescriptions=["Linux/UNIX"],
            StartTime=datetime.now(timezone.utc),
        )
        for page in pages:
            for entry in page.get("SpotPriceHistory", []):
                itype = entry["InstanceType"]
                price = float(entry["SpotPrice"])
                # Keep lowest price seen (multiple AZs may differ)
                if itype not in prices or price < prices[itype]:
                    prices[itype] = price
    except Exception as e:
        logger.warning(f"[AWS] describe_spot_price_history failed: {e}")
    return prices


def _fetch_instance_specs(ec2, instance_types: list[str]) -> dict[str, dict]:
    """
    Fetch vcpus, ram, GPU info from describe_instance_types.
    Skips unsupported instance types for the current region.
    """
    specs = {}
    for itype in instance_types:
        try:
            response = ec2.describe_instance_types(InstanceTypes=[itype])
            for it in response.get("InstanceTypes", []):
                vcpus  = it.get("VCpuInfo", {}).get("DefaultVCpus", 4)
                ram_gb = it.get("MemoryInfo", {}).get("SizeInMiB", 16384) / 1024
                gpus   = it.get("GpuInfo", {}).get("Gpus", [])

                gpu_model  = gpus[0].get("Name") if gpus else None
                gpu_mem_gb = (
                    gpus[0].get("MemoryInfo", {}).get("SizeInMiB", 0) / 1024
                    if gpus else 0
                )
                gpu_count = sum(g.get("Count", 0) for g in gpus)

                specs[itype] = {
                    "vcpus":      vcpus,
                    "ram_gb":     round(ram_gb, 1),
                    "gpu_model":  gpu_model,
                    "gpu_mem_gb": round(gpu_mem_gb, 1),
                    "gpu_count":  gpu_count,
                }
        except Exception as e:
            logger.warning(f"[AWS] Skipping unsupported instance type {itype}: {e}")
    return specs


# ══════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════

def list_instances() -> list[dict]:
    """
    Return list of AWS spot instance candidates for selector.py.

    Uses bulk live pricing via boto3 paginator (Doc 5 approach —
    one API call for all instances, more efficient than per-instance
    calls). Falls back to catalog prices on any API failure.

    GPU instances are excluded unless AWS_ENABLE_GPU=true in env.
    Risk score is NOT set here — selector.py calls the LSTM+XGBoost
    model via score_instance_from_api() separately for each candidate.
    """
    catalog      = _active_catalog()
    instance_types = list(catalog.keys())
    live_prices  = {}
    live_specs   = {}
    price_source = "catalog"

    ec2 = _get_boto3_client("ec2")
    if ec2:
        try:
            live_prices  = _fetch_live_spot_prices(ec2, instance_types)
            live_specs   = _fetch_instance_specs(ec2, instance_types)
            price_source = "live" if live_prices else "catalog"
            logger.info(
                f"[AWS] Live prices fetched for "
                f"{len(live_prices)}/{len(instance_types)} instances"
            )
        except Exception as e:
            logger.warning(f"[AWS] boto3 fetch failed: {e} — using catalog prices")

    candidates = []
    for instance_type, (catalog_price, cat_vcpus, cat_ram, cat_gpu, cat_gpu_mem, cat_gpu_count) in catalog.items():
        price = live_prices.get(instance_type, catalog_price)
        spec  = live_specs.get(instance_type, {})

        candidates.append({
            "cloud":         "aws",
            "instance_type": instance_type,
            "region":        AWS_REGION,
            "zone":          f"{AWS_REGION}a",
            "price_usd_hr":  price,
            "gpu_model":     spec.get("gpu_model",  cat_gpu),
            "gpu_mem_gb":    spec.get("gpu_mem_gb", cat_gpu_mem),
            "gpu_count":     spec.get("gpu_count",  cat_gpu_count),
            "vcpus":         spec.get("vcpus",      cat_vcpus),
            "ram_gb":        spec.get("ram_gb",      cat_ram),
            "is_spot":       True,
            "_price_source": price_source,
            "ondemand_price": _ONDEMAND_PRICES.get(instance_type, 0),
        })

    logger.info(
        f"[AWS] {len(candidates)} spot instances loaded "
        f"({price_source} pricing, GPU={'enabled' if AWS_ENABLE_GPU else 'disabled'})"
    )
    return candidates


def get_spot_price(instance_type: str) -> tuple[float, str]:
    """
    Fetch current EC2 Spot price for a single instance type.
    Returns (price, source) where source is "live" or "fallback".
    Falls back to ~30% of on-demand if the API is unavailable.

    Note: list_instances() uses bulk fetching for efficiency.
    This helper is for one-off lookups (e.g. resume_job checks).
    """
    ec2 = _get_boto3_client("ec2")
    if ec2:
        try:
            resp = ec2.describe_spot_price_history(
                InstanceTypes=[instance_type],
                ProductDescriptions=["Linux/UNIX"],
                StartTime=datetime.now(timezone.utc) - timedelta(hours=1),
                MaxResults=5,
            )
            prices = resp.get("SpotPriceHistory", [])
            if prices:
                avg = sum(float(p["SpotPrice"]) for p in prices) / len(prices)
                logger.debug(f"[AWS] Live spot price {instance_type}: ${avg:.4f}/hr")
                return round(avg, 4), "live"
        except Exception as e:
            logger.warning(f"[AWS] Spot price fetch failed for {instance_type}: {e}")

    # Fallback: ~30% of on-demand (typical spot discount)
    fallback = round(_ONDEMAND_PRICES.get(instance_type, 0.50) * 0.30, 4)
    logger.debug(f"[AWS] Using fallback price for {instance_type}: ${fallback}/hr")
    return fallback, "fallback"


def estimate_preemption_risk(instance_type: str) -> float:
    """
    Return estimated interruption probability [0.0, 1.0].
    Based on AWS Spot Advisor historical frequency data, adjusted
    for current UTC hour (peak hours → higher competition → higher risk).

    Note: selector.py overrides this with the live LSTM+XGBoost score
    from score_instance_from_api(). This is the heuristic fallback
    used when the ML model is unavailable.
    """
    base = _INTERRUPTION_FREQ.get(instance_type, 0.12)

    # Peak US business hours → higher spot market competition
    hour = datetime.now(timezone.utc).hour
    if 13 <= hour <= 23:
        base = min(base * 1.25, 0.95)
    elif 0 <= hour <= 6:
        base = base * 0.75

    return round(base, 3)


def launch_vm(decision: dict, startup_script: str) -> dict:
    """Delegate to aws_launcher.create_aws_vm()."""
    from scheduler.launchers.aws_launcher import create_aws_vm
    return create_aws_vm(decision, startup_script)
    