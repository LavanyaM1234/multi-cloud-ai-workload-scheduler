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

No extra AWS permissions needed beyond:
    ec2:DescribeSpotPriceHistory
    ec2:DescribeInstanceTypes
"""

from __future__ import annotations
import os
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

AWS_REGION = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION", "us-east-1")

# ── Instance catalog ─────────────────────────────────────────────
# (instance_type, vcpus, ram_gb, gpu_model, gpu_mem_gb, gpu_count)
# Prices fetched live; these are fallback on API failure.
_AWS_CATALOG: dict[str, tuple] = {
    # CPU
    "t3.medium":     (0.014,  2,   4,  None,  0, 0),
    "t3.large":      (0.025,  2,   8,  None,  0, 0),
    "m5.large":      (0.048,  2,   8,  None,  0, 0),
    "m5.xlarge":     (0.096,  4,  16,  None,  0, 0),
    "m5.2xlarge":    (0.192,  8,  32,  None,  0, 0),
    "c5.xlarge":     (0.085,  4,   8,  None,  0, 0),
    "c5.2xlarge":    (0.170,  8,  16,  None,  0, 0),
    # GPU
    "g4dn.xlarge":   (0.166,  4,  16, "T4",   16, 1),
    "g4dn.2xlarge":  (0.338,  8,  32, "T4",   16, 1),
    "g4dn.4xlarge":  (0.602, 16,  64, "T4",   16, 1),
    "p3.2xlarge":    (0.918,  8,  61, "V100", 16, 1),
    "p3.8xlarge":    (3.672, 32, 244, "V100", 16, 4),
    "p4d.24xlarge": (16.912, 96, 1152,"A100", 40, 8),
}

# Baseline preemption risk — overridden by availability score when live
_AWS_RISK: dict[str, float] = {
    "t3.medium":     0.04,
    "t3.large":      0.04,
    "m5.large":      0.07,
    "m5.xlarge":     0.08,
    "m5.2xlarge":    0.09,
    "c5.xlarge":     0.08,
    "c5.2xlarge":    0.09,
    "g4dn.xlarge":   0.14,
    "g4dn.2xlarge":  0.15,
    "g4dn.4xlarge":  0.16,
    "p3.2xlarge":    0.20,
    "p3.8xlarge":    0.22,
    "p4d.24xlarge":  0.25,
}


def _fetch_live_spot_prices(ec2, instance_types: list[str]) -> dict[str, float]:
    """
    Fetch current spot prices from describe_spot_price_history.
    Returns dict of instance_type → spot_price_usd_hr.
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

from risk.predictor import score_instance_from_api




def _fetch_instance_specs(ec2, instance_types: list[str]) -> dict[str, dict]:
    """
    Fetch vcpus, ram, GPU info from describe_instance_types.
    Skips unsupported instance types for the current region.
    """
    specs = {}

    for itype in instance_types:
        try:
            response = ec2.describe_instance_types(
                InstanceTypes=[itype]
            )

            for it in response.get("InstanceTypes", []):
                vcpus  = it.get("VCpuInfo", {}).get("DefaultVCpus", 4)
                ram_gb = it.get("MemoryInfo", {}).get("SizeInMiB", 16384) / 1024

                gpus = it.get("GpuInfo", {}).get("Gpus", [])

                gpu_model = gpus[0].get("Name") if gpus else None

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


def list_instances() -> list[dict]:
    """
    Return list of AWS spot instance candidates for selector.py.

    Attempts live pricing via boto3; falls back to catalog on failure.
    Risk score comes from catalog baseline (live risk model from your
    LSTM+XGBoost is applied separately in predictor.py via the poller).
    """
    instance_types = list(_AWS_CATALOG.keys())
    live_prices    = {}
    live_specs     = {}
    price_source   = "catalog"

    try:
        import boto3  # type: ignore
        ec2 = boto3.client(
            "ec2",
            region_name            = AWS_REGION,
            aws_access_key_id      = os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key  = os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
        live_prices  = _fetch_live_spot_prices(ec2, instance_types)
        live_specs   = _fetch_instance_specs(ec2, instance_types)
        price_source = "live" if live_prices else "catalog"
        logger.info(
            f"[AWS] Live prices fetched for {len(live_prices)}/{len(instance_types)} instances"
        )
    except ImportError:
        logger.warning("[AWS] boto3 not installed — using catalog prices")
    except Exception as e:
        logger.warning(f"[AWS] boto3 client failed: {e} — using catalog prices")

    candidates = []
    for instance_type, (catalog_price, cat_vcpus, cat_ram, cat_gpu, cat_gpu_mem, cat_gpu_count) in _AWS_CATALOG.items():
        price = live_prices.get(instance_type, catalog_price)
        spec  = live_specs.get(instance_type, {})

        candidates.append({
            "cloud":           "aws",
            "instance_type":   instance_type,
            "region":          AWS_REGION,
            "zone":            f"{AWS_REGION}a",   # default AZ
            "price_usd_hr":    price,
            
            "gpu_model":       spec.get("gpu_model",  cat_gpu),
            "gpu_mem_gb":      spec.get("gpu_mem_gb", cat_gpu_mem),
            "gpu_count":       spec.get("gpu_count",  cat_gpu_count),
            "vcpus":           spec.get("vcpus",      cat_vcpus),
            "ram_gb":          spec.get("ram_gb",     cat_ram),
            "is_spot":         True,
            "_price_source":   price_source,
        })

    logger.info(f"[AWS] {len(candidates)} spot instances loaded ({price_source} pricing)")
    return candidates