"""
scheduler/providers/azure_provider.py
────────────────────────────────────────────────────────────────────
Fetches Azure spot VM prices using the public Retail Prices API.
No Azure SDK required — uses only requests.

AZURE CREDENTIALS YOU NEED
───────────────────────────
For SPOT PRICING (this file):
    → Nothing required. Azure Retail Prices API is public/unauthenticated.

For LAUNCHING VMs (launcher.py):
    1. portal.azure.com → Azure Active Directory → App registrations
       → New registration → name it "ml-scheduler" → Register
    2. Note: AZURE_CLIENT_ID, AZURE_TENANT_ID
    3. App registrations → Certificates & secrets → New client secret
       AZURE_CLIENT_SECRET = the secret value
    4. Subscription → Access control (IAM) → Add "Contributor" role
    5. AZURE_SUBSCRIPTION_ID = your subscription ID
    6. AZURE_LOCATION = your preferred region (e.g. eastus)
"""

from __future__ import annotations
import os
import logging

logger = logging.getLogger(__name__)

AZURE_LOCATION = os.getenv("AZURE_LOCATION", "eastus")

_RETAIL_API = "https://prices.azure.com/api/retail/prices"

# ── Instance catalog ─────────────────────────────────────────────
# (instance_type, vcpus, ram_gb, gpu_model, gpu_mem_gb, gpu_count)
_AZURE_CATALOG: dict[str, tuple] = {
    # CPU — D-series
    "Standard_D2s_v3":       (0.024,  2,   8,  None,  0, 0),
    "Standard_D4s_v3":       (0.048,  4,  16,  None,  0, 0),
    "Standard_D8s_v3":       (0.096,  8,  32,  None,  0, 0),
    "Standard_D16s_v3":      (0.192, 16,  64,  None,  0, 0),
    # CPU — F-series (compute optimized)
    "Standard_F4s_v2":       (0.042,  4,   8,  None,  0, 0),
    "Standard_F8s_v2":       (0.084,  8,  16,  None,  0, 0),
    # GPU — NC-series (T4)
    "Standard_NC4as_T4_v3":  (0.132,  4,  28, "T4",   16, 1),
    "Standard_NC8as_T4_v3":  (0.264,  8,  56, "T4",   16, 1),
    "Standard_NC16as_T4_v3": (0.528, 16, 110, "T4",   16, 1),
    # GPU — NC-series (V100)
    "Standard_NC6s_v3":      (0.756,  6, 112, "V100", 16, 1),
    "Standard_NC12s_v3":     (1.512, 12, 224, "V100", 16, 2),
    # GPU — ND-series (A100)
    "Standard_ND96asr_v4":   (14.418, 96, 900, "A100", 80, 8),
}

_AZURE_RISK: dict[str, float] = {
    "Standard_D2s_v3":       0.05,
    "Standard_D4s_v3":       0.06,
    "Standard_D8s_v3":       0.07,
    "Standard_D16s_v3":      0.08,
    "Standard_F4s_v2":       0.07,
    "Standard_F8s_v2":       0.08,
    "Standard_NC4as_T4_v3":  0.13,
    "Standard_NC8as_T4_v3":  0.14,
    "Standard_NC16as_T4_v3": 0.15,
    "Standard_NC6s_v3":      0.18,
    "Standard_NC12s_v3":     0.20,
    "Standard_ND96asr_v4":   0.28,
}

# Map Azure location string → region key used in selector's carbon table
_LOCATION_TO_REGION = {
    "eastus":        "eastus",
    "eastus2":       "eastus2",
    "westus":        "westus",
    "westus2":       "westus2",
    "westus3":       "westus3",
    "northeurope":   "northeurope",
    "westeurope":    "westeurope",
    "swedencentral": "swedencentral",
    "southeastasia": "southeastasia",
    "japaneast":     "japaneast",
}


def _fetch_live_spot_prices(instance_types: list[str]) -> dict[str, float]:
    """
    Fetch Azure spot prices from the public Retail Prices API.
    No auth required.
    Returns dict of instance_type → spot_price_usd_hr.
    """
    prices = {}
    try:
        import requests

        filter_str = (
            f"armRegionName eq '{AZURE_LOCATION}' "
            f"and serviceName eq 'Virtual Machines' "
            f"and contains(skuName, 'Spot')"
        )
        params = {
            "api-version": "2023-01-01-preview",
            "$filter":     filter_str,
        }
        resp  = requests.get(_RETAIL_API, params=params, timeout=10)
        resp.raise_for_status()
        items = resp.json().get("Items", [])

        for item in items:
            sku   = item.get("skuName", "").replace(" Spot", "").strip()
            price = float(item.get("retailPrice", 0))
            for itype in instance_types:
                normalized = itype.replace("Standard_", "").replace("_", " ")
                if normalized.lower() in sku.lower() and price > 0:
                    if itype not in prices or price < prices[itype]:
                        prices[itype] = price

        logger.info(f"[Azure] Retail API returned {len(items)} SKUs, matched {len(prices)} instances")
    except ImportError:
        logger.warning("[Azure] requests not installed — using catalog prices")
    except Exception as e:
        logger.warning(f"[Azure] Retail Prices API failed: {e} — using catalog prices")
    return prices


def list_instances() -> list[dict]:
    """
    Return list of Azure spot instance candidates for selector.py.
    Attempts live pricing via public Retail API (no auth needed).
    Falls back to catalog on failure.
    """
    instance_types = list(_AZURE_CATALOG.keys())
    live_prices    = _fetch_live_spot_prices(instance_types)
    price_source   = "live" if live_prices else "catalog"
    region         = _LOCATION_TO_REGION.get(AZURE_LOCATION, AZURE_LOCATION)

    candidates = []
    for instance_type, (catalog_price, vcpus, ram_gb, gpu_model, gpu_mem_gb, gpu_count) in _AZURE_CATALOG.items():
        price = live_prices.get(instance_type, catalog_price)
        risk  = _AZURE_RISK.get(instance_type, 0.12)

        candidates.append({
            "cloud":           "azure",
            "instance_type":   instance_type,
            "region":          region,
            "zone":            f"{AZURE_LOCATION}-1",
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

    logger.info(f"[Azure] {len(candidates)} spot instances loaded ({price_source} pricing)")
    return candidates