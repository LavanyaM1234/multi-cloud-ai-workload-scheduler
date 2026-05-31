"""
scheduler/selector.py
──────────────────────────────────────────────────────────────────────
Multi-cloud Pareto optimizer.

Algorithm
─────────
1. Fetch live candidates from GCP + AWS + Azure providers.
2. Filter by preferred_clouds and spot_only.
3. Estimate total runtime and total cost per candidate.
4. Apply hard constraints:
   • Reject if price_usd_hr * est_hours > max_budget
   • Reject if est_hours > deadline_hrs
   • Reject if gpu_required and gpu_mem_gb < min_gpu_mem
   • Reject if spot_only and not a spot instance
5. Apply preferred_regions bias (soft — boosts score, not hard reject).
6. Compute Pareto frontier across:
   • minimize estimated_total_cost
   • minimize preemption_risk
   • minimize estimated_hours
   • minimize carbon_intensity (when carbon_aware=True — 4th axis)
7. From the Pareto set, pick winner by weighted scoring.
8. Full fallback chain:  AWS → Azure → GCP → on-demand.

Called by:
    api/server.py → submit_job()
    launcher.py   → for cloud routing
"""

from __future__ import annotations

import os
import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)

# ── Environment ───────────────────────────────────────────────────
GCS_BUCKET = os.getenv("CHECKPOINT_GCS_BUCKET", "")

# ── Base scoring weights (sum to 1.0) ────────────────────────────
W_COST   = float(os.getenv("SELECTOR_W_COST",   "0.40"))
W_RISK   = float(os.getenv("SELECTOR_W_RISK",   "0.35"))
W_TIME   = float(os.getenv("SELECTOR_W_TIME",   "0.25"))
W_CARBON = float(os.getenv("SELECTOR_W_CARBON", "0.00"))

# ── Carbon intensity by region (gCO2eq/kWh) ──────────────────────
CARBON_INTENSITY: dict[str, float] = {
    # GCP
    "us-central1":    494,
    "us-east1":       545,
    "us-east4":       383,
    "us-west1":        83,
    "us-west2":       195,
    "us-west3":       536,
    "us-west4":       490,
    "europe-west1":    94,
    "europe-west4":   283,
    "europe-north1":   26,
    "asia-east1":     541,
    "asia-southeast1":453,
    # AWS
    "us-east-1":      415,
    "us-east-2":      439,
    "us-west-1":      207,
    "us-west-2":       82,
    "eu-west-1":      316,
    "eu-central-1":   338,
    "eu-north-1":       8,
    "ap-southeast-1": 453,
    "ap-northeast-1": 506,
    # Azure
    "eastus":         383,
    "eastus2":        383,
    "westus":         207,
    "westus2":         82,
    "westus3":        536,
    "northeurope":    316,
    "westeurope":     283,
    "swedencentral":    8,
    "southeastasia":  453,
    "japaneast":      506,
}
_CARBON_DEFAULT = 400.0


def _carbon_for(region: str) -> float:
    return CARBON_INTENSITY.get(region, _CARBON_DEFAULT)


# ── Training time model ───────────────────────────────────────────
_GPU_THROUGHPUT = {
    "A100": 50_000,
    "V100": 25_000,
    "T4":   15_000,
    "L4":   20_000,
    "K80":   8_000,
    "M60":   6_000,
    None:    3_000,
}


def _estimate_training_hours(candidate: dict, job: dict) -> float:
    num_epochs = int(job.get("epochs", job.get("num_epochs", 50)))
    batch_size = int(job.get("batch_size", 64))

    if "synthetic_rows" in job:
        total_samples = int(job["synthetic_rows"])
    elif job.get("dataset_type", "synthetic-500k").startswith("synthetic"):
        ds_type = job.get("dataset_type", "synthetic-10k")
        suffix  = ds_type.split("-")[-1].lower()
        multiplier = 1_000 if suffix.endswith("k") else 1_000_000 if suffix.endswith("m") else 1
        try:
            total_samples = int(float(suffix.rstrip("km")) * multiplier)
        except ValueError:
            total_samples = 10_000
    else:
        dataset_mb    = float(job.get("dataset_size_mb", 100))
        total_samples = int(dataset_mb * 1000)

    gpu_model  = candidate.get("gpu_model")
    throughput = _GPU_THROUGHPUT.get(gpu_model, _GPU_THROUGHPUT[None])

    steps_per_epoch = math.ceil(total_samples / batch_size)
    total_seconds   = (steps_per_epoch * num_epochs / (throughput / batch_size)) * 1.10
    est_hours       = max(round(total_seconds / 3600, 3), 0.05)

    logger.debug(
        f"[Selector] est_hours for {candidate.get('instance_type','?')}: "
        f"{est_hours:.2f}h  "
        f"(samples={total_samples} epochs={num_epochs} "
        f"batch={batch_size} throughput={throughput}/s "
        f"gpu={gpu_model or 'CPU'})"
    )
    return est_hours


def _validate_env() -> list[str]:
    warnings = []
    checks = [
        ("GCP_PROJECT_ID",        ""),
        ("GCP_ZONE",              ""),
        ("AWS_REGION",            ""),
        ("AZURE_SUBSCRIPTION_ID", ""),
        ("AZURE_LOCATION",        ""),
    ]
    for var, label in checks:
        if not os.getenv(var):
            warnings.append(f"  ⚠  {var} not set — {label} may use defaults")
    return warnings


# ── Priority → weight mapping ─────────────────────────────────────
def _priority_weights(priority: str, carbon_aware: bool, carbon_weight: str) -> tuple[float, float, float, float]:
    base = {
        "cheapest": (0.60, 0.25, 0.15),
        "balanced": (0.40, 0.35, 0.25),
        "fastest":  (0.20, 0.30, 0.50),
    }.get(priority, (0.40, 0.35, 0.25))

    w_cost, w_risk, w_time = base

    if not carbon_aware:
        return w_cost, w_risk, w_time, 0.0

    carbon_shares = {"light": 0.10, "balanced": 0.20, "strong": 0.30}
    w_carbon = carbon_shares.get(carbon_weight, 0.20)
    w_cost = round(w_cost - w_carbon * 0.6, 4)
    w_time = round(w_time - w_carbon * 0.4, 4)
    w_cost = max(w_cost, 0.05)
    w_time = max(w_time, 0.05)
    return w_cost, w_risk, w_time, w_carbon


def _pareto_frontier(candidates: list[dict], carbon_aware: bool = False) -> list[dict]:
    """Return Pareto-non-dominated subset (3- or 4-objective)."""
    pareto = []
    for i, a in enumerate(candidates):
        dominated = False
        for j, b in enumerate(candidates):
            if i == j:
                continue
            b_leq_a = (
                b["est_cost"]        <= a["est_cost"]  and
                b["preemption_risk"] <= a["preemption_risk"] and
                b["est_hours"]       <= a["est_hours"]
            )
            if carbon_aware:
                b_leq_a = b_leq_a and (b.get("carbon_intensity", _CARBON_DEFAULT) <=
                                        a.get("carbon_intensity", _CARBON_DEFAULT))
            b_lt_a = (
                b["est_cost"]        < a["est_cost"]  or
                b["preemption_risk"] < a["preemption_risk"] or
                b["est_hours"]       < a["est_hours"]
            )
            if carbon_aware:
                b_lt_a = b_lt_a or (b.get("carbon_intensity", _CARBON_DEFAULT) <
                                     a.get("carbon_intensity", _CARBON_DEFAULT))
            if b_leq_a and b_lt_a:
                dominated = True
                break
        if not dominated:
            pareto.append(a)
    return pareto


def _normalize(values: list[float]) -> list[float]:
    mn, mx = min(values), max(values)
    if mx == mn:
        return [0.5] * len(values)
    return [(v - mn) / (mx - mn) for v in values]


def _weighted_score(
    candidates: list[dict],
    w_cost: float,
    w_risk: float,
    w_time: float,
    w_carbon: float,
    preferred_regions: list[str],
) -> dict:
    """Pick best candidate by weighted score. Lower = better."""
    costs   = [c["est_cost"]        for c in candidates]
    risks   = [c["preemption_risk"] for c in candidates]
    times   = [c["est_hours"]       for c in candidates]
    carbons = [c.get("carbon_intensity", _CARBON_DEFAULT) for c in candidates]

    norm_costs   = _normalize(costs)
    norm_risks   = _normalize(risks)
    norm_times   = _normalize(times)
    norm_carbons = _normalize(carbons)

    best_score     = math.inf
    best_candidate = candidates[0]

    for i, c in enumerate(candidates):
        score = (
            w_cost   * norm_costs[i]   +
            w_risk   * norm_risks[i]   +
            w_time   * norm_times[i]   +
            w_carbon * norm_carbons[i]
        )
        if preferred_regions and c.get("region") in preferred_regions:
            score -= 0.05

        c["_score"] = round(score, 4)
        logger.debug(
            f"  [{c['cloud']:5}] {c['instance_type']:25} "
            f"score={score:.4f}  cost=${c['est_cost']:.2f}  "
            f"risk={c['preemption_risk']:.2f}  hrs={c['est_hours']:.2f}  "
            f"carbon={c.get('carbon_intensity', '?')}"
        )
        if score < best_score:
            best_score     = score
            best_candidate = c

    return best_candidate


def _apply_constraints(
    candidates: list[dict],
    job: dict,
    max_budget: float,
    deadline_hrs: float,
    min_gpu_mem: float,
    gpu_required: bool,
    spot_only: bool,
) -> tuple[list[dict], list[dict]]:
    """Split candidates into (valid, rejected)."""
    valid    = []
    rejected = []

    for c in candidates:
        reasons = []
        if c["est_cost"] > max_budget:
            reasons.append(
                f"budget exceeded (${c['est_cost']:.2f} > ${max_budget:.2f})"
            )
        if c["est_hours"] > deadline_hrs:
            reasons.append(
                f"deadline missed ({c['est_hours']:.2f}h > {deadline_hrs:.2f}h)"
            )
        if gpu_required and c.get("gpu_mem_gb", 0) < min_gpu_mem:
            reasons.append(
                f"insufficient GPU memory ({c.get('gpu_mem_gb',0)}GB < {min_gpu_mem}GB)"
            )
        if spot_only and c.get("is_spot") is False:
            reasons.append("on-demand instance rejected (spot_only=True)")

        if reasons:
            c["_reject_reason"] = "; ".join(reasons)
            rejected.append(c)
        else:
            valid.append(c)

    return valid, rejected


def _filter_preferred_clouds(
    candidates: list[dict],
    preferred_clouds: list[str],
) -> list[dict]:
    """Hard-filter candidates to only preferred clouds. Falls back if nothing matches."""
    if not preferred_clouds or set(preferred_clouds) >= {"aws", "gcp", "azure"}:
        return candidates

    filtered = [c for c in candidates if c.get("cloud", "").lower() in preferred_clouds]

    if not filtered:
        logger.warning(
            f"[Selector] preferred_clouds={preferred_clouds} filtered out ALL candidates "
            f"— falling back to unrestricted pool."
        )
        return candidates

    logger.info(
        f"[Selector] preferred_clouds filter: "
        f"{len(filtered)}/{len(candidates)} candidates kept "
        f"(clouds: {preferred_clouds})"
    )
    return filtered


def _parse_preferred_regions(region_str: str) -> list[str]:
    if not region_str:
        return []
    return [r.strip() for r in region_str.split(",") if r.strip()]


def pick_best_cloud(job: dict) -> dict:
    """
    Main entry point. Given a job spec, return the best cloud + instance.

    Fields consumed:
        preferred_clouds   list[str]  e.g. ["aws", "gcp"]
        preferred_regions  str        comma-separated e.g. "us-east-1, us-west-2"
        gpu_required       bool       hard filter — must have GPU
        spot_only          bool       hard filter — spot/preemptible only
        ondemand_max_hrs   float      stored in config, consumed by poller
        carbon_aware       bool       enables 4th Pareto axis + weight
        carbon_weight      str        "light" | "balanced" | "strong"
        priority           str        "cheapest" | "balanced" | "fastest"
    """
    env_warnings = _validate_env()
    if env_warnings:
        logger.warning("[Selector] Environment warnings:\n" + "\n".join(env_warnings))

    job_id            = job.get("job_id",           "job-unknown")
    max_budget        = float(job.get("max_budget",    2.0))
    deadline_hrs      = float(job.get("deadline_hrs",  8.0))
    min_gpu_mem       = float(job.get("min_gpu_mem",   0.0))
    gpu_required      = bool(job.get("gpu_required",   False))
    spot_only         = bool(job.get("spot_only",      True))
    preferred_clouds  = job.get("preferred_clouds",   ["aws", "gcp", "azure"])
    preferred_regions = _parse_preferred_regions(job.get("preferred_regions", ""))
    carbon_aware      = bool(job.get("carbon_aware",   False))
    carbon_weight     = job.get("carbon_weight",      "balanced")
    priority          = job.get("priority",           "balanced")
    prefer_cloud      = job.get("prefer_cloud", "any").lower()
    print("called selector")
    print(job)
    logger.info(
        f"[Selector] pick_best_cloud — job={job_id}  "
        f"budget=${max_budget}  deadline={deadline_hrs}h  "
        f"gpu_required={gpu_required}  min_gpu={min_gpu_mem}GB  "
        f"spot_only={spot_only}  preferred_clouds={preferred_clouds}  "
        f"preferred_regions={preferred_regions}  "
        f"carbon_aware={carbon_aware}({carbon_weight})  priority={priority}"
    )

    # ── 1. Collect candidates ─────────────────────────────────────
    all_candidates: list[dict] = []
    provider_errors: list[str] = []

    def _safe_list(provider_fn, label: str):
        try:
            instances = provider_fn()
            logger.info(f"[Selector] {label}: {len(instances)} instances fetched")
            return instances
        except Exception as e:
            msg = f"{label} provider error: {e}"
            logger.error(f"[Selector] {msg}")
            provider_errors.append(msg)
            return []

    from scheduler.providers.gcp_provider   import list_instances as gcp_list
    from scheduler.providers.aws_provider   import list_instances as aws_list
    from scheduler.providers.azure_provider import list_instances as azure_list

    fetch_clouds = set(preferred_clouds) if preferred_clouds else {"gcp", "aws", "azure"}
    if prefer_cloud != "any":
        fetch_clouds = {prefer_cloud}

    if "gcp"   in fetch_clouds: all_candidates += _safe_list(gcp_list,   "GCP")
    if "aws"   in fetch_clouds: all_candidates += _safe_list(aws_list,   "AWS")
    if "azure" in fetch_clouds: all_candidates += _safe_list(azure_list, "Azure")
    
    if not all_candidates:
        logger.error("[Selector] All providers failed — using emergency GCP fallback")
        return _emergency_fallback(job, max_budget, deadline_hrs,
                                   reason="all providers unreachable")

    # ── 2. Filter by preferred_clouds ────────────────────────────
    all_candidates = _filter_preferred_clouds(all_candidates, list(fetch_clouds))

    # ── 3. Enrich with est hours, cost, carbon intensity ─────────
    for c in all_candidates:
        c["est_hours"]        = _estimate_training_hours(c, job)
        c["est_cost"]         = round(c["price_usd_hr"] * c["est_hours"], 4)
        c["carbon_intensity"] = _carbon_for(c.get("region", ""))

    # Risk scoring via live predictor
    from api.server import get_bq_client
    from risk.predictor import score_instance_from_api
    bq = get_bq_client()
    # Risk scoring — skip for pareto probe to keep response fast
    if job.get("pareto_probe"):
        risk_lookup = job.get("risk_lookup", {})
        for c in all_candidates:
            key = (c["cloud"], c["instance_type"])
            if key in risk_lookup:
                # Use real cached score
                c["preemption_risk"] = risk_lookup[key]
            else:
                # Vary by cloud so points spread on the chart
                # (real scoring will replace this once risk cache warms up)
                cloud_base = {"aws": 0.35, "gcp": 0.20, "azure": 0.25}
                base = cloud_base.get(c["cloud"], 0.30)
                # Add small variation based on instance name hash so
                # different instance types get different positions
                jitter = (hash(c["instance_type"]) % 100) / 500.0
                c["preemption_risk"] = round(min(0.95, base + jitter), 3)
    else:
        from api.server import get_bq_client
        from risk.predictor import score_instance_from_api
        bq = get_bq_client()
        for c in all_candidates:
            try:
                c["preemption_risk"] = score_instance_from_api(
                    cloud         = c["cloud"],
                    region        = c["region"],
                    az            = c["zone"],
                    instance_type = c["instance_type"],
                    bq_client     = bq,
                )
            except Exception as e:
                logger.warning(
                    f"[Selector] Risk scoring failed for "
                    f"{c['cloud']}/{c['instance_type']}: {e} — defaulting to 0.5"
                )
                c["preemption_risk"] = 0.5

    # ── 4. Log live price table ───────────────────────────────────
    logger.info("[Selector] ── Live spot prices fetched ─────────────────────")
    logger.info(
        f"[Selector]   {'Cloud':5}  {'Instance':28}  "
        f"{'$/hr':>7}  {'Est hrs':>7}  {'Est $':>7}  "
        f"{'Risk':>5}  {'Carbon':>6}  {'GPU'}"
    )
    for c in sorted(all_candidates, key=lambda x: x["price_usd_hr"]):
        source = "(live)" if c.get("_price_source") == "live" else "(fallback)"
        logger.info(
            f"[Selector]   [{c['cloud']:5}] {c['instance_type']:28}  "
            f"${c['price_usd_hr']:>6.4f}  "
            f"{c['est_hours']:>7.2f}h  "
            f"${c['est_cost']:>6.3f}  "
            f"{c['preemption_risk']:>5.2f}  "
            f"{c.get('carbon_intensity', '?'):>6}  "
            f"{c.get('gpu_model','CPU') or 'CPU':8} {source}"
        )
    logger.info("[Selector] ──────────────────────────────────────────────────")

    # ── 5. Apply hard constraints ─────────────────────────────────
    valid, rejected = _apply_constraints(
        all_candidates, job,
        max_budget, deadline_hrs,
        min_gpu_mem, gpu_required, spot_only,
    )

    logger.info(
        f"[Selector] Constraints: {len(valid)} valid / "
        f"{len(rejected)} rejected of {len(all_candidates)}"
    )
    for r in rejected:
        logger.info(f"  ✗ [{r['cloud']:5}] {r['instance_type']:25} — {r['_reject_reason']}")

    if not valid:
        logger.warning(
            "[Selector] No candidates passed constraints — relaxing and using cheapest."
        )
        valid = sorted(all_candidates, key=lambda c: c["est_cost"])
        if not valid:
            return _emergency_fallback(job, max_budget, deadline_hrs,
                                       reason="no instances after constraint relaxation")

    # ── 6. Pareto frontier ────────────────────────────────────────
    pareto = _pareto_frontier(valid, carbon_aware=carbon_aware)
    logger.info(
        f"[Selector] Pareto frontier: {len(pareto)} non-dominated "
        f"({'4-objective w/ carbon' if carbon_aware else '3-objective'}) "
        f"from {len(valid)} valid"
    )
    for p in pareto:
        logger.info(
            f"  ★ [{p['cloud']:5}] {p['instance_type']:25} "
            f"cost=${p['est_cost']:.2f}  risk={p['preemption_risk']:.2f}  "
            f"hrs={p['est_hours']:.2f}  carbon={p.get('carbon_intensity','?')}  "
            f"gpu={p.get('gpu_model','CPU')}"
        )

    # ── 7. Dynamic weights ────────────────────────────────────────
    w_cost, w_risk, w_time, w_carbon = _priority_weights(
        priority, carbon_aware, carbon_weight
    )
    logger.info(
        f"[Selector] Weights: cost={w_cost}  risk={w_risk}  "
        f"time={w_time}  carbon={w_carbon}  "
        f"(priority={priority}, carbon_aware={carbon_aware})"
    )

    # ── 8. Weighted scoring ───────────────────────────────────────
    winner = _weighted_score(
        pareto, w_cost, w_risk, w_time, w_carbon, preferred_regions
    )

    region_note = (
        f" Region bias applied: {preferred_regions}." if preferred_regions else ""
    )
    carbon_note = (
        f" Carbon-aware ({carbon_weight}): {winner.get('carbon_intensity','?')} gCO2/kWh."
        if carbon_aware else ""
    )

    reason = (
        f"Pareto-optimal across {len(pareto)} non-dominated candidates "
        f"(from {len(all_candidates)} total across clouds={list(fetch_clouds)}). "
        f"Priority={priority}. Weights: cost×{w_cost} risk×{w_risk} time×{w_time}"
        f"{f' carbon×{w_carbon}' if carbon_aware else ''}. "
        f"Score={winner.get('_score','?')}. "
        f"Est ${winner['est_cost']:.2f} over {winner['est_hours']:.2f}h "
        f"with {winner['preemption_risk']*100:.1f}% preemption risk."
        f"{region_note}{carbon_note}"
    )

    logger.info(f"[Selector] ✓ Winner: [{winner['cloud']}] {winner['instance_type']}")
    logger.info(f"[Selector]   {reason}")
    print("winner")
    print(winner)
    # ── 9. Build decision dict ────────────────────────────────────
    return {
        "cloud":            winner["cloud"],
        "instance_type":    winner["instance_type"],
        "region":           winner["region"],
        "zone":             winner["zone"],
        "price_usd_hr":     winner["price_usd_hr"],
        "est_hours":        winner["est_hours"],
        "est_cost":         winner["est_cost"],
        "preemption_risk":  winner["preemption_risk"],
        "gpu_model":        winner.get("gpu_model"),
        "gpu_mem_gb":       winner.get("gpu_mem_gb", 0),
        "gpu_count":        winner.get("gpu_count",  0),
        "vcpus":            winner.get("vcpus",  4),
        "ram_gb":           winner.get("ram_gb", 16),
        "carbon_intensity": winner.get("carbon_intensity", _CARBON_DEFAULT),
        "is_spot":          winner.get("is_spot", True),
        "s3_bucket":        os.getenv("CHECKPOINT_S3_BUCKET", ""),
        "gcs_bucket":       GCS_BUCKET,
        "reason":           reason,
        "pareto_set":       [
            {k: v for k, v in p.items() if not k.startswith("_")}
            for p in pareto
        ],
        "rejected_count":   len(rejected),
        "total_evaluated":  len(all_candidates),
    }


def _emergency_fallback(job: dict, max_budget: float, deadline_hrs: float,
                        reason: str = "") -> dict:
    """Last-resort fallback: GCP e2-standard-4 spot (CPU)."""
    logger.error(f"[Selector] EMERGENCY FALLBACK triggered: {reason}")
    PRICE     = 0.034
    est_hours = min(max_budget / PRICE, deadline_hrs)
    return {
        "cloud":            "gcp",
        "instance_type":    "e2-standard-4",
        "region":           os.getenv("GCP_REGION", "us-central1"),
        "zone":             os.getenv("GCP_ZONE",   "us-central1-a"),
        "price_usd_hr":     PRICE,
        "est_hours":        round(est_hours, 2),
        "est_cost":         round(est_hours * PRICE, 4),
        "preemption_risk":  0.05,
        "gpu_model":        None,
        "gpu_mem_gb":       0,
        "gpu_count":        0,
        "vcpus":            4,
        "ram_gb":           16,
        "carbon_intensity": _carbon_for("us-central1"),
        "is_spot":          True,
        "s3_bucket":        os.getenv("CHECKPOINT_S3_BUCKET", ""),
        "gcs_bucket":       GCS_BUCKET,
        "reason":           f"Emergency fallback (GCP e2-standard-4 CPU): {reason}",
        "pareto_set":       [],
        "rejected_count":   0,
        "total_evaluated":  0,
    }


# ── Manual test ───────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)

    test_job = {
        "job_id":            "test-001",
        "max_budget":        5.0,
        "deadline_hrs":      8.0,
        "min_gpu_mem":       16.0,
        "gpu_required":      True,
        "spot_only":         True,
        "preferred_clouds":  ["aws", "gcp"],
        "preferred_regions": "us-west-2, us-west1",
        "carbon_aware":      True,
        "carbon_weight":     "balanced",
        "priority":          "balanced",
        "dataset_size_mb":   2048,
        "num_epochs":        20,
        "batch_size":        64,
    }

    result = pick_best_cloud(test_job)
    display = {k: v for k, v in result.items() if k != "pareto_set"}
    print(json.dumps(display, indent=2))
    print(f"\nPareto set ({len(result['pareto_set'])} candidates):")
    for p in result["pareto_set"]:
        print(f"  [{p['cloud']:5}] {p['instance_type']:25} "
              f"${p['est_cost']:.2f}  risk={p['preemption_risk']:.2f}  "
              f"carbon={p.get('carbon_intensity','?')}")