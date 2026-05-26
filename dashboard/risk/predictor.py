"""
risk/predictor.py
──────────────────
Loads the trained LSTM + XGBoost hybrid model and scores
preemption risk for any (cloud, region, az, instance_type).

Model details (from hybrid_meta.json):
  - Type:        LSTM + XGBoost Hybrid Regression
  - Output:      continuous risk score 0.0 → 1.0
  - Sequence:    last 60 timesteps per cloud/region/az
  - Features:    29 (see feature_cols.pkl)
  - Test R²:     0.992
  - Test RMSE:   0.00123

LSTM architecture (must match train_hybrid_model.py):
  LSTMFeatureExtractor(input_size=29, hidden=64, layers=2, out_dim=8)
  with attention mechanism + compress head
"""

import os
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Singleton cache ────────────────────────────────────────────────
_lstm      = None
_xgb       = None
_scaler    = None
_feat_cols = None
_encoders  = None
_meta      = None

_HERE        = Path(__file__).parent
PROJECT_ROOT = _HERE.parent
MODELS_DIR     = PROJECT_ROOT / "models"
MODEL_DATA_DIR = PROJECT_ROOT / "model_data"

SEQUENCE_LEN = 60    # from hybrid_meta.json
LSTM_OUT_DIM = 8     # from hybrid_meta.json


# ══════════════════════════════════════════════════════════════════
# LSTM ARCHITECTURE — must match train_hybrid_model.py exactly
# ══════════════════════════════════════════════════════════════════

def _build_lstm(n_features: int):
    import torch.nn as nn

    class LSTMFeatureExtractor(nn.Module):
        def __init__(self, input_size, hidden=64, layers=2, out_dim=LSTM_OUT_DIM):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size, hidden_size=hidden,
                num_layers=layers, dropout=0.3, batch_first=True
            )
            self.attn = nn.Sequential(
                nn.Linear(hidden, 32), nn.Tanh(), nn.Linear(32, 1)
            )
            self.compress = nn.Sequential(
                nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(32, out_dim), nn.Tanh(),
            )
            self.head = nn.Sequential(
                nn.Linear(out_dim, 1)
            )

        def forward(self, x):
            out, _   = self.lstm(x)
            attn_w   = __import__('torch').softmax(self.attn(out), dim=1)
            ctx      = (attn_w * out).sum(dim=1)
            features = self.compress(ctx)
            logits   = self.head(features).squeeze(-1)
            return features, logits

    return LSTMFeatureExtractor(input_size=n_features)


# ══════════════════════════════════════════════════════════════════
# LOAD
# ══════════════════════════════════════════════════════════════════

def _check_files():
    required = [
        MODELS_DIR     / "lstm_extractor.pt",
        MODELS_DIR     / "xgb_hybrid.pkl",
        MODELS_DIR     / "hybrid_meta.json",
        MODEL_DATA_DIR / "scaler.pkl",
        MODEL_DATA_DIR / "feature_cols.pkl",
        MODEL_DATA_DIR / "encoders.pkl",
    ]
    print("[risk] Checking model files:")
    missing = []
    for p in required:
        if p.exists():
            print(f"[risk]   OK  {p}  ({p.stat().st_size // 1024} KB)")
        else:
            print(f"[risk]   !!  MISSING: {p}")
            missing.append(str(p))
    if missing:
        raise FileNotFoundError(
            "[risk] Missing files:\n" + "\n".join(missing)
        )


def load_models():
    """Load all artifacts. Safe to call multiple times."""
    global _lstm, _xgb, _scaler, _feat_cols, _encoders, _meta

    if _lstm is not None:
        return

    _check_files()

    import torch
    import joblib

    # Meta
    with open(MODELS_DIR / "hybrid_meta.json") as f:
        _meta = json.load(f)
    print(f"[risk] Model: {_meta['model_type']}")
    print(f"[risk]   n_features={_meta['n_features']}  "
          f"seq_len={_meta['sequence_len']}  "
          f"R²={_meta['test_r2']:.4f}")

    # Feature cols + scaler + encoders
    _feat_cols = joblib.load(MODEL_DATA_DIR / "feature_cols.pkl")
    _scaler    = joblib.load(MODEL_DATA_DIR / "scaler.pkl")
    _encoders  = joblib.load(MODEL_DATA_DIR / "encoders.pkl")
    print(f"[risk]   {len(_feat_cols)} features: {list(_feat_cols)}")
    print(f"[risk]   Encoded cols: {list(_encoders.keys())}")

    # XGBoost — regression model, use .predict() not .predict_proba()
    _xgb = joblib.load(MODELS_DIR / "xgb_hybrid.pkl")
    print(f"[risk]   XGBoost: {type(_xgb).__name__}  "
          f"n_estimators={getattr(_xgb, 'n_estimators', '?')}")

    # LSTM
    n_features = _meta["n_features"]
    lstm_model = _build_lstm(n_features)
    state = torch.load(
        MODELS_DIR / "lstm_extractor.pt",
        map_location="cpu",
        weights_only=True,
    )
    lstm_model.load_state_dict(state)
    lstm_model.eval()
    _lstm = lstm_model

    # Smoke test
    dummy = torch.zeros(1, SEQUENCE_LEN, n_features)
    with torch.no_grad():
        feats, logit = lstm_model(dummy)
    print(f"[risk]   LSTM smoke test: feats={tuple(feats.shape)}  "
          f"logit={logit.item():.4f}")
    print(f"[risk] All models loaded\n")


# ══════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING — mirrors prepare_data.py exactly
# ══════════════════════════════════════════════════════════════════

def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reproduce every feature from prepare_data.py FEATURE_COLS.
    Column names must match exactly — including zone_enc not availability_zone_enc.
    """
    df  = df.copy()
    eps = 1e-9

    # ── Timestamp ──────────────────────────────────────────────────
    df["collected_at"] = pd.to_datetime(df["collected_at"], utc=True)
    df["hour"]              = df["collected_at"].dt.hour
    df["day_of_week"]       = df["collected_at"].dt.dayofweek
    df["is_weekend"]        = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_business_hours"] = df["hour"].between(9, 17).astype(int)

    # ── Price ratios ───────────────────────────────────────────────
    df["spot_ratio"]   = df["price_usd_per_hr"] / df["ondemand_price_usd_hr"].replace(0, eps)
    mean_price = df["price_usd_per_hr"].mean()
    if mean_price == 0:
            mean_price = eps

    df["price_vs_mean"] = df["price_usd_per_hr"] / mean_price

    # ── Sort (prepare_data groups by cloud/region/az) ──────────────
    df = df.sort_values("collected_at").reset_index(drop=True)

    # ── Lag features ───────────────────────────────────────────────
    df["price_lag_1"]  = df["price_usd_per_hr"].shift(1)
    df["price_lag_3"]  = df["price_usd_per_hr"].shift(3)
    df["price_lag_5"]  = df["price_usd_per_hr"].shift(5)
    df["price_lag_10"] = df["price_usd_per_hr"].shift(10)

    # ── Price changes ──────────────────────────────────────────────
    df["price_change_1"]  = df["price_usd_per_hr"] - df["price_lag_1"]
    df["price_pct_chg_1"] = df["price_change_1"] / df["price_lag_1"].replace(0, eps)

    # ── Rolling stats ──────────────────────────────────────────────
    df["price_roll_mean_5"]  = df["price_usd_per_hr"].rolling(5,  min_periods=1).mean()
    df["price_roll_mean_10"] = df["price_usd_per_hr"].rolling(10, min_periods=1).mean()
    df["price_roll_std_5"]   = df["price_usd_per_hr"].rolling(5,  min_periods=1).std().fillna(0)
    df["price_roll_max_5"]   = df["price_usd_per_hr"].rolling(5,  min_periods=1).max()

    # ── Preemption rate features ───────────────────────────────────
    if "preempted" in df.columns:
        p = df["preempted"].astype(float)
        df["recent_preempt_rate_5"]  = p.rolling(5,  min_periods=1).mean()
        df["recent_preempt_rate_10"] = p.rolling(10, min_periods=1).mean()
        # Consecutive preemption streak
        df["preempt_streak"] = p.groupby(
            (p != p.shift()).cumsum()
        ).cumcount().where(p == 1, 0)
    else:
        df["recent_preempt_rate_5"]  = 0.0
        df["recent_preempt_rate_10"] = 0.0
        df["preempt_streak"]         = 0

    # ── Categorical encoding ───────────────────────────────────────
    # Note: prepare_data.py uses "zone_enc" for availability_zone
    cat_map = {
        "cloud":             "cloud_enc",
        "region":            "region_enc",
        "availability_zone": "zone_enc",        # ← zone_enc not availability_zone_enc
        "instance_type":     "instance_type_enc",
        "gpu_class":         "gpu_class_enc",
    }
    for col, enc_col in cat_map.items():
        if col in df.columns and col in _encoders:
            le    = _encoders[col]
            known = set(le.classes_)
            df[enc_col] = df[col].astype(str).apply(
                lambda x: int(le.transform([x])[0]) if x in known else 0
            )
        else:
            df[enc_col] = 0

    # ── Fill NaN ───────────────────────────────────────────────────
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    return df


# ══════════════════════════════════════════════════════════════════
# FETCH + SCORE
# ══════════════════════════════════════════════════════════════════

def _fetch_rows(cloud, region, az, instance_type, bq_client) -> pd.DataFrame:
    """Fetch raw rows from BigQuery — only columns that exist in the table."""
    from google.cloud.bigquery import QueryJobConfig, ScalarQueryParameter

    PROJECT_ID = os.getenv("GCP_PROJECT_ID",  "tensile-method-459009-k2")
    DATASET    = os.getenv("BIGQUERY_DATASET", "spot_prices")
    TABLE      = os.getenv("BIGQUERY_TABLE",   "price_history")

    RAW_COLS = [
        "price_usd_per_hr", "ondemand_price_usd_hr",
        "gpu_count", "vcpu_count", "ram_gb",
        "cloud", "region", "availability_zone",
        "instance_type", "gpu_class",
        "preempted", "collected_at",
    ]

    query = f"""
        SELECT {', '.join(RAW_COLS)}
        FROM `{PROJECT_ID}.{DATASET}.{TABLE}`
        WHERE cloud             = @cloud
          AND region            = @region
          AND availability_zone = @az
          AND instance_type     = @instance_type
        ORDER BY collected_at DESC
        LIMIT {SEQUENCE_LEN + 15}
    """
    cfg  = QueryJobConfig(query_parameters=[
        ScalarQueryParameter("cloud",         "STRING", cloud),
        ScalarQueryParameter("region",        "STRING", region),
        ScalarQueryParameter("az",            "STRING", az),
        ScalarQueryParameter("instance_type", "STRING", instance_type),
    ])
    rows = list(bq_client.query(query, job_config=cfg))
    return pd.DataFrame([dict(r) for r in rows])


def score_instance(cloud, region, az, instance_type, bq_client) -> float:
    """
    Score preemption risk for one instance.
    Returns float 0.0–1.0. Returns 0.5 if not enough data.
    """
    import torch

    load_models()

    print(f"[risk] Scoring: {cloud}/{instance_type}/{region}/{az}")

    df_raw = _fetch_rows(cloud, region, az, instance_type, bq_client)
    print(f"[risk]   BQ rows: {len(df_raw)}  (need {SEQUENCE_LEN})")

    if len(df_raw) < SEQUENCE_LEN:
        print(f"[risk]   Not enough rows → returning 0.5")
        return 0.5

    # Feature engineering
    df = _engineer_features(df_raw)

    # Validate all features present
    missing = [c for c in _feat_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"[risk] Missing features: {missing}\n"
            f"Add to _engineer_features() to match prepare_data.py"
        )

    # Build arrays
    X_raw  = df[list(_feat_cols)].values.astype(np.float32)
    X_sc   = _scaler.transform(X_raw)
    X_seq  = X_sc[-SEQUENCE_LEN:]   # (60, 29) — sequence for LSTM
    X_flat = X_sc[-1]               # (29,)    — latest row for XGBoost

    print(f"[risk]   X_seq: {X_seq.shape}  price_last={df['price_usd_per_hr'].iloc[-1]:.4f}")
    print(f"[risk]   spot_ratio={df['spot_ratio'].iloc[-1]:.4f}  "
          f"preempt_rate_5={df['recent_preempt_rate_5'].iloc[-1]:.4f}")

    # LSTM → temporal features
    seq_t = torch.tensor(X_seq).unsqueeze(0)   # (1, 60, 29)
    with torch.no_grad():
        lstm_feats, lstm_logit = _lstm(seq_t)
    lstm_feats = lstm_feats.squeeze(0).numpy()  # (8,)
    print(f"[risk]   LSTM logit={lstm_logit.item():.4f}  "
          f"feats={lstm_feats.round(3).tolist()}")

    # XGBoost regression → risk score directly (no predict_proba)
    X_hybrid = np.hstack([lstm_feats, X_flat]).reshape(1, -1)  # (1, 37)
    risk     = float(_xgb.predict(X_hybrid)[0])
    risk     = float(np.clip(risk, 0.0, 1.0))   # safety clip
    print(f"[risk]   RISK SCORE = {risk:.4f}")

    return round(risk, 4)


def score_all_instances(bq_client) -> list[dict]:
    """Score every distinct active instance. Returns list sorted by risk desc."""
    
    PROJECT_ID = os.getenv("GCP_PROJECT_ID",  "tensile-method-459009-k2")
    DATASET    = os.getenv("BIGQUERY_DATASET", "spot_prices")
    TABLE      = os.getenv("BIGQUERY_TABLE",   "price_history")

    query = f"""
        SELECT DISTINCT cloud, region, availability_zone, instance_type
        FROM `{PROJECT_ID}.{DATASET}.{TABLE}`
        WHERE preempted = FALSE
          AND collected_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 3 HOUR)
        ORDER BY cloud, instance_type
    """

    print("\n[risk] Running instance discovery query...")
    instances = list(bq_client.query(query))

    print(f"[risk] Found {len(instances)} instances")

    for i, row in enumerate(instances, 1):
        print(f"[risk] Instance {i}: {dict(row)}")

    results = []

    for i, row in enumerate(instances, 1):
        try:
            print("\n" + "="*60)
            print(f"[risk] START SCORING #{i}")
            print(f"[risk] cloud         = {row['cloud']}")
            print(f"[risk] region        = {row['region']}")
            print(f"[risk] az            = {row['availability_zone']}")
            print(f"[risk] instance_type = {row['instance_type']}")

            score = score_instance(
                cloud=row["cloud"],
                region=row["region"],
                az=row["availability_zone"],
                instance_type=row["instance_type"],
                bq_client=bq_client,
            )

            print(f"[risk] FINAL SCORE = {score}")

            results.append({
                "cloud": row["cloud"],
                "region": row["region"],
                "az": row["availability_zone"],
                "instance_type": row["instance_type"],
                "risk_score": score,
            })

        except Exception as e:
            print(f"[risk] ERROR: {type(e).__name__}: {e}")

    results.sort(key=lambda x: x["risk_score"], reverse=True)

    print("\n[risk] FINAL SORTED RESULTS")
    for r in results:
        print(r)

    return results




def _fetch_live_prices(cloud: str, region: str, az: str,
                        instance_type: str) -> pd.DataFrame:
    """
    Fetch live spot price history from cloud APIs.
    Returns a DataFrame with the same columns as BigQuery would.
    """
    rows = []

    if cloud == "aws":
        import boto3
        from datetime import datetime, timezone, timedelta

        ec2 = boto3.client("ec2", region_name=region)

        if not az:
            zones_resp = ec2.describe_availability_zones(
                Filters=[{"Name": "region-name", "Values": [region]}]
            )

            zones = [
                z["ZoneName"]
                for z in zones_resp["AvailabilityZones"]
                if z["State"] == "available"
            ]

            az = zones[0] if zones else None
            print(f"[risk]   Auto-selected AZ: {az}")

        params = {
            "InstanceTypes": [instance_type],
            "ProductDescriptions": ["Linux/UNIX"],
            "StartTime": datetime.now(timezone.utc) - timedelta(hours=1080),#45 days
            "MaxResults": 100,
        }

        if az:
            params["AvailabilityZone"] = az

        resp = ec2.describe_spot_price_history(**params)

        print(f"[risk]   Retrieved {len(resp.get('SpotPriceHistory', []))} spot records")

        for entry in resp.get("SpotPriceHistory", []):
            rows.append({
                "price_usd_per_hr":      float(entry["SpotPrice"]),
                "ondemand_price_usd_hr": _get_aws_ondemand(instance_type),
                "gpu_count":             0,
                "vcpu_count":            0,
                "ram_gb":                0.0,
                "cloud":                 "aws",
                "region":                region,
                "availability_zone":     entry.get("AvailabilityZone", az),
                "instance_type":         instance_type,
                "gpu_class":             "none",
                "preempted":             False,
                "collected_at":          entry["Timestamp"],
            })


    elif cloud == "gcp":
        # GCP preemptible prices are quasi-static — generate synthetic history
        # from the known price with small noise to fill the sequence
        import random
        base_price = _get_gcp_preemptible_price(instance_type, region)
        now        = pd.Timestamp.now(tz="UTC")
        for i in range(SEQUENCE_LEN + 10):
            noise = random.uniform(-0.002, 0.002)
            rows.append({
                "price_usd_per_hr":      round(base_price + noise, 4),
                "ondemand_price_usd_hr": base_price * 4,
                "gpu_count":             0,
                "vcpu_count":            0,
                "ram_gb":                0.0,
                "cloud":                 "gcp",
                "region":                region,
                "availability_zone":     az,
                "instance_type":         instance_type,
                "gpu_class":             "none",
                "preempted":             False,
                "collected_at":          now - pd.Timedelta(minutes=i),
            })

    elif cloud == "azure":
        # Azure spot prices via REST API
        price = _get_azure_spot_price(instance_type, region)
        now   = pd.Timestamp.now(tz="UTC")
        for i in range(SEQUENCE_LEN + 10):
            rows.append({
                "price_usd_per_hr":      price,
                "ondemand_price_usd_hr": price * 3,
                "gpu_count":             0,
                "vcpu_count":            0,
                "ram_gb":                0.0,
                "cloud":                 "azure",
                "region":                region,
                "availability_zone":     az,
                "instance_type":         instance_type,
                "gpu_class":             "none",
                "preempted":             False,
                "collected_at":          now - pd.Timedelta(minutes=i),
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["collected_at"] = pd.to_datetime(df["collected_at"], utc=True)
    df = df.sort_values("collected_at").reset_index(drop=True)
    return df


def _get_aws_ondemand(instance_type: str) -> float:
    """Approximate on-demand prices for discount ratio calculation."""
    prices = {
        "t3.small":     0.0208,
        "t3.medium":    0.0416,
        "g4dn.xlarge":  0.526,
        "g4dn.2xlarge": 0.752,
        "g5.xlarge":    1.006,
        "p3.2xlarge":   3.060,
    }
    return prices.get(instance_type, 0.10)


def _get_gcp_preemptible_price(instance_type: str, region: str) -> float:
    """GCP preemptible prices (quasi-static)."""
    prices = {
        "e2-standard-4":  0.067,
        "g2-standard-4":  0.2818,
        "a2-highgpu-1g":  0.75,
        "n1-standard-4":  0.048,
    }
    return prices.get(instance_type, 0.05)


def _get_azure_spot_price(instance_type: str, region: str) -> float:
    """
    Fetch Azure spot price from Retail Prices API.
    Falls back to approximate if API fails.
    """
    try:
        import requests
        url    = "https://prices.azure.com/api/retail/prices"
        params = {
            "$filter": (
                f"serviceName eq 'Virtual Machines' and "
                f"priceType eq 'Spot' and "
                f"armRegionName eq '{region}' and "
                f"armSkuName eq '{instance_type}'"
            )
        }
        resp  = requests.get(url, params=params, timeout=5)
        items = resp.json().get("Items", [])
        if items:
            return float(items[0]["retailPrice"])
    except Exception:
        pass
    # Fallback approximations
    fallbacks = {
        "Standard_NC4as_T4_v3": 0.21,
        "Standard_NC8as_T4_v3": 0.36,
    }
    return fallbacks.get(instance_type, 0.15)


def score_instance_from_api(
    cloud:         str,
    region:        str,
    az:            str,
    instance_type: str,
    bq_client=None,       # ← add this
) -> float:
    import torch

    load_models()

    print(f"[risk]   Fetching live prices: {cloud}/{instance_type}/{region}")
    df_raw = _fetch_live_prices(cloud, region, az, instance_type)

    # ── Fallback to BQ if live API didn't return enough ───────────
    if (df_raw.empty or len(df_raw) < SEQUENCE_LEN) and bq_client is not None:
        print(f"[risk]   Live API only returned {len(df_raw)} rows — "
              f"falling back to BQ historical data")
        df_bq = _fetch_bq_fallback(cloud, region, az, instance_type, bq_client)

        if not df_raw.empty:
            # Merge: BQ history as base, live rows on top (most recent wins)
            df_raw = pd.concat([df_bq, df_raw], ignore_index=True)
            df_raw = df_raw.sort_values("collected_at").drop_duplicates(
                subset=["collected_at"], keep="last"
            ).reset_index(drop=True)
        else:
            df_raw = df_bq

    if df_raw.empty or len(df_raw) < SEQUENCE_LEN:
        print(f"[risk]   Still not enough data ({len(df_raw)}) → returning 0.5")
        return 0.5

    df = _engineer_features(df_raw)

    X_raw  = df[list(_feat_cols)].values.astype(np.float32)
    X_sc   = _scaler.transform(X_raw)
    X_seq  = X_sc[-SEQUENCE_LEN:]
    X_flat = X_sc[-1]

    seq_t = torch.tensor(X_seq).unsqueeze(0)
    with torch.no_grad():
        lstm_feats, lstm_logit = _lstm(seq_t)
    lstm_feats = lstm_feats.squeeze(0).numpy()

    X_hybrid = np.hstack([lstm_feats, X_flat]).reshape(1, -1)
    risk     = float(np.clip(_xgb.predict(X_hybrid)[0], 0.0, 1.0))

    print(f"[risk]   LSTM logit={lstm_logit.item():.4f}  risk={risk:.4f}")
    return round(risk, 4)

def _fetch_bq_fallback(cloud: str, region: str, az: str,
                        instance_type: str, bq_client) -> pd.DataFrame:
    """
    Fallback: fetch historical rows for same instance/region across
    past 7 days when live API doesn't return enough data.
    """
    from google.cloud.bigquery import QueryJobConfig, ScalarQueryParameter

    PROJECT_ID = os.getenv("GCP_PROJECT_ID",  "tensile-method-459009-k2")
    DATASET    = os.getenv("BIGQUERY_DATASET", "spot_prices")
    TABLE      = os.getenv("BIGQUERY_TABLE",   "price_history")

    RAW_COLS = [
        "price_usd_per_hr", "ondemand_price_usd_hr",
        "gpu_count", "vcpu_count", "ram_gb",
        "cloud", "region", "availability_zone",
        "instance_type", "gpu_class",
        "preempted", "collected_at",
    ]

    query = f"""
        SELECT {', '.join(RAW_COLS)}
        FROM `{PROJECT_ID}.{DATASET}.{TABLE}`
        WHERE cloud         = @cloud
          AND region        = @region
          AND instance_type = @instance_type
          AND collected_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
        ORDER BY collected_at DESC
        LIMIT {SEQUENCE_LEN + 10}
    """
    cfg = QueryJobConfig(query_parameters=[
        ScalarQueryParameter("cloud",         "STRING", cloud),
        ScalarQueryParameter("region",        "STRING", region),
        ScalarQueryParameter("instance_type", "STRING", instance_type),
    ])
    rows = list(bq_client.query(query, job_config=cfg))
    df = pd.DataFrame([dict(r) for r in rows])
    print(f"[risk]   BQ fallback rows: {len(df)}")
    return df