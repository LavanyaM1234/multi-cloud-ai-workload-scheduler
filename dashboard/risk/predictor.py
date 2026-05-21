"""
risk/predictor.py
──────────────────
Loads trained model artifacts and scores preemption risk.

ROOT CAUSE FIX:
  The old predictor tried to SELECT spot_ratio, price_lag_1, hour etc
  directly from BigQuery → BadRequest because those are DERIVED features
  computed in 01_data_prep.py, not stored columns.

  Fix: fetch only RAW columns that exist in BigQuery, then run the same
  feature engineering pipeline as 01_data_prep.py before passing to model.

RAW columns fetched from BQ:
  gpu_count, vcpu_count, ram_gb,
  price_usd_per_hr, ondemand_price_usd_hr,
  preempted, collected_at,
  cloud, region, availability_zone, instance_type, gpu_class

Derived features computed here (matching 01_data_prep.py exactly):
  spot_ratio, price_lag_1/5/10, price_change, price_pct_chg,
  price_rolling_mean_5/10, price_rolling_std_5,
  recent_preempt_rate, hour, day_of_week, is_weekend,
  cloud_enc, region_enc, availability_zone_enc,
  instance_type_enc, gpu_class_enc
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path

log = logging.getLogger(__name__)

# ── Singleton cache ───────────────────────────────────────────────
_lstm      = None
_xgb       = None
_scaler    = None
_encoders  = None   # dict of {col: LabelEncoder} from 01_data_prep.py
_feat_cols = None
_meta      = None

_HERE        = Path(__file__).parent   # dashboard/risk/
PROJECT_ROOT = _HERE.parent            # dashboard/
MODELS_DIR     = PROJECT_ROOT / "model"
MODEL_DATA_DIR = PROJECT_ROOT / "model_data"

SEQUENCE_LEN = 10  # must match 02_train_hybrid_model.py

# ── Raw columns to fetch from BigQuery ────────────────────────────
# ONLY columns that actually exist in the price_history table.
# All other features are derived below in _engineer_features().
RAW_BQ_COLS = [
    "gpu_count", "vcpu_count", "ram_gb",
    "price_usd_per_hr", "ondemand_price_usd_hr",
    "preempted", "collected_at",
    "cloud", "region", "availability_zone",
    "instance_type", "gpu_class",
]

# ── Feature columns — must match FEATURE_COLS in 01_data_prep.py ──
FEATURE_COLS = [
    "gpu_count", "vcpu_count", "ram_gb",
    "price_usd_per_hr", "ondemand_price_usd_hr",
    "spot_ratio",
    "price_lag_1", "price_lag_5", "price_lag_10",
    "price_change", "price_pct_chg",
    "price_rolling_mean_5", "price_rolling_mean_10", "price_rolling_std_5",
    "recent_preempt_rate",
    "hour", "day_of_week", "is_weekend",
    "cloud_enc", "region_enc", "availability_zone_enc",
    "instance_type_enc", "gpu_class_enc",
]


# ══════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING — mirrors 01_data_prep.py exactly
# ══════════════════════════════════════════════════════════════════

def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a DataFrame with RAW_BQ_COLS and returns one with FEATURE_COLS.
    Must mirror 01_data_prep.py engineer_features() exactly so the
    scaler and model see the same distribution at inference time.

    df must be sorted by collected_at ascending before calling this.
    """

    df = df.copy()
    print(f"[debug] after copy: {df.shape}")
    df["collected_at"] = pd.to_datetime(df["collected_at"], utc=True)
    print(f"[debug] after to_datetime: {df.shape}")
    df = df.sort_values(["cloud", "region", "availability_zone", "collected_at"])
    print(f"[debug] after sort_values: {df.shape}")
    df = df.reset_index(drop=True)
    print(f"[debug] after reset_index: {df.shape}")


    grp = df.groupby(["cloud", "region", "availability_zone"])
    print(f"[debug] after groupby: {df.shape}")

    # Price lag features

    df["price_lag_1"]  = grp["price_usd_per_hr"].shift(1)
    df["price_lag_3"]  = grp["price_usd_per_hr"].shift(3)
    df["price_lag_5"]  = grp["price_usd_per_hr"].shift(5)
    df["price_lag_10"] = grp["price_usd_per_hr"].shift(10)
    print(f"[debug] after lag features: {df.shape}")

    # Price change

    df["price_change_1"]  = df["price_usd_per_hr"] - df["price_lag_1"]
    df["price_pct_chg_1"] = df["price_change_1"] / df["price_lag_1"].replace(0, 1e-6)
    print(f"[debug] after price change: {df.shape}")

    # Rolling stats

    df["price_roll_mean_5"]  = grp["price_usd_per_hr"].transform(lambda x: x.rolling(5,  min_periods=1).mean())
    df["price_roll_mean_10"] = grp["price_usd_per_hr"].transform(lambda x: x.rolling(10, min_periods=1).mean())
    df["price_roll_std_5"]   = grp["price_usd_per_hr"].transform(lambda x: x.rolling(5,  min_periods=1).std().fillna(0))
    df["price_roll_max_5"]   = grp["price_usd_per_hr"].transform(lambda x: x.rolling(5,  min_periods=1).max())
    print(f"[debug] after rolling stats: {df.shape}")

    # Price ratios

    df["spot_ratio"]    = df["price_usd_per_hr"] / df["ondemand_price_usd_hr"].replace(0, 1e-6)
    df["price_vs_mean"] = df["price_usd_per_hr"] / df["price_roll_mean_5"].replace(0, 1e-6)
    print(f"[debug] after price ratios: {df.shape}")

    # Preemption history

    df["recent_preempt_rate_5"]  = grp["preempted"].transform(lambda x: x.rolling(5,  min_periods=1).mean())
    df["recent_preempt_rate_10"] = grp["preempted"].transform(lambda x: x.rolling(10, min_periods=1).mean())
    print(f"[debug] after preempt rates: {df.shape}")

    # Preemption streak — how many consecutive preemptions
    def calc_streak(series):
        streak = np.zeros(len(series), dtype=int)
        count  = 0
        for i, v in enumerate(series):
            count     = count + 1 if v else 0
            streak[i] = count
        return streak


    df["preempt_streak"] = grp["preempted"].transform(
        lambda x: pd.Series(calc_streak(x.values), index=x.index)
    )
    print(f"[debug] after preempt streak: {df.shape}")

    # Time features

    df["hour"]               = df["collected_at"].dt.hour
    df["day_of_week"]        = df["collected_at"].dt.dayofweek
    df["is_weekend"]         = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_business_hours"]  = df["hour"].between(9, 17).astype(int)
    print(f"[debug] after time features: {df.shape}")

    # Cloud+region identity (for routing)

    df["cloud_region_zone"] = df["cloud"] + "|" + df["region"] + "|" + df["availability_zone"]
    print(f"[debug] after cloud_region_zone: {df.shape}")

    # Encode categoricals

    cat_map = {
        "cloud":             "cloud_enc",
        "region":            "region_enc",
        "availability_zone": "zone_enc",
        "instance_type":     "instance_type_enc",
        "gpu_class":         "gpu_class_enc",
    }
    for col, enc_col in cat_map.items():
        if _encoders and col in _encoders:
            le = _encoders[col]
            known = set(le.classes_)
            df[enc_col] = df[col].astype(str).apply(
                lambda v: le.transform([v])[0] if v in known else 0
            )
        else:
            df[enc_col] = df[col].astype("category").cat.codes
    print(f"[debug] after encoding categoricals: {df.shape}")


    # Drop rows with NaN from lag features
    print(f"[debug] before dropna price_lag_1: {df.shape}")
    df = df.dropna(subset=["price_lag_1"]).reset_index(drop=True)
    print(f"[debug] after dropna price_lag_1: {df.shape}")


    # Fill remaining NaN/Inf
    print(f"[debug] before fillna: {df.shape}")
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    print(f"[debug] after fillna: {df.shape}")

    return df


# ══════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════

def _check_model_files():
    required = [
        MODELS_DIR     / "lstm_extractor.pt",
        MODELS_DIR     / "xgb_hybrid.pkl",
        MODELS_DIR     / "hybrid_meta.json",
        MODEL_DATA_DIR / "scaler.pkl",
        MODEL_DATA_DIR / "feature_cols.pkl",
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
            "[risk] Missing model files:\n" + "\n".join(f"  {m}" for m in missing)
        )


def load_models():
    """Load all model artifacts into memory. Safe to call multiple times."""
    global _lstm, _xgb, _scaler, _encoders, _feat_cols, _meta

    if _lstm is not None:
        print("[risk] load_models() — already loaded, skipping")
        return

    _check_model_files()

    import torch
    import joblib
    from risk.lstm_arch import LSTMFeatureExtractor

    # ── Metadata ──────────────────────────────────────────────────
    print("\n[risk] Loading hybrid_meta.json ...")
    with open(MODELS_DIR / "hybrid_meta.json") as f:
        _meta = json.load(f)
    print(f"[risk]   n_features   : {_meta['n_features']}")
    print(f"[risk]   test_roc_auc : {_meta.get('test_roc_auc', 'n/a')}")

    # ── Feature columns ───────────────────────────────────────────
    print("\n[risk] Loading feature_cols.pkl ...")
    _feat_cols = joblib.load(MODEL_DATA_DIR / "feature_cols.pkl")
    print(f"[risk]   {len(_feat_cols)} feature cols: {list(_feat_cols)}")

    # Verify matches hardcoded FEATURE_COLS
    if list(_feat_cols) != FEATURE_COLS:
        print(f"[risk]   ⚠ WARNING: feature_cols.pkl differs from FEATURE_COLS")
        print(f"[risk]   pkl:  {list(_feat_cols)}")
        print(f"[risk]   code: {FEATURE_COLS}")

    # ── Scaler ────────────────────────────────────────────────────
    print("\n[risk] Loading scaler.pkl ...")
    _scaler = joblib.load(MODEL_DATA_DIR / "scaler.pkl")
    print(f"[risk]   type: {type(_scaler).__name__}")

    # ── Encoders (from 01_data_prep.py) ───────────────────────────
    encoders_path = MODEL_DATA_DIR / "encoders.pkl"
    if encoders_path.exists():
        print("\n[risk] Loading encoders.pkl ...")
        _encoders = joblib.load(encoders_path)
        print(f"[risk]   encoder keys: {list(_encoders.keys())}")
    else:
        print("\n[risk] ⚠ encoders.pkl not found — using fallback hash encoding")
        _encoders = {}

    # ── XGBoost ───────────────────────────────────────────────────
    print("\n[risk] Loading xgb_hybrid.pkl ...")
    _xgb = joblib.load(MODELS_DIR / "xgb_hybrid.pkl")
    print(f"[risk]   type             : {type(_xgb).__name__}")
    print(f"[risk]   n_estimators     : {getattr(_xgb, 'n_estimators', '?')}")
    print(f"[risk]   n_features_in_   : {getattr(_xgb, 'n_features_in_', '?')}")

    # ── LSTM ──────────────────────────────────────────────────────
    print("\n[risk] Loading lstm_extractor.pt ...")
    n_features = _meta["n_features"]
    lstm_model = LSTMFeatureExtractor(input_size=n_features)
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
        feats, logit = _lstm(dummy)
    print(f"[risk]   smoke test: feats={tuple(feats.shape)} logit={logit.item():.4f}")
    print(f"\n[risk] ✓ All models loaded\n")


# ══════════════════════════════════════════════════════════════════
# BQ DATA FETCH — raw columns only
# ══════════════════════════════════════════════════════════════════

def _fetch_raw_rows(cloud, region, az, instance_type, bq_client) -> pd.DataFrame:
    """
    Fetch RAW rows from BigQuery for one instance.
    Only selects columns that actually exist in the table.
    Returns DataFrame with at least SEQUENCE_LEN rows, or empty.
    """
    PROJECT_ID = os.getenv("GCP_PROJECT_ID",  "tensile-method-459009-k2")
    DATASET    = os.getenv("BIGQUERY_DATASET", "spot_prices")
    TABLE      = os.getenv("BIGQUERY_TABLE",   "price_history")

    from google.cloud.bigquery import QueryJobConfig, ScalarQueryParameter

    # Fetch SEQUENCE_LEN + 10 extra rows so lag/rolling features
    # can be computed for the last SEQUENCE_LEN rows without NaN issues
    n_fetch = SEQUENCE_LEN + 15

    # Build SELECT only from columns that exist in BQ
    select_cols = ", ".join(RAW_BQ_COLS)

    # az can be None for GCP (no availability_zone in their table)
    if az and az != "None":
        az_filter = "AND availability_zone = @az"
        az_param  = [ScalarQueryParameter("az", "STRING", az)]
    else:
        az_filter = ""
        az_param  = []

    query = f"""
        SELECT {select_cols}
        FROM `{PROJECT_ID}.{DATASET}.{TABLE}`
        WHERE cloud         = @cloud
          AND region        = @region
          AND instance_type = @instance_type
          AND preempted     = FALSE
          {az_filter}
        ORDER BY collected_at DESC
        LIMIT {n_fetch}
    """

    params = [
        ScalarQueryParameter("cloud",         "STRING", cloud),
        ScalarQueryParameter("region",        "STRING", region),
        ScalarQueryParameter("instance_type", "STRING", instance_type),
    ] + az_param

    job_config = QueryJobConfig(query_parameters=params)
    rows = list(bq_client.query(query, job_config=job_config))
    print(f"[risk]   BQ raw rows fetched: {len(rows)}  (need {SEQUENCE_LEN})")

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame([dict(r) for r in rows])


# ══════════════════════════════════════════════════════════════════
# SCORE ONE INSTANCE
# ══════════════════════════════════════════════════════════════════

def score_instance(
    cloud:         str,
    region:        str,
    az:            str,
    instance_type: str,
    bq_client,
) -> float:
    """
    Score preemption risk for one instance.
    Returns float 0.0–1.0.
    Returns 0.5 (neutral) if not enough data.

    Pipeline:
      1. Fetch RAW rows from BQ (only existing columns)
      2. Run _engineer_features() to compute derived features
      3. Scale with saved scaler
      4. Pass last SEQUENCE_LEN rows through LSTM → feature vector
      5. Combine LSTM features + last tabular row → XGBoost input
      6. Return XGBoost preemption probability
    """
    import torch

    load_models()  # no-op if already loaded

    # ── Step 1: fetch raw data ────────────────────────────────────
    df_raw = _fetch_raw_rows(cloud, region, az, instance_type, bq_client)
    if df_raw.empty or len(df_raw) < SEQUENCE_LEN:
        print(f"[risk]   ⚠ Not enough rows ({len(df_raw)}/{SEQUENCE_LEN}) — returning 0.5")
        return 0.5

    # ── Step 2: engineer features (same as 01_data_prep.py) ───────
    df_eng = _engineer_features(df_raw)

    # Use the saved feature_cols order (from pkl file) not hardcoded
    feat_cols = list(_feat_cols) if _feat_cols is not None else FEATURE_COLS

    # Verify all feature cols are present after engineering
    missing_cols = [c for c in feat_cols if c not in df_eng.columns]
    if missing_cols:
        print(f"[risk]   ✗ Missing engineered cols: {missing_cols} — returning 0.5")
        return 0.5

    # ── Step 3: extract feature matrix ───────────────────────────
    X_raw = df_eng[feat_cols].values.astype(np.float32)
    # Take the last n_fetch rows after engineering (lags may shrink available rows)
    # Ensure we have enough for the sequence
    if len(X_raw) < SEQUENCE_LEN:
        print(f"[risk]   ⚠ After engineering only {len(X_raw)} rows — returning 0.5")
        return 0.5

    # ── Step 4: scale ─────────────────────────────────────────────
    X_scaled = _scaler.transform(X_raw)

    # Take the last SEQUENCE_LEN rows for the LSTM window
    X_seq  = X_scaled[-SEQUENCE_LEN:]   # shape: (SEQUENCE_LEN, n_features)
    X_flat = X_scaled[-1]               # shape: (n_features,) — last row for XGB tabular

    print(f"[risk]   X_seq shape : {X_seq.shape}")
    print(f"[risk]   X_flat shape: {X_flat.shape}")
    print(f"[risk]   X_seq[-1] sample (first 4): {X_seq[-1][:4].round(4).tolist()}")

    # ── Step 5: LSTM forward pass ──────────────────────────────────
    seq_tensor = torch.tensor(X_seq).unsqueeze(0)   # (1, seq_len, n_features)
    with torch.no_grad():
        lstm_feats, lstm_logit = _lstm(seq_tensor)
    lstm_feats = lstm_feats.squeeze(0).numpy()       # (lstm_out_dim,)
    print(f"[risk]   LSTM feats shape: {lstm_feats.shape}  logit: {lstm_logit.item():.4f}")

    # ── Step 6: XGBoost hybrid forward pass ───────────────────────
    X_hybrid = np.hstack([lstm_feats, X_flat]).reshape(1, -1)
    print(f"[risk]   X_hybrid shape: {X_hybrid.shape}")

    proba = _xgb.predict_proba(X_hybrid)[0]
    risk  = float(proba[1])   # class 1 = preempted
    print(f"[risk]   XGB proba: {proba.round(4).tolist()}  → risk={risk:.4f}")

    return round(risk, 4)


# ══════════════════════════════════════════════════════════════════
# SCORE ALL ACTIVE INSTANCES — called by /api/risk
# ══════════════════════════════════════════════════════════════════

def score_all_instances(bq_client) -> list:
    """
    Score every distinct active instance.
    Returns list of dicts sorted by risk descending.
    """
    PROJECT_ID = os.getenv("GCP_PROJECT_ID",  "tensile-method-459009-k2")
    DATASET    = os.getenv("BIGQUERY_DATASET", "spot_prices")
    TABLE      = os.getenv("BIGQUERY_TABLE",   "price_history")

    query = f"""
        SELECT DISTINCT cloud, region, availability_zone, instance_type
        FROM `{PROJECT_ID}.{DATASET}.{TABLE}`
        WHERE preempted    = FALSE
          AND gpu_class   != 'none'
          AND collected_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 3 HOUR)
        ORDER BY cloud, instance_type
    """
    instances = list(bq_client.query(query))
    results   = []

    for row in instances:
        try:
            score = score_instance(
                cloud         = row["cloud"],
                region        = row["region"],
                az            = row["availability_zone"],
                instance_type = row["instance_type"],
                bq_client     = bq_client,
            )
            results.append({
                "cloud":         row["cloud"],
                "region":        row["region"],
                "az":            row["availability_zone"],
                "instance_type": row["instance_type"],
                "risk":          score,
            })
        except Exception as e:
            log.warning(f"[risk] Skipping {row['instance_type']}: {e}")

    results.sort(key=lambda x: x["risk"], reverse=True)
    return results