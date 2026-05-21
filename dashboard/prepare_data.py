"""
01_prepare_data.py
──────────────────
Loads data from BigQuery OR a CSV/JSONL file.
Engineers features, saves train/val/test splits.

Usage:
    python 01_prepare_data.py --source bigquery
    python 01_prepare_data.py --source csv --file my_data.csv
    python 01_prepare_data.py --source jsonl --file dataset_*.jsonl
"""

import argparse
import glob
import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

OUTPUT_DIR = "model_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────── FEATURE COLUMNS ────────────────────────────────

FEATURE_COLS = [
    # Raw
    "gpu_count", "vcpu_count", "ram_gb",
    "price_usd_per_hr", "ondemand_price_usd_hr",
    # Price ratios
    "spot_ratio", "price_vs_mean",
    # Lag features
    "price_lag_1", "price_lag_3", "price_lag_5", "price_lag_10",
    "price_change_1", "price_pct_chg_1",
    # Rolling stats
    "price_roll_mean_5", "price_roll_mean_10",
    "price_roll_std_5",  "price_roll_max_5",
    # Preemption history
    "recent_preempt_rate_5", "recent_preempt_rate_10",
    "preempt_streak",
    # Time
    "hour", "day_of_week", "is_weekend", "is_business_hours",
    # Encoded categoricals
    "cloud_enc", "region_enc", "zone_enc",
    "instance_type_enc", "gpu_class_enc",
]

TARGET_COL    = "preempted"
SEQUENCE_LEN  = 10    # LSTM looks back 10 timesteps


# ─────────────────────────── LOADERS ────────────────────────────────────────

def load_from_bigquery() -> pd.DataFrame:
    from dotenv import load_dotenv
    from google.cloud import bigquery
    load_dotenv()
    table_id = os.getenv("BQ_TABLE")
    if not table_id:
        raise ValueError("BQ_TABLE not set in .env")
    project  = table_id.split(".")[0]
    query    = f"SELECT * FROM `{table_id}` ORDER BY collected_at ASC"
    print(f"Loading from BigQuery: {table_id}")
    df = pd.read_gbq(query, project_id=project)
    print(f"  ✔ {len(df)} rows loaded")
    return df


def load_from_csv(filepath: str) -> pd.DataFrame:
    print(f"Loading from CSV: {filepath}")
    df = pd.read_csv(filepath)
    print(f"  ✔ {len(df)} rows loaded")
    return df


def load_from_jsonl(pattern: str) -> pd.DataFrame:
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No files matching: {pattern}")
    print(f"Loading {len(files)} JSONL files matching: {pattern}")
    dfs = [pd.read_json(f, lines=True) for f in sorted(files)]
    df  = pd.concat(dfs, ignore_index=True)
    print(f"  ✔ {len(df)} rows loaded from {len(files)} files")
    return df


# ─────────────────────────── FEATURE ENGINEERING ────────────────────────────

def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()

    # Parse timestamp
    df["collected_at"] = pd.to_datetime(df["collected_at"], utc=True)
    df = df.sort_values(["cloud", "region", "availability_zone", "collected_at"])
    df = df.reset_index(drop=True)

    grp = df.groupby(["cloud", "region", "availability_zone"])

    # Price lag features
    df["price_lag_1"]  = grp["price_usd_per_hr"].shift(1)
    df["price_lag_3"]  = grp["price_usd_per_hr"].shift(3)
    df["price_lag_5"]  = grp["price_usd_per_hr"].shift(5)
    df["price_lag_10"] = grp["price_usd_per_hr"].shift(10)

    # Price change
    df["price_change_1"]  = df["price_usd_per_hr"] - df["price_lag_1"]
    df["price_pct_chg_1"] = df["price_change_1"] / df["price_lag_1"].replace(0, 1e-6)

    # Rolling stats
    df["price_roll_mean_5"]  = grp["price_usd_per_hr"].transform(lambda x: x.rolling(5,  min_periods=1).mean())
    df["price_roll_mean_10"] = grp["price_usd_per_hr"].transform(lambda x: x.rolling(10, min_periods=1).mean())
    df["price_roll_std_5"]   = grp["price_usd_per_hr"].transform(lambda x: x.rolling(5,  min_periods=1).std().fillna(0))
    df["price_roll_max_5"]   = grp["price_usd_per_hr"].transform(lambda x: x.rolling(5,  min_periods=1).max())

    # Price ratios
    df["spot_ratio"]    = df["price_usd_per_hr"] / df["ondemand_price_usd_hr"].replace(0, 1e-6)
    df["price_vs_mean"] = df["price_usd_per_hr"] / df["price_roll_mean_5"].replace(0, 1e-6)

    # Preemption history
    df["recent_preempt_rate_5"]  = grp["preempted"].transform(lambda x: x.rolling(5,  min_periods=1).mean())
    df["recent_preempt_rate_10"] = grp["preempted"].transform(lambda x: x.rolling(10, min_periods=1).mean())

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

    # Time features
    df["hour"]               = df["collected_at"].dt.hour
    df["day_of_week"]        = df["collected_at"].dt.dayofweek
    df["is_weekend"]         = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_business_hours"]  = df["hour"].between(9, 17).astype(int)

    # Cloud+region identity (for routing)
    df["cloud_region_zone"] = df["cloud"] + "|" + df["region"] + "|" + df["availability_zone"]

    # Encode categoricals
    cat_map = {
        "cloud":             "cloud_enc",
        "region":            "region_enc",
        "availability_zone": "zone_enc",
        "instance_type":     "instance_type_enc",
        "gpu_class":         "gpu_class_enc",
    }
    encoders = {}
    for col, enc_col in cat_map.items():
        le = LabelEncoder()
        df[enc_col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # Drop rows with NaN from lag features
    df = df.dropna(subset=["price_lag_1"]).reset_index(drop=True)

    print(f"  ✔ Features engineered: {len(df)} rows, {len(FEATURE_COLS)} features")
    return df, encoders


# ─────────────────────────── SPLIT & SAVE ───────────────────────────────────

def split_and_save(df: pd.DataFrame, encoders: dict):
    # Drop any remaining NaN values in features or target
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).reset_index(drop=True)

    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[TARGET_COL].astype(int).values

    # Time-ordered split (never shuffle — preserve temporal order)
    n         = len(X)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)

    X_train, y_train = X[:train_end],       y[:train_end]
    X_val,   y_val   = X[train_end:val_end], y[train_end:val_end]
    X_test,  y_test  = X[val_end:],          y[val_end:]

    # Scale (fit on train only)
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # Save arrays
    np.save(f"{OUTPUT_DIR}/X_train.npy", X_train)
    np.save(f"{OUTPUT_DIR}/y_train.npy", y_train)
    np.save(f"{OUTPUT_DIR}/X_val.npy",   X_val)
    np.save(f"{OUTPUT_DIR}/y_val.npy",   y_val)
    np.save(f"{OUTPUT_DIR}/X_test.npy",  X_test)
    np.save(f"{OUTPUT_DIR}/y_test.npy",  y_test)

    # Save full df for LSTM sequence building
    df.to_parquet(f"{OUTPUT_DIR}/full_df.parquet", index=False)

    # Save scaler and encoders
    joblib.dump(scaler,   f"{OUTPUT_DIR}/scaler.pkl")
    joblib.dump(encoders, f"{OUTPUT_DIR}/encoders.pkl")
    joblib.dump(FEATURE_COLS, f"{OUTPUT_DIR}/feature_cols.pkl")

    pos_rate = y.mean()
    print(f"\n  Train: {len(X_train):>6}  Val: {len(X_val):>6}  Test: {len(X_test):>6}")
    print(f"  Preemption rate: {pos_rate:.3%}")
    print(f"  Class imbalance ratio: {(1-pos_rate)/pos_rate:.1f}:1")
    print(f"  Saved to {OUTPUT_DIR}/")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ─────────────────────────── MAIN ───────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["bigquery", "csv", "jsonl"], default="bigquery")
    parser.add_argument("--file",   type=str, default="dataset_*.jsonl",
                        help="CSV/JSONL path or glob pattern")
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print("  SPOT PREEMPTION — Data Preparation")
    print(f"{'='*55}")

    if args.source == "bigquery":
        df = load_from_bigquery()
    elif args.source == "csv":
        df = load_from_csv(args.file)
    else:
        df = load_from_jsonl(args.file)

    print(f"\nEngineering features...")
    df, encoders = engineer_features(df)

    print(f"\nSplitting and saving...")
    split_and_save(df, encoders)

    print(f"\n✅ Data preparation complete.")
    print(f"   Next: python 02_train_hybrid_model.py")