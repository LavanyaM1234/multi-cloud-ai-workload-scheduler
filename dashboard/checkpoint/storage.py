"""
checkpoint/storage.py
──────────────────────
Handles upload and download of checkpoint files to GCS and S3.

engine.py calls this — it doesn't know which cloud it's talking to.
Both uploads happen simultaneously using asyncio so total upload
time = max(GCS time, S3 time), not GCS time + S3 time.

Buckets are configured via .env:
    CHECKPOINT_GCS_BUCKET = your-gcs-bucket
    CHECKPOINT_S3_BUCKET  = your-s3-bucket
    AWS_ACCESS_KEY_ID     = ...
    AWS_SECRET_ACCESS_KEY = ...
"""

import os
import io
import logging
import asyncio
from pathlib import Path

log = logging.getLogger(__name__)


# ── GCS ───────────────────────────────────────────────────────────────────────

def upload_to_gcs(local_path: str, gcs_path: str) -> bool:
    """
    Upload a local file to GCS.
    gcs_path: path inside the bucket, e.g. 'checkpoints/job1/step_000500.pt'
    """
    bucket_name = os.getenv("CHECKPOINT_GCS_BUCKET")
    if not bucket_name:
        log.warning("CHECKPOINT_GCS_BUCKET not set — skipping GCS upload")
        return False
    try:
        from google.cloud import storage
        client  = storage.Client()
        bucket  = client.bucket(bucket_name)
        blob    = bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)
        size_mb = Path(local_path).stat().st_size / 1024 / 1024
        log.info(f"GCS upload OK — gs://{bucket_name}/{gcs_path} ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        log.error(f"GCS upload FAILED: {e}")
        return False


def download_from_gcs(gcs_path: str, local_path: str) -> bool:
    """Download a file from GCS to local disk."""
    bucket_name = os.getenv("CHECKPOINT_GCS_BUCKET")
    if not bucket_name:
        log.warning("CHECKPOINT_GCS_BUCKET not set — cannot download from GCS")
        return False
    try:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob   = bucket.blob(gcs_path)
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(local_path)
        log.info(f"GCS download OK — {gcs_path} → {local_path}")
        return True
    except Exception as e:
        log.error(f"GCS download FAILED: {e}")
        return False


def gcs_file_exists(gcs_path: str) -> bool:
    bucket_name = os.getenv("CHECKPOINT_GCS_BUCKET")
    if not bucket_name:
        return False
    try:
        from google.cloud import storage
        client = storage.Client()
        blob   = client.bucket(bucket_name).blob(gcs_path)
        return blob.exists()
    except Exception:
        return False


# ── S3 ────────────────────────────────────────────────────────────────────────

def upload_to_s3(local_path: str, s3_path: str) -> bool:
    """
    Upload a local file to S3.
    s3_path: path inside the bucket, e.g. 'checkpoints/job1/step_000500.pt'
    """
    bucket_name = os.getenv("CHECKPOINT_S3_BUCKET")
    if not bucket_name:
        log.warning("CHECKPOINT_S3_BUCKET not set — skipping S3 upload")
        return False
    try:
        import boto3
        s3 = boto3.client(
            "s3",
            aws_access_key_id     = os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name           = os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        )
        s3.upload_file(local_path, bucket_name, s3_path)
        size_mb = Path(local_path).stat().st_size / 1024 / 1024
        log.info(f"S3 upload OK — s3://{bucket_name}/{s3_path} ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        log.error(f"S3 upload FAILED: {e}")
        return False


def download_from_s3(s3_path: str, local_path: str) -> bool:
    """Download a file from S3 to local disk."""
    bucket_name = os.getenv("CHECKPOINT_S3_BUCKET")
    if not bucket_name:
        log.warning("CHECKPOINT_S3_BUCKET not set — cannot download from S3")
        return False
    try:
        import boto3
        s3 = boto3.client(
            "s3",
            aws_access_key_id     = os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name           = os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        )
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket_name, s3_path, local_path)
        log.info(f"S3 download OK — {s3_path} → {local_path}")
        return True
    except Exception as e:
        log.error(f"S3 download FAILED: {e}")
        return False


def s3_file_exists(s3_path: str) -> bool:
    bucket_name = os.getenv("CHECKPOINT_S3_BUCKET")
    if not bucket_name:
        return False
    try:
        import boto3
        s3 = boto3.client(
            "s3",
            aws_access_key_id     = os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name           = os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        )
        s3.head_object(Bucket=bucket_name, Key=s3_path)
        return True
    except Exception:
        return False


# ── Dual upload (GCS + S3 simultaneously) ─────────────────────────────────────

async def upload_to_both(local_path: str, remote_path: str) -> dict:
    """
    Upload the same file to GCS and S3 at the same time.
    Uses asyncio to run both uploads concurrently — total time
    is max(gcs_time, s3_time) instead of gcs_time + s3_time.

    Returns: {"gcs": True/False, "s3": True/False}
    """
    loop = asyncio.get_event_loop()

    gcs_result, s3_result = await asyncio.gather(
        loop.run_in_executor(None, upload_to_gcs, local_path, remote_path),
        loop.run_in_executor(None, upload_to_s3,  local_path, remote_path),
    )

    return {"gcs": gcs_result, "s3": s3_result}


def download_best_available(remote_path: str, local_path: str) -> str:
    """
    Download checkpoint from whichever cloud is available.
    Tries GCS first (we're usually on GCP), then S3 as fallback.

    Returns: "gcs" | "s3" | None
    """
    # Try GCS first
    if gcs_file_exists(remote_path):
        log.info(f"Checkpoint found on GCS — downloading from GCS")
        if download_from_gcs(remote_path, local_path):
            return "gcs"

    # Fall back to S3
    if s3_file_exists(remote_path):
        log.info(f"Checkpoint found on S3 — downloading from S3")
        if download_from_s3(remote_path, local_path):
            return "s3"

    log.warning(f"Checkpoint not found on GCS or S3: {remote_path}")
    return None
