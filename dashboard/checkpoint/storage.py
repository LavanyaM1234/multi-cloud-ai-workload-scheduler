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


def gcs_file_exists(gcs_path: str) -> bool:


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



# ── S3 async upload for compatibility ──
async def upload_to_s3_async(local_path: str, s3_path: str) -> dict:
    """
    Upload the file to S3 asynchronously (for compatibility with engine.py).
    Returns: {"s3": True/False}
    """
    loop = asyncio.get_event_loop()
    s3_result = await loop.run_in_executor(None, upload_to_s3, local_path, s3_path)
    return {"s3": s3_result}


def download_best_available(remote_path: str, local_path: str) -> str:
    """
    Download checkpoint from S3 only.
    Returns: "s3" | None
    """
    if s3_file_exists(remote_path):
        log.info(f"Checkpoint found on S3 — downloading from S3")
        if download_from_s3(remote_path, local_path):
            return "s3"
    log.warning(f"Checkpoint not found on S3: {remote_path}")
    return None
