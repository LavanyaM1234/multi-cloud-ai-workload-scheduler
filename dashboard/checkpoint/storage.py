"""
checkpoint/storage.py
──────────────────────
Handles upload and download of checkpoint files.
S3 is the single source of truth — no GCS.

engine.py calls this — it doesn't know or care which cloud the VM runs on.
All three clouds (GCP, AWS, Azure) write checkpoints and job state to S3.

Buckets are configured via env vars (set in startup script at boot):
    CHECKPOINT_S3_BUCKET  = your-s3-bucket
    AWS_ACCESS_KEY_ID     = ...
    AWS_SECRET_ACCESS_KEY = ...
    AWS_DEFAULT_REGION    = us-east-1
"""

import os
import logging
import asyncio
from pathlib import Path

log = logging.getLogger(__name__)


# ── S3 ────────────────────────────────────────────────────────────

def upload_to_s3(local_path: str, s3_path: str) -> bool:
    """
    Upload a local file to S3.
    s3_path: key inside the bucket, e.g. 'checkpoints/job1/step_000500.pt'
    Returns True on success, False on failure.
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
    """
    Download a file from S3 to local disk.
    Returns True on success, False on failure.
    """
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
    """Check if a key exists in S3. Returns True/False."""
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


async def upload_to_s3_async(local_path: str, s3_path: str) -> dict:
    """
    Upload a file to S3 asynchronously (called by engine.save()).
    Returns: {"s3": True/False}
    """
    loop      = asyncio.get_event_loop()
    s3_result = await loop.run_in_executor(None, upload_to_s3, local_path, s3_path)
    return {"s3": s3_result}


def download_best_available(remote_path: str, local_path: str) :
    """
    Download checkpoint from S3.
    Returns: "s3" on success, None if not found.
    """
    if s3_file_exists(remote_path):
        log.info(f"Checkpoint found on S3 — downloading")
        if download_from_s3(remote_path, local_path):
            return "s3"
    log.warning(f"Checkpoint not found on S3: {remote_path}")
    return None