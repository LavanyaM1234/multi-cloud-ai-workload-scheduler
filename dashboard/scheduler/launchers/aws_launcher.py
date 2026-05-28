"""
scheduler/launchers/aws_launcher.py
─────────────────────────────────────
AWS EC2 Spot Instance launcher.

Changes from previous version:
    - _update_job_state() writes launch_result to S3 (job_state.json)
      instead of a local file on disk.
    - UserData preamble exports CHECKPOINT_S3_BUCKET (was S3_BUCKET) so
      the startup script's bucket guard works correctly.
    - Improved Free Tier error detection with actionable message.
    - AWS_IAM_INSTANCE_PROFILE no longer required — launches without
      instance profile and falls back to UserData-injected creds.

Required env vars:
    AWS_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY
    AWS_REGION                  (default: us-east-1)
    AWS_AMI_ID                  (Ubuntu 22.04 AMI for your region)
    AWS_SUBNET_ID               (VPC subnet for the instance)
    AWS_SECURITY_GROUP_ID       (SG with SSH + training ports)
    AWS_IAM_INSTANCE_PROFILE    (optional — instance profile ARN or name)
    AWS_KEY_PAIR_NAME           (EC2 key pair for SSH access, optional)
    CHECKPOINT_S3_BUCKET        (S3 bucket for checkpoints)
"""

import os
import json
import logging
import time
import base64
from datetime import datetime, timezone

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

AWS_REGION            = os.getenv("AWS_REGION",               "us-east-1")
AWS_AMI_ID            = os.getenv("AWS_AMI_ID",               "")
AWS_SUBNET_ID         = os.getenv("AWS_SUBNET_ID",            "")
AWS_SECURITY_GROUP_ID = os.getenv("AWS_SECURITY_GROUP_ID",    "")
AWS_IAM_PROFILE       = os.getenv("AWS_IAM_INSTANCE_PROFILE", "")
AWS_KEY_PAIR          = os.getenv("AWS_KEY_PAIR_NAME",        "")
S3_BUCKET             = os.getenv("CHECKPOINT_S3_BUCKET",     "")

# Spot interruption notice lead time (seconds) — AWS gives 2 min warning
INTERRUPTION_LEAD_S = 110


def _get_ec2_client():
    """Return a boto3 EC2 client."""
    try:
        import boto3
        return boto3.client("ec2", region_name=AWS_REGION)
    except ImportError:
        logger.error("[AWS Launcher] boto3 not installed — run: pip install boto3")
        raise
    except Exception as e:
        logger.error(f"[AWS Launcher] boto3 client error: {e}")
        raise


def _s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id     = os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name           = AWS_REGION,
    )


def _s3_put_json(key: str, data: dict) -> bool:
    if not S3_BUCKET:
        logger.warning(f"[AWS Launcher] S3_BUCKET not set — skipping S3 write for {key}")
        return False
    try:
        _s3_client().put_object(
            Bucket      = S3_BUCKET,
            Key         = key,
            Body        = json.dumps(data, indent=2).encode(),
            ContentType = "application/json",
        )
        return True
    except Exception as e:
        logger.warning(f"[AWS Launcher] S3 put failed for {key}: {e}")
        return False


def _s3_get_json(key: str) -> dict:
    if not S3_BUCKET:
        return {}
    try:
        obj = _s3_client().get_object(Bucket=S3_BUCKET, Key=key)
        return json.loads(obj["Body"].read())
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return {}
        logger.warning(f"[AWS Launcher] S3 get failed for {key}: {e}")
        return {}
    except Exception as e:
        logger.warning(f"[AWS Launcher] S3 get failed for {key}: {e}")
        return {}


def _build_user_data(startup_script: str, job_id: str) -> str:
    """
    Wrap startup_script with:
      - CHECKPOINT_S3_BUCKET + AWS creds exported as env vars
        (so awscli and boto3 work immediately without any metadata curl)
      - AWS 2-minute spot interruption notice handler (syncs checkpoint to S3)
    Then base64-encode for UserData.
    """
    full_script = f"""#!/bin/bash
set -euo pipefail

# ── Injected by Multi-Cloud Scheduler ─────────────────────────────
export JOB_ID="{job_id}"
export CLOUD="aws"
export CHECKPOINT_S3_BUCKET="{S3_BUCKET}"
export AWS_ACCESS_KEY_ID="{os.getenv('AWS_ACCESS_KEY_ID', '')}"
export AWS_SECRET_ACCESS_KEY="{os.getenv('AWS_SECRET_ACCESS_KEY', '')}"
export AWS_DEFAULT_REGION="{AWS_REGION}"
export CHECKPOINT_LEAD_S="{INTERRUPTION_LEAD_S}"

# ── AWS Spot interruption notice handler ──────────────────────────
# Polls instance metadata every 5s; signals train.py on 2-min notice
(
  while true; do
    HTTP_CODE=$(curl -s -o /dev/null -w "%{{http_code}}" \\
      http://169.254.169.254/latest/meta-data/spot/termination-time \\
      --max-time 2)
    if [ "$HTTP_CODE" = "200" ]; then
      echo "[SPOT] Interruption notice received — triggering checkpoint"
      kill -USR1 $(cat /tmp/training_pid.txt 2>/dev/null || echo 1) 2>/dev/null || true
      aws s3 sync /checkpoints/ s3://$CHECKPOINT_S3_BUCKET/checkpoints/$JOB_ID/ --quiet || true
      echo "[SPOT] Checkpoint synced to S3. Shutting down."
      break
    fi
    sleep 5
  done
) &

# ── User training script ──────────────────────────────────────────
{startup_script}
"""
    return base64.b64encode(full_script.encode("utf-8")).decode("utf-8")


def create_aws_vm(decision: dict, startup_script: str) -> dict:
    """
    Launch an EC2 Spot Instance.

    Args:
        decision:       dict from selector.pick_best_cloud()
        startup_script: bash training script (from launcher._build_training_script)

    Returns:
        dict with instance_id, instance_type, public_ip,
             private_ip, status, s3_bucket, cloud
    """
    job_id        = decision.get("job_id", f"job-{int(time.time())}")
    instance_type = decision["instance_type"]
    region        = decision.get("region", AWS_REGION)

    # AWS_IAM_INSTANCE_PROFILE is optional — S3 access falls back to UserData creds
    missing = []
    if not AWS_AMI_ID:            missing.append("AWS_AMI_ID")
    if not AWS_SUBNET_ID:         missing.append("AWS_SUBNET_ID")
    if not AWS_SECURITY_GROUP_ID: missing.append("AWS_SECURITY_GROUP_ID")
    if missing:
        raise EnvironmentError(
            f"[AWS Launcher] Missing required env vars: {', '.join(missing)}"
        )

    logger.info(
        f"[AWS Launcher] Launching EC2 Spot: {instance_type} in {region}  job={job_id}"
    )

    user_data_b64 = _build_user_data(startup_script, job_id)

    # ── Resolve IAM instance profile ─────────────────────────────
    # AWS_IAM_INSTANCE_PROFILE must be either:
    #   arn:aws:iam::ACCT:instance-profile/NAME  ← full ARN (correct)
    #   ml-training-profile                      ← name only (also correct)
    #
    # Common mistake: setting it to a user ARN (arn:aws:iam::ACCT:user/NAME)
    # which is NOT an instance profile and will always fail.
    #
    # If the value is a user ARN or empty, we launch WITHOUT an instance
    # profile. The VM can still access S3 via the UserData-injected creds.
    iam_profile_ref = None
    iam_val = (AWS_IAM_PROFILE or "").strip()

    if not iam_val:
        logger.warning(
            "[AWS Launcher] AWS_IAM_INSTANCE_PROFILE not set — "
            "launching without instance profile. S3 access via UserData creds."
        )
    elif ":user/" in iam_val:
        logger.warning(
            f"[AWS Launcher] AWS_IAM_INSTANCE_PROFILE='{iam_val}' is a USER ARN, "
            f"not an instance profile ARN. Launching without instance profile. "
            f"To fix: IAM → Roles → Create Role → EC2 use case → attach "
            f"AmazonS3FullAccess → copy the instance profile ARN "
            f"(arn:aws:iam::ACCT:instance-profile/NAME) into your .env."
        )
    elif iam_val.startswith("arn:"):
        iam_profile_ref = {"Arn": iam_val}
        logger.info(f"[AWS Launcher] IAM profile ARN: {iam_val}")
    else:
        iam_profile_ref = {"Name": iam_val}
        logger.info(f"[AWS Launcher] IAM profile name: {iam_val}")

    launch_spec: dict = {
        "ImageId":      AWS_AMI_ID,
        "InstanceType": instance_type,
        "UserData":     user_data_b64,
        "NetworkInterfaces": [{
            "DeviceIndex":              0,
            "SubnetId":                 AWS_SUBNET_ID,
            "Groups":                   [AWS_SECURITY_GROUP_ID],
            "AssociatePublicIpAddress": True,
        }],
        "BlockDeviceMappings": [{
            "DeviceName": "/dev/sda1",
            "Ebs": {
                "VolumeSize":          100,
                "VolumeType":          "gp3",
                "DeleteOnTermination": True,
            },
        }],
        "TagSpecifications": [{
            "ResourceType": "instance",
            "Tags": [
                {"Key": "Name",    "Value": f"ml-job-{job_id}"},
                {"Key": "JobId",   "Value": job_id},
                {"Key": "Cloud",   "Value": "aws"},
                {"Key": "SpotJob", "Value": "true"},
            ],
        }],
    }
    if iam_profile_ref:
        launch_spec["IamInstanceProfile"] = iam_profile_ref
    if AWS_KEY_PAIR:
        launch_spec["KeyName"] = AWS_KEY_PAIR

    try:
        ec2 = _get_ec2_client()

        resp = ec2.run_instances(
            **{k: v for k, v in launch_spec.items()
               if k not in ("TagSpecifications",)},
            TagSpecifications=launch_spec["TagSpecifications"],
            InstanceMarketOptions={
                "MarketType": "spot",
                "SpotOptions": {
                    "SpotInstanceType":             "one-time",
                    "InstanceInterruptionBehavior": "terminate",
                },
            },
            MinCount=1,
            MaxCount=1,
        )

        instance    = resp["Instances"][0]
        instance_id = instance["InstanceId"]

        logger.info(f"[AWS Launcher] Instance launched: {instance_id}  "
                    f"(waiting for running state...)")

        waiter = ec2.get_waiter("instance_running")
        waiter.wait(
            InstanceIds=[instance_id],
            WaiterConfig={"Delay": 10, "MaxAttempts": 30},
        )

        desc       = ec2.describe_instances(InstanceIds=[instance_id])
        inst       = desc["Reservations"][0]["Instances"][0]
        public_ip  = inst.get("PublicIpAddress",  "")
        private_ip = inst.get("PrivateIpAddress", "")

        logger.info(
            f"[AWS Launcher] ✓ Running: {instance_id}  "
            f"public={public_ip}  private={private_ip}"
        )

    except Exception as e:
        err_str = str(e)
        if "not eligible for Free Tier" in err_str or "InvalidParameterCombination" in err_str:
            raise RuntimeError(
                f"[AWS] Free-tier account cannot launch '{instance_type}' as Spot. "
                f"Only t2.micro/t3.micro on-demand are free-tier eligible. Options:\n"
                f"  1. Upgrade to a paid AWS account\n"
                f"  2. Use t3.micro (note: even t3.micro Spot may require paid account)\n"
                f"  Original error: {e}"
            ) from e
        logger.error(f"[AWS Launcher] Launch failed: {e}")
        raise

    result = {
        "cloud":         "aws",
        "instance_id":   instance_id,
        "instance_type": instance_type,
        "region":        region,
        "public_ip":     public_ip,
        "private_ip":    private_ip,
        "status":        "running",
        "s3_bucket":     S3_BUCKET,
        "launched_at":   datetime.now(timezone.utc).isoformat(),
    }

    _update_job_state(job_id, result)
    return result


def terminate_aws_vm(instance_id: str) -> bool:
    """Terminate an EC2 instance by ID. Returns True on success."""
    try:
        ec2 = _get_ec2_client()
        ec2.terminate_instances(InstanceIds=[instance_id])
        logger.info(f"[AWS Launcher] Terminated instance: {instance_id}")
        return True
    except Exception as e:
        logger.error(f"[AWS Launcher] Termination failed for {instance_id}: {e}")
        return False


def _update_job_state(job_id: str, launch_result: dict):
    """Merge launch_result into job_state.json on S3."""
    key      = f"checkpoints/{job_id}/job_state.json"
    existing = _s3_get_json(key)
    existing.update({
        "launch_result": launch_result,
        "updated_at":    datetime.now(timezone.utc).isoformat(),
    })
    if _s3_put_json(key, existing):
        logger.info(f"[AWS Launcher] job_state.json updated on S3 for {job_id}")
    else:
        logger.warning(f"[AWS Launcher] Could not update job_state.json on S3 for {job_id}")