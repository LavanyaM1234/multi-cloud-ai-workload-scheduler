"""
scheduler/launchers/azure_launcher.py
───────────────────────────────────────
Azure Spot VM launcher.

All job state (job_state.json) written to S3 only. No GCS.

Required env vars:
    AZURE_SUBSCRIPTION_ID
    AZURE_TENANT_ID
    AZURE_CLIENT_ID
    AZURE_CLIENT_SECRET       (or Managed Identity if running in Azure)
    AZURE_RESOURCE_GROUP      (pre-created resource group)
    AZURE_LOCATION            (default: eastus)
    AZURE_VNET_NAME           (pre-created VNet)
    AZURE_SUBNET_NAME         (subnet within VNet)
    AZURE_NSG_NAME            (Network Security Group name, optional)
    AZURE_VM_USERNAME         (admin username for the VM)
    AZURE_VM_PASSWORD         (admin password — min 12 chars)
    CHECKPOINT_S3_BUCKET      (S3 bucket — single source of truth)
    AWS_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY
    AWS_REGION                (default: us-east-1)

Optional:
    AZURE_NIC_ID              (reuse an existing NIC instead of creating one)
    AZURE_IMAGE_PUBLISHER     (default: Canonical)
    AZURE_IMAGE_OFFER         (default: 0001-com-ubuntu-server-jammy)
    AZURE_IMAGE_SKU           (default: 22_04-lts-gen2)

Public IP note:
    Each VM gets a Standard-tier Static public IP by default.
    This requires "Network Contributor" on the resource group.
    If you only have "Virtual Machine Contributor", set AZURE_NIC_ID
    to a pre-created NIC that already has a public IP attached, OR
    grant Network Contributor:
        az role assignment create \\
          --assignee <AZURE_CLIENT_ID> \\
          --role "Network Contributor" \\
          --scope /subscriptions/<SUB>/resourceGroups/<RG>
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

# ── Config from env ────────────────────────────────────────────────
AZURE_SUBSCRIPTION  = os.getenv("AZURE_SUBSCRIPTION_ID",  "")
AZURE_TENANT_ID     = os.getenv("AZURE_TENANT_ID",         "")
AZURE_CLIENT_ID     = os.getenv("AZURE_CLIENT_ID",         "")
AZURE_CLIENT_SECRET = os.getenv("AZURE_CLIENT_SECRET",     "")
AZURE_RG            = os.getenv("AZURE_RESOURCE_GROUP",    "ml-scheduler-rg")
AZURE_LOCATION      = os.getenv("AZURE_LOCATION",          "eastus")
AZURE_VNET          = os.getenv("AZURE_VNET_NAME",         "ml-vnet")
AZURE_SUBNET        = os.getenv("AZURE_SUBNET_NAME",       "ml-subnet")
AZURE_NSG           = os.getenv("AZURE_NSG_NAME",          "")
AZURE_VM_USER       = os.getenv("AZURE_VM_USERNAME",       "azureuser")
AZURE_VM_PASS       = os.getenv("AZURE_VM_PASSWORD",       "")
S3_BUCKET           = os.getenv("CHECKPOINT_S3_BUCKET",    "")
AWS_REGION          = os.getenv("AWS_REGION",              "us-east-1")


# ══════════════════════════════════════════════════════════════════
# AZURE AUTH
# ══════════════════════════════════════════════════════════════════

def _get_credentials():
    """Return ClientSecretCredential or DefaultAzureCredential."""
    try:
        from azure.identity import ClientSecretCredential, DefaultAzureCredential
    except ImportError:
        raise RuntimeError(
            "[azure_launcher] azure-identity not installed.\n"
            "Run: pip install azure-identity azure-mgmt-compute azure-mgmt-network"
        )

    if all([AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET]):
        return ClientSecretCredential(
            tenant_id     = AZURE_TENANT_ID,
            client_id     = AZURE_CLIENT_ID,
            client_secret = AZURE_CLIENT_SECRET,
        )

    logger.info("[azure_launcher] Using DefaultAzureCredential (Managed Identity / CLI)")
    return DefaultAzureCredential()


# ══════════════════════════════════════════════════════════════════
# S3 HELPERS
# ══════════════════════════════════════════════════════════════════

def _s3():
    return boto3.client(
        "s3",
        aws_access_key_id     = os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name           = AWS_REGION,
    )


def _s3_put_json(key: str, data: dict) -> bool:
    if not S3_BUCKET:
        logger.warning(f"[azure_launcher] S3_BUCKET not set — skipping write to {key}")
        return False
    try:
        _s3().put_object(
            Bucket      = S3_BUCKET,
            Key         = key,
            Body        = json.dumps(data, indent=2).encode(),
            ContentType = "application/json",
        )
        return True
    except Exception as e:
        logger.warning(f"[azure_launcher] S3 put failed for {key}: {e}")
        return False


def _s3_get_json(key: str) -> dict:
    if not S3_BUCKET:
        return {}
    try:
        obj = _s3().get_object(Bucket=S3_BUCKET, Key=key)
        return json.loads(obj["Body"].read())
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("NoSuchKey", "404"):
            return {}
        logger.warning(f"[azure_launcher] S3 get failed for {key}: {e}")
        return {}
    except Exception as e:
        logger.warning(f"[azure_launcher] S3 get failed for {key}: {e}")
        return {}


# ══════════════════════════════════════════════════════════════════
# CLOUD-INIT SCRIPT
# ══════════════════════════════════════════════════════════════════

def _build_custom_data(startup_script: str, job_id: str) -> str:
    """
    Build base64-encoded cloud-init for the Azure VM.
    Exports S3 + AWS creds so the startup script can reach S3 immediately.
    Also starts a background loop that monitors Azure scheduled-events for
    preemption notices and syncs the checkpoint to S3 before the VM dies.
    """
    cloud_init = f"""#!/bin/bash
set -euo pipefail

# ── Job env ───────────────────────────────────────────────────────
export JOB_ID="{job_id}"
export CLOUD="azure"
export CHECKPOINT_S3_BUCKET="{S3_BUCKET}"
export AWS_ACCESS_KEY_ID="{os.getenv('AWS_ACCESS_KEY_ID', '')}"
export AWS_SECRET_ACCESS_KEY="{os.getenv('AWS_SECRET_ACCESS_KEY', '')}"
export AWS_DEFAULT_REGION="{AWS_REGION}"

# ── Preemption monitor (background) ──────────────────────────────
(
  EVENTS_URL="http://169.254.169.254/metadata/scheduledevents?api-version=2020-07-01"
  while true; do
    RESP=$(curl -s -H "Metadata:true" "$EVENTS_URL" --max-time 3 2>/dev/null || echo "")
    if echo "$RESP" | grep -q '"Preempt"'; then
      echo "[SPOT] Azure Preempt notice received"
      kill -USR1 $(cat /tmp/training_pid.txt 2>/dev/null || echo 1) 2>/dev/null || true
      aws s3 sync /checkpoints/ s3://$CHECKPOINT_S3_BUCKET/checkpoints/$JOB_ID/ \
          --quiet 2>/dev/null || true
      break
    fi
    sleep 5
  done
) &

# ── Main startup script ───────────────────────────────────────────
{startup_script}
"""
    return base64.b64encode(cloud_init.encode("utf-8")).decode("utf-8")


# ══════════════════════════════════════════════════════════════════
# JOB STATE  — S3 only
# ══════════════════════════════════════════════════════════════════

def _update_job_state(job_id: str, launch_result: dict):
    """Patch job_state.json on S3 with VM launch details."""
    key      = f"checkpoints/{job_id}/job_state.json"
    existing = _s3_get_json(key)
    existing.update({
        "launch_result": launch_result,
        "updated_at":    datetime.now(timezone.utc).isoformat(),
    })
    if _s3_put_json(key, existing):
        logger.info(f"[azure_launcher] job_state.json updated on S3 for {job_id}")
    else:
        logger.warning(f"[azure_launcher] Could not update job_state.json on S3 for {job_id}")


# ══════════════════════════════════════════════════════════════════
# CREATE VM
# ══════════════════════════════════════════════════════════════════

def create_azure_vm(decision: dict, startup_script: str) -> dict:
    """
    Launch an Azure Spot VM with a public IP.

    Network resource creation order:
        1. Public IP address  (Standard SKU, Static allocation)
        2. NIC               (attached to subnet + public IP + optional NSG)
        3. VM                (attached to NIC)

    All three require "Network Contributor" on the resource group.
    Set AZURE_NIC_ID to skip steps 1+2 and reuse an existing NIC.

    Args:
        decision:       output of selector.pick_best_cloud()
        startup_script: shell script string (from launcher._build_training_script)

    Returns:
        dict with vm_name, vm_id, instance_type, location,
        private_ip, public_ip, status, s3_bucket, launched_at
    """
    job_id   = decision.get("job_id", f"job-{int(time.time())}")
    sku      = decision["instance_type"]
    location = decision.get("region", AZURE_LOCATION)

    # VM names: max 15 chars, alphanumeric + hyphens
    ts       = int(time.time()) % 100000
    vm_name  = f"ml-{job_id[:8]}-{ts}"[:15].rstrip("-")
    pip_name = f"{vm_name}-pip"    # public IP resource name
    nic_name = f"{vm_name}-nic"

    # ── Validate required env vars ────────────────────────────────
    missing = []
    if not AZURE_SUBSCRIPTION: missing.append("AZURE_SUBSCRIPTION_ID")
    if not AZURE_RG:           missing.append("AZURE_RESOURCE_GROUP")
    if not AZURE_VM_PASS:      missing.append("AZURE_VM_PASSWORD")
    if missing:
        raise EnvironmentError(
            f"[azure_launcher] Missing required env vars: {', '.join(missing)}"
        )

    logger.info(
        f"[azure_launcher] Launching Spot VM: {vm_name} ({sku}) "
        f"in {location}/{AZURE_RG} job={job_id}"
    )

    try:
        from azure.mgmt.compute import ComputeManagementClient
        from azure.mgmt.network import NetworkManagementClient
    except ImportError:
        raise RuntimeError(
            "[azure_launcher] azure-mgmt-compute / azure-mgmt-network not installed.\n"
            "Run: pip install azure-mgmt-compute azure-mgmt-network"
        )

    creds          = _get_credentials()
    compute_client = ComputeManagementClient(creds, AZURE_SUBSCRIPTION)
    network_client = NetworkManagementClient(creds, AZURE_SUBSCRIPTION)

    # ── NIC strategy ─────────────────────────────────────────────
    # Option A: reuse a pre-created NIC (AZURE_NIC_ID set in .env)
    # Option B: create public IP → NIC with public IP attached
    existing_nic_id = os.getenv("AZURE_NIC_ID", "").strip()

    if existing_nic_id:
        # ── Option A: pre-existing NIC ────────────────────────────
        logger.info(f"[azure_launcher] Reusing existing NIC: {existing_nic_id}")
        nic_id     = existing_nic_id
        private_ip = ""
        public_ip  = ""

    else:
        # ── Option B: create Public IP then NIC ───────────────────
        # Step 1: create public IP
        logger.info(f"[azure_launcher] Creating public IP: {pip_name}")
        pip_params = {
            "location": location,
            "sku":      {"name": "Standard", "tier": "Regional"},
            "properties": {
                "publicIPAllocationMethod": "Static",
                "publicIPAddressVersion":   "IPv4",
            },
        }
        pip_op     = network_client.public_ip_addresses.begin_create_or_update(
            AZURE_RG, pip_name, pip_params
        )
        pip_result = pip_op.result()
        public_ip  = pip_result.ip_address or ""
        logger.info(f"[azure_launcher] Public IP created: {pip_name}  ip={public_ip}")

        # Step 2: get subnet (and optional NSG)
        subnet  = network_client.subnets.get(AZURE_RG, AZURE_VNET, AZURE_SUBNET)
        nsg_ref = None
        if AZURE_NSG:
            try:
                nsg     = network_client.network_security_groups.get(AZURE_RG, AZURE_NSG)
                nsg_ref = {"id": nsg.id}
            except Exception:
                logger.warning(f"[azure_launcher] NSG '{AZURE_NSG}' not found — skipping")

        # Step 3: create NIC with public IP attached
        logger.info(f"[azure_launcher] Creating NIC: {nic_name}")
        nic_body = {
            "location": location,
            "properties": {
                "ipConfigurations": [{
                    "name": "ipconfig1",
                    "properties": {
                        "subnet":                    {"id": subnet.id},
                        "publicIPAddress":           {"id": pip_result.id},
                        "privateIPAllocationMethod": "Dynamic",
                    },
                }],
            },
        }
        if nsg_ref:
            nic_body["properties"]["networkSecurityGroup"] = nsg_ref

        nic_op  = network_client.network_interfaces.begin_create_or_update(
            AZURE_RG, nic_name, nic_body
        )
        nic     = nic_op.result()
        nic_id  = nic.id

        # Read back assigned private IP
        nic_detail = network_client.network_interfaces.get(AZURE_RG, nic_name)
        private_ip = (
            nic_detail.ip_configurations[0].private_ip_address
            if nic_detail.ip_configurations else ""
        )
        logger.info(
            f"[azure_launcher] NIC created: {nic_name}  "
            f"private={private_ip}  public={public_ip}"
        )

    # ── cloud-init ────────────────────────────────────────────────
    custom_data = _build_custom_data(startup_script, job_id)

    # ── VM parameters (ARM camelCase) ─────────────────────────────
    vm_params = {
        "location": location,
        "tags": {
            "job_id": job_id,
            "cloud":  "azure",
            "spot":   "true",
        },
        "identity": {
            "type": "SystemAssigned",
        },
        "properties": {
            "hardwareProfile": {
                "vmSize": sku,
            },
            "storageProfile": {
                "imageReference": {
                    "publisher": os.getenv("AZURE_IMAGE_PUBLISHER", "Canonical"),
                    "offer":     os.getenv("AZURE_IMAGE_OFFER",
                                           "0001-com-ubuntu-server-jammy"),
                    "sku":       os.getenv("AZURE_IMAGE_SKU", "22_04-lts-gen2"),
                    "version":   "latest",
                },
                "osDisk": {
                    "osType":       "Linux",
                    "createOption": "FromImage",
                    "diskSizeGB":   128,
                    "managedDisk":  {"storageAccountType": "Premium_LRS"},
                    "deleteOption": "Delete",
                },
            },
            "osProfile": {
                "computerName":  vm_name,
                "adminUsername": AZURE_VM_USER,
                "adminPassword": AZURE_VM_PASS,
                "customData":    custom_data,
                "linuxConfiguration": {
                    "disablePasswordAuthentication": False,
                    "provisionVMAgent":              True,
                },
            },
            "networkProfile": {
                "networkInterfaces": [{"id": nic_id, "properties": {"primary": True}}],
            },
            "priority":       "Spot",
            "evictionPolicy": "Deallocate",
            "billingProfile": {"maxPrice": -1},
        },
    }

    logger.info(f"[azure_launcher] Submitting VM creation: {vm_name}")

    try:
        vm_op     = compute_client.virtual_machines.begin_create_or_update(
            AZURE_RG, vm_name, vm_params
        )
        vm_result = vm_op.result()
        vm_id     = vm_result.id
    except Exception as e:
        logger.error(f"[azure_launcher] VM creation failed: {e}")
        raise

    # If public IP wasn't available at NIC creation time, fetch it now
    if not public_ip and not existing_nic_id:
        try:
            pip_detail = network_client.public_ip_addresses.get(AZURE_RG, pip_name)
            public_ip  = pip_detail.ip_address or ""
        except Exception:
            pass

    logger.info(
        f"[azure_launcher] ✓ VM running: {vm_name}  "
        f"private={private_ip}  public={public_ip or '(pending)'}"
    )

    result = {
        "cloud":          "azure",
        "vm_name":        vm_name,
        "vm_id":          vm_id,
        "instance_type":  sku,
        "resource_group": AZURE_RG,
        "location":       location,
        "private_ip":     private_ip,
        "public_ip":      public_ip,
        "pip_name":       pip_name,     # stored so it can be deleted on cleanup
        "nic_name":       nic_name,
        "status":         "running",
        "s3_bucket":      S3_BUCKET,
        "launched_at":    datetime.now(timezone.utc).isoformat(),
    }

    _update_job_state(job_id, result)
    return result


# ══════════════════════════════════════════════════════════════════
# DEALLOCATE VM
# ══════════════════════════════════════════════════════════════════

def deallocate_azure_vm(vm_name: str, cleanup_network: bool = False) -> bool:
    """
    Deallocate (stop + release) an Azure VM.
    Optionally delete the associated NIC and public IP.
    Returns True on success, False on error.
    """
    try:
        from azure.mgmt.compute import ComputeManagementClient
        from azure.mgmt.network import NetworkManagementClient
        creds          = _get_credentials()
        compute_client = ComputeManagementClient(creds, AZURE_SUBSCRIPTION)

        op = compute_client.virtual_machines.begin_deallocate(AZURE_RG, vm_name)
        op.result()
        logger.info(f"[azure_launcher] Deallocated VM: {vm_name}")

        if cleanup_network:
            net_client = NetworkManagementClient(creds, AZURE_SUBSCRIPTION)
            # Delete NIC
            try:
                nic_name = f"{vm_name}-nic"
                net_client.network_interfaces.begin_delete(AZURE_RG, nic_name).result()
                logger.info(f"[azure_launcher] Deleted NIC: {nic_name}")
            except Exception as e:
                logger.warning(f"[azure_launcher] NIC delete failed: {e}")
            # Delete public IP
            try:
                pip_name = f"{vm_name}-pip"
                net_client.public_ip_addresses.begin_delete(AZURE_RG, pip_name).result()
                logger.info(f"[azure_launcher] Deleted public IP: {pip_name}")
            except Exception as e:
                logger.warning(f"[azure_launcher] Public IP delete failed: {e}")

        return True
    except Exception as e:
        logger.error(f"[azure_launcher] Deallocation failed for {vm_name}: {e}")
        return False