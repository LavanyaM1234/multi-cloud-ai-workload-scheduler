"""
checkpoint/engine.py
─────────────────────
Saves and loads PyTorch checkpoints.

What gets saved:
  - model.state_dict()      all learned weights
  - optimizer.state_dict()  momentum + adaptive LR (same size as model)
  - epoch                   which epoch we were on
  - global_step             exact step within that epoch
  - loss                    last recorded loss value
  - job_id                  links this checkpoint to a specific run

Save flow:
  1. torch.save() → /tmp/checkpoint_{step}.pt   (local disk)
  2. upload_to_both() → GCS + S3 simultaneously
  3. write job_state.json → GCS (for orchestrator to find on restart)
  4. delete local tmp file

Load flow:
  1. read CHECKPOINT_PATH from env  (set when VM is launched)
  2. download_best_available()      GCS first, S3 fallback
  3. torch.load() → restore model + optimizer + step

Usage:
    engine = CheckpointEngine(job_id="resnet-run-1")

    # save
    path = await engine.save(model, optimizer, epoch=3, step=1500, loss=0.42)

    # load on new VM (CHECKPOINT_PATH set in env)
    meta = engine.load(model, optimizer)
    if meta:
        print(f"Resuming from epoch {meta['epoch']} step {meta['step']}")
"""

import os
import json
import asyncio
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)


class CheckpointEngine:

    def __init__(self, job_id: str):
        """
        job_id: unique name for this training run.
                e.g. "resnet18-cifar10-run1"
                All checkpoints for this run go under:
                  checkpoints/{job_id}/step_{N:08d}.pt
        """
        self.job_id     = job_id
        self.prefix     = f"checkpoints/{job_id}"
        self.state_path = f"{self.prefix}/job_state.json"

    # ── SAVE ──────────────────────────────────────────────────────────────────

    async def save(
        self,
        model,
        optimizer,
        epoch:      int,
        step:       int,
        loss:       float,
        scheduler   = None,
    ) -> str:
        """
        Save checkpoint to GCS + S3.
        Returns the remote path string, or None on failure.

        IMPORTANT: Only call this at a step boundary (after optimizer.step()).
        Never call mid-step — optimizer state will be inconsistent.
        """
        import torch

        remote_path = f"{self.prefix}/step_{step:08d}.pt"

        # Build checkpoint dict
        ckpt = {
            "job_id":     self.job_id,
            "epoch":      epoch,
            "step":       step,
            "loss":       round(loss, 6),
            "saved_at":   datetime.now(timezone.utc).isoformat(),
            "model":      model.state_dict(),
            "optimizer":  optimizer.state_dict(),
        }
        if scheduler is not None:
            ckpt["scheduler"] = scheduler.state_dict()

        # Save to local temp file first
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            tmp_path = f.name

        torch.save(ckpt, tmp_path)
        size_mb = Path(tmp_path).stat().st_size / 1024 / 1024
        log.info(f"Checkpoint serialised — {size_mb:.1f} MB — step {step}")

        # Upload to GCS + S3 simultaneously
        from checkpoint.storage import upload_to_both
        results = await upload_to_both(tmp_path, remote_path)

        # Clean up local temp file
        Path(tmp_path).unlink(missing_ok=True)

        if not any(results.values()):
            log.error("Checkpoint upload failed on ALL storage backends")
            return None

        log.info(f"Checkpoint saved — GCS:{results['gcs']} S3:{results['s3']} — {remote_path}")

        # Write job_state.json so the orchestrator can find this checkpoint
        await self._write_job_state(epoch, step, loss, remote_path)

        return remote_path

    # ── LOAD ──────────────────────────────────────────────────────────────────

    def load(self, model, optimizer, scheduler=None) -> dict | None:
        """
        Load checkpoint into model and optimizer.

        Reads CHECKPOINT_PATH from environment — this is set by the
        orchestrator when launching the new VM after migration.

        Returns dict with {epoch, step, loss} on success, None if no checkpoint.
        """
        import torch

        checkpoint_path = os.getenv("CHECKPOINT_PATH")
        if not checkpoint_path:
            log.info("CHECKPOINT_PATH not set — starting training from scratch")
            return None

        log.info(f"Loading checkpoint from: {checkpoint_path}")

        # Download from GCS or S3
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            tmp_path = f.name

        from checkpoint.storage import download_best_available
        source = download_best_available(checkpoint_path, tmp_path)

        if source is None:
            log.error(f"Could not download checkpoint: {checkpoint_path}")
            Path(tmp_path).unlink(missing_ok=True)
            return None

        # Load into model
        ckpt = torch.load(tmp_path, map_location="cpu")
        Path(tmp_path).unlink(missing_ok=True)

        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler and "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])

        meta = {
            "epoch": ckpt["epoch"],
            "step":  ckpt["step"],
            "loss":  ckpt["loss"],
        }
        log.info(
            f"Resumed from {source.upper()} — "
            f"epoch {meta['epoch']}, step {meta['step']}, loss {meta['loss']:.4f}"
        )
        return meta

    # ── JOB STATE ─────────────────────────────────────────────────────────────

    async def _write_job_state(
        self,
        epoch:           int,
        step:            int,
        loss:            float,
        checkpoint_path: str,
    ):
        """
        Write a small JSON to GCS that records where this job is.
        The orchestrator reads this to know:
          - Is the job still running?
          - What checkpoint path should the new VM use?
        """
        state = {
            "job_id":          self.job_id,
            "epoch":           epoch,
            "step":            step,
            "loss":            round(loss, 6),
            "checkpoint_path": checkpoint_path,
            "updated_at":      datetime.now(timezone.utc).isoformat(),
            "status":          "running",
        }

        # Write to temp file then upload
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(state, f, indent=2)
            tmp_path = f.name

        from checkpoint.storage import upload_to_gcs
        upload_to_gcs(tmp_path, self.state_path)
        Path(tmp_path).unlink(missing_ok=True)

    def get_latest_job_state(self) -> dict | None:
        """
        Read job_state.json from GCS.
        Returns the state dict or None if not found.
        """
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = f.name

        from checkpoint.storage import download_from_gcs
        ok = download_from_gcs(self.state_path, tmp_path)
        if not ok:
            return None

        with open(tmp_path) as f:
            state = json.load(f)
        Path(tmp_path).unlink(missing_ok=True)
        return state
