"""
checkpoint/engine.py
─────────────────────
Saves and loads PyTorch checkpoints to GCS + S3 simultaneously.

Changes from original:
  - _write_job_state() now writes ALL fields the dashboard needs:
      total_epochs, accuracy, cost_usd, cloud, instance,
      elapsed_hrs, resumed_from, migration_count
  - save() accepts extra_state dict so train.py can pass
      accuracy, cost_usd etc without engine.py knowing about them
  - load() returns None cleanly if no checkpoint exists
    (original raised on missing CHECKPOINT_PATH)
  - get_latest_job_state() made sync (was already sync, kept as-is)

What stays the same:
  - upload_to_both() via storage.py — GCS + S3 simultaneously
  - download_best_available() — GCS first, S3 fallback
  - checkpoint file layout:
      checkpoints/{job_id}/step_{N:08d}.pt   milestone saves
      checkpoints/{job_id}/checkpoint_latest.pt  always latest
      checkpoints/{job_id}/job_state.json    dashboard reads this
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
        self.job_id     = job_id
        self.prefix     = f"checkpoints/{job_id}"
        self.state_path = f"{self.prefix}/job_state.json"
        self.latest_path = f"{self.prefix}/checkpoint_latest.pt"

    # ══════════════════════════════════════════════════════════════
    # SAVE
    # ══════════════════════════════════════════════════════════════

    async def save(
        self,
        model,
        optimizer,
        epoch:       int,
        step:        int,
        loss:        float,
        scheduler    = None,
        extra_state: dict = None,   # ← NEW: pass accuracy, cost_usd, etc.
    ) -> str:
        """
        Save checkpoint to GCS + S3 simultaneously.

        extra_state keys recognised for job_state.json:
            accuracy, cost_usd, elapsed_hrs, cloud, instance,
            total_epochs, best_val_acc, train_loss,
            resumed_from, migration_count, status

        Returns remote path string, or None on failure.
        """
        import torch

        # ── Build checkpoint dict ──────────────────────────────────
        ckpt = {
            "job_id":    self.job_id,
            "epoch":     epoch,
            "step":      step,
            "loss":      round(float(loss), 6),
            "saved_at":  datetime.now(timezone.utc).isoformat(),
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        if scheduler is not None:
            ckpt["scheduler"] = scheduler.state_dict()
        if extra_state:
            # Store training metadata inside checkpoint too
            # so it can be recovered even without job_state.json
            ckpt["training_meta"] = {
                k: v for k, v in extra_state.items()
                if k not in ("model", "optimizer", "scheduler")
            }

        # ── Write to local temp file ───────────────────────────────
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            tmp_path = f.name
        torch.save(ckpt, tmp_path)
        size_mb = Path(tmp_path).stat().st_size / 1024 / 1024
        log.info(f"[ckpt] Serialised {size_mb:.1f}MB — epoch={epoch} step={step}")

        # ── Upload milestone + latest simultaneously ───────────────
        milestone_path = f"{self.prefix}/step_{step:08d}.pt"

        from checkpoint.storage import upload_to_both, upload_to_gcs

        # Upload milestone (GCS + S3)
        milestone_results = await upload_to_both(tmp_path, milestone_path)

        # Upload latest (GCS + S3) — always overwrite, resume reads this
        latest_results = await upload_to_both(tmp_path, self.latest_path)

        Path(tmp_path).unlink(missing_ok=True)

        if not any(milestone_results.values()):
            log.error("[ckpt] Upload failed on ALL backends")
            return None

        log.info(
            f"[ckpt] Saved — GCS:{milestone_results['gcs']} "
            f"S3:{milestone_results['s3']} — {milestone_path}"
        )

        # ── Write job_state.json ───────────────────────────────────
        await self._write_job_state(epoch, step, loss, milestone_path,
                                    extra_state or {})
        return milestone_path

    # ══════════════════════════════════════════════════════════════
    # LOAD
    # ══════════════════════════════════════════════════════════════

    def load(self, model, optimizer, scheduler=None) -> dict | None:
        """
        Load checkpoint_latest.pt from GCS (S3 fallback).

        Reads RESUME_STEP env var:
          - "0" or unset → return None (fresh start)
          - any other value → download and load checkpoint

        Decision: we use RESUME_STEP not CHECKPOINT_PATH because
        train.py already knows the job_id and can construct the path.
        CHECKPOINT_PATH from original engine.py is not used.

        Returns dict {epoch, step, loss} on success, None if no checkpoint.
        """
        import torch

        resume_step = int(os.environ.get("RESUME_STEP", "0"))
        if resume_step == 0:
            log.info("[ckpt] RESUME_STEP=0 — fresh start")
            return None

        prev_cloud = os.environ.get("PREV_CLOUD", "")
        curr_cloud = os.environ.get("CLOUD", "gcp")
        if prev_cloud and prev_cloud != curr_cloud:
            log.info(f"[ckpt] Cross-cloud resume: {prev_cloud} → {curr_cloud}")
        else:
            log.info(f"[ckpt] Resuming from step {resume_step} on {curr_cloud}")

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            tmp_path = f.name

        from checkpoint.storage import download_best_available
        source = download_best_available(self.latest_path, tmp_path)

        if source is None:
            log.warning("[ckpt] checkpoint_latest.pt not found on GCS or S3 — fresh start")
            Path(tmp_path).unlink(missing_ok=True)
            return None

        try:
            ckpt = torch.load(tmp_path, map_location="cpu")
            Path(tmp_path).unlink(missing_ok=True)

            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            if scheduler and "scheduler" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])

            meta = {
                "epoch":         ckpt["epoch"],
                "step":          ckpt["step"],
                "loss":          ckpt.get("loss", 999.0),
                "training_meta": ckpt.get("training_meta", {}),
            }
            log.info(
                f"[ckpt] Loaded from {source.upper()} — "
                f"epoch={meta['epoch']} step={meta['step']} loss={meta['loss']:.4f}"
            )
            return meta

        except Exception as e:
            log.error(f"[ckpt] Load failed: {e} — fresh start")
            Path(tmp_path).unlink(missing_ok=True)
            return None

    # ══════════════════════════════════════════════════════════════
    # JOB STATE  — written to GCS only (dashboard reads from GCS)
    # ══════════════════════════════════════════════════════════════

    async def _write_job_state(
        self,
        epoch:           int,
        step:            int,
        loss:            float,
        checkpoint_path: str,
        extra:           dict,
    ):
        """
        Write job_state.json to GCS.
        This is what the dashboard reads every 60s via /api/jobs.

        Fields written (all used by ui.js mergeRealJobs/updateJobFromState):
            job_id, task_name, status,
            epoch, total_epochs,          ← progress % = epoch/total_epochs
            step, loss, accuracy,
            elapsed_hrs, cost_usd,        ← cost tracking
            cloud, instance,              ← which VM is running
            resumed_from, migration_count,← migration history
            checkpoint_path,              ← so poller knows where to resume
            updated_at
        """
        state = {
            "job_id":           self.job_id,
            "task_name":        extra.get("task_name", self.job_id),
            "status":           extra.get("status",    "running"),
            "epoch":            epoch,
            "total_epochs":     extra.get("total_epochs",     50),
            "step":             step,
            "loss":             round(float(loss), 6),
            "accuracy":         round(float(extra.get("accuracy",   0.0)), 4),
            "best_val_acc":     round(float(extra.get("best_val_acc", 0.0)), 4),
            "train_loss":       round(float(extra.get("train_loss",  loss)), 6),
            "elapsed_hrs":      round(float(extra.get("elapsed_hrs", 0.0)), 4),
            "cost_usd":         round(float(extra.get("cost_usd",    0.0)), 4),
            "cloud":            extra.get("cloud",    os.environ.get("CLOUD",         "gcp")),
            "instance":         extra.get("instance", os.environ.get("INSTANCE_TYPE", "e2-standard-4")),
            "resumed_from":     extra.get("resumed_from",     0),
            "migration_count":  extra.get("migration_count",  0),
            "checkpoint_path":  checkpoint_path,
            "updated_at":       datetime.now(timezone.utc).isoformat(),
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(state, f, indent=2)
            tmp_path = f.name

        # job_state.json goes to GCS only — S3 not needed for dashboard
        from checkpoint.storage import upload_to_gcs
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, upload_to_gcs, tmp_path, self.state_path
        )
        Path(tmp_path).unlink(missing_ok=True)
        log.info(f"[state] job_state.json written — status={state['status']} epoch={epoch}")

    async def write_terminal_state(self, epoch, step, loss, extra: dict):
        """
        Write a final job_state.json for terminal statuses:
        preempted, done, budget_exceeded, failed.
        Called directly by train.py at exit points.
        """
        await self._write_job_state(epoch, step, loss,
                                    f"{self.prefix}/checkpoint_latest.pt",
                                    extra)

    # ══════════════════════════════════════════════════════════════
    # READ STATE  (used by server.py poller)
    # ══════════════════════════════════════════════════════════════

    def get_latest_job_state(self) -> dict | None:
        """Read job_state.json from GCS. Returns dict or None."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = f.name
        from checkpoint.storage import download_from_gcs
        ok = download_from_gcs(self.state_path, tmp_path)
        if not ok:
            Path(tmp_path).unlink(missing_ok=True)
            return None
        with open(tmp_path) as f:
            state = json.load(f)
        Path(tmp_path).unlink(missing_ok=True)
        return state