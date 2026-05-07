"""
checkpoint/trainer.py
──────────────────────
PyTorch training loop wrapper with automatic checkpointing.

Triggers a checkpoint when:
  1. Every CHECKPOINT_EVERY_N_STEPS steps (configurable via .env)
  2. SIGTERM received (GCP/AWS/Azure all send this before killing VM)
  3. trigger_preemption() called manually (by metadata_watcher)
  4. [Phase 3 ready hook] risk_score > PREEMPTION_RISK_THRESHOLD
     — plug in Phase 2 model here when ready

Usage:
    trainer = ResumableTrainer(
        model=model,
        optimizer=optimizer,
        job_id="resnet18-cifar10-run1",
    )
    trainer.fit(train_loader, num_epochs=20)
"""

import os
import signal
import asyncio
import logging
from datetime import datetime, timezone

log = logging.getLogger(__name__)

CHECKPOINT_EVERY  = int(os.getenv("CHECKPOINT_EVERY_N_STEPS", "500"))
RISK_THRESHOLD    = float(os.getenv("PREEMPTION_RISK_THRESHOLD", "0.70"))


class ResumableTrainer:

    def __init__(
        self,
        model,
        optimizer,
        job_id:     str,
        loss_fn     = None,
        scheduler   = None,
        device:     str = "auto",
    ):
        import torch
        self.model      = model
        self.optimizer  = optimizer
        self.scheduler  = scheduler
        self.loss_fn    = loss_fn
        self.job_id     = job_id

        # Device selection
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Checkpoint engine
        from checkpoint.engine import CheckpointEngine
        self.engine = CheckpointEngine(job_id=job_id)

        # Training state
        self.global_step  = 0
        self.start_epoch  = 0
        self.loss_history = []

        # Preemption flag — set by SIGTERM handler or manual trigger
        self._preemption_flag = False

        # Register OS signal handlers
        signal.signal(signal.SIGTERM, self._on_sigterm)
        signal.signal(signal.SIGINT,  self._on_sigterm)

        self.model.to(self.device)
        log.info(f"ResumableTrainer ready — device: {self.device} | job: {job_id}")

    # ── PUBLIC: fit ───────────────────────────────────────────────────────────

    def fit(self, train_loader, num_epochs: int):
        """
        Main training entry point.
        Automatically resumes from checkpoint if CHECKPOINT_PATH is set.
        """
        # Try to resume
        self._maybe_resume()

        log.info(
            f"Training start — epochs {self.start_epoch}→{num_epochs} | "
            f"step {self.global_step} | device {self.device}"
        )

        for epoch in range(self.start_epoch, num_epochs):

            if self._preemption_flag:
                log.warning(f"Preemption flag set — stopping before epoch {epoch}")
                break

            avg_loss = self._run_epoch(train_loader, epoch)
            self.loss_history.append({"epoch": epoch, "loss": avg_loss})

            # End-of-epoch checkpoint
            asyncio.run(
                self.engine.save(
                    self.model, self.optimizer,
                    epoch=epoch, step=self.global_step,
                    loss=avg_loss, scheduler=self.scheduler,
                )
            )
            log.info(f"Epoch {epoch} complete — avg loss: {avg_loss:.4f}")

        log.info(f"Training finished — total steps: {self.global_step}")

    # ── PUBLIC: trigger preemption manually (called by metadata_watcher) ──────

    def trigger_preemption(self):
        """
        Call this when preemption is detected.
        Sets the flag — training loop saves at next step boundary.
        """
        log.warning("Preemption triggered — will checkpoint at next step boundary")
        self._preemption_flag = True

    # ── PUBLIC: set risk score (called by Phase 2 predictor when ready) ───────

    def update_risk_score(self, score: float):
        """
        Called every 60s with latest risk score from Phase 2 model.
        When score > RISK_THRESHOLD, sets the preemption flag.

        This method is a no-op until Phase 2 is ready.
        To activate: just start calling this with real scores.
        """
        if score > RISK_THRESHOLD:
            log.warning(f"Risk score {score:.2f} > threshold {RISK_THRESHOLD} — triggering checkpoint")
            self._preemption_flag = True

    # ── PRIVATE: one epoch ────────────────────────────────────────────────────

    def _run_epoch(self, train_loader, epoch: int) -> float:
        import torch
        import torch.nn.functional as F

        self.model.train()
        total_loss = 0.0
        n_batches  = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):

            # Check preemption flag at EVERY step boundary
            if self._preemption_flag:
                log.warning(
                    f"Emergency checkpoint — epoch {epoch}, "
                    f"step {self.global_step}, batch {batch_idx}"
                )
                asyncio.run(
                    self.engine.save(
                        self.model, self.optimizer,
                        epoch=epoch, step=self.global_step,
                        loss=total_loss / max(n_batches, 1),
                        scheduler=self.scheduler,
                    )
                )
                return total_loss / max(n_batches, 1)

            inputs  = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward + backward + update
            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            if self.loss_fn is not None:
                loss = self.loss_fn(outputs, targets)
            else:
                loss = F.cross_entropy(outputs, targets)

            loss.backward()
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            total_loss   += loss.item()
            n_batches    += 1
            self.global_step += 1

            # Periodic checkpoint
            if self.global_step % CHECKPOINT_EVERY == 0:
                log.info(f"Periodic checkpoint at step {self.global_step}")
                asyncio.run(
                    self.engine.save(
                        self.model, self.optimizer,
                        epoch=epoch, step=self.global_step,
                        loss=loss.item(), scheduler=self.scheduler,
                    )
                )

            if batch_idx % 50 == 0:
                log.info(
                    f"Epoch {epoch} | batch {batch_idx}/{len(train_loader)} | "
                    f"loss {loss.item():.4f} | step {self.global_step}"
                )

        return total_loss / max(n_batches, 1)

    # ── PRIVATE: resume ───────────────────────────────────────────────────────

    def _maybe_resume(self):
        meta = self.engine.load(self.model, self.optimizer, self.scheduler)
        if meta is None:
            log.info("No checkpoint found — starting fresh")
            return
        self.start_epoch = meta["epoch"] + 1
        self.global_step = meta["step"]
        log.info(
            f"Resumed — starting from epoch {self.start_epoch}, "
            f"step {self.global_step}"
        )

    # ── PRIVATE: signal handler ───────────────────────────────────────────────

    def _on_sigterm(self, signum, frame):
        """
        GCP sends SIGTERM 30s before killing VM.
        AWS  sends SIGTERM 2min before termination.
        Azure sends SIGTERM 30s before eviction.
        All three are caught here.
        """
        log.warning(f"SIGTERM received (signal {signum}) — emergency checkpoint queued")
        self._preemption_flag = True
