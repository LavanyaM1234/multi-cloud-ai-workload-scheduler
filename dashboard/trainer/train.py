#!/usr/bin/env python3
"""
trainer/train.py
─────────────────
MLP training script. All checkpointing delegated to CheckpointEngine.

Flow:
  1. Load job_config.json from GCS
  2. Build model + optimizer
  3. engine.load() → if RESUME_STEP > 0, downloads checkpoint_latest.pt
     from GCS (S3 fallback), restores model + optimizer weights
  4. Train loop:
       - every ckpt_every steps → engine.save() → GCS + S3 simultaneously
       - every 10 batches → check GCP preemption metadata
       - every epoch → engine.save() + write_terminal_state if needed
  5. On preemption / budget exceeded → engine.write_terminal_state()
     with status=preempted → server.py poller picks it up → relaunches

Checkpoint files written to GCS + S3:
  checkpoints/{job_id}/
    job_config.json            ← written by launcher before VM starts
    job_state.json             ← updated by engine after every save
    checkpoint_latest.pt       ← always latest, used by engine.load()
    step_{N:08d}.pt            ← milestone saves every ckpt_every steps
"""

import os, sys, json, time, asyncio
from datetime import datetime, timezone

# ── torch ─────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    print("[ERROR] pip install torch --index-url https://download.pytorch.org/whl/cpu")
    sys.exit(1)

# ── GCS (only used to read job_config.json at startup) ────────────
try:
    from google.cloud import storage as gcs_lib
    GCS_OK = True
except ImportError:
    GCS_OK = False
    print("[WARN] pip install google-cloud-storage")


# ══════════════════════════════════════════════════════════════════
# CONFIG  — read job_config.json from GCS
# ══════════════════════════════════════════════════════════════════

def load_config() -> dict:
    """
    Load job config written by launcher.launch_job() before this VM booted.

    Decision: train.py reads config from GCS directly using a minimal
    gcs_lib call here (not via CheckpointEngine) because engine needs
    job_id to init, which comes from the config. Chicken-and-egg.
    Fallback: local job_config.json for local testing.
    """
    bucket = os.environ.get("GCS_BUCKET", "")
    job_id = os.environ.get("JOB_ID",     "local-test")

    config = None

    if bucket and GCS_OK:
        try:
            blob   = gcs_lib.Client().bucket(bucket).blob(
                        f"checkpoints/{job_id}/job_config.json")
            config = json.loads(blob.download_as_text())
            print(f"[config] Loaded from GCS: checkpoints/{job_id}/job_config.json")
        except Exception as e:
            print(f"[config] GCS read failed: {e}")

    if config is None and os.path.exists("job_config.json"):
        with open("job_config.json") as f:
            config = json.load(f)
        print("[config] Loaded from local job_config.json")

    if config is None:
        print("[config] No config found — using defaults")
        config = {}

    # Defaults — all keys the training loop needs
    config.setdefault("job_id",       job_id)
    config.setdefault("task_name",    "Untitled")
    config.setdefault("lr",           0.001)
    config.setdefault("hidden_dim",   256)
    config.setdefault("dropout",      0.3)
    config.setdefault("batch_size",   64)
    config.setdefault("epochs",       50)
    config.setdefault("ckpt_every",   50)
    config.setdefault("input_dim",    50)
    config.setdefault("num_classes",  5)
    config.setdefault("max_budget",   2.0)
    config.setdefault("gcs_bucket",   bucket)
    config.setdefault("price_usd_hr", 0.067)   # e2-standard-4 spot approx
    config.setdefault("migration_count", 0)

    # env always wins over config file
    if os.environ.get("JOB_ID"):     config["job_id"]     = os.environ["JOB_ID"]
    if os.environ.get("GCS_BUCKET"): config["gcs_bucket"] = os.environ["GCS_BUCKET"]

    return config


# ══════════════════════════════════════════════════════════════════
# PREEMPTION + BUDGET CHECKS
# ══════════════════════════════════════════════════════════════════

def check_preemption() -> bool:
    """
    Poll GCP instance metadata server for preemption notice.
    GCP gives ~30s warning before killing a spot VM.
    Returns False on any error (non-GCP machine, network issue).
    """
    try:
        import urllib.request
        req = urllib.request.Request(
            "http://metadata.google.internal/computeMetadata/v1/instance/preempted",
            headers={"Metadata-Flavor": "Google"}
        )
        with urllib.request.urlopen(req, timeout=1) as r:
            return r.read().decode().strip().lower() == "true"
    except Exception:
        return False


def check_budget(cfg: dict, start_time: float) -> bool:
    elapsed_hrs = (time.time() - start_time) / 3600
    cost_so_far = elapsed_hrs * float(cfg["price_usd_hr"])
    return cost_so_far >= float(cfg["max_budget"])


def runtime_stats(cfg: dict, start_time: float) -> dict:
    """Compute cost/time stats to pass into engine.save() extra_state."""
    elapsed_hrs = (time.time() - start_time) / 3600
    return {
        "elapsed_hrs": round(elapsed_hrs, 4),
        "cost_usd":    round(elapsed_hrs * float(cfg["price_usd_hr"]), 4),
        "cloud":       os.environ.get("CLOUD",         "gcp"),
        "instance":    os.environ.get("INSTANCE_TYPE", "e2-standard-4"),
    }


# ══════════════════════════════════════════════════════════════════
# DATASET  — synthetic tabular, matches modal's synthetic-500k
# ══════════════════════════════════════════════════════════════════

def make_dataset(cfg: dict):
    """
    Decision: use 10k/2k synthetic rows for CPU speed.
    The modal says synthetic-500k but training all 500k rows on
    an e2-standard-4 CPU would take hours per epoch.
    10k rows gives a realistic training loop that demonstrates
    checkpointing and preemption without burning budget.
    """
    n_tr  = 10_000
    n_va  =  2_000
    feat  = int(cfg["input_dim"])
    cls   = int(cfg["num_classes"])

    print(f"[data] Synthetic: {n_tr} train / {n_va} val | feat={feat} cls={cls}")
    torch.manual_seed(42)

    tr = DataLoader(
        TensorDataset(torch.randn(n_tr, feat), torch.randint(0, cls, (n_tr,))),
        batch_size=int(cfg["batch_size"]), shuffle=True
    )
    va = DataLoader(
        TensorDataset(torch.randn(n_va, feat), torch.randint(0, cls, (n_va,))),
        batch_size=256
    )
    return tr, va


# ══════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════

class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,  hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden,  hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden,  out_dim),
        )
    def forward(self, x):
        return self.net(x)


# ══════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════════════════

def train(cfg: dict):
    print("\n" + "="*60)
    print(f"  job_id   : {cfg['job_id']}")
    print(f"  task     : {cfg['task_name']}")
    print(f"  cloud    : {os.environ.get('CLOUD','gcp')} / "
          f"{os.environ.get('INSTANCE_TYPE','e2-standard-4')}")
    print(f"  lr={cfg['lr']}  hidden={cfg['hidden_dim']}  "
          f"dropout={cfg['dropout']}  batch={cfg['batch_size']}")
    print(f"  epochs={cfg['epochs']}  ckpt_every={cfg['ckpt_every']}")
    print(f"  budget=${cfg['max_budget']}  "
          f"price=${cfg['price_usd_hr']}/hr")
    print(f"  RESUME_STEP={os.environ.get('RESUME_STEP','0')}")
    print("="*60 + "\n")

    start_time = time.time()

    # ── Model + optimizer ─────────────────────────────────────────
    model     = MLP(int(cfg["input_dim"]), int(cfg["hidden_dim"]),
                    int(cfg["num_classes"]), float(cfg["dropout"]))
    optimizer = optim.Adam(model.parameters(), lr=float(cfg["lr"]))
    criterion = nn.CrossEntropyLoss()

    # ── CheckpointEngine ─────────────────────────────────────────
    # Decision: init engine AFTER model so load() can restore weights
    from checkpoint.engine import CheckpointEngine
    engine = CheckpointEngine(job_id=cfg["job_id"])

    # ── Resume if RESUME_STEP > 0 ─────────────────────────────────
    # engine.load() downloads checkpoint_latest.pt from GCS (S3 fallback)
    # restores model.state_dict() + optimizer.state_dict() in-place
    meta = engine.load(model, optimizer)
    if meta:
        start_epoch  = meta["epoch"]       # resume from this epoch
        global_step  = meta["step"]        # exact step count restored
        last_loss    = meta["loss"]
        resumed_from = meta["step"]
        print(f"[resume] Restored to epoch={start_epoch} step={global_step}\n")
    else:
        start_epoch  = 1
        global_step  = 0
        last_loss    = 999.0
        resumed_from = 0

    train_loader, val_loader = make_dataset(cfg)

    # Write initial running state so dashboard shows job immediately
    asyncio.run(engine.write_terminal_state(
        epoch=start_epoch, step=global_step, loss=last_loss,
        extra={
            "task_name":       cfg["task_name"],
            "status":          "running",
            "total_epochs":    cfg["epochs"],
            "resumed_from":    resumed_from,
            "migration_count": cfg.get("migration_count", 0),
            **runtime_stats(cfg, start_time),
        }
    ))
    print("[state] Initial job_state.json written\n")

    best_val_acc = 0.0
    avg_val_loss = last_loss
    preempted    = False

    # ── Epoch loop ────────────────────────────────────────────────
    for epoch in range(start_epoch, int(cfg["epochs"]) + 1):

        # Pre-epoch preemption check
        if check_preemption():
            print(f"\n[!!!] PREEMPTION before epoch {epoch} step {global_step}")
            asyncio.run(engine.save(
                model, optimizer,
                epoch=epoch, step=global_step, loss=avg_val_loss,
                extra_state={
                    "task_name":        cfg["task_name"],
                    "status":           "preempted",
                    "total_epochs":     cfg["epochs"],
                    "accuracy":         best_val_acc,
                    "best_val_acc":     best_val_acc,
                    "resumed_from":     resumed_from,
                    "migration_count":  cfg.get("migration_count", 0),
                    "preemption_source": "gcp_metadata_pre_epoch",
                    **runtime_stats(cfg, start_time),
                }
            ))
            preempted = True
            break

        # Pre-epoch budget check
        if check_budget(cfg, start_time):
            stats = runtime_stats(cfg, start_time)
            print(f"\n[budget] Limit reached: ${stats['cost_usd']:.4f} "
                  f">= ${cfg['max_budget']}")
            asyncio.run(engine.save(
                model, optimizer,
                epoch=epoch, step=global_step, loss=avg_val_loss,
                extra_state={
                    "task_name":       cfg["task_name"],
                    "status":          "budget_exceeded",
                    "total_epochs":    cfg["epochs"],
                    "accuracy":        best_val_acc,
                    "best_val_acc":    best_val_acc,
                    "resumed_from":    resumed_from,
                    "migration_count": cfg.get("migration_count", 0),
                    **stats,
                }
            ))
            break

        # ── Batch loop ────────────────────────────────────────────
        model.train()
        run_loss = 0.0
        correct  = 0
        total    = 0

        for i, (X, y) in enumerate(train_loader):

            # Preemption check every 10 batches
            # Decision: every 10 batches gives ~3-5s polling frequency
            # on a CPU VM with 10k dataset. GCP gives 30s warning
            # so we have 6+ polls before VM dies.
            if i % 10 == 0 and check_preemption():
                print(f"\n[!!!] PREEMPTION mid-epoch "
                      f"epoch={epoch} batch={i} step={global_step}")
                avg = run_loss / max(i, 1)
                asyncio.run(engine.save(
                    model, optimizer,
                    epoch=epoch, step=global_step, loss=avg,
                    extra_state={
                        "task_name":         cfg["task_name"],
                        "status":            "preempted",
                        "total_epochs":      cfg["epochs"],
                        "accuracy":          best_val_acc,
                        "best_val_acc":      best_val_acc,
                        "resumed_from":      resumed_from,
                        "migration_count":   cfg.get("migration_count", 0),
                        "preemption_source": "gcp_metadata_mid_epoch",
                        **runtime_stats(cfg, start_time),
                    }
                ))
                preempted = True
                break

            # Forward + backward
            optimizer.zero_grad()
            out  = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            run_loss    += loss.item()
            correct     += (out.argmax(1) == y).sum().item()
            total       += y.size(0)
            global_step += 1

            # Periodic checkpoint every ckpt_every steps
            if global_step % int(cfg["ckpt_every"]) == 0:
                avg  = run_loss / (i + 1)
                acc  = correct / total
                stats = runtime_stats(cfg, start_time)
                print(f"  e{epoch:3d} | step {global_step:5d} | "
                      f"loss {avg:.4f} | acc {acc:.3f} | "
                      f"${stats['cost_usd']:.4f}")

                # engine.save() writes:
                #   step_{N:08d}.pt  → GCS + S3
                #   checkpoint_latest.pt → GCS + S3
                #   job_state.json   → GCS only
                asyncio.run(engine.save(
                    model, optimizer,
                    epoch=epoch, step=global_step, loss=avg,
                    extra_state={
                        "task_name":       cfg["task_name"],
                        "status":          "running",
                        "total_epochs":    cfg["epochs"],
                        "accuracy":        acc,
                        "best_val_acc":    best_val_acc,
                        "resumed_from":    resumed_from,
                        "migration_count": cfg.get("migration_count", 0),
                        **stats,
                    }
                ))

        if preempted:
            break

        # ── Validation ────────────────────────────────────────────
        model.eval()
        v_loss = 0.0
        v_ok   = 0
        v_tot  = 0
        with torch.no_grad():
            for X, y in val_loader:
                out     = model(X)
                v_loss += criterion(out, y).item()
                v_ok   += (out.argmax(1) == y).sum().item()
                v_tot  += y.size(0)

        val_acc      = v_ok / v_tot
        avg_val_loss = v_loss / len(val_loader)
        avg_tr_loss  = run_loss / len(train_loader)
        if val_acc > best_val_acc:
            best_val_acc = val_acc

        stats = runtime_stats(cfg, start_time)
        print(f"Epoch {epoch:3d}/{cfg['epochs']} | "
              f"tr={avg_tr_loss:.4f} val={avg_val_loss:.4f} "
              f"val_acc={val_acc:.3f} best={best_val_acc:.3f} "
              f"${stats['cost_usd']:.4f}")

        # End-of-epoch checkpoint via engine
        asyncio.run(engine.save(
            model, optimizer,
            epoch=epoch, step=global_step, loss=avg_val_loss,
            extra_state={
                "task_name":       cfg["task_name"],
                "status":          "running",
                "total_epochs":    cfg["epochs"],
                "accuracy":        val_acc,
                "best_val_acc":    best_val_acc,
                "train_loss":      avg_tr_loss,
                "resumed_from":    resumed_from,
                "migration_count": cfg.get("migration_count", 0),
                **stats,
            }
        ))

    # ── Terminal states ───────────────────────────────────────────
    if not preempted:
        stats = runtime_stats(cfg, start_time)
        print(f"\n{'='*60}")
        print(f"  Done!  best_val_acc={best_val_acc:.4f}")
        print(f"  Time:  {stats['elapsed_hrs']*60:.1f} min")
        print(f"  Cost:  ${stats['cost_usd']:.4f}")
        print(f"{'='*60}\n")

        # Final state — status=done
        asyncio.run(engine.write_terminal_state(
            epoch=int(cfg["epochs"]), step=global_step,
            loss=avg_val_loss,
            extra={
                "task_name":       cfg["task_name"],
                "status":          "done",
                "total_epochs":    cfg["epochs"],
                "accuracy":        best_val_acc,
                "best_val_acc":    best_val_acc,
                "train_loss":      avg_tr_loss,
                "resumed_from":    resumed_from,
                "migration_count": cfg.get("migration_count", 0),
                **stats,
            }
        ))


# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cfg = load_config()
    train(cfg)