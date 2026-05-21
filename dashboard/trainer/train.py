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
       - top of every epoch → check_command() for server-side commands
       - every 10 batches → check GCP preemption metadata
       - every ckpt_every steps → engine.save() → GCS + S3 simultaneously
       - every epoch → engine.save() + write_terminal_state if needed
  5. On preemption / budget exceeded → engine.write_terminal_state()
     with status=preempted → server.py poller picks it up → relaunches

Commands (written by server.py to GCS, consumed once):
    migrate    → save checkpoint, exit with status=preempted
    stop       → save final checkpoint, exit cleanly
    reduce_lr  → multiply all param group LRs by 0.1, continue training

Checkpoint files written to GCS + S3:
  checkpoints/{job_id}/
    job_config.json            ← written by launcher before VM starts
    job_state.json             ← updated by engine after every save
    job_command.json           ← written by server.py, consumed here
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
# COMMAND POLLING  — server.py writes these, train.py consumes once
# ══════════════════════════════════════════════════════════════════

def check_command(cfg: dict) -> str | None:
    """
    Check GCS for a command written by server.py poller.
    Returns command string or None.
    Deletes the command file after reading (so it fires only once).

    Supported commands:
        migrate    → save checkpoint, exit with status=preempted
        stop       → save final checkpoint, exit cleanly
        reduce_lr  → multiply all LRs by 0.1, continue training
    """
    if not GCS_OK or not cfg.get("gcs_bucket"):
        return None
    try:
        bucket = gcs_lib.Client().bucket(cfg["gcs_bucket"])
        blob   = bucket.blob(f"checkpoints/{cfg['job_id']}/job_command.json")
        if not blob.exists():
            return None
        cmd = json.loads(blob.download_as_text()).get("command")
        blob.delete()   # consume command — won't fire again
        print(f"[command] Received: {cmd}")
        return cmd
    except Exception:
        return None


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
# DATASET
# ══════════════════════════════════════════════════════════════════

def _download_s3_dataset(s3_path: str, local_dir: str) -> list:
    """
    Download all CSV files from an S3 path prefix to local_dir.
    s3_path format: s3://bucket-name/path/to/data/
    Returns list of local file paths downloaded.
    """
    import boto3, re
    from pathlib import Path

    match = re.match(r"s3://([^/]+)/?(.*)", s3_path.rstrip("/") + "/")
    if not match:
        raise ValueError(f"Invalid S3 path: {s3_path!r}  expected s3://bucket/prefix/")
    bucket_name = match.group(1)
    prefix      = match.group(2)

    print(f"[data] Connecting to S3 bucket: {bucket_name}  prefix: {prefix!r}")

    s3 = boto3.client(
        "s3",
        aws_access_key_id     = os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region_name           = os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    )

    Path(local_dir).mkdir(parents=True, exist_ok=True)

    paginator   = s3.get_paginator("list_objects_v2")
    local_files = []
    total_bytes = 0

    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".csv"):
                continue
            filename   = Path(key).name
            local_path = os.path.join(local_dir, filename)
            print(f"  [s3] downloading s3://{bucket_name}/{key} ({obj['Size']//1024}KB)")
            s3.download_file(bucket_name, key, local_path)
            local_files.append(local_path)
            total_bytes += obj["Size"]

    if not local_files:
        raise FileNotFoundError(
            f"No .csv files found at s3://{bucket_name}/{prefix} — "
            f"make sure your S3 path ends with / and contains .csv files."
        )

    print(f"[data] Downloaded {len(local_files)} files "
          f"({total_bytes // 1024}KB total) to {local_dir}")
    return local_files


def _load_csv_dataset(csv_files: list, cfg: dict):
    """
    Load CSV files into PyTorch tensors.
    Assumes last column is the label, all others are features.
    """
    try:
        import pandas as pd
    except ImportError:
        raise RuntimeError("pip install pandas")

    dfs = [pd.read_csv(f) for f in csv_files]
    df  = pd.concat(dfs, ignore_index=True)

    print(f"[data] Loaded {len(df)} rows, {len(df.columns)} columns from CSV")

    df_numeric = df.select_dtypes(include="number")
    if df_numeric.shape[1] < 2:
        raise ValueError(
            f"CSV has fewer than 2 numeric columns: {list(df.columns)} — "
            "need at least 1 feature + 1 label column."
        )

    X = df_numeric.iloc[:, :-1].values.astype("float32")
    y = df_numeric.iloc[:,  -1].values.astype("int64")

    if X.shape[1] != int(cfg["input_dim"]):
        print(f"[data] WARN: CSV has {X.shape[1]} features but "
              f"input_dim={cfg['input_dim']}. Updating config.")
        cfg["input_dim"] = X.shape[1]

    n_classes = len(set(y))
    if n_classes != int(cfg["num_classes"]):
        print(f"[data] WARN: CSV has {n_classes} classes but "
              f"num_classes={cfg['num_classes']}. Updating config.")
        cfg["num_classes"] = n_classes

    print(f"[data] Final: {X.shape[0]} rows, {X.shape[1]} features, "
          f"{n_classes} classes")
    return torch.tensor(X), torch.tensor(y)


def make_dataset(cfg: dict):
    """
    Returns (train_loader, val_loader).

    Routing:
      dataset_type == "custom"      → download from S3, load CSV
      dataset_type == "synthetic-*" → generate synthetic data
    """
    dataset_type = cfg.get("dataset_type", "synthetic-500k")
    feat         = int(cfg["input_dim"])
    cls          = int(cfg["num_classes"])
    batch        = int(cfg["batch_size"])

    if dataset_type == "custom":
        s3_path = cfg.get("s3_dataset_path", "").strip()
        if not s3_path:
            raise ValueError(
                "dataset_type=custom but s3_dataset_path is empty in job_config.json"
            )
        print(f"[data] Loading custom dataset from S3: {s3_path}")
        local_dir  = "/tmp/dataset"
        csv_files  = _download_s3_dataset(s3_path, local_dir)
        X_t, y_t   = _load_csv_dataset(csv_files, cfg)
        n          = len(X_t)
        n_train    = int(n * 0.8)
        indices    = torch.randperm(n)
        X_tr, y_tr = X_t[indices[:n_train]],  y_t[indices[:n_train]]
        X_va, y_va = X_t[indices[n_train:]], y_t[indices[n_train:]]
        print(f"[data] Split: {len(X_tr)} train / {len(X_va)} val")
    else:
        n_tr = 10_000
        n_va =  2_000
        print(f"[data] Synthetic: {n_tr} train / {n_va} val | feat={feat} cls={cls}")
        torch.manual_seed(42)
        X_tr = torch.randn(n_tr, feat)
        y_tr = torch.randint(0, cls, (n_tr,))
        X_va = torch.randn(n_va, feat)
        y_va = torch.randint(0, cls, (n_va,))

    tr = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch, shuffle=True)
    va = DataLoader(TensorDataset(X_va, y_va), batch_size=256)
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
    print(f"  budget=${cfg['max_budget']}  price=${cfg['price_usd_hr']}/hr")
    print(f"  RESUME_STEP={os.environ.get('RESUME_STEP','0')}")
    print("="*60 + "\n")

    start_time = time.time()

    # ── Model + optimizer ─────────────────────────────────────────
    model     = MLP(int(cfg["input_dim"]), int(cfg["hidden_dim"]),
                    int(cfg["num_classes"]), float(cfg["dropout"]))
    optimizer = optim.Adam(model.parameters(), lr=float(cfg["lr"]))
    criterion = nn.CrossEntropyLoss()

    # ── CheckpointEngine ──────────────────────────────────────────
    from checkpoint.engine import CheckpointEngine
    engine = CheckpointEngine(job_id=cfg["job_id"])

    # ── Resume if RESUME_STEP > 0 ─────────────────────────────────
    meta = engine.load(model, optimizer)
    if meta:
        start_epoch  = meta["epoch"]
        global_step  = meta["step"]
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
    avg_tr_loss  = last_loss
    preempted    = False

    # ── Epoch loop ────────────────────────────────────────────────
    for epoch in range(start_epoch, int(cfg["epochs"]) + 1):

        # ── Check for server-side command at top of every epoch ───
        # server.py writes job_command.json; we consume it once here.
        cmd = check_command(cfg)
        if cmd == "migrate":
            print(f"[command] migrate — saving checkpoint and exiting")
            asyncio.run(engine.save(
                model, optimizer,
                epoch=epoch, step=global_step, loss=avg_val_loss,
                extra_state={
                    "task_name":         cfg["task_name"],
                    "status":            "preempted",
                    "total_epochs":      cfg["epochs"],
                    "accuracy":          best_val_acc,
                    "best_val_acc":      best_val_acc,
                    "resumed_from":      resumed_from,
                    "migration_count":   cfg.get("migration_count", 0),
                    "preemption_source": "server_command_migrate",
                    **runtime_stats(cfg, start_time),
                }
            ))
            preempted = True
            break

        elif cmd == "stop":
            print(f"[command] stop — saving final checkpoint")
            asyncio.run(engine.write_terminal_state(
                epoch=epoch, step=global_step, loss=avg_val_loss,
                extra={
                    "task_name":       cfg["task_name"],
                    "status":          "done",
                    "total_epochs":    cfg["epochs"],
                    "accuracy":        best_val_acc,
                    "best_val_acc":    best_val_acc,
                    "train_loss":      avg_tr_loss,
                    "resumed_from":    resumed_from,
                    "migration_count": cfg.get("migration_count", 0),
                    **runtime_stats(cfg, start_time),
                }
            ))
            break

        elif cmd == "reduce_lr":
            for g in optimizer.param_groups:
                g["lr"] *= 0.1
            new_lr = optimizer.param_groups[0]["lr"]
            print(f"[command] reduce_lr — new lr={new_lr:.6f}")
            # Continue training — do not break

        # ── Pre-epoch preemption check ────────────────────────────
        if check_preemption():
            print(f"\n[!!!] PREEMPTION before epoch {epoch} step {global_step}")
            asyncio.run(engine.save(
                model, optimizer,
                epoch=epoch, step=global_step, loss=avg_val_loss,
                extra_state={
                    "task_name":         cfg["task_name"],
                    "status":            "preempted",
                    "total_epochs":      cfg["epochs"],
                    "accuracy":          best_val_acc,
                    "best_val_acc":      best_val_acc,
                    "resumed_from":      resumed_from,
                    "migration_count":   cfg.get("migration_count", 0),
                    "preemption_source": "gcp_metadata_pre_epoch",
                    **runtime_stats(cfg, start_time),
                }
            ))
            preempted = True
            break

        # ── Pre-epoch budget check ────────────────────────────────
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
                avg   = run_loss / (i + 1)
                acc   = correct / total
                stats = runtime_stats(cfg, start_time)
                print(f"  e{epoch:3d} | step {global_step:5d} | "
                      f"loss {avg:.4f} | acc {acc:.3f} | "
                      f"${stats['cost_usd']:.4f}")

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

        # End-of-epoch checkpoint
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

    # ── Terminal state ────────────────────────────────────────────
    if not preempted:
        stats = runtime_stats(cfg, start_time)
        print(f"\n{'='*60}")
        print(f"  Done!  best_val_acc={best_val_acc:.4f}")
        print(f"  Time:  {stats['elapsed_hrs']*60:.1f} min")
        print(f"  Cost:  ${stats['cost_usd']:.4f}")
        print(f"{'='*60}\n")

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