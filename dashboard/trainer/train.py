"""
trainer/train.py
─────────────────
Training script that runs on the spawned VM.
Reads job_config.json from S3, trains the model, checkpoints every N steps.

Features:
  - model_arch: MLP / Transformer / CNN / RNN
  - precision:  fp32 / fp16 / bf16 / int8
  - train_mode: manual (existing) / sweep (Hyperband over lr + hidden_dim)
  - training_paradigm: fine-tuning / pre-training / rl / distillation

Flow:
  1. Load job_config.json from S3
  2. Apply precision (cast model)
  3. Build model from model_arch
  4. If train_mode=sweep → run Hyperband, pick best config, train final model
     If train_mode=manual → existing training loop unchanged
  5. Checkpoint every ckpt_every steps to S3
  6. On preemption / budget exceeded → write terminal state → poller relaunches

Commands (written by server.py to S3, consumed once):
    migrate    → save checkpoint, exit with status=preempted
    stop       → save final checkpoint, exit cleanly
    reduce_lr  → multiply all param group LRs by 0.1, continue training

All checkpoint files written to S3:
  checkpoints/{job_id}/
    job_config.json            ← written by launcher before VM starts
    job_state.json             ← updated by engine after every save
    job_command.json           ← written by server.py, consumed here
    checkpoint_latest.pt       ← always latest, used by engine.load()
    step_{N:08d}.pt            ← milestone saves every ckpt_every steps
"""

import os, sys, json, time, asyncio, math
from datetime import datetime, timezone

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    print("[ERROR] pip install torch --index-url https://download.pytorch.org/whl/cpu")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════
# S3 CLIENT HELPER
# ══════════════════════════════════════════════════════════════════

def _s3_client():
    import boto3
    return boto3.client(
        "s3",
        aws_access_key_id     = os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region_name           = os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    )


def _s3_read_json(s3_path: str):
    """Read a JSON file from S3. Returns dict or None on any error."""
    bucket = os.environ.get("CHECKPOINT_S3_BUCKET", "")
    if not bucket:
        return None
    try:
        obj  = _s3_client().get_object(Bucket=bucket, Key=s3_path)
        return json.loads(obj["Body"].read().decode())
    except Exception as e:
        print(f"[s3] read_json({s3_path}) failed: {e}")
        return None


def _s3_delete(s3_path: str):
    """Delete a key from S3 (used to consume job_command.json)."""
    bucket = os.environ.get("CHECKPOINT_S3_BUCKET", "")
    if not bucket:
        return
    try:
        _s3_client().delete_object(Bucket=bucket, Key=s3_path)
    except Exception as e:
        print(f"[s3] delete({s3_path}) failed: {e}")


def _s3_key_exists(s3_path: str) -> bool:
    bucket = os.environ.get("CHECKPOINT_S3_BUCKET", "")
    if not bucket:
        return False
    try:
        _s3_client().head_object(Bucket=bucket, Key=s3_path)
        return True
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════
# CONFIG  — read job_config.json from S3
# ══════════════════════════════════════════════════════════════════

def load_config() -> dict:
    """
    Load job config written by launcher.submit_job() before this VM booted.
    Reads from S3: s3://CHECKPOINT_S3_BUCKET/checkpoints/{job_id}/job_config.json
    Fallback: local job_config.json for local testing.
    """
    job_id = os.environ.get("JOB_ID", "local-test")
    bucket = os.environ.get("CHECKPOINT_S3_BUCKET", "")

    config = None

    if bucket:
        s3_key = f"checkpoints/{job_id}/job_config.json"
        config = _s3_read_json(s3_key)
        if config:
            print(f"[config] Loaded from S3: s3://{bucket}/{s3_key}")
        else:
            print(f"[config] S3 read failed for {s3_key}")

    if config is None and os.path.exists("job_config.json"):
        with open("job_config.json") as f:
            config = json.load(f)
        print("[config] Loaded from local job_config.json")

    if config is None:
        print("[config] No config found — using defaults")
        config = {}

    # Defaults
    config.setdefault("job_id",            job_id)
    config.setdefault("task_name",         "Untitled")
    config.setdefault("train_mode",        "manual")
    config.setdefault("lr",                0.001)
    config.setdefault("hidden_dim",        256)
    config.setdefault("dropout",           0.3)
    config.setdefault("batch_size",        64)
    config.setdefault("epochs",            50)
    config.setdefault("ckpt_every",        50)
    config.setdefault("input_dim",         50)
    config.setdefault("num_classes",       5)
    config.setdefault("max_budget",        2.0)
    config.setdefault("price_usd_hr",      0.034)
    config.setdefault("migration_count",   0)
    config.setdefault("model_arch",        "mlp")
    config.setdefault("precision",         "fp32")
    config.setdefault("training_paradigm", "fine-tuning")
    config.setdefault("sweep_lr_min",      0.0001)
    config.setdefault("sweep_lr_max",      0.01)
    config.setdefault("sweep_hidden",      [256])
    config.setdefault("sweep_trials",      5)
    config.setdefault("sweep_budget",      5.0)
    config.setdefault("dataset_type",      "synthetic-500k")
    config.setdefault("s3_dataset_path",   "")

    if os.environ.get("JOB_ID"):
        config["job_id"] = os.environ["JOB_ID"]

    return config


# ══════════════════════════════════════════════════════════════════
# PRECISION
# ══════════════════════════════════════════════════════════════════

def apply_precision(model: nn.Module, precision: str, device: torch.device):
    """Cast model to the requested precision."""
    precision = precision.lower()

    if precision == "fp16":
        model = model.to(device).half()
        print(f"[precision] fp16 — model cast to float16")

    elif precision == "bf16":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            model = model.to(device).to(torch.bfloat16)
            print(f"[precision] bf16 — model cast to bfloat16")
        else:
            model = model.to(device).half()
            print(f"[precision] bf16 requested but not supported — falling back to fp16")

    elif precision == "int8":
        model = model.to(device)
        print(f"[precision] int8 — training in fp32, quantization applied post-training")

    else:
        model = model.to(device).float()
        print(f"[precision] fp32 — standard float32")

    return model


def get_autocast_context(precision: str, device: torch.device):
    """Returns a context manager for mixed precision forward passes."""
    if precision in ("fp16", "bf16") and device.type == "cuda":
        dtype = torch.float16 if precision == "fp16" else torch.bfloat16
        return torch.amp.autocast(device_type="cuda", dtype=dtype)
    import contextlib
    return contextlib.nullcontext()


def cast_batch(X, y, precision: str, device: torch.device):
    """Move batch to device and cast X to match model precision."""
    X = X.to(device)
    y = y.to(device)
    if precision == "fp16":
        X = X.half()
    elif precision == "bf16" and device.type == "cuda" and torch.cuda.is_bf16_supported():
        X = X.to(torch.bfloat16)
    else:
        X = X.float()
    return X, y


# ══════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURES
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


class TabularTransformer(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout, n_heads=4, n_layers=2):
        super().__init__()
        self.embed   = nn.Linear(1, hidden)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=hidden * 2,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pool    = nn.AdaptiveAvgPool1d(1)
        self.head    = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, out_dim)
        )
        self.in_dim  = in_dim

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.embed(x)
        x = self.transformer(x)
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        return self.head(x)


class CNN1D(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, hidden // 2, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(hidden // 2, hidden, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, out_dim)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        return self.head(x)


class RNNClassifier(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout, n_layers=2):
        super().__init__()
        self.gru  = nn.GRU(
            input_size=1, hidden_size=hidden,
            num_layers=n_layers, dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, out_dim)
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        _, h = self.gru(x)
        return self.head(h[-1])


def build_model(cfg: dict) -> nn.Module:
    arch    = cfg.get("model_arch", "mlp").lower()
    in_dim  = int(cfg["input_dim"])
    hidden  = int(cfg["hidden_dim"])
    out_dim = int(cfg["num_classes"])
    dropout = float(cfg["dropout"])

    if arch == "transformer":
        n_heads = 4 if hidden >= 64 else 2 if hidden >= 32 else 1
        model   = TabularTransformer(in_dim, hidden, out_dim, dropout, n_heads=n_heads)
        print(f"[model] TabularTransformer — hidden={hidden} heads={n_heads}")
    elif arch == "cnn":
        model = CNN1D(in_dim, hidden, out_dim, dropout)
        print(f"[model] CNN1D — hidden={hidden}")
    elif arch == "rnn":
        model = RNNClassifier(in_dim, hidden, out_dim, dropout)
        print(f"[model] RNNClassifier (GRU) — hidden={hidden}")
    else:
        model = MLP(in_dim, hidden, out_dim, dropout)
        print(f"[model] MLP — hidden={hidden}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] Parameters: {n_params:,}")
    return model


# ══════════════════════════════════════════════════════════════════
# COMMAND / PREEMPTION / BUDGET CHECKS
# ══════════════════════════════════════════════════════════════════

def check_command(cfg: dict):
    """
    Check S3 for a command written by server.py.
    Returns command string or None. Deletes the file after reading.
    """
    s3_key = f"checkpoints/{cfg['job_id']}/job_command.json"
    if not _s3_key_exists(s3_key):
        return None
    try:
        data = _s3_read_json(s3_key)
        if not data:
            return None
        cmd = data.get("command")
        _s3_delete(s3_key)
        print(f"[command] Received: {cmd}")
        return cmd
    except Exception as e:
        print(f"[command] check_command failed: {e}")
        return None


def check_preemption() -> bool:
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
    elapsed_hrs = (time.time() - start_time) / 3600
    return {
        "elapsed_hrs": round(elapsed_hrs, 4),
        "cost_usd":    round(elapsed_hrs * float(cfg["price_usd_hr"]), 4),
        "cloud":       os.environ.get("CLOUD",         "unknown"),
        "instance":    os.environ.get("INSTANCE_TYPE", "unknown"),
    }


# ══════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════

def _download_s3_dataset(s3_path: str, local_dir: str) -> list:
    """Download all CSV files from an S3 path prefix to local_dir."""
    import re
    from pathlib import Path
    match = re.match(r"s3://([^/]+)/?(.*)", s3_path.rstrip("/") + "/")
    if not match:
        raise ValueError(f"Invalid S3 path: {s3_path!r}")
    bucket_name = match.group(1)
    prefix      = match.group(2)

    print(f"[data] Connecting to S3 bucket: {bucket_name}  prefix: {prefix!r}")
    s3 = _s3_client()

    Path(local_dir).mkdir(parents=True, exist_ok=True)
    paginator   = s3.get_paginator("list_objects_v2")
    local_files = []
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".csv"):
                continue
            local_path = os.path.join(local_dir, Path(key).name)
            s3.download_file(bucket_name, key, local_path)
            local_files.append(local_path)
    if not local_files:
        raise FileNotFoundError(f"No .csv files at {s3_path}")
    return local_files


def _load_csv_dataset(csv_files: list, cfg: dict):
    import pandas as pd
    dfs = [pd.read_csv(f) for f in csv_files]
    df  = pd.concat(dfs, ignore_index=True).select_dtypes(include="number")
    X   = df.iloc[:, :-1].values.astype("float32")
    y   = df.iloc[:,  -1].values.astype("int64")
    if X.shape[1] != int(cfg["input_dim"]):
        cfg["input_dim"] = X.shape[1]
    cfg["num_classes"] = len(set(y))
    return torch.tensor(X), torch.tensor(y)


def make_dataset(cfg: dict):
    dataset_type = cfg.get("dataset_type", "synthetic-500k")
    feat  = int(cfg["input_dim"])
    cls   = int(cfg["num_classes"])
    batch = int(cfg["batch_size"])

    if dataset_type == "custom":
        s3_path   = cfg.get("s3_dataset_path", "").strip()
        csv_files = _download_s3_dataset(s3_path, "/tmp/dataset")
        X_t, y_t  = _load_csv_dataset(csv_files, cfg)
        n         = len(X_t)
        n_train   = int(n * 0.8)
        idx       = torch.randperm(n)
        X_tr, y_tr = X_t[idx[:n_train]],  y_t[idx[:n_train]]
        X_va, y_va = X_t[idx[n_train:]], y_t[idx[n_train:]]
    else:
        torch.manual_seed(42)
        X_tr = torch.randn(10_000, feat); y_tr = torch.randint(0, cls, (10_000,))
        X_va = torch.randn(2_000,  feat); y_va = torch.randint(0, cls, (2_000,))

    tr = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch, shuffle=True)
    va = DataLoader(TensorDataset(X_va, y_va), batch_size=256)
    return tr, va


# ══════════════════════════════════════════════════════════════════
# HYPERBAND SWEEP
# ══════════════════════════════════════════════════════════════════

def _sample_configs(cfg: dict) -> list[dict]:
    import random
    lr_min    = float(cfg["sweep_lr_min"])
    lr_max    = float(cfg["sweep_lr_max"])
    hiddens   = [int(h) for h in cfg.get("sweep_hidden", [256])]
    n_trials  = int(cfg["sweep_trials"])

    log_lrs = [
        math.exp(math.log(lr_min) + i * (math.log(lr_max) - math.log(lr_min)) / max(n_trials - 1, 1))
        for i in range(n_trials)
    ]
    random.shuffle(log_lrs)

    configs = []
    for i in range(n_trials):
        configs.append({
            **cfg,
            "lr":         round(log_lrs[i], 6),
            "hidden_dim": hiddens[i % len(hiddens)],
            "_trial_id":  i,
        })
    return configs


def _train_trial(trial_cfg: dict, train_loader, val_loader,
                 steps: int, device: torch.device) -> tuple[float, dict]:
    """Train a single Hyperband trial for `steps` gradient steps."""
    precision = trial_cfg.get("precision", "fp32")
    model     = build_model(trial_cfg)
    model     = apply_precision(model, precision, device)
    optimizer = optim.Adam(model.parameters(), lr=float(trial_cfg["lr"]))
    criterion = nn.CrossEntropyLoss()
    autocast  = get_autocast_context(precision, device)

    model.train()
    step = 0
    for X, y in train_loader:
        if step >= steps:
            break
        X, y = cast_batch(X, y, precision, device)
        optimizer.zero_grad()
        with autocast:
            loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        step += 1

    model.eval()
    v_loss = 0.0
    with torch.no_grad():
        for X, y in val_loader:
            X, y = cast_batch(X, y, precision, device)
            with autocast:
                v_loss += criterion(model(X), y).item()

    avg_val_loss = v_loss / max(len(val_loader), 1)
    trial_id     = trial_cfg.get("_trial_id", "?")
    print(f"  [sweep] trial={trial_id} lr={trial_cfg['lr']:.5f} "
          f"hidden={trial_cfg['hidden_dim']} steps={steps} "
          f"val_loss={avg_val_loss:.4f}")
    return avg_val_loss, trial_cfg


def run_hyperband(cfg: dict, train_loader, val_loader, device: torch.device) -> dict:
    """
    Successive Halving (Hyperband bracket 0).
    Returns the best config dict to use for final training.
    """
    n_trials  = int(cfg["sweep_trials"])
    min_steps = max(10, len(train_loader) // 2)
    eta       = 3
    trials    = _sample_configs(cfg)
    n_rounds  = math.ceil(math.log(n_trials, eta))

    print(f"\n[sweep] Hyperband: {n_trials} trials  "
          f"{n_rounds} rounds  min_steps={min_steps}  eta={eta}")

    for rnd in range(n_rounds):
        steps  = min_steps * (eta ** rnd)
        n_keep = max(1, math.ceil(len(trials) / eta))
        print(f"\n[sweep] Round {rnd+1}/{n_rounds}: "
              f"{len(trials)} trials × {int(steps)} steps → keep top {n_keep}")

        scored = []
        for trial_cfg in trials:
            val_loss, tc = _train_trial(trial_cfg, train_loader, val_loader,
                                        int(steps), device)
            scored.append((val_loss, tc))

        scored.sort(key=lambda x: x[0])
        trials = [tc for _, tc in scored[:n_keep]]

        if len(trials) == 1:
            break

    best = trials[0]
    print(f"\n[sweep] ✓ Best config: lr={best['lr']:.5f}  "
          f"hidden={best['hidden_dim']}  "
          f"(from {n_trials} trials)")
    return best


# ══════════════════════════════════════════════════════════════════
# CORE TRAINING LOOP
# ══════════════════════════════════════════════════════════════════

def _run_training_loop(cfg: dict, model: nn.Module, optimizer, criterion,
                       train_loader, val_loader, engine,
                       start_epoch: int, global_step: int, resumed_from: int,
                       start_time: float, device: torch.device,
                       precision: str) -> tuple[bool, float, float]:
    """Core epoch/batch loop. Returns (preempted, best_val_acc, avg_val_loss)."""
    autocast     = get_autocast_context(precision, device)
    best_val_acc = 0.0
    avg_val_loss = 999.0
    avg_tr_loss  = 999.0
    preempted    = False

    for epoch in range(start_epoch, int(cfg["epochs"]) + 1):

        # ── Command check ─────────────────────────────────────────
        cmd = check_command(cfg)
        if cmd == "migrate":
            asyncio.run(engine.save(
                model, optimizer, epoch=epoch, step=global_step,
                loss=avg_val_loss,
                extra_state={
                    "task_name": cfg["task_name"], "status": "preempted",
                    "total_epochs": cfg["epochs"], "accuracy": best_val_acc,
                    "best_val_acc": best_val_acc, "resumed_from": resumed_from,
                    "migration_count": cfg.get("migration_count", 0),
                    "preemption_source": "server_command_migrate",
                    **runtime_stats(cfg, start_time),
                }
            ))
            preempted = True
            break

        elif cmd == "stop":
            asyncio.run(engine.write_terminal_state(
                epoch=epoch, step=global_step, loss=avg_val_loss,
                extra={"task_name": cfg["task_name"], "status": "done",
                       "total_epochs": cfg["epochs"], "accuracy": best_val_acc,
                       "best_val_acc": best_val_acc, "train_loss": avg_tr_loss,
                       "resumed_from": resumed_from,
                       "migration_count": cfg.get("migration_count", 0),
                       **runtime_stats(cfg, start_time)}
            ))
            break

        elif cmd == "reduce_lr":
            for g in optimizer.param_groups:
                g["lr"] *= 0.1
            print(f"[command] reduce_lr → {optimizer.param_groups[0]['lr']:.6f}")

        # ── Preemption + budget checks ────────────────────────────
        if check_preemption():
            print(f"\n[!!!] PREEMPTION before epoch {epoch}")
            asyncio.run(engine.save(
                model, optimizer, epoch=epoch, step=global_step,
                loss=avg_val_loss,
                extra_state={
                    "task_name": cfg["task_name"], "status": "preempted",
                    "total_epochs": cfg["epochs"], "accuracy": best_val_acc,
                    "best_val_acc": best_val_acc, "resumed_from": resumed_from,
                    "migration_count": cfg.get("migration_count", 0),
                    "preemption_source": "gcp_metadata_pre_epoch",
                    **runtime_stats(cfg, start_time),
                }
            ))
            preempted = True
            break

        if check_budget(cfg, start_time):
            stats = runtime_stats(cfg, start_time)
            print(f"\n[budget] Limit reached: ${stats['cost_usd']:.4f}")
            asyncio.run(engine.save(
                model, optimizer, epoch=epoch, step=global_step,
                loss=avg_val_loss,
                extra_state={
                    "task_name": cfg["task_name"], "status": "budget_exceeded",
                    "total_epochs": cfg["epochs"], "accuracy": best_val_acc,
                    "best_val_acc": best_val_acc, "resumed_from": resumed_from,
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

            if i % 10 == 0 and check_preemption():
                print(f"\n[!!!] PREEMPTION mid-epoch e={epoch} b={i}")
                avg = run_loss / max(i, 1)
                asyncio.run(engine.save(
                    model, optimizer, epoch=epoch, step=global_step, loss=avg,
                    extra_state={
                        "task_name": cfg["task_name"], "status": "preempted",
                        "total_epochs": cfg["epochs"], "accuracy": best_val_acc,
                        "best_val_acc": best_val_acc, "resumed_from": resumed_from,
                        "migration_count": cfg.get("migration_count", 0),
                        "preemption_source": "gcp_metadata_mid_epoch",
                        **runtime_stats(cfg, start_time),
                    }
                ))
                preempted = True
                break

            X, y = cast_batch(X, y, precision, device)
            optimizer.zero_grad()
            with autocast:
                out  = model(X)
                loss = criterion(out, y)
            loss.backward()
            grad_clip = cfg.get("_grad_clip")
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            run_loss    += loss.item()
            correct     += (out.float().argmax(1) == y).sum().item()
            total       += y.size(0)
            global_step += 1

            if global_step % int(cfg["ckpt_every"]) == 0:
                avg   = run_loss / (i + 1)
                acc   = correct / total
                stats = runtime_stats(cfg, start_time)
                print(f"  e{epoch:3d} | step {global_step:5d} | "
                      f"loss {avg:.4f} | acc {acc:.3f} | ${stats['cost_usd']:.4f}")
                asyncio.run(engine.save(
                    model, optimizer, epoch=epoch, step=global_step, loss=avg,
                    extra_state={
                        "task_name": cfg["task_name"], "status": "running",
                        "total_epochs": cfg["epochs"], "accuracy": acc,
                        "best_val_acc": best_val_acc, "resumed_from": resumed_from,
                        "migration_count": cfg.get("migration_count", 0),
                        **stats,
                    }
                ))

        if preempted:
            break

        # ── Validation ────────────────────────────────────────────
        model.eval()
        v_loss = v_ok = v_tot = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = cast_batch(X, y, precision, device)
                with autocast:
                    out     = model(X)
                    v_loss += criterion(out, y).item()
                v_ok  += (out.float().argmax(1) == y).sum().item()
                v_tot += y.size(0)

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

        asyncio.run(engine.save(
            model, optimizer, epoch=epoch, step=global_step, loss=avg_val_loss,
            extra_state={
                "task_name": cfg["task_name"], "status": "running",
                "total_epochs": cfg["epochs"], "accuracy": val_acc,
                "best_val_acc": best_val_acc, "train_loss": avg_tr_loss,
                "resumed_from": resumed_from,
                "migration_count": cfg.get("migration_count", 0),
                **stats,
            }
        ))

    return preempted, best_val_acc, avg_val_loss


# ══════════════════════════════════════════════════════════════════
# TRAINING PARADIGM
# ══════════════════════════════════════════════════════════════════

def _apply_paradigm(cfg: dict, paradigm: str):
    """Mutate cfg in-place based on training paradigm."""
    if paradigm == "rl":
        original = cfg["ckpt_every"]
        cfg["ckpt_every"] = max(10, original // 5)
        cfg["lr"] = float(cfg["lr"]) * 0.1
        cfg["_grad_clip"] = 1.0
        print(f"[paradigm] RL — ckpt_every {original}→{cfg['ckpt_every']}  "
              f"lr→{cfg['lr']:.5f}  grad_clip=1.0")

    elif paradigm == "pre-training":
        original = cfg["ckpt_every"]
        cfg["ckpt_every"] = original * 2
        cfg["_grad_clip"] = 5.0
        print(f"[paradigm] Pre-training — ckpt_every {original}→{cfg['ckpt_every']}  "
              f"grad_clip=5.0")

    elif paradigm == "distillation":
        original = cfg["ckpt_every"]
        cfg["ckpt_every"] = max(20, original // 2)
        cfg["_grad_clip"] = None
        print(f"[paradigm] Distillation — ckpt_every {original}→{cfg['ckpt_every']}")

    else:
        cfg["_grad_clip"] = None
        print(f"[paradigm] Fine-tuning — ckpt_every={cfg['ckpt_every']} (unchanged)")


# ══════════════════════════════════════════════════════════════════
# MAIN ENTRY
# ══════════════════════════════════════════════════════════════════

def train(cfg: dict):
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = cfg.get("precision", "fp32").lower()
    mode      = cfg.get("train_mode", "manual").lower()

    print("\n" + "="*60)
    print(f"  job_id     : {cfg['job_id']}")
    print(f"  task       : {cfg['task_name']}")
    print(f"  arch       : {cfg.get('model_arch','mlp')}  "
          f"precision={precision}  paradigm={cfg.get('training_paradigm','fine-tuning')}")
    print(f"  train_mode : {mode}")
    print(f"  device     : {device}")
    print(f"  cloud      : {os.environ.get('CLOUD','gcp')} / "
          f"{os.environ.get('INSTANCE_TYPE','e2-standard-4')}")
    print(f"  budget     : ${cfg['max_budget']}  price=${cfg['price_usd_hr']}/hr")
    print(f"  RESUME_STEP: {os.environ.get('RESUME_STEP','0')}")
    print("="*60 + "\n")

    start_time = time.time()

    from checkpoint.engine import CheckpointEngine
    engine = CheckpointEngine(job_id=cfg["job_id"])

    train_loader, val_loader = make_dataset(cfg)

    # ── SWEEP MODE ────────────────────────────────────────────────
    if mode == "sweep":
        print(f"[sweep] Auto Sweep mode — "
              f"lr=[{cfg['sweep_lr_min']},{cfg['sweep_lr_max']}]  "
              f"hidden={cfg['sweep_hidden']}  trials={cfg['sweep_trials']}")

        best_cfg = run_hyperband(cfg, train_loader, val_loader, device)
        cfg["lr"]         = best_cfg["lr"]
        cfg["hidden_dim"] = best_cfg["hidden_dim"]
        print(f"\n[sweep] Final training with best config: "
              f"lr={cfg['lr']}  hidden_dim={cfg['hidden_dim']}")

    # ── TRAINING PARADIGM ─────────────────────────────────────────
    paradigm = cfg.get("training_paradigm", "fine-tuning").lower()
    _apply_paradigm(cfg, paradigm)

    # ── BUILD FINAL MODEL ─────────────────────────────────────────
    model     = build_model(cfg)
    model     = apply_precision(model, precision, device)
    optimizer = optim.Adam(model.parameters(), lr=float(cfg["lr"]))
    criterion = nn.CrossEntropyLoss()

    # ── RESUME ────────────────────────────────────────────────────
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

    # Write initial state
    asyncio.run(engine.write_terminal_state(
        epoch=start_epoch, step=global_step, loss=last_loss,
        extra={
            "task_name":       cfg["task_name"],
            "status":          "running",
            "total_epochs":    cfg["epochs"],
            "resumed_from":    resumed_from,
            "migration_count": cfg.get("migration_count", 0),
            "model_arch":      cfg.get("model_arch", "mlp"),
            "precision":       precision,
            "train_mode":      mode,
            **runtime_stats(cfg, start_time),
        }
    ))

    # ── TRAINING LOOP ─────────────────────────────────────────────
    preempted, best_val_acc, avg_val_loss = _run_training_loop(
        cfg, model, optimizer, criterion,
        train_loader, val_loader, engine,
        start_epoch, global_step, resumed_from,
        start_time, device, precision,
    )

    # ── TERMINAL STATE ────────────────────────────────────────────
    if not preempted:
        stats = runtime_stats(cfg, start_time)
        print(f"\n{'='*60}")
        print(f"  Done!  best_val_acc={best_val_acc:.4f}")
        print(f"  Time:  {stats['elapsed_hrs']*60:.1f} min  Cost: ${stats['cost_usd']:.4f}")
        print(f"{'='*60}\n")

        if precision == "int8":
            model_fp32 = model.cpu().float()
            model_q    = torch.quantization.quantize_dynamic(
                model_fp32, {nn.Linear}, dtype=torch.qint8
            )
            print(f"[precision] int8 quantization applied to final model")

        asyncio.run(engine.write_terminal_state(
            epoch=int(cfg["epochs"]), step=global_step,
            loss=avg_val_loss,
            extra={
                "task_name":       cfg["task_name"],
                "status":          "done",
                "total_epochs":    cfg["epochs"],
                "accuracy":        best_val_acc,
                "best_val_acc":    best_val_acc,
                "resumed_from":    resumed_from,
                "migration_count": cfg.get("migration_count", 0),
                "model_arch":      cfg.get("model_arch", "mlp"),
                "precision":       precision,
                "train_mode":      mode,
                "sweep_best_lr":   cfg["lr"]         if mode == "sweep" else None,
                "sweep_best_h":    cfg["hidden_dim"] if mode == "sweep" else None,
                **stats,
            }
        ))


if __name__ == "__main__":
    cfg = load_config()
    train(cfg)