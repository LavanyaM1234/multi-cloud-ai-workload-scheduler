#!/usr/bin/env python3
"""
trainer/train.py
─────────────────
Generated MLP trainer with full preemption + resume support.

Flow:
  1. Load job_config.json from GCS
  2. If RESUME_STEP env var set → download checkpoint_latest.pt, resume
  3. Train, saving checkpoints to GCS every N steps
  4. Watch for GCP preemption notice every 10 batches (~30s warning)
  5. On preemption → emergency checkpoint → job_state status=preempted
     → scheduler sees it → relaunches on cheapest available cloud

Checkpoint layout in GCS:
  gs://{bucket}/checkpoints/{job_id}/
    job_config.json             ← written by launcher before VM starts
    job_state.json              ← updated every epoch (dashboard reads)
    checkpoint_latest.pt        ← always overwritten, used for resume
    checkpoint_step_{N}.pt      ← milestone saves every ckpt_every steps
"""

import os, sys, json, time
from datetime import datetime, timezone

# ── Dependency checks ──────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    print("[ERROR] pip install torch --index-url https://download.pytorch.org/whl/cpu")
    sys.exit(1)

try:
    from google.cloud import storage as gcs_lib
    GCS_OK = True
except ImportError:
    GCS_OK = False
    print("[WARN] pip install google-cloud-storage — GCS writes disabled")


# ══════════════════════════════════════════════════════════════════
# GCS HELPERS
# ══════════════════════════════════════════════════════════════════

def gcs_read_json(bucket_name, path):
    client = gcs_lib.Client()
    return json.loads(client.bucket(bucket_name).blob(path).download_as_text())

def gcs_write_json(bucket_name, path, data):
    if not GCS_OK or not bucket_name:
        with open(os.path.basename(path), "w") as f:
            json.dump(data, f, indent=2)
        return
    try:
        gcs_lib.Client().bucket(bucket_name).blob(path).upload_from_string(
            json.dumps(data, indent=2), content_type="application/json"
        )
    except Exception as e:
        print(f"[WARN] GCS write {path}: {e}")

def gcs_upload_file(bucket_name, gcs_path, local_path):
    if not GCS_OK or not bucket_name:
        return
    try:
        gcs_lib.Client().bucket(bucket_name).blob(gcs_path).upload_from_filename(local_path)
        print(f"  [gcs] → gs://{bucket_name}/{gcs_path}")
    except Exception as e:
        print(f"[WARN] GCS upload {gcs_path}: {e}")

def gcs_blob_exists(bucket_name, gcs_path):
    if not GCS_OK or not bucket_name:
        return False
    try:
        return gcs_lib.Client().bucket(bucket_name).blob(gcs_path).exists()
    except Exception:
        return False

def gcs_download_file(bucket_name, gcs_path, local_path):
    if not GCS_OK or not bucket_name:
        return False
    try:
        blob = gcs_lib.Client().bucket(bucket_name).blob(gcs_path)
        if not blob.exists():
            return False
        blob.download_to_filename(local_path)
        print(f"  [gcs] downloaded gs://{bucket_name}/{gcs_path}")
        return True
    except Exception as e:
        print(f"[WARN] GCS download {gcs_path}: {e}")
        return False


# ══════════════════════════════════════════════════════════════════
# CONFIG LOADING
# ══════════════════════════════════════════════════════════════════

def load_config():
    bucket = os.environ.get("GCS_BUCKET", "")
    job_id = os.environ.get("JOB_ID",     "local-test")
    config = None

    # Try GCS first
    if bucket and GCS_OK:
        try:
            config = gcs_read_json(bucket, f"checkpoints/{job_id}/job_config.json")
            print(f"[OK] Config from GCS: checkpoints/{job_id}/job_config.json")
        except Exception as e:
            print(f"[WARN] GCS config: {e}")

    # Local fallback
    if config is None and os.path.exists("job_config.json"):
        with open("job_config.json") as f:
            config = json.load(f)
        print("[OK] Config from local job_config.json")

    if config is None:
        print("[WARN] No config found — using defaults")
        config = {}

    # Defaults
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
    config.setdefault("price_usd_hr", 0.067)

    if os.environ.get("JOB_ID"):      config["job_id"]     = os.environ["JOB_ID"]
    if os.environ.get("GCS_BUCKET"):  config["gcs_bucket"] = os.environ["GCS_BUCKET"]

    return config


# ══════════════════════════════════════════════════════════════════
# JOB STATE  — what the dashboard reads every 60s
# ══════════════════════════════════════════════════════════════════

def write_state(cfg, epoch, step, loss, acc, status="running", extra=None):
    elapsed = (time.time() - cfg["_start"]) / 3600
    state = {
        "job_id":          cfg["job_id"],
        "task_name":       cfg["task_name"],
        "status":          status,
        "epoch":           epoch,
        "total_epochs":    cfg["epochs"],        # progress % = epoch/total_epochs
        "step":            step,
        "loss":            round(float(loss), 6),
        "accuracy":        round(float(acc),  4),
        "elapsed_hrs":     round(elapsed, 4),
        "cost_usd":        round(elapsed * cfg["price_usd_hr"], 4),
        "cloud":           os.environ.get("CLOUD",         "gcp"),
        "instance":        os.environ.get("INSTANCE_TYPE", "e2-standard-4"),
        "resumed_from":    cfg.get("_resume_step", 0),
        "updated_at":      datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        state.update(extra)
    gcs_write_json(cfg["gcs_bucket"],
                   f"checkpoints/{cfg['job_id']}/job_state.json", state)
    return state


# ══════════════════════════════════════════════════════════════════
# CHECKPOINT SAVE / LOAD
# ══════════════════════════════════════════════════════════════════

def save_checkpoint(cfg, model, optimizer, epoch, step, loss, is_emergency=False):
    tag = "EMERGENCY" if is_emergency else f"step {step}"
    print(f"  [ckpt] Saving {tag}...")

    ckpt = {
        "epoch":      epoch,
        "step":       step,
        "loss":       float(loss),
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "config":     {k: v for k, v in cfg.items() if not k.startswith("_")},
        "saved_at":   datetime.now(timezone.utc).isoformat(),
        "cloud":      os.environ.get("CLOUD", "gcp"),
        "instance":   os.environ.get("INSTANCE_TYPE", "e2-standard-4"),
    }

    local = "/tmp/checkpoint_latest.pt"
    torch.save(ckpt, local)

    bucket = cfg["gcs_bucket"]
    job_id = cfg["job_id"]

    # checkpoint_latest.pt — always overwrite, this is what resume loads
    gcs_upload_file(bucket, f"checkpoints/{job_id}/checkpoint_latest.pt", local)

    # Step-specific file for history (skip on emergency to save time)
    if not is_emergency:
        gcs_upload_file(bucket, f"checkpoints/{job_id}/checkpoint_step_{step}.pt", local)

    print(f"  [ckpt] ✓  epoch={epoch} step={step}")


def load_checkpoint(cfg, model, optimizer):
    """
    Returns (start_epoch, start_step, last_loss).
    Loads checkpoint_latest.pt from GCS if RESUME_STEP > 0.
    This works regardless of which cloud we're resuming ON —
    the checkpoint is in GCS, accessible from anywhere.
    """
    resume_step = int(os.environ.get("RESUME_STEP", "0"))
    if resume_step == 0:
        print("[train] Fresh start (RESUME_STEP=0)")
        return 1, 0, 999.0

    print(f"\n[RESUME] Resuming from step {resume_step}...")
    prev_cloud = os.environ.get("PREV_CLOUD", "unknown")
    curr_cloud = os.environ.get("CLOUD", "gcp")
    if prev_cloud != curr_cloud:
        print(f"[RESUME] Cross-cloud migration: {prev_cloud} → {curr_cloud}")

    local = "/tmp/checkpoint_latest.pt"
    ok = gcs_download_file(cfg["gcs_bucket"],
                           f"checkpoints/{cfg['job_id']}/checkpoint_latest.pt",
                           local)
    if not ok:
        print("[RESUME] No checkpoint in GCS — starting from scratch")
        return 1, 0, 999.0

    try:
        ckpt = torch.load(local, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        epoch = ckpt["epoch"]
        step  = ckpt["step"]
        loss  = ckpt.get("loss", 999.0)
        cfg["_resume_step"] = step
        print(f"[RESUME] ✓  epoch={epoch} step={step} loss={loss:.4f}")
        return epoch, step, loss
    except Exception as e:
        print(f"[RESUME] Load failed: {e} — starting from scratch")
        return 1, 0, 999.0


# ══════════════════════════════════════════════════════════════════
# PREEMPTION + BUDGET
# ══════════════════════════════════════════════════════════════════

def check_preemption():
    """
    Polls GCP metadata server. GCP gives ~30s notice before killing VM.
    Returns False on non-GCP machines (safe to call anywhere).
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

def check_budget(cfg):
    elapsed = (time.time() - cfg["_start"]) / 3600
    return (elapsed * cfg["price_usd_hr"]) >= float(cfg["max_budget"])


# ══════════════════════════════════════════════════════════════════
# DATASET  — synthetic tabular (matches modal's synthetic-500k)
# ══════════════════════════════════════════════════════════════════

def make_dataset(cfg):
    n_tr, n_va = 10_000, 2_000
    feat, cls  = cfg["input_dim"], cfg["num_classes"]
    print(f"[data] Synthetic: {n_tr} train / {n_va} val | feat={feat} cls={cls}")
    torch.manual_seed(42)
    return (
        DataLoader(TensorDataset(torch.randn(n_tr, feat), torch.randint(0, cls, (n_tr,))),
                   batch_size=cfg["batch_size"], shuffle=True),
        DataLoader(TensorDataset(torch.randn(n_va, feat), torch.randint(0, cls, (n_va,))),
                   batch_size=256)
    )


# ══════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════

class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),  nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden),  nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x): return self.net(x)


# ══════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════════════════

def train(cfg):
    print("\n" + "="*60)
    print(f"  Job     : {cfg['job_id']}")
    print(f"  Task    : {cfg['task_name']}")
    print(f"  Cloud   : {os.environ.get('CLOUD','gcp')} / {os.environ.get('INSTANCE_TYPE','e2-standard-4')}")
    print(f"  Params  : lr={cfg['lr']} hidden={cfg['hidden_dim']} dropout={cfg['dropout']}")
    print(f"  Train   : epochs={cfg['epochs']} batch={cfg['batch_size']} ckpt_every={cfg['ckpt_every']}")
    print(f"  Budget  : ${cfg['max_budget']} @ ${cfg['price_usd_hr']}/hr")
    print(f"  Resume  : step {os.environ.get('RESUME_STEP','0')}")
    print("="*60 + "\n")

    cfg["_start"]       = time.time()
    cfg["_resume_step"] = 0

    model     = MLP(cfg["input_dim"], cfg["hidden_dim"], cfg["num_classes"], cfg["dropout"])
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])
    criterion = nn.CrossEntropyLoss()

    # ── Resume ────────────────────────────────────────────────────
    start_epoch, global_step, last_loss = load_checkpoint(cfg, model, optimizer)
    train_loader, val_loader            = make_dataset(cfg)

    write_state(cfg, start_epoch, global_step, last_loss, 0.0,
                status="running",
                extra={"resumed_from": cfg["_resume_step"]})
    print("[OK] Initial job_state.json written\n")

    best_val_acc = 0.0
    avg_val_loss = last_loss
    preempted    = False

    for epoch in range(start_epoch, cfg["epochs"] + 1):

        # ── Pre-epoch checks ──────────────────────────────────────
        if check_preemption():
            print(f"\n[!!!] PREEMPTION at epoch={epoch} step={global_step}")
            save_checkpoint(cfg, model, optimizer, epoch, global_step,
                            avg_val_loss, is_emergency=True)
            write_state(cfg, epoch, global_step, avg_val_loss, best_val_acc,
                        status="preempted",
                        extra={"preemption_source": "gcp_metadata",
                               "resume_from_step":  global_step})
            preempted = True
            break

        if check_budget(cfg):
            print(f"\n[BUDGET] Limit reached at step {global_step}")
            save_checkpoint(cfg, model, optimizer, epoch, global_step,
                            avg_val_loss, is_emergency=True)
            write_state(cfg, epoch, global_step, avg_val_loss, best_val_acc,
                        status="budget_exceeded",
                        extra={"resume_from_step": global_step})
            break

        # ── Train epoch ───────────────────────────────────────────
        model.train()
        run_loss = 0.0
        correct  = 0
        total    = 0

        for i, (X, y) in enumerate(train_loader):
            # Preemption check mid-epoch every 10 batches
            if i % 10 == 0 and check_preemption():
                print(f"\n[!!!] PREEMPTION mid-epoch batch={i} step={global_step}")
                save_checkpoint(cfg, model, optimizer, epoch, global_step,
                                run_loss / max(i, 1), is_emergency=True)
                write_state(cfg, epoch, global_step, run_loss / max(i, 1), best_val_acc,
                            status="preempted",
                            extra={"preemption_source": "gcp_metadata_mid_epoch",
                                   "resume_from_step":  global_step})
                preempted = True
                break

            optimizer.zero_grad()
            out  = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            run_loss  += loss.item()
            correct   += (out.argmax(1) == y).sum().item()
            total     += y.size(0)
            global_step += 1

            # Periodic checkpoint
            if global_step % cfg["ckpt_every"] == 0:
                avg = run_loss / (i + 1)
                acc = correct / total
                print(f"  e{epoch:3d} | step {global_step:5d} | loss {avg:.4f} | acc {acc:.3f}")
                save_checkpoint(cfg, model, optimizer, epoch, global_step, avg)
                write_state(cfg, epoch, global_step, avg, acc)

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

        elapsed = (time.time() - cfg["_start"]) / 3600
        print(f"Epoch {epoch:3d}/{cfg['epochs']} | "
              f"tr={avg_tr_loss:.4f} val={avg_val_loss:.4f} "
              f"val_acc={val_acc:.3f} best={best_val_acc:.3f} "
              f"${elapsed * cfg['price_usd_hr']:.4f}")

        write_state(cfg, epoch, global_step, avg_val_loss, val_acc,
                    extra={"best_val_acc": round(best_val_acc, 4),
                           "train_loss":   round(avg_tr_loss, 4)})

    # ── Completion ────────────────────────────────────────────────
    if not preempted:
        elapsed = (time.time() - cfg["_start"]) / 3600
        print(f"\n{'='*60}")
        print(f"  Done! best_val_acc={best_val_acc:.4f} "
              f"time={elapsed*60:.1f}min cost=${elapsed*cfg['price_usd_hr']:.4f}")
        print(f"{'='*60}\n")
        write_state(cfg, cfg["epochs"], global_step, avg_val_loss, best_val_acc,
                    status="done",
                    extra={"total_cost_usd":  round(elapsed * cfg["price_usd_hr"], 4),
                           "elapsed_minutes": round(elapsed * 60, 1),
                           "best_val_acc":    round(best_val_acc, 4)})

if __name__ == "__main__":
    train(load_config())
