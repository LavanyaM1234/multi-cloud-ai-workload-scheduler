"""
fake_trainer.py
───────────────
Simulates a real train.py process writing to S3 epoch-by-epoch,
then self-preempts at a chosen epoch so the live poller in server.py
triggers the full migration cycle automatically.

Run this alongside a running server.py:

    Terminal 1:  cd dashboard && python api/server.py
    Terminal 2:  cd dashboard && python fake_trainer.py

Full cycle you'll see:
    queued → launched → running (epochs tick) → preempted
        → poller detects → _migrate() → pick_best_cloud()
        → resume_job() → running on new cloud (repeat if desired)

Usage:
    python fake_trainer.py                          # defaults
    python fake_trainer.py --epochs 20 --preempt-at 8
    python fake_trainer.py --cloud gcp --region us-central1
    python fake_trainer.py --job-id my-existing-job  # resume a real job
    python fake_trainer.py --cycles 3               # auto-preempt 3 times
"""

import os
import sys
import json
import time
import math
import signal
import argparse
import random
from datetime import datetime, timezone

import boto3
from dotenv import load_dotenv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────
S3_BUCKET  = os.getenv("CHECKPOINT_S3_BUCKET", "ml-scheduler-checkpoints")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

# ── Colours ────────────────────────────────────────────────────────
R = "\033[91m"; G = "\033[92m"; Y = "\033[93m"
C = "\033[96m"; B = "\033[1m";  X = "\033[0m"

STATUS_COLOUR = {
    "queued": Y, "launched": Y, "running": G,
    "preempted": R, "migrating": C, "done": G,
}

_shutdown = False
def _handle_sigint(sig, frame):
    global _shutdown
    print(f"\n{Y}[trainer] Ctrl-C received — writing preempted state and exiting{X}")
    _shutdown = True
signal.signal(signal.SIGINT, _handle_sigint)


# ══════════════════════════════════════════════════════════════════
# S3 helpers
# ══════════════════════════════════════════════════════════════════

def _s3():
    return boto3.client(
        "s3",
        aws_access_key_id     = os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name           = AWS_REGION,
    )

def s3_write(key: str, data: dict):
    _s3().put_object(
        Bucket      = S3_BUCKET,
        Key         = key,
        Body        = json.dumps(data, indent=2).encode(),
        ContentType = "application/json",
    )

def s3_read(key: str) -> dict | None:
    try:
        obj = _s3().get_object(Bucket=S3_BUCKET, Key=key)
        return json.loads(obj["Body"].read().decode())
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════
# Fake training loop  — one "cloud run" until preemption
# ══════════════════════════════════════════════════════════════════

def fake_loss(epoch: int, total: int) -> float:
    """Decaying loss with a tiny bit of noise — looks realistic on the dashboard."""
    base = 2.5 * math.exp(-3.5 * epoch / total)
    return round(base + random.uniform(-0.02, 0.02), 4)


def run_training_loop(
    job_id:        str,
    cloud:         str,
    region:        str,
    az:            str,
    instance_type: str,
    start_epoch:   int,
    total_epochs:  int,
    preempt_at:    int,      # epoch index at which to self-preempt
    epoch_secs:    float,    # wall-clock seconds per simulated epoch
    cost_per_hr:   float,
    task_name:     str,
) -> dict:
    """
    Simulates one cloud run of the training job.
    Returns the final job_state dict written to S3.
    """
    global _shutdown

    prefix       = f"checkpoints/{job_id}"
    state_key    = f"{prefix}/job_state.json"
    command_key  = f"{prefix}/job_command.json"
    cost_so_far  = float((s3_read(state_key) or {}).get("cost_usd", 0.0))
    epoch_cost   = cost_per_hr * (epoch_secs / 3600)

    print(f"\n{B}[trainer]{X} Starting loop  "
          f"{C}{cloud}{X}/{instance_type}  "
          f"epochs {start_epoch}–{preempt_at - 1} of {total_epochs}")
    print(f"  epoch_secs={epoch_secs:.1f}s  "
          f"preempt_at={preempt_at}  "
          f"cost_per_hr=${cost_per_hr:.3f}")
    print(f"  {Y}─────────────────────────────────────────────{X}")

    def write_state(status: str, epoch: int, extra: dict = {}):
        state = {
            "job_id":        job_id,
            "task_name":     task_name,
            "status":        status,
            "cloud":         cloud,
            "region":        region,
            "availability_zone": az,
            "instance_type": instance_type,
            "is_spot":       True,
            "epoch":         epoch,
            "total_epochs":  total_epochs,
            "step":          epoch * 100,
            "loss":          fake_loss(epoch, total_epochs),
            "cost_usd":      round(cost_so_far, 4),
            "updated_at":    datetime.now(timezone.utc).isoformat(),
            "launch_result": {
                "cloud": cloud, "region": region,
                "az": az, "instance_type": instance_type,
            },
            **extra,
        }
        s3_write(state_key, state)
        return state

    # ── queued → launched → running ────────────────────────────────
    write_state("queued",   start_epoch)
    _print_status("queued",   start_epoch, total_epochs, 0.0, cloud, instance_type)
    time.sleep(0.5)

    write_state("launched", start_epoch)
    _print_status("launched", start_epoch, total_epochs, 0.0, cloud, instance_type)
    time.sleep(1.0)

    last_state = None

    for epoch in range(start_epoch, total_epochs):
        if _shutdown:
            break

        # Check for dashboard commands (migrate / stop / reduce_lr)
        cmd_obj = s3_read(command_key)
        if cmd_obj:
            cmd = cmd_obj.get("command")
            print(f"\n  {Y}[trainer] Command received: {cmd}{X}")
            _s3().delete_object(Bucket=S3_BUCKET, Key=command_key)
            if cmd == "stop":
                last_state = write_state("done", epoch)
                _print_status("done", epoch, total_epochs, cost_so_far, cloud, instance_type)
                return last_state
            elif cmd == "migrate":
                print(f"  {C}[trainer] Checkpointing and triggering preemption...{X}")
                break   # fall through to preempt block below

        cost_so_far += epoch_cost
        last_state   = write_state("running", epoch)
        _print_status("running", epoch, total_epochs, cost_so_far, cloud, instance_type)

        # ── Self-preempt at target epoch ───────────────────────────
        if epoch >= preempt_at - 1:
            break

        time.sleep(epoch_secs)

    # ── Write preempted state ──────────────────────────────────────
    final_epoch = last_state["epoch"] if last_state else start_epoch
    last_state = write_state("preempted", final_epoch, {
        "preemption_source": "simulated_spot_interruption",
        "cost_usd":          round(cost_so_far, 4),
    })
    _print_status("preempted", final_epoch, total_epochs, cost_so_far, cloud, instance_type)
    print(f"\n  {R}[trainer] Preempted at epoch {final_epoch} — exiting VM{X}")
    print(f"  {Y}Poller will detect this in ≤30s and trigger migration...{X}\n")

    return last_state


# ══════════════════════════════════════════════════════════════════
# Watch S3 for the poller to migrate and relaunch
# ══════════════════════════════════════════════════════════════════

def wait_for_relaunch(job_id: str, timeout: int = 180) -> dict | None:
    """
    After preemption, block until job_state.json transitions away from
    'preempted' (i.e. the poller has picked it up and relaunched).
    Returns the new state, or None on timeout.
    """
    key      = f"checkpoints/{job_id}/job_state.json"
    deadline = time.time() + timeout
    dots     = 0

    print(f"{C}[watcher]{X} Waiting for poller to detect preemption", end="", flush=True)

    while time.time() < deadline:
        state  = s3_read(key) or {}
        status = state.get("status", "")

        if status != "preempted":
            print(f"\n{G}[watcher]{X} Status changed → {B}{status}{X}")
            print(f"  cloud   : {state.get('cloud')}")
            print(f"  instance: {state.get('instance_type') or state.get('launch_result', {}).get('instance_type')}")
            return state

        # Print a dot every 5s so it's clear things are alive
        time.sleep(5)
        dots += 1
        if dots % 6 == 0:
            elapsed = int(time.time() - (deadline - timeout))
            print(f" ({elapsed}s)", end="", flush=True)
        else:
            print(".", end="", flush=True)

    print(f"\n{R}[watcher]{X} Timeout — poller never changed status from 'preempted'.")
    print("  Is server.py running?  Check its logs.")
    return None


# ══════════════════════════════════════════════════════════════════
# Pretty status line
# ══════════════════════════════════════════════════════════════════

def _print_status(status, epoch, total, cost, cloud, instance):
    pct    = min(99, round(epoch / max(total, 1) * 100))
    bar_w  = 20
    filled = round(bar_w * pct / 100)
    bar    = "█" * filled + "░" * (bar_w - filled)
    colour = STATUS_COLOUR.get(status, "")
    ts     = datetime.now().strftime("%H:%M:%S")
    print(
        f"  {ts}  {colour}{status:<10}{X}  "
        f"[{bar}] {pct:>3}%  "
        f"epoch={epoch:>3}  "
        f"cost=${cost:.3f}  "
        f"{C}{cloud}{X}/{instance}"
    )


# ══════════════════════════════════════════════════════════════════
# Initial S3 seed  — job_config.json only (state written by loop)
# ══════════════════════════════════════════════════════════════════

def seed_config(job_id: str, args):
    config = {
        "job_id":            job_id,
        "task_name":         args.task_name,
        "max_budget":        args.budget,
        "deadline_hrs":      args.deadline_hrs,
        "priority":          args.priority,
        "spot_only":         True,
        "ondemand_max_hrs":  1.0,
        "preferred_clouds":  args.preferred_clouds,
        "preferred_regions": "",
        "gpu_required":      False,
        "min_gpu_mem":       0,
        "carbon_aware":      False,
        "carbon_weight":     "balanced",
        "epochs":            args.epochs,
        "batch_size":        64,
        "dataset_type":      "synthetic-500k",
        "synthetic_rows":    500_000,
        "migration_count":   0,
    }
    s3_write(f"checkpoints/{job_id}/job_config.json", config)
    print(f"  {G}✓{X} wrote job_config.json  "
          f"(budget=${args.budget:.2f}  epochs={args.epochs})")


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Fake training loop that self-preempts so the real poller migrates it",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--job-id",      default=f"fake-train-{int(time.time())}",
                   help="S3 job prefix; use an existing job_id to resume")
    p.add_argument("--task-name",   default="Fake Training Run")
    p.add_argument("--cloud",       default="aws", choices=["aws", "gcp", "azure"],
                   help="Cloud to simulate running on initially")
    p.add_argument("--region",      default="us-east-1")
    p.add_argument("--az",          default="us-east-1a")
    p.add_argument("--instance",    default="p3.2xlarge")
    p.add_argument("--epochs",      type=int, default=30)
    p.add_argument("--preempt-at",  type=int, default=8,
                   help="Epoch index at which to self-preempt (0-indexed)")
    p.add_argument("--epoch-secs",  type=float, default=3.0,
                   help="Wall-clock seconds per simulated epoch")
    p.add_argument("--cost-per-hr", type=float, default=0.90,
                   help="Simulated $/hr for cost accumulation")
    p.add_argument("--budget",      type=float, default=5.0)
    p.add_argument("--deadline-hrs",type=float, default=8.0)
    p.add_argument("--priority",    default="balanced",
                   choices=["cost", "speed", "balanced"])
    p.add_argument("--preferred-clouds", nargs="+",
                   default=["aws", "gcp", "azure"], metavar="CLOUD")
    p.add_argument("--cycles",      type=int, default=1,
                   help="How many preemption→migration cycles to simulate. "
                        "Set >1 to watch repeated migrations. 0 = run forever.")
    p.add_argument("--wait-timeout",type=int, default=180,
                   help="Seconds to wait for poller to detect preemption")
    return p.parse_args()


def main():
    args   = parse_args()
    job_id = args.job_id

    if not S3_BUCKET:
        print(f"{R}[error]{X} CHECKPOINT_S3_BUCKET env var not set.")
        sys.exit(1)

    print(f"\n{'═'*60}")
    print(f"{B}  Fake Trainer — full preemption/migration cycle test{X}")
    print(f"{'─'*60}")
    print(f"  job_id     : {job_id}")
    print(f"  cloud      : {args.cloud} / {args.instance}")
    print(f"  epochs     : {args.epochs}  preempt_at={args.preempt_at}")
    print(f"  epoch_secs : {args.epoch_secs}s  "
          f"(~{args.epoch_secs * args.preempt_at:.0f}s until preemption)")
    print(f"  cycles     : {'∞' if args.cycles == 0 else args.cycles}")
    print(f"{'═'*60}\n")

    # Write config once — poller reads this for migration decisions
    print(f"{B}[seed]{X} Writing job_config.json to S3 ...")
    seed_config(job_id, args)

    cloud       = args.cloud
    region      = args.region
    az          = args.az
    instance    = args.instance
    start_epoch = 0
    cycle       = 0

    while True:
        cycle += 1
        if args.cycles > 0 and cycle > args.cycles:
            break

        print(f"\n{'─'*60}")
        print(f"{B}  Cycle {cycle}{'/' + str(args.cycles) if args.cycles else ''}{X}  "
              f"cloud={C}{cloud}{X}  start_epoch={start_epoch}")
        print(f"{'─'*60}")

        # ── Simulate training until self-preemption ────────────────
        preempt_at = min(args.preempt_at + start_epoch, args.epochs)
        state = run_training_loop(
            job_id        = job_id,
            cloud         = cloud,
            region        = region,
            az            = az,
            instance_type = instance,
            start_epoch   = start_epoch,
            total_epochs  = args.epochs,
            preempt_at    = preempt_at,
            epoch_secs    = args.epoch_secs,
            cost_per_hr   = args.cost_per_hr,
            task_name     = args.task_name,
        )

        if _shutdown or state.get("status") in ("done", "budget_exceeded"):
            break

        # ── Wait for poller to pick it up and migrate ──────────────
        new_state = wait_for_relaunch(job_id, timeout=args.wait_timeout)
        if new_state is None:
            print(f"\n{R}[trainer]{X} Poller never resumed the job — aborting cycles.")
            break

        # ── Extract new cloud/instance for the next simulated run ──
        launch      = new_state.get("launch_result", {})
        cloud       = new_state.get("cloud")    or launch.get("cloud",    cloud)
        region      = launch.get("region")      or region
        az          = launch.get("az")          or az
        instance    = (launch.get("instance_type")
                       or new_state.get("instance_type") or instance)
        start_epoch = new_state.get("epoch", state.get("epoch", 0))

        print(f"\n{G}[trainer]{X} Picking up on {C}{cloud}{X}/{instance} "
              f"from epoch {start_epoch}")

        # Give the new VM a moment to "boot" (poller sets launched briefly)
        time.sleep(2)

        if args.cycles > 0 and cycle >= args.cycles:
            print(f"\n{G}[trainer]{X} {cycle} cycle(s) complete — stopping.")
            break

    print(f"\n{'═'*60}")
    print(f"{B}  Fake Trainer finished{X}")
    final = s3_read(f"checkpoints/{job_id}/job_state.json") or {}
    print(f"  final status : {STATUS_COLOUR.get(final.get('status',''), '')}"
          f"{final.get('status', '?')}{X}")
    print(f"  final cloud  : {final.get('cloud', '?')}")
    print(f"  total cost   : ${final.get('cost_usd', 0):.3f}")
    print(f"  epoch        : {final.get('epoch', '?')}/{args.epochs}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()