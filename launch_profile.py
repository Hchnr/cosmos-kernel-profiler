"""Launch an isolated eight-GPU capture using the user's original USR recipe."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from validate_reports import fingerprint
from workspace import cosmos_repo

TASK = Path(__file__).resolve().parent
REPO = cosmos_repo()
RECIPE = (
    REPO
    / "examples/toml/sft_config/vision_sft_nano_usr_128gpu_fixed_static_und3k_gen96k_ga2.toml"
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def prepare(run, *, wait=20, warmup=2, active=3, start=12600, dryrun=False):
    cache = run / "cache"
    for path in (cache / "tmp", run / "job"):
        path.mkdir(parents=True, exist_ok=True)
    # UNIX-domain socket addresses are limited to ~108 bytes. Keep the payloads
    # in the task directory while giving multiprocessing a short pathname.
    alias = Path(tempfile.mkdtemp(prefix="ckl-", dir="/tmp"))
    short_tmp = alias / "tmp"
    short_tmp.symlink_to(cache / "tmp", target_is_directory=True)
    env = os.environ.copy()
    for name in list(env):
        if name in {
            "RANK",
            "LOCAL_RANK",
            "WORLD_SIZE",
            "LOCAL_WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
            "GROUP_RANK",
            "ROLE_RANK",
            "WANDB_API_KEY",
            "WANDB_ENTITY",
        } or name.startswith(
            ("TORCHELASTIC_", "PET_", "RDZV_", "COSMOS_TOKENIZER_AOT_")
        ):
            env.pop(name)
    env.update(
        {
            "LD_LIBRARY_PATH": "",
            "COSMOS_REPO": str(REPO),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "42",
            "PYTHONUNBUFFERED": "1",
            "WANDB_MODE": "disabled",
            "COSMOS_TRAINING": "true",
            "PYTHONPATH": os.pathsep.join(map(str, (TASK / "env", TASK, REPO))),
            "IMAGINAIRE_OUTPUT_ROOT": str(run / "job"),
            "COSMOS_KERNEL_CACHE_ROOT": str(cache),
            "COSMOS_KERNEL_REPORT_DIR": str(run / "reports"),
            "COSMOS_KERNEL_START_STEP": str(start),
            "TMPDIR": str(short_tmp),
            "TMP": str(short_tmp),
            "TEMP": str(short_tmp),
            "HF_HOME": str(cache / "huggingface"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_MODEL_PATH": "/share/project/eai_pwm/models/Qwen3-VL-8B-Instruct",
            "USR_DATASET_ROOT": "/share/project/liyue/data/parquet-data/train-usr-v2-20260922/full-r3-80-r6b-5-r6-15-epochs4-seed0",
            "OMP_NUM_THREADS": "8",
            "TORCHINDUCTOR_COMPILE_THREADS": "4",
            "NCCL_DEBUG": "WARN",
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
            "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
            "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC": "1800",
            "COSMOS_NCCL_SUBGROUP_TIMEOUT_SEC": "1800",
        }
    )
    count = wait + warmup + active
    overrides = [
        "job.project=kernel-list",
        "job.group=v0.2.3c",
        "job.name=profile",
        "job.wandb_mode=disabled",
        "upload_reproducible_setup=false",
        "model.config.parallelism.data_parallel_shard_degree=8",
        "model.config.parallelism.data_parallel_replicate_degree=1",
        "model.config.parallelism.context_parallel_shard_degree=1",
        f"trainer.max_iter={start + count}",
        "trainer.seed=42",
        "trainer.run_validation=false",
        "trainer.save_zero_checkpoint=false",
        f"trainer.compile_cache.path={cache}/inductor/rank${{oc.env:LOCAL_RANK,0}}",
        "trainer.profiling.enable_profiling=true",
        "trainer.profiling.enable_nsys=false",
        "trainer.profiling.enable_memory_snapshot=false",
        f"trainer.profiling.profile_freq={count}",
        f"trainer.profiling.profile_warmup={warmup}",
        f"trainer.profiling.profile_active={active}",
        "trainer.profiling.target_ranks=[0]",
        "trainer.profiling.record_shape=true",
        "trainer.profiling.profile_memory=false",
        "trainer.profiling.with_stack=false",
        "trainer.profiling.with_modules=false",
        "checkpoint.load_training_state=true",
        "checkpoint.save_iter=1000000",
        "+trainer.callbacks.kernel_phases._target_=phase_audit.PhaseAudit",
        "+trainer.callbacks.kernel_batches._target_=cosmos_framework.callbacks.load_balance_trace.LoadBalanceTrace",
        "+trainer.callbacks.kernel_batches.enabled=true",
        f"+trainer.callbacks.kernel_batches.output_dir={run}/batches",
        "+trainer.callbacks.kernel_batches.record_cuda_timing=false",
        "+trainer.callbacks.kernel_batches.flush_every_n_batches=1",
    ]
    entry = [str(TASK / "run_profile.py"), f"--sft-toml={RECIPE}"]
    if dryrun:
        command = [sys.executable, *entry, "--dryrun", "--", *overrides]
    else:
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc-per-node=8",
            *entry,
            "--",
            *overrides,
        ]
    return command, env, overrides


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dryrun", action="store_true")
    parser.add_argument("--wait", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--active", type=int, default=3)
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id or args.run_id in {".", ".."}:
        parser.error("run-id must be a single directory name")
    if args.wait < 0 or args.warmup < 0 or args.active < 1:
        parser.error("invalid capture schedule")
    run = TASK / "runs" / args.run_id
    run.mkdir(parents=True, exist_ok=False)
    busy = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_memory",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    if busy and not args.dryrun:
        (run / "metadata.json").write_text(
            json.dumps(
                {
                    "run_id": args.run_id,
                    "checked_utc": utc_now(),
                    "status": "resources_busy",
                    "launched": False,
                    "gpu_processes": busy,
                },
                indent=2,
            )
            + "\n"
        )
        print(
            f"CAPTURE_NOT_STARTED: GPU compute processes already exist: {busy}",
            flush=True,
        )
        return 75
    command, env, overrides = prepare(
        run, wait=args.wait, warmup=args.warmup, active=args.active, dryrun=args.dryrun
    )
    metadata = {
        "started_utc": utc_now(),
        "profiler_git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=TASK, text=True
        ).strip(),
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "recipe": str(RECIPE),
        "recipe_fingerprint": fingerprint(RECIPE),
        "command": command,
        "overrides": overrides,
        "dryrun": args.dryrun,
        "run_id": args.run_id,
        "schedule": {
            "wait": args.wait,
            "warmup": args.warmup,
            "active": args.active,
            "repeat": 1,
        },
        "gpu_before": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,memory.total,memory.used,driver_version",
                "--format=csv",
            ],
            text=True,
        ),
        "implementation": [
            fingerprint(TASK / filename)
            for filename in (
                "run_profile.py",
                "profile_hook.py",
                "phase_audit.py",
                "profiler_reports.py",
                "validate_reports.py",
                "launch_profile.py",
                "checkpoint_compat.py",
                "workspace.py",
            )
        ],
    }
    # Only explicitly managed, non-secret variables go into the reproducible shell command.
    names = {
        "COSMOS_REPO",
        "LD_LIBRARY_PATH",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONHASHSEED",
        "PYTHONUNBUFFERED",
        "PYTHONPATH",
        "WANDB_MODE",
        "COSMOS_TRAINING",
        "IMAGINAIRE_OUTPUT_ROOT",
        "COSMOS_KERNEL_CACHE_ROOT",
        "COSMOS_KERNEL_REPORT_DIR",
        "COSMOS_KERNEL_START_STEP",
        "TMPDIR",
        "TMP",
        "TEMP",
        "HF_HOME",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_MODEL_PATH",
        "USR_DATASET_ROOT",
        "OMP_NUM_THREADS",
        "TORCHINDUCTOR_COMPILE_THREADS",
        "NCCL_DEBUG",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING",
        "CUDA_VISIBLE_DEVICES",
        "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC",
        "COSMOS_NCCL_SUBGROUP_TIMEOUT_SEC",
    }
    metadata["managed_environment"] = {name: env[name] for name in sorted(names)}
    shell = (
        "#!/usr/bin/env bash\nset -euo pipefail\ncd " + shlex.quote(str(REPO)) + "\n"
    )
    shell += "# Audit snapshot; regenerate temporary IPC alias using launch_profile.py with a new run-id.\n"
    shell += (
        "\n".join(f"export {name}={shlex.quote(env[name])}" for name in sorted(names))
        + "\n"
    )
    shell += shlex.join(command) + "\n"
    (run / "launch.sh").write_text(shell)
    (run / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"RUN_DIRECTORY={run}", flush=True)
    try:
        with (run / "train.log").open("w") as log:
            process = subprocess.Popen(
                command,
                cwd=REPO,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            metadata["launcher_pid"] = process.pid
            (run / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
            print(f"TORCHRUN_PID={process.pid}", flush=True)
            returncode = process.wait()
        metadata.update(returncode=returncode, ended_utc=utc_now())
        (run / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    finally:
        short_tmp = Path(env["TMPDIR"])
        short_tmp.unlink()
        short_tmp.parent.rmdir()
    print(f"RETURN_CODE={returncode}", flush=True)
    return returncode


if __name__ == "__main__":
    sys.exit(main())
