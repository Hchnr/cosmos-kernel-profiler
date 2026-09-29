"""Profile the reviewed pure VLM recipe with the shared kernel report exporter."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import shlex
import subprocess
import sys
import tempfile
import tomllib
from datetime import datetime, timezone

from validate_reports import fingerprint
from workspace import cosmos_repo


TASK = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dryrun", action="store_true")
    parser.add_argument("--wait", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--active", type=int, default=3)
    parser.add_argument(
        "--dataset-recipe",
        type=Path,
        help="Use only the dataset list/ratios from another reviewed TOML",
    )
    parser.add_argument(
        "--dataset-name", help="Select one dataset directory name from --dataset-recipe"
    )
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id or args.run_id in {".", ".."}:
        parser.error("run-id must be a single directory name")
    if args.wait < 0 or args.warmup < 0 or args.active < 1:
        parser.error("invalid capture schedule")
    if args.dataset_name and not args.dataset_recipe:
        parser.error("--dataset-name requires --dataset-recipe")
    repo = cosmos_repo()
    source = repo / "examples/understanding_review/grounding100k_four_node_ga3.toml"
    run = TASK / "runs" / args.run_id
    run.mkdir(parents=True, exist_ok=False)
    recipe = run / "recipe.toml"
    recipe_text = source.read_text()
    if args.dataset_recipe:
        datasets = tomllib.loads(args.dataset_recipe.read_text())["dataloader_train"][
            "datasets"
        ]
        if args.dataset_name:
            datasets = [
                d for d in datasets if Path(d["path"]).name == args.dataset_name
            ]
            if len(datasets) != 1:
                raise ValueError(f"Expected one dataset named {args.dataset_name}")
        recipe_text = recipe_text.split("[[dataloader_train.datasets]]", 1)[0]
        for dataset in datasets:
            recipe_text += (
                "\n[[dataloader_train.datasets]]\npath = "
                + json.dumps(dataset["path"])
                + "\nratio = "
                + str(dataset["ratio"])
                + "\n"
            )
    recipe.write_text(recipe_text)
    cache = run / "cache"
    (cache / "tmp").mkdir(parents=True)
    alias = Path(tempfile.mkdtemp(prefix="vkl-", dir="/tmp"))
    (alias / "tmp").symlink_to(cache / "tmp", target_is_directory=True)
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
        } or name.startswith(
            ("TORCHELASTIC_", "PET_", "RDZV_", "WANDB_", "COSMOS_TOKENIZER_AOT_")
        ):
            env.pop(name)
    managed = {
        "LD_LIBRARY_PATH": "",
        "COSMOS_REPO": str(repo),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "42",
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": os.pathsep.join(map(str, (TASK, repo))),
        "WANDB_MODE": "disabled",
        "COSMOS_TRAINING": "true",
        "IMAGINAIRE_OUTPUT_ROOT": str(run / "job"),
        "COSMOS_KERNEL_CACHE_ROOT": str(cache),
        "COSMOS_KERNEL_REPORT_DIR": str(run / "reports"),
        "COSMOS_KERNEL_START_STEP": "0",
        "COSMOS_KERNEL_STACK_DIR": str(run / "stacks"),
        "TMPDIR": str(alias / "tmp"),
        "TMP": str(alias / "tmp"),
        "TEMP": str(alias / "tmp"),
        "HF_HOME": str(cache / "huggingface"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_MODEL_PATH": "/share/project/eai_pwm/models/Qwen3-VL-8B-Instruct",
        "OMP_NUM_THREADS": "8",
        "TORCHINDUCTOR_COMPILE_THREADS": "4",
        "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
        "NCCL_DEBUG": "WARN",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC": "1800",
    }
    env.update(managed)
    count = args.wait + args.warmup + args.active
    overrides = [
        "job.project=kernel-list",
        "job.group=vlm",
        f"job.name={args.run_id}",
        "job.wandb_mode=disabled",
        "upload_reproducible_setup=false",
        "model.config.parallelism.data_parallel_shard_degree=8",
        "model.config.parallelism.data_parallel_replicate_degree=1",
        "model.config.parallelism.context_parallel_shard_degree=1",
        "+model.config.vlm_config.model_instance.config.include_visual=true",
        "model.config.vlm_config.pretrained_weights.enabled=true",
        f"model.config.vlm_config.pretrained_weights.backbone_path={managed['HF_MODEL_PATH']}",
        f"model.config.vlm_config.tokenizer.pretrained_model_name={managed['HF_MODEL_PATH']}",
        "checkpoint.load_training_state=false",
        "checkpoint.only_load_scheduler_state=false",
        "checkpoint.save_iter=1000000",
        "checkpoint.save_to_object_store.enabled=false",
        f"trainer.max_iter={count}",
        "trainer.seed=42",
        "trainer.run_validation=false",
        "trainer.save_zero_checkpoint=false",
        "trainer.profiling.enable_profiling=true",
        "trainer.profiling.enable_nsys=false",
        "trainer.profiling.enable_memory_snapshot=false",
        f"trainer.profiling.profile_freq={count}",
        f"trainer.profiling.profile_warmup={args.warmup}",
        f"trainer.profiling.profile_active={args.active}",
        "trainer.profiling.target_ranks=[0]",
        "trainer.profiling.record_shape=true",
        "trainer.profiling.profile_memory=false",
        "trainer.profiling.with_stack=false",
        "trainer.profiling.with_modules=false",
        "~trainer.callbacks.every_n_sample_reg=null",
        "~trainer.callbacks.every_n_sample_ema=null",
        "+trainer.callbacks.kernel_phases._target_=phase_audit.PhaseAudit",
        "+trainer.callbacks.kernel_vlm_audit._target_=vlm_audit.VLMAudit",
        f"+trainer.callbacks.kernel_vlm_audit.output_dir={run}/batches",
    ]
    entry = [str(TASK / "run_profile.py"), f"--sft-toml={recipe}"]
    if args.dryrun:
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
    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(
        resource.RLIMIT_NOFILE,
        (min(65536, hard) if hard != resource.RLIM_INFINITY else 65536, hard),
    )
    metadata = {
        "run_id": args.run_id,
        "workload": "pure_vlm_v2_reference"
        if args.dataset_recipe
        else "pure_vlm_grounding100k",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "profiler_git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=TASK, text=True
        ).strip(),
        "recipe": str(source),
        "recipe_fingerprint": fingerprint(source),
        "effective_recipe_fingerprint": fingerprint(recipe),
        "dataset_recipe": str(args.dataset_recipe) if args.dataset_recipe else None,
        "dataset_name": args.dataset_name,
        "dataset_recipe_fingerprint": fingerprint(args.dataset_recipe)
        if args.dataset_recipe
        else None,
        "command": command,
        "overrides": overrides,
        "managed_environment": managed,
        "dryrun": args.dryrun,
        "rlimit_nofile": resource.getrlimit(resource.RLIMIT_NOFILE),
        "schedule": {
            "wait": args.wait,
            "warmup": args.warmup,
            "active": args.active,
            "repeat": 1,
        },
        "implementation": [
            fingerprint(TASK / name)
            for name in (
                "launch_vlm_profile.py",
                "run_profile.py",
                "profile_hook.py",
                "phase_audit.py",
                "vlm_audit.py",
                "profiler_reports.py",
                "validate_reports.py",
                "validate_vlm_run.py",
            )
        ],
    }
    metadata["gpu_before"] = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,memory.used,driver_version",
            "--format=csv",
        ],
        text=True,
    )
    metadata_path = run / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    shell = (
        "#!/usr/bin/env bash\nset -euo pipefail\nulimit -n 65536\ncd "
        + shlex.quote(str(repo))
        + "\n"
    )
    shell += (
        "# Audit snapshot: regenerate the temporary IPC alias with a fresh run-id.\n"
    )
    shell += (
        "\n".join(
            f"export {name}={shlex.quote(value)}" for name, value in managed.items()
        )
        + "\n"
    )
    (run / "launch.sh").write_text(shell + shlex.join(command) + "\n")
    print(f"RUN_DIRECTORY={run}", flush=True)
    try:
        if not args.dryrun:
            busy = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                text=True,
            ).strip()
            if busy:
                raise RuntimeError(f"GPU compute processes already exist: {busy}")
        with (run / "train.log").open("w") as log:
            process = subprocess.Popen(
                command,
                cwd=repo,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            metadata["launcher_pid"] = process.pid
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
            print(f"LAUNCHER_PID={process.pid}", flush=True)
            metadata["returncode"] = process.wait()
    finally:
        metadata["ended_utc"] = datetime.now(timezone.utc).isoformat()
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        (alias / "tmp").unlink()
        alias.rmdir()
    print(f"RETURN_CODE={metadata['returncode']}", flush=True)
    return metadata["returncode"]


if __name__ == "__main__":
    sys.exit(main())
