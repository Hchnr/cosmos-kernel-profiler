"""Wait for two idle observations, then run the existing guarded launcher."""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from workspace import cosmos_repo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--probe-attention", action="store_true")
    parser.add_argument("--idle-observations", type=int, default=2)
    args = parser.parse_args()
    if args.idle_observations < 2:
        parser.error("at least two idle observations are required")
    task = Path(__file__).resolve().parent
    log_path = task / "validation" / "resource_wait.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    repo = cosmos_repo()
    idle = 0
    print(f"RESOURCE_WAITER_PID={os.getpid()}", flush=True)
    while idle < args.idle_observations:
        processes = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid,used_memory",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
        idle = idle + 1 if not processes else 0
        with log_path.open("a") as log:
            log.write(
                json.dumps(
                    {
                        "utc": datetime.now(timezone.utc).isoformat(),
                        "processes": processes,
                        "idle_observations": idle,
                    }
                )
                + "\n"
            )
        if idle < args.idle_observations:
            time.sleep(30)
    print("GPUS_IDLE_LAUNCHING", flush=True)
    if args.probe_attention:
        env = os.environ.copy()
        env.update(
            {
                "LD_LIBRARY_PATH": "",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": os.pathsep.join(map(str, (task / "env", task, repo))),
                "TORCHINDUCTOR_COMPILE_THREADS": "4",
                "OMP_NUM_THREADS": "8",
                "TMPDIR": str(task / "validation"),
            }
        )
        with (task / "validation" / f"{args.run_id}-attention.log").open("w") as log:
            result = subprocess.call(
                [
                    sys.executable,
                    str(task / "probe_attention.py"),
                    str(task / "validation" / f"{args.run_id}-attention"),
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if result:
            return result
    return subprocess.call(
        [sys.executable, str(task / "launch_profile.py"), "--run-id", args.run_id]
    )


if __name__ == "__main__":
    sys.exit(main())
