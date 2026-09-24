"""Check completed distributed training and describe the actually captured workload."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path


def validate_run(run):
    metadata = json.loads((run / "metadata.json").read_text())
    window = json.loads((run / "profile_window.json").read_text())
    quality = json.loads((run / "report_quality.json").read_text())
    if metadata.get("returncode") != 0:
        raise ValueError("Training did not complete successfully")
    if quality.get("compile_timed_ranges"):
        raise ValueError(
            "Active window contains compilation; extend warmup and recapture"
        )
    active_steps = window["active_completed_steps"]
    ga = window["grad_accum_iter"]
    start = window["global_start_step"]
    end = active_steps[-1]
    ranks, captured = [], []
    for rank in range(8):
        path = run / "batches" / f"rank_{rank:05d}.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        starts = [row for row in records if row["record_type"] == "trace_start"]
        ends = [row for row in records if row["record_type"] == "trace_end"]
        batches = [row for row in records if row["record_type"] == "microbatch"]
        if len(starts) != 1 or starts[0]["iteration"] != start:
            raise ValueError(f"Rank {rank} resume mismatch")
        if len(ends) != 1 or ends[0]["iteration"] != end:
            raise ValueError(f"Rank {rank} did not reach capture end")
        counts = Counter(row["iteration"] for row in batches)
        if counts != {step: ga for step in range(start, end)}:
            raise ValueError(f"Rank {rank} has missing/duplicated microbatches")
        active = [row for row in batches if row["iteration"] + 1 in active_steps]
        ranks.append(
            {
                "rank": rank,
                "microbatches": len(batches),
                "active_microbatches": len(active),
            }
        )
        if rank == window["rank"]:
            captured = active
    expected_phases = {
        "kernel_list.forward": len(active_steps) * ga,
        "kernel_list.backward": len(active_steps) * ga,
        "kernel_list.optimizer": len(active_steps),
    }
    if quality["phase_annotations"] != expected_phases:
        raise ValueError(f"Unexpected phase counts: {quality['phase_annotations']}")
    log = (run / "train.log").read_text(errors="replace")
    losses = [
        (int(step), float(value))
        for step, value in re.findall(
            r"Iteration: (\d+), average iter time: [^,]+, total loss ([^\s]+)", log
        )
    ]
    active_losses = [(step, loss) for step, loss in losses if step in active_steps]
    if set(step for step, _ in active_losses) != set(active_steps) or not all(
        math.isfinite(loss) for _, loss in active_losses
    ):
        raise ValueError("Missing/non-finite active-step losses")
    datasets, conditions, modes, shapes = Counter(), Counter(), Counter(), Counter()
    sample_count = 0
    for batch in captured:
        datasets.update(
            batch.get("source_dataset_names") or batch.get("dataset_names", [])
        )
        modes.update(batch.get("vision_input_mode", []))
        sample_count += batch.get("sample_count_from_ids", 0)
        for plan in batch.get("sequence_plans", []):
            conditions[str(plan.get("vision_conditioning", "unknown"))] += 1
        for shape in batch.get("video_latent_shapes", []):
            shapes[json.dumps(shape)] += 1
    return {
        "distributed_training": "passed",
        "run_id": metadata["run_id"],
        "ranks": ranks,
        "global_start_step": start,
        "active_completed_steps": active_steps,
        "profile_rank": window["rank"],
        "phase_annotations": expected_phases,
        "active_losses": active_losses,
        "captured_samples": sample_count,
        "captured_dataset_sample_counts": dict(datasets),
        "captured_conditions": dict(conditions),
        "captured_vision_input_modes": dict(modes),
        "captured_latent_shapes": dict(shapes),
        "scope": "One rank and three optimizer updates with the original weighted USR sampler. "
        "Observed variants only; no claim of all 29 datasets or every model input shape.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    result = validate_run(args.run)
    output = args.run / "training_validation.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(output)
