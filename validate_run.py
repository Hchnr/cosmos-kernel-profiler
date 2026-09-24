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
        for row in active:
            if (
                row["world_size"],
                row["dp_shard"],
                row["dp_replicate"],
                row["cp_size"],
            ) != (8, 8, 1, 1):
                raise ValueError(f"Rank {rank} has unexpected parallelism")
            output = row["output"]
            if (
                output["physical_und_token_length"],
                output["physical_gen_token_length"],
            ) != (3072, 98304):
                raise ValueError(
                    f"Rank {rank} has unexpected physical sequence lengths"
                )

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
    per_rank_matches = re.findall(
        r"\[RANK (\d+)\] Iteration (\d+): Loss: ([^\s|]+)", log
    )
    active_losses_by_rank = {}
    if per_rank_matches:
        for rank in range(8):
            active_losses_by_rank[str(rank)] = [
                (int(step), float(value))
                for r, step, value in per_rank_matches
                if int(r) == rank and int(step) in active_steps
            ]
    else:
        active_losses_by_rank["0"] = [
            (int(step), float(value))
            for step, value in re.findall(
                r"Iteration: (\d+), average iter time: [^,]+, total loss ([^\s]+)", log
            )
            if int(step) in active_steps
        ]
    for rank, losses in active_losses_by_rank.items():
        if Counter(step for step, _ in losses) != Counter(active_steps) or not all(
            math.isfinite(loss) for _, loss in losses
        ):
            raise ValueError(
                f"Missing/duplicate/non-finite active-step losses on rank {rank}"
            )
    active_losses = active_losses_by_rank["0"]
    datasets, conditions, modes, shapes = Counter(), Counter(), Counter(), Counter()
    sample_count = 0
    for batch_record in captured:
        batch = batch_record.get("data", batch_record)
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
        "active_losses_by_rank": active_losses_by_rank,
        "captured_samples": sample_count,
        "captured_dataset_sample_counts": dict(datasets),
        "captured_conditions": dict(conditions),
        "captured_vision_input_modes": dict(modes),
        "captured_latent_shapes": dict(shapes),
        "scope": f"One rank and {len(active_steps)} optimizer updates with the original weighted USR sampler. "
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
