"""Validate the actual pure-VLM workload, distributed completion and report coverage."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import yaml

from validate_reports import validate


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_vlm_run(run, *, report_coverage_gaps=False):
    metadata = json.loads((run / "metadata.json").read_text())
    window = json.loads((run / "profile_window.json").read_text())
    require(
        metadata.get("returncode") == 0 and not metadata.get("dryrun"),
        "Training did not complete",
    )
    config_paths = (
        [run / "config.yaml"]
        if (run / "config.yaml").exists()
        else list((run / "job").rglob("config.yaml"))
    )
    require(len(config_paths) == 1, "Expected one resolved training config")
    config = yaml.safe_load(config_paths[0].read_text())
    model = config["model"]["config"]
    for key, expected in {
        "vision_gen": False,
        "action_gen": False,
        "load_vision_tokenizer": False,
        "load_vlm_visual": True,
        "train_vlm_visual": True,
        "predict_text_tokens": True,
        "und_loss_reduction": "sample_mean",
        "und_loss_normalization": "accumulation_window",
    }.items():
        require(model[key] == expected, f"Unexpected model setting {key}")
    require(model["parallelism"]["data_parallel_shard_degree"] == 8, "Expected FSDP8")
    require(
        model["parallelism"]["data_parallel_replicate_degree"] == 1,
        "Expected DP replicate 1",
    )
    require(model["parallelism"]["context_parallel_shard_degree"] == 1, "Expected CP1")
    require(not model["compile"]["enabled"], "Reference recipe has compile disabled")
    require(model["activation_checkpointing"]["mode"] == "full", "Expected full AC")
    require(
        not config["checkpoint"]["load_training_state"],
        "Expected fresh HF initialization",
    )
    require(
        model["vlm_config"]["pretrained_weights"]["enabled"],
        "HF initialization is disabled",
    )
    quality = validate(run / "reports", window["rank"])
    require(
        report_coverage_gaps or not quality["kernel_name_count_mismatches"],
        "Trace/report kernel counts differ",
    )
    require(
        not quality["excess_report_kernel_names"], "Report contains excess kernel calls"
    )
    require(not quality["compile_timed_ranges"], "Active window contains compilation")
    steps = window["active_completed_steps"]
    ga = window["grad_accum_iter"]
    require(ga == 3, "Expected GA3")
    expected_phases = {
        "kernel_list.forward": len(steps) * ga,
        "kernel_list.backward": len(steps) * ga,
        "kernel_list.optimizer": len(steps),
    }
    require(
        quality["phase_annotations"] == expected_phases, "Incomplete training phases"
    )
    ranks = []
    captured = []
    for rank in range(8):
        records = [
            json.loads(line)
            for line in (run / "batches" / f"rank_{rank:05d}.jsonl")
            .read_text()
            .splitlines()
        ]
        starts = [row for row in records if row["record_type"] == "trace_start"]
        ends = [row for row in records if row["record_type"] == "trace_end"]
        batches = [row for row in records if row["record_type"] == "microbatch"]
        updates = [row for row in records if row["record_type"] == "optimizer_step"]
        require(
            len(starts) == 1
            and starts[0]["iteration"] == 0
            and starts[0]["world_size"] == 8,
            f"Rank {rank} start mismatch",
        )
        require(
            len(ends) == 1 and ends[0]["iteration"] == steps[-1],
            f"Rank {rank} incomplete",
        )
        require(
            Counter(row["iteration"] for row in batches)
            == {i: ga for i in range(steps[-1])},
            f"Rank {rank} missing microbatches",
        )
        require(
            Counter(row["iteration"] for row in updates)
            == Counter(range(1, steps[-1] + 1)),
            f"Rank {rank} missing updates",
        )
        for row in updates:
            require(
                all(
                    math.isfinite(row[key])
                    for key in ("loss", "loss_window", "loss_window_weight")
                ),
                f"Rank {rank} non-finite loss",
            )
            require(
                row["loss_window_weight"] > 0,
                f"Rank {rank} unsupervised optimizer window",
            )
        active = [row for row in batches if row["iteration"] + 1 in steps]
        for row in active:
            data = row["data"]
            packed = data["packed_sequence"]
            require(
                not packed["has_generation_vision"]
                and not packed["has_generation_action"],
                "Generation payload in pure VLM run",
            )
            require(
                0 < packed["sequence_length"] <= 36000, "Unexpected sequence length"
            )
            require(bool(data["vlm_inputs"]), "Missing visual inputs")
            require(
                "cross_entropy_loss" in row["output"], "Expected Understanding CE loss"
            )
        parameters = json.loads(
            (run / "batches" / f"rank_{rank:05d}_parameters.json").read_text()
        )
        visual = [p for p in parameters if ".visual." in p["name"]]
        require(
            bool(visual) and all(p["requires_grad"] for p in visual),
            "ViT is absent or frozen",
        )
        ranks.append(
            {
                "rank": rank,
                "microbatches": len(batches),
                "active_microbatches": len(active),
                "active_losses": [row for row in updates if row["iteration"] in steps],
                "trainable_visual_parameter_tensors": len(visual),
            }
        )
        if rank == window["rank"]:
            captured = active
    return {
        "distributed_training": "passed",
        "kernel_count_coverage": (
            "incomplete" if quality["kernel_name_count_mismatches"] else "passed"
        ),
        "kernel_name_count_mismatches": quality["kernel_name_count_mismatches"],
        "workload": metadata.get("workload", "pure_vlm"),
        "run_id": metadata["run_id"],
        "profile_rank": window["rank"],
        "active_completed_steps": steps,
        "phase_annotations": expected_phases,
        "ranks": ranks,
        "captured_sequence_lengths": [
            row["data"]["packed_sequence"]["sequence_length"] for row in captured
        ],
        "captured_sample_count": sum(
            len(row["data"]["packed_sequence"]["sample_lens"]) for row in captured
        ),
        "report_kernel_calls": quality["report_kernel_calls"],
        "unique_report_kernels": quality["unique_report_kernels"],
        "scope": "Observed rank-0 variants in a short training window; not full-dataset coverage or a throughput benchmark.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument(
        "--report-coverage-gaps",
        action="store_true",
        help="Validate training and report incomplete kernel counts explicitly, without claiming exact coverage.",
    )
    args = parser.parse_args()
    result = validate_vlm_run(args.run, report_coverage_gaps=args.report_coverage_gaps)
    path = args.run / "training_validation.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(path)
