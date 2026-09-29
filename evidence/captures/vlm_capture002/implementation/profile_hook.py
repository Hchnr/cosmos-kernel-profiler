"""Task-local, single-window profiler; never modifies the training source files."""

from __future__ import annotations

import contextlib
import gzip
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from profiler_reports import export_kernel_reports
from validate_reports import validate


def schedule_parameters(settings):
    warmup, active, frequency = (
        settings.profile_warmup,
        settings.profile_active,
        settings.profile_freq,
    )
    if active <= 0 or warmup < 0 or frequency < warmup + active:
        raise ValueError("Invalid profiler schedule")
    return {
        "wait": frequency - warmup - active,
        "warmup": warmup,
        "active": active,
        "repeat": 1,
    }


def event_snapshot(events):
    """Keep the report's public event attributes for auditable offline re-export."""
    for event in events:
        yield {
            "key": str(getattr(event, "key", None) or getattr(event, "name", "null")),
            "input_shapes": getattr(event, "input_shapes", None),
            "structured_input_shapes": getattr(event, "structured_input_shapes", None),
            "input_dtypes": getattr(event, "input_dtypes", None),
            "is_user_annotation": bool(getattr(event, "is_user_annotation", False)),
            "kernels": [
                {
                    "name": kernel.name,
                    "duration": float(
                        kernel.duration()
                        if callable(kernel.duration)
                        else kernel.duration
                    ),
                }
                for kernel in (getattr(event, "kernels", None) or ())
            ],
        }


def restore_events(rows):
    for row in rows:
        yield SimpleNamespace(
            **{
                **row,
                "kernels": [SimpleNamespace(**kernel) for kernel in row["kernels"]],
            }
        )


@contextlib.contextmanager
def maybe_enable_profiling(config, *, global_step=0):
    settings = config.trainer.profiling
    rank = dist.get_rank() if dist.is_initialized() else 0
    if not settings.enable_profiling or rank not in settings.target_ranks:
        yield None
        return
    expected_start = int(os.environ["COSMOS_KERNEL_START_STEP"])
    if global_step != expected_start:
        raise RuntimeError(
            f"Unexpected resume step {global_step}; expected {expected_start}"
        )
    schedule = schedule_parameters(settings)
    if config.trainer.max_iter - global_step != settings.profile_freq:
        raise RuntimeError("Training must end at the end of the single capture window")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA profiling requires an available CUDA device")
    output = Path(os.environ["COSMOS_KERNEL_REPORT_DIR"])
    staging = output.with_name(output.name + ".pending")
    if output.exists() or staging.exists():
        raise FileExistsError(f"Refusing to overwrite a capture: {output}")
    staging.mkdir(parents=True)
    metadata = {
        "rank": rank,
        "global_start_step": global_step,
        "schedule": schedule,
        "active_completed_steps": list(
            range(
                global_step + schedule["wait"] + schedule["warmup"] + 1,
                global_step + settings.profile_freq + 1,
            )
        ),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "grad_accum_iter": config.trainer.grad_accum_iter,
    }
    (output.parent / "profile_window.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    captures = 0

    def handler(profiler):
        nonlocal captures
        if captures:
            raise RuntimeError("Expected only one trace callback")
        captures += 1
        trace = staging / f"rank-{rank}.json.gz"
        print(
            f"KERNEL_LIST_EXPORT_BEGIN local_step={profiler.step_num} rank={rank}",
            flush=True,
        )
        profiler.export_chrome_trace(str(trace))
        events = profiler.events()
        with gzip.open(
            output.parent / f"rank-{rank}_report_events.json.gz", "wt"
        ) as stream:
            json.dump(list(event_snapshot(events)), stream)
        if profiler.step_num != settings.profile_freq:
            raise RuntimeError(
                f"Incomplete active window: local step {profiler.step_num}"
            )
        export_kernel_reports(events, staging, rank, trace)
        quality = validate(staging, rank)
        (output.parent / "report_quality.json").write_text(
            json.dumps(quality, indent=2, ensure_ascii=False) + "\n"
        )
        staging.rename(output)
        print(
            f"KERNEL_LIST_EXPORT_DONE {output} calls={quality['report_kernel_calls']}",
            flush=True,
        )

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(**schedule),
        on_trace_ready=handler,
        record_shapes=settings.record_shape,
        profile_memory=settings.profile_memory,
        with_stack=settings.with_stack,
        with_modules=settings.with_modules,
    ) as profiler:
        # Keep this counter local: assigning a resumed global step breaks repeat=1.
        yield profiler
    if captures != 1:
        raise RuntimeError(f"Capture incomplete: {captures} trace callbacks")
