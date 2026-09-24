"""Small CUDA probe of resumed-step scheduling and all four report artifacts."""

import argparse
import os
from pathlib import Path
from types import SimpleNamespace as NS

import torch
from profile_hook import maybe_enable_profiling


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["COSMOS_KERNEL_REPORT_DIR"] = str(args.output / "reports")
    os.environ["COSMOS_KERNEL_START_STEP"] = "12600"
    settings = NS(
        enable_profiling=True,
        target_ranks=[0],
        profile_freq=5,
        profile_warmup=1,
        profile_active=2,
        record_shape=True,
        profile_memory=False,
        with_stack=False,
        with_modules=False,
    )
    config = NS(trainer=NS(profiling=settings, max_iter=12605, grad_accum_iter=1))
    torch.manual_seed(42)
    weight = torch.nn.Parameter(
        torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    )
    optimizer = torch.optim.AdamW([weight], lr=0.0001, fused=True)
    value = torch.randn_like(weight)
    with maybe_enable_profiling(config, global_step=12600) as profiler:
        for _ in range(5):
            with torch.profiler.record_function("kernel_list.forward"):
                loss = (value @ weight).float().square().mean()
            with torch.profiler.record_function("kernel_list.backward"):
                loss.backward()
            with torch.profiler.record_function("kernel_list.optimizer"):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            profiler.step()
    print("CUDA_PROBE_PASSED", flush=True)


if __name__ == "__main__":
    main()
