"""Preserve the production CLI and replace only its process-local profiler binding."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import runpy
import sys
from pathlib import Path

from workspace import cosmos_repo


def main():
    task = Path(__file__).resolve().parent
    repo = cosmos_repo()
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(task))
    local_rank = os.environ.get("LOCAL_RANK", "0")
    cache = Path(os.environ["COSMOS_KERNEL_CACHE_ROOT"]) / f"rank{local_rank}"
    for name, subdirectory in (
        ("TRITON_CACHE_DIR", "triton"),
        ("TORCH_EXTENSIONS_DIR", "extensions"),
    ):
        directory = cache / subdirectory
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(directory)

    import torch
    from checkpoint_compat import checkpoint_pathlib_compat
    from profile_hook import maybe_enable_profiling

    import cosmos_framework
    import cosmos_framework.trainer as trainer
    from cosmos_framework.model.attention.flash3 import FLASH3_SUPPORTED
    from cosmos_framework.model.attention.natten import NATTEN_SUPPORTED

    if not (FLASH3_SUPPORTED or NATTEN_SUPPORTED):
        raise RuntimeError(
            "USR packed varlen attention requires a compatible FA3 or NATTEN backend"
        )
    if local_rank == "0":
        versions = {}
        for package in (
            "torch",
            "triton",
            "natten",
            "flash-attn",
            "flash-attn-3-nv",
            "transformers",
            "huggingface-hub",
        ):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        runtime = {
            "python": platform.python_version(),
            "packages": versions,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "flash3_supported": FLASH3_SUPPORTED,
            "natten_supported": NATTEN_SUPPORTED,
        }
        (cache.parent.parent / "runtime.json").write_text(
            json.dumps(runtime, indent=2) + "\n"
        )

    if Path(cosmos_framework.__file__).resolve().parent != repo / "cosmos_framework":
        raise RuntimeError("Imported a different Cosmos checkout")
    if (
        trainer.ImaginaireTrainer.train.__globals__["maybe_enable_profiling"]
        is not trainer.maybe_enable_profiling
    ):
        raise RuntimeError("Trainer does not use the expected profiler binding")
    original = trainer.maybe_enable_profiling
    trainer.maybe_enable_profiling = maybe_enable_profiling
    try:
        with checkpoint_pathlib_compat():
            runpy.run_module("cosmos_framework.scripts.train", run_name="__main__")
    finally:
        trainer.maybe_enable_profiling = original


if __name__ == "__main__":
    main()
