"""Locate the external Cosmos checkout without assuming this repository's layout."""

import os
from pathlib import Path


def cosmos_repo():
    value = os.environ.get("COSMOS_REPO")
    if not value:
        raise RuntimeError(
            "Set COSMOS_REPO to the Cosmos framework checkout before launching training"
        )
    path = Path(value).expanduser().resolve()
    if (
        not (path / "cosmos_framework/trainer/__init__.py").is_file()
        or not (path / "pyproject.toml").is_file()
    ):
        raise ValueError(f"Not a Cosmos framework checkout: {path}")
    return path
