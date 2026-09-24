"""Resolve Python 3.13 pathlib pickle globals when reading checkpoints on 3.12."""

import contextlib
import pathlib
import sys


@contextlib.contextmanager
def checkpoint_pathlib_compat():
    name = "pathlib._local"
    install = sys.version_info < (3, 13) and name not in sys.modules
    if install:
        # Both modules expose the same public Path classes. Only the module name
        # moved; tensor payloads, metadata values and filesystem paths are intact.
        sys.modules[name] = pathlib
    try:
        yield
    finally:
        if install:
            del sys.modules[name]
