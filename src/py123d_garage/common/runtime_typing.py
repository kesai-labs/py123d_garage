# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI
# Adapted from kesai-labs/lead (MIT License).

from __future__ import annotations

import importlib
import importlib.util
import sys
from typing import Any

from beartype import BeartypeConf, beartype

# PEP 484 numeric tower, so int values pass float parameters as they do for static checkers.
typechecker = beartype(conf=BeartypeConf(is_pep484_tower=True))

# The jaxtyping import hook set by py123d_garage/__init__.py; None when PY123D_GARAGE_RUNTIME_TYPE_CHECKING is off.
hook_manager = None


def import_unwrapped(module_name: str) -> None:
    """
    Imports a module with the jaxtyping import hook paused, for numba @njit(cache=True) and Triton @jit kernels that cannot be wrapped.

    Args:
        module_name: dotted path of the module to import unwrapped.
    """
    if hook_manager is None:
        importlib.import_module(module_name)
        return
    # A hooked package __init__ runs inside the jaxtyping loader, which patches cache_from_source to its own .pyc tag;
    # the unwrapped import reads and writes the plain .pyc instead, or it loads instrumented bytecode of an earlier
    # wrapped import.
    bootstrap_external: Any = sys.modules["importlib._bootstrap_external"]
    patched_cache_from_source = bootstrap_external.cache_from_source
    bootstrap_external.cache_from_source = importlib.util.cache_from_source
    sys.meta_path.remove(hook_manager.hook)
    try:
        importlib.import_module(module_name)
    finally:
        sys.meta_path.insert(0, hook_manager.hook)
        bootstrap_external.cache_from_source = patched_cache_from_source
