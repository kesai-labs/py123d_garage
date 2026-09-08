from __future__ import annotations

import importlib
import sys

from beartype import BeartypeConf, beartype

# PEP 484 numeric tower (int satisfies float), matching static checkers; beartype
# would otherwise reject int-valued configs passed to float parameters.
typechecker = beartype(conf=BeartypeConf(is_pep484_tower=True))

# Set by py123d_garage/__init__.py once the jaxtyping import hook is installed;
# stays None when PY123D_GARAGE_RUNTIME_TYPE_CHECKING is off.
hook_manager = None


def import_unwrapped(module_name: str) -> None:
    """
    Import a module with the jaxtyping import hook paused, then restore it.

    For numba @njit(cache=True) kernel modules: wrapping a kernel with @jaxtyped
    makes numba pickle the wrapper's closure for the on-disk cache, which fails.

    Args:
        module_name: dotted path of the module to import unwrapped
    """
    if hook_manager is None:
        importlib.import_module(module_name)
        return
    sys.meta_path.remove(hook_manager.hook)
    try:
        importlib.import_module(module_name)
    finally:
        sys.meta_path.insert(0, hook_manager.hook)
