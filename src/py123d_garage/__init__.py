from __future__ import annotations

import os

if os.environ.get("PY123D_GARAGE_RUNTIME_TYPE_CHECKING", "false").lower() == "true":
    from jaxtyping import install_import_hook

    from py123d_garage.common import runtime_typing

    # Applies @jaxtyped(typechecker=...) to every function and dataclass py123d_garage.* imported after this point.
    runtime_typing.hook_manager = install_import_hook(
        "py123d_garage",
        "py123d_garage.common.runtime_typing.typechecker",
    )
