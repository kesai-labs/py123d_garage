"""The beartype+jaxtyping import hook: installed when enabled, and actually checking."""

from __future__ import annotations

import os

import pytest

_ENABLED = os.environ.get("PY123D_GARAGE_RUNTIME_TYPE_CHECKING", "false").lower() == "true"


def test_hook_state_matches_environment() -> None:
    from py123d_garage.common import runtime_typing

    if _ENABLED:
        assert runtime_typing.hook_manager is not None
    else:
        assert runtime_typing.hook_manager is None


@pytest.mark.skipif(not _ENABLED, reason="runtime type checking disabled")
def test_annotated_call_is_checked() -> None:
    """A wrong argument type raises through beartype instead of failing deep inside."""
    from beartype.roar import BeartypeCallHintViolation

    from py123d_garage.cache.codec import ZlibCodec

    with pytest.raises((BeartypeCallHintViolation, TypeError)):
        ZlibCodec().encode("not a tensor")  # type: ignore[arg-type]
