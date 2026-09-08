"""Resolution of the module:name import strings the configs carry instead of objects."""

from __future__ import annotations

import importlib
from typing import Any, Protocol, TypeVar, runtime_checkable

T = TypeVar("T")


def import_string(target: str) -> Any:
    """Imports the object named by a module:name path."""
    module_name, _, object_name = target.partition(":")
    return getattr(importlib.import_module(module_name), object_name)


@runtime_checkable
class _HasTarget(Protocol):
    target: str


def build_from_string(config: _HasTarget, base_class: type[T]) -> T:
    """Constructs the base_class subclass config.target names, passing it the config."""
    target_class = import_string(config.target)
    if not issubclass(target_class, base_class):
        raise TypeError(f"target {config.target!r} is not a {base_class.__name__} subclass.")
    return target_class(config)  # pyright: ignore[reportCallIssue]
