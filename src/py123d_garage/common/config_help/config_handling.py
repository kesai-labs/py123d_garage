"""Typing, validating, and saving the config Hydra composed."""

from __future__ import annotations

import dataclasses
import os
import re
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, TypeVar, cast, get_args, get_origin, get_type_hints

import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

T = TypeVar("T")


def command_line_overrides(args: list[str] | None = None) -> list[str]:
    """The PY123D_GARAGE_CONFIG overrides followed by the command line's, or by args for tests."""
    return [*os.environ.get("PY123D_GARAGE_CONFIG", "").split(), *(sys.argv[1:] if args is None else args)]


def apply_override(merged: DictConfig, override: str) -> None:
    """Applies one key=value override onto a typed node."""
    key, separator, value = override.partition("=")
    if not separator:
        raise ValueError(f"override {override!r} is not of the form key=value")
    try:
        parsed = yaml.safe_load(value)
    except yaml.YAMLError:
        parsed = value
    OmegaConf.update(merged, key, parsed, merge=True)


def replay_values(merged: DictConfig, values: Mapping[str, Any], path: str = "") -> None:
    """Writes plain values into the typed node under path, one leaf at a time."""
    for key, value in values.items():
        # Hydra's _target_ is the module.Class spelling of target.
        if key == "_target_" and isinstance(value, str):
            module_name, _, class_name = value.rpartition(".")
            key, value = "target", f"{module_name}:{class_name}"
        child_path = f"{path}.{key}" if path else str(key)
        children = cast("Mapping[Any, Any]", value) if isinstance(value, Mapping) else None
        if children is not None and all(isinstance(child_key, str) for child_key in children):
            # A block the schema leaves None gets its typed node first, so the leaves below land typed.
            if OmegaConf.select(merged, child_path) is None:
                block_class = _block_class(merged, path, str(key))
                if block_class is not None:
                    OmegaConf.update(merged, child_path, OmegaConf.structured(block_class()), merge=False)
            replay_values(merged, children, child_path)
        else:
            # Merging a whole node skips the enum coercion inside dict[str, list[Enum]]; a leaf assignment does not.
            OmegaConf.update(merged, child_path, value, merge=True)


def _block_class(merged: DictConfig, path: str, key: str) -> type[Any] | None:
    """The dataclass the block at path.key is typed with, or None when it is untyped."""
    parent = OmegaConf.select(merged, path) if path else merged
    owner = OmegaConf.get_type(parent)
    if owner is not None and dataclasses.is_dataclass(owner):
        return _dataclass_in(get_type_hints(owner)[key])
    # An element of a dict[str, Dataclass] field: the parent is a plain dict, the hint sits one level up.
    grandparent_path, _, parent_key = path.rpartition(".")
    grandparent = OmegaConf.select(merged, grandparent_path) if grandparent_path else merged
    grandparent_owner = OmegaConf.get_type(grandparent)
    if parent_key and grandparent_owner is not None and dataclasses.is_dataclass(grandparent_owner):
        return _dataclass_in(get_type_hints(grandparent_owner)[parent_key])
    return None


def _dataclass_in(hint: Any) -> type[Any] | None:
    """The first dataclass in a type hint, through unions and container arguments."""
    # dict[str, X] passes isinstance(hint, type), so the origin decides whether to descend.
    if get_origin(hint) is None:
        return hint if isinstance(hint, type) and dataclasses.is_dataclass(hint) else None
    for argument in get_args(hint):
        found = _dataclass_in(argument)
        if found is not None:
            return found
    return None


def reject_cross_key_interpolations(container: object, path: str = "") -> None:
    """Rejects ${other_key} references; wiring belongs in Python, only ${oc.env:...} is allowed."""
    items: Iterable[tuple[object, object]]
    if isinstance(container, Mapping):
        items = cast("Mapping[object, object]", container).items()
    elif isinstance(container, list):
        items = enumerate(cast("list[object]", container))
    else:
        return
    for key, value in items:
        child = f"{path}.{key}" if path else str(key)
        if isinstance(value, str):
            if re.search(r"\$\{(?!oc\.env:)", value):
                raise ValueError(
                    f"{child}={value!r} uses a cross-key interpolation; "
                    "set the value explicitly, only ${oc.env:...} is allowed.",
                )
        else:
            reject_cross_key_interpolations(value, child)


def typed_config(cfg: DictConfig, config_class: type[T]) -> DictConfig:
    """The composed config replayed onto the schema's structured node; unknown keys and wrong types fail here."""
    merged = cast("DictConfig", OmegaConf.structured(config_class))
    replay_values(merged, composed_values(cfg))
    return merged


def composed_values(cfg: DictConfig) -> dict[str, Any]:
    """The composed config as plain values, interpolations unresolved and Hydra's own node dropped."""
    values = cast("dict[str, Any]", OmegaConf.to_container(cfg, resolve=False))
    values.pop("hydra", None)
    return values


def complete(merged: DictConfig, config_class: type[T]) -> T:
    """The typed node as the dataclass, after the cross-key check."""
    reject_cross_key_interpolations(OmegaConf.to_container(merged, resolve=False))
    config = OmegaConf.to_object(merged)
    assert isinstance(config, config_class)
    return config


def finalize(cfg: DictConfig, config_class: type[T]) -> T:
    """The composed config as the entry point's dataclass."""
    return complete(typed_config(cfg, config_class), config_class)


def run_dir() -> Path:
    """The run's output dir: hydra.run.dir, or the job's subdir of hydra.sweep.dir in a multirun."""
    hydra_config = cast("DictConfig", HydraConfig.get())
    # runtime.output_dir is only set by a running job; the compose path falls back to run.dir.
    output_dir = cast("str | None", OmegaConf.select(hydra_config, "runtime.output_dir"))
    return Path(output_dir or hydra_config.run.dir)


def save_config(config: object) -> None:
    """Writes the resolved config into the run dir as config.yaml."""
    output_dir = run_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.structured(config), output_dir / "config.yaml")
