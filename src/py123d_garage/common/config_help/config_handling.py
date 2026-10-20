# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""Typing, validating, and saving the config Hydra composed."""

from __future__ import annotations

import dataclasses
import logging
import os
import re
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from types import UnionType
from typing import Any, TypeVar, Union, cast, get_args, get_origin, get_type_hints

import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

LOG = logging.getLogger(__name__)

T = TypeVar("T")


def command_line_overrides(args: list[str] | None = None) -> list[str]:
    """
    Collects the PY123D_GARAGE_CONFIG overrides followed by the command line's.

    Args:
        args: overrides that replace the command line's, for tests; None reads sys.argv.

    Returns:
        the overrides in the order they apply.
    """
    return [*os.environ.get("PY123D_GARAGE_CONFIG", "").split(), *(sys.argv[1:] if args is None else args)]


def apply_override(merged: DictConfig, override: str) -> None:
    """
    Applies one key=value override onto a typed node.

    Args:
        merged: typed node to write into.
        override: the override as key=value, the value parsed as YAML.
    """
    key, separator, value = override.partition("=")
    if not separator:
        raise ValueError(f"override {override!r} is not of the form key=value")
    try:
        parsed = yaml.safe_load(value)
    except yaml.YAMLError:
        parsed = value
    OmegaConf.update(merged, key, parsed, merge=True)


def replay_values(
    merged: DictConfig,
    values: Mapping[str, Any],
    path: str = "",
    skip_unknown: bool = False,
) -> None:
    """
    Writes plain values into the typed node under path, one leaf at a time.

    Args:
        merged: typed node to write into.
        values: plain values, nested like the node.
        path: dotted path of values inside merged.
        skip_unknown: drop keys the schema does not know, for configs saved by older code.
    """
    for key, value in values.items():
        # Hydra's _target_ is written as module.Class, our target as module:Class.
        if key == "_target_" and isinstance(value, str):
            module_name, _, class_name = value.rpartition(".")
            key, value = "target", f"{module_name}:{class_name}"
        child_path = f"{path}.{key}" if path else str(key)
        if skip_unknown and not _schema_has_field(merged, path, str(key)):
            LOG.warning(f"Skipping {child_path}: the schema has no such field")
            continue
        children = cast("Mapping[Any, Any]", value) if isinstance(value, Mapping) else None
        if children is not None and _is_dict_field(merged, path, str(key)):
            # A dict field holds data, not a block: the preset's entries replace the schema's instead of joining them.
            OmegaConf.update(merged, child_path, {}, merge=False)
        if children is not None and all(isinstance(child_key, str) for child_key in children):
            # Create the typed node of a block the schema leaves None, so its leaves get typed.
            if OmegaConf.select(merged, child_path) is None:
                block_class = _block_class(merged, path, str(key))
                if block_class is not None:
                    OmegaConf.update(merged, child_path, OmegaConf.structured(block_class()), merge=False)
            replay_values(merged, children, child_path, skip_unknown)
        elif children is not None:
            # Entries of a dict with non-string keys, set one by one so OmegaConf converts their enum names.
            node = cast(DictConfig, OmegaConf.select(merged, child_path))
            for child_key, child in children.items():
                node[child_key] = child
        else:
            # Merging a whole node skips the enum conversion inside dict[str, list[Enum]]; setting a leaf does not.
            OmegaConf.update(merged, child_path, value, merge=True)


def _schema_has_field(merged: DictConfig, path: str, key: str) -> bool:
    """
    Checks whether the schema allows a key under a path.

    Args:
        merged: typed root node.
        path: dotted path of the parent node.
        key: name of the child.

    Returns:
        False only when the node at path is a dataclass without a field named key.
    """
    owner = OmegaConf.get_type(OmegaConf.select(merged, path) if path else merged)
    if owner is None or not dataclasses.is_dataclass(owner):
        return True
    return key in {field.name for field in dataclasses.fields(owner)}


def _is_dict_field(merged: DictConfig, path: str, key: str) -> bool:
    """
    Checks whether path.key is a dict-typed field of a dataclass.

    Args:
        merged: typed root node.
        path: dotted path of the parent node.
        key: name of the field.

    Returns:
        True for a dict field, whose entries are data rather than schema blocks.
    """
    owner = OmegaConf.get_type(OmegaConf.select(merged, path) if path else merged)
    if owner is None or not dataclasses.is_dataclass(owner):
        return False
    hint = get_type_hints(owner).get(key)
    return hint is not None and _is_dict_hint(hint)


def _is_dict_hint(hint: Any) -> bool:
    """
    Checks whether a type hint is a dict, possibly inside a union.

    Args:
        hint: the type hint to check.

    Returns:
        True for dict[...] and for unions holding one.
    """
    origin = get_origin(hint)
    if origin is dict:
        return True
    if origin in (Union, UnionType):
        return any(_is_dict_hint(argument) for argument in get_args(hint))
    return False


def _block_class(merged: DictConfig, path: str, key: str) -> type[Any] | None:
    """
    Finds the dataclass that types the block at path.key.

    Args:
        merged: typed root node.
        path: dotted path of the parent node.
        key: name of the block.

    Returns:
        the dataclass of the block, or None when it is untyped.
    """
    parent = OmegaConf.select(merged, path) if path else merged
    owner = OmegaConf.get_type(parent)
    if owner is not None and dataclasses.is_dataclass(owner):
        return _dataclass_in(get_type_hints(owner)[key])
    # For an element of a dict[str, Dataclass] field, the type hint sits on the grandparent.
    grandparent_path, _, parent_key = path.rpartition(".")
    grandparent = OmegaConf.select(merged, grandparent_path) if grandparent_path else merged
    grandparent_owner = OmegaConf.get_type(grandparent)
    if parent_key and grandparent_owner is not None and dataclasses.is_dataclass(grandparent_owner):
        return _dataclass_in(get_type_hints(grandparent_owner)[parent_key])
    return None


def _dataclass_in(hint: Any) -> type[Any] | None:
    """
    Finds the first dataclass in a type hint, through unions and container arguments.

    Args:
        hint: the type hint to search.

    Returns:
        the first dataclass found, or None.
    """
    # dict[str, X] passes isinstance(hint, type), so check the origin first.
    if get_origin(hint) is None:
        return hint if isinstance(hint, type) and dataclasses.is_dataclass(hint) else None
    for argument in get_args(hint):
        found = _dataclass_in(argument)
        if found is not None:
            return found
    return None


def reject_cross_key_interpolations(container: object, path: str = "") -> None:
    """
    Rejects ${other_key} references, since wiring belongs in Python; only ${oc.env:...} is allowed.

    Args:
        container: plain config values to check, recursively.
        path: dotted path of container, for the error message.
    """
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
    """
    Writes the composed config onto the schema's typed node, so unknown keys and wrong types fail here.

    Args:
        cfg: the config Hydra composed.
        config_class: the entry point's config dataclass.

    Returns:
        the typed node.
    """
    merged = cast("DictConfig", OmegaConf.structured(config_class))
    replay_values(merged, composed_values(cfg))
    return merged


def composed_values(cfg: DictConfig) -> dict[str, Any]:
    """
    Converts the composed config to plain values, interpolations unresolved and Hydra's own node dropped.

    Args:
        cfg: the config Hydra composed.

    Returns:
        the plain values.
    """
    values = cast("dict[str, Any]", OmegaConf.to_container(cfg, resolve=False))
    values.pop("hydra", None)
    return values


def complete(merged: DictConfig, config_class: type[T]) -> T:
    """
    Converts the typed node to the dataclass, after the cross-key check.

    Args:
        merged: the typed node.
        config_class: the dataclass the node is typed with.

    Returns:
        the config dataclass.
    """
    reject_cross_key_interpolations(OmegaConf.to_container(merged, resolve=False))
    config = OmegaConf.to_object(merged)
    assert isinstance(config, config_class)
    return config


def finalize(cfg: DictConfig, config_class: type[T]) -> T:
    """
    Turns the composed config into the entry point's dataclass.

    Args:
        cfg: the config Hydra composed.
        config_class: the entry point's config dataclass.

    Returns:
        the config dataclass.
    """
    return complete(typed_config(cfg, config_class), config_class)


def run_dir() -> Path:
    """
    Finds the run's output dir.

    Returns:
        hydra.run.dir, or the job's subdir of hydra.sweep.dir in a multirun.
    """
    hydra_config = cast("DictConfig", HydraConfig.get())
    # runtime.output_dir is only set inside a running job, so fall back to run.dir.
    output_dir = cast("str | None", OmegaConf.select(hydra_config, "runtime.output_dir"))
    return Path(output_dir or hydra_config.run.dir)


def save_config(config: object) -> None:
    """
    Writes the resolved config into the run dir as config.yaml.

    Args:
        config: the config dataclass to save.
    """
    output_dir = run_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.structured(config), output_dir / "config.yaml")
