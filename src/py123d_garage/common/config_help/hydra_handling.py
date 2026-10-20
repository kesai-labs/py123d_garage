# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""The entry points' schema in Hydra's config store, and composing their config outside of Hydra's runner."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, TypeVar

from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from py123d_garage.common.config_help.config_handling import command_line_overrides, finalize
from py123d_garage.common.config_help.replay import finalize_evaluation

T = TypeVar("T")

# The primary configs, one per entry point.
CONFIG_PATH = Path(__file__).parents[2] / "config" / "hydra"


def register_schema(config_name: str, config_class: type[Any]) -> None:
    """
    Stores the dataclass defaults as schema/<config_name>, the first entry of the primary config's defaults.

    Args:
        config_name: name of the entry point's primary config.
        config_class: the entry point's config dataclass.
    """
    # Hydra's one-line error summary hides the raising frame.
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")
    ConfigStore.instance().store(
        group="schema",
        name=config_name,
        node=OmegaConf.to_container(OmegaConf.structured(config_class), resolve=False),
        package="_global_",
    )


def hydra_overrides() -> list[str]:
    """
    Reads the command line's overrides of the running job.

    Returns:
        the job's overrides; the sweep's own for a multirun job.
    """
    return list(HydraConfig.get().overrides.task)


def compose_config(config_name: str, config_class: type[Any], overrides: list[str]) -> DictConfig:
    """
    Composes the entry point's config without Hydra's runner.

    Args:
        config_name: name of the entry point's primary config.
        config_class: the entry point's config dataclass.
        overrides: the Hydra overrides.

    Returns:
        the composed config, with the hydra node.
    """
    register_schema(config_name, config_class)
    with initialize_config_dir(config_dir=str(CONFIG_PATH), version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides, return_hydra_config=True)
    # Publishes the hydra node like a running job would, so run_dir() works on this path too.
    HydraConfig.instance().set_config(cfg)
    return cfg


def load_config(config_class: type[T], config_name: str, args: list[str] | None = None) -> T:
    """
    Builds the config the entry point would build from PY123D_GARAGE_CONFIG and the command line.

    Args:
        config_class: the entry point's config dataclass.
        config_name: name of the entry point's primary config.
        args: overrides that replace the command line's; None reads sys.argv.

    Returns:
        the config dataclass.
    """
    return finalize(compose_config(config_name, config_class, command_line_overrides(args)), config_class)


def load_evaluation_config(config_class: type[T], config_name: str, args: list[str] | None = None) -> T:
    """
    Builds the benchmark config the evaluation entry point would build, the policy replayed from its checkpoint.

    Args:
        config_class: the benchmark config dataclass.
        config_name: name of the entry point's primary config.
        args: overrides that replace the command line's; None reads sys.argv.

    Returns:
        the benchmark config dataclass.
    """
    overrides = command_line_overrides(args)
    return finalize_evaluation(compose_config(config_name, config_class, overrides), config_class, overrides)
