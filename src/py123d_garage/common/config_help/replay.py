# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""Loads the policy config saved next to a checkpoint, so evaluation runs the policy as it was trained."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar, cast

from omegaconf import DictConfig, OmegaConf

from py123d_garage.common.config_help.config_handling import (
    apply_override,
    complete,
    replay_values,
    typed_config,
)

T = TypeVar("T")

_CHECKPOINTS_KEY = "policy_config.evaluation_checkpoint_files"


def finalize_evaluation(cfg: DictConfig, config_class: type[T], overrides: list[str]) -> T:
    """
    Builds the evaluation config, with the policy config taken from the checkpoints' config.yaml.

    Args:
        cfg: the config Hydra composed.
        config_class: the benchmark config dataclass.
        overrides: the overrides, whose policy_config ones are applied again over the saved config.

    Returns:
        the benchmark config dataclass.
    """
    merged = typed_config(cfg, config_class)
    checkpoints = cast("list[str] | None", OmegaConf.select(merged, _CHECKPOINTS_KEY))
    if not checkpoints:
        raise ValueError(
            f"evaluation needs {_CHECKPOINTS_KEY}=[<checkpoint>, ...]; the policy replays the config.yaml next to it.",
        )
    saved_policy_configs = [_saved_policy_config(checkpoint) for checkpoint in checkpoints]
    for checkpoint, saved in zip(checkpoints[1:], saved_policy_configs[1:], strict=True):
        if saved != saved_policy_configs[0]:
            raise ValueError(
                f"{checkpoint} was trained with a different policy_config than {checkpoints[0]}; "
                "an ensemble averages checkpoints of one recipe.",
            )
    replay_values(merged, {"policy_config": saved_policy_configs[0]}, skip_unknown=True)
    # The saved config overwrote the policy_config overrides, so apply them again without their + prefix.
    for override in overrides:
        appended = override.lstrip("+")
        if appended.startswith("policy_config."):
            apply_override(merged, appended)
    return complete(merged, config_class)


def _saved_policy_config(checkpoint: str) -> dict[str, Any]:
    """
    Loads the policy config from the config.yaml next to a checkpoint.

    Args:
        checkpoint: path of the checkpoint file.

    Returns:
        the saved policy config as plain values, without its drawing styles.
    """
    config_file = Path(checkpoint).parent / "config.yaml"
    if not config_file.is_file():
        raise ValueError(
            f"no config.yaml next to {checkpoint}; evaluation rebuilds the policy from the run config saved with the checkpoint.",
        )
    saved = cast("dict[str, Any]", OmegaConf.to_container(OmegaConf.load(config_file)))
    return cast("dict[str, Any]", _without_visualization(saved["policy_config"]))


def _without_visualization(config: Any) -> Any:
    """
    Drops the drawing styles from a saved config, since they come from the current code, not the training run.

    Args:
        config: the saved config values.

    Returns:
        the values without any visualization_config.
    """
    if isinstance(config, dict):
        return {
            key: _without_visualization(value)
            for key, value in cast("dict[str, Any]", config).items()
            if key != "visualization_config"
        }
    return config
