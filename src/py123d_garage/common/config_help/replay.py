"""Replay of the config.yaml saved next to a checkpoint, so evaluation runs the trained policy configuration."""

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

_CHECKPOINT_KEY = "policy_config.evaluation_checkpoint_file"


def finalize_evaluation(cfg: DictConfig, config_class: type[T], overrides: list[str]) -> T:
    """The composed benchmark config, its policy replayed from the checkpoint's config.yaml, the overrides back on top."""
    merged = typed_config(cfg, config_class)
    checkpoint = cast("str | None", OmegaConf.select(merged, _CHECKPOINT_KEY))
    if not checkpoint:
        raise ValueError(
            f"evaluation needs {_CHECKPOINT_KEY}=<checkpoint>; the policy replays the config.yaml next to it.",
        )
    config_file = Path(checkpoint).parent / "config.yaml"
    if not config_file.is_file():
        raise ValueError(
            f"no config.yaml next to {checkpoint}; evaluation rebuilds the policy from the run config saved with the checkpoint.",
        )
    saved = cast("dict[str, Any]", OmegaConf.to_container(OmegaConf.load(config_file)))
    replay_values(merged, {"policy_config": saved["policy_config"]})
    # Hydra applied the overrides before the replay overwrote them; a block the schema leaves empty arrives with a + prefix.
    for override in overrides:
        appended = override.lstrip("+")
        if appended.startswith("policy_config."):
            apply_override(merged, appended)
    return complete(merged, config_class)
