"""A Python preset as plain values for Hydra, and a training plan densified into a cache plan."""

from __future__ import annotations

from typing import Any, TypeVar, cast

from omegaconf import DictConfig, OmegaConf

from py123d_garage.common.config_help.config_handling import complete, composed_values, replay_values
from py123d_garage.common.config_help.import_string import import_string

T = TypeVar("T")

_PLAN_KEY = "plan"


def preset_values(target: str) -> dict[str, Any]:
    """What the module:function factory builds, as plain values with interpolations unresolved."""
    return cast("dict[str, Any]", OmegaConf.to_container(OmegaConf.structured(import_string(target)()), resolve=False))


def finalize_cache(cfg: DictConfig, config_class: type[T]) -> T:
    """The composed cache config, its policy and sources taken from the training plan under the plan key."""
    values = composed_values(cfg)
    plan = cast("dict[str, Any] | None", values.pop(_PLAN_KEY, None))
    if plan is None:
        raise ValueError(
            "caching requires the training plan whose stores it builds: "
            "training@plan=<name> naming a yaml preset, or training@plan=<module:function> naming a Python one.",
        )
    sources = cast("dict[str, dict[str, Any]]", plan["offline_data_sources"])
    if values["build_dense_cache"]:
        for source in sources.values():
            source["garage_scene_filter"]["timestamp_threshold_s"] = None
    values["policy_config"] = plan["policy_config"]
    values["offline_data_sources"] = sources
    merged = cast("DictConfig", OmegaConf.structured(config_class))
    replay_values(merged, values)
    return complete(merged, config_class)
