# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""Prints the override that points an offline evaluation entry point at a slice of one dataset."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from py123d_garage.api.abstract_benchmark_config import AbstractBenchmarkConfig
from py123d_garage.api.abstract_offline_data_source_config import AbstractOfflineDataSourceConfig
from py123d_garage.api.abstract_policy import AbstractPolicy, AnyPolicy
from py123d_garage.api.abstract_policy_config import AbstractPolicyConfig
from py123d_garage.common.config_help import build_from_string, load_evaluation_config
from py123d_garage.config.presets.offline_data_sources.navsim import navtest
from py123d_garage.config.presets.offline_data_sources.physical_ai_av import physical_ai_av_test
from py123d_garage.config.schema.evaluation.open_loop_config import OpenLoopBenchmarkConfig

# The open-loop guarantees (plan horizon, route) also satisfy navsim's, so one slice serves both protocols.
SOURCE_BUILDERS: dict[
    str,
    Callable[[AbstractPolicyConfig, AbstractBenchmarkConfig], AbstractOfflineDataSourceConfig],
] = {
    "navtest": navtest,
    "physical_ai_av": lambda policy_config, _benchmark_config: physical_ai_av_test(
        policy_config,
        cache_store="smoke",
        timestamp_threshold_s=None,
    ),
}
# The fields a training source carries beyond an evaluation source.
_TRAINING_ONLY_FIELDS = ("source_weight", "has_box_annotations")


def main() -> None:
    """Prints the source override, for the caller to pass on verbatim."""
    checkpoint_file, dataset, max_num_scenes = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
    # The same replay the entry points run, so the slice is built against the architecture the checkpoint was trained with.
    benchmark_config: OpenLoopBenchmarkConfig = load_evaluation_config(
        OpenLoopBenchmarkConfig,
        "evaluate_open_loop",
        args=[f"policy_config.evaluation_checkpoint_files=[{checkpoint_file}]"],
    )
    policy: AnyPolicy = build_from_string(benchmark_config.policy_config, AbstractPolicy)
    source = SOURCE_BUILDERS[dataset](policy.policy_config, benchmark_config)
    served: dict[str, Any] = OmegaConf.to_container(OmegaConf.structured(source), resolve=True)  # pyright: ignore[reportAssignmentType]
    for field in _TRAINING_ONLY_FIELDS:
        served.pop(field, None)
    served["cache_root"] = None
    served["garage_scene_filter"]["max_num_scenes"] = max_num_scenes
    print(f"benchmark_offline_data_sources=[{_override_literal(served)}]")


def _override_literal(value: Any) -> str:
    """
    The value in Hydra's override grammar, every string quoted.

    Args:
        value: plain config values.

    Returns:
        the override text.
    """
    if isinstance(value, dict):
        return "{" + ",".join(f"{key}:{_override_literal(item)}" for key, item in value.items()) + "}"
    if isinstance(value, list):
        return "[" + ",".join(_override_literal(item) for item in value) + "]"
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    return "'" + str(value).replace("\\", "\\\\").replace("'", "\\'") + "'"


if __name__ == "__main__":
    main()
