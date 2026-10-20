# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

from __future__ import annotations

import os

from py123d_garage.api.abstract_benchmark_config import AbstractBenchmarkConfig
from py123d_garage.api.abstract_offline_data_source_config import OfflineEvaluationDataSourceConfig
from py123d_garage.api.abstract_policy_config import AbstractPolicyConfig
from py123d_garage.py123d_help.scene_filters import GarageSceneFilter


def navtest(
    policy_config: AbstractPolicyConfig,
    benchmark_config: AbstractBenchmarkConfig,
) -> OfflineEvaluationDataSourceConfig:
    """
    The navtest samples, as defined in https://github.com/autonomousvision/navsim/blob/main/docs/splits.md.

    The filter guarantees the policy's modalities and history, and what the benchmark asks of a scored scene: its
    scored future window, or the policy's plan horizon for benchmarks that score against the logged future, and the
    route out to the farthest target point only for benchmarks that need it.

    Args:
        policy_config: the architecture block of the policy under evaluation.
        benchmark_config: the benchmark the scenes are scored by.

    Returns:
        the source config.

    Raises:
        ValueError: if PY123D_GARAGE_DATA_ROOT is unset.
    """
    data_root = os.environ.get("PY123D_GARAGE_DATA_ROOT")
    if data_root is None:
        raise ValueError(
            "PY123D_GARAGE_DATA_ROOT must point at the dataset root; the navtest samples live at <root>/nuplan/123D.",
        )
    lidar_scope = "history+initial" if policy_config.required_past_lidar_interval_us else "initial"
    camera_scope = "history+initial" if policy_config.required_past_camera_interval_us else "initial"
    future_duration_us = benchmark_config.scored_scene_future_duration_us or policy_config.trajectory_horizon_us
    min_remaining_route_m = (
        max(policy_config.required_target_point_distances_m, default=0.0)
        if benchmark_config.scored_scene_needs_route
        else None
    )
    return OfflineEvaluationDataSourceConfig(
        garage_scene_filter=GarageSceneFilter(
            split_names=["nuplan_test"],
            future_duration_s=future_duration_us / 1e6,
            history_duration_s=policy_config.required_history_duration_us / 1e6 or None,
            min_remaining_route_m=min_remaining_route_m,
            custom_anchor_filter_fns=["py123d_garage.py123d_help.scene_filters:is_navtest_scene"],
            required_scene_modalities=[
                "ego_state_se3",
                *[
                    f"lidar.{lidar.name.lower()}@{lidar_scope}"
                    for lidar in policy_config.required_lidars.get("nuplan", [])
                ],
                *[
                    f"camera.{camera.name.lower()}@{camera_scope}"
                    for camera in policy_config.required_cameras["nuplan"]
                ],
            ],
        ),
        data_root=f"{data_root}/nuplan/123D",
        served_ego_state_interval_us=100_000,
        served_camera_interval_us=100_000,
        served_lidar_interval_us=100_000,
    )
