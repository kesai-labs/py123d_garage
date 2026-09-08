from __future__ import annotations

import os

from py123d_garage.api.abstract_offline_data_source_config import OfflineEvaluationDataSourceConfig
from py123d_garage.api.abstract_policy_config import AbstractPolicyConfig
from py123d_garage.config.schema.evaluation.navsim_config import NavsimBenchmarkConfig
from py123d_garage.py123d_help.scene_filters import GarageSceneFilter


def navtest(policy_config: AbstractPolicyConfig) -> OfflineEvaluationDataSourceConfig:
    """
    The navtest samples. See https://github.com/autonomousvision/navsim/blob/main/docs/splits.md
    for the official split definitions.

    Args:
        policy_config: the architecture block of the policy under evaluation.

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
    return OfflineEvaluationDataSourceConfig(
        garage_scene_filter=GarageSceneFilter(
            split_names=["nuplan_test"],
            future_duration_s=NavsimBenchmarkConfig.navsim_scoring_future_duration_us / 1e6,
            history_duration_s=policy_config.required_history_duration_us / 1e6 or None,
            min_remaining_route_m=max(policy_config.required_target_point_distances_m),
            custom_anchor_filter_fns=["py123d_garage.py123d_help.scene_filters:is_navtest_scene"],
            required_scene_modalities=[
                "ego_state_se3",
                *[
                    f"lidar.{lidar.name.lower()}@{lidar_scope}"
                    for lidar in policy_config.required_lidars.get("nuplan", [])
                ],
                *[f"camera.{camera.name.lower()}@initial" for camera in policy_config.required_cameras["nuplan"]],
            ],
        ),
        data_root=f"{data_root}/nuplan/123D",
        served_ego_state_interval_us=100_000,
        served_camera_interval_us=100_000,
        served_lidar_interval_us=100_000,
    )
