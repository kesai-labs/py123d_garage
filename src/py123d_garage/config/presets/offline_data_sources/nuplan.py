# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

from __future__ import annotations

from typing import Final

from py123d_garage.api.abstract_offline_data_source_config import OfflineTrainingDataSourceConfig
from py123d_garage.api.abstract_policy_config import AbstractPolicyConfig
from py123d_garage.py123d_help.scene_filters import GarageSceneFilter

_PLAUSIBILITY_FILTER_FNS: Final[list[str]] = [
    "py123d_garage.py123d_help.scene_filters:plausible_ego_motion",
    "py123d_garage.py123d_help.scene_filters:plausible_route",
]


def _nuplan(
    policy_config: AbstractPolicyConfig,
    split_name: str,
    cache_store: str,
    custom_anchor_filter_fns: list[str],
    timestamp_threshold_s: float | None,
) -> OfflineTrainingDataSourceConfig:
    """
    Builds the nuplan training source of one split for a policy.

    Args:
        policy_config: the policy the source will serve.
        split_name: name of the 123D split.
        cache_store: name of the feature store under the dataset's cache dir.
        custom_anchor_filter_fns: import paths of extra anchor filters.
        timestamp_threshold_s: minimum time between selected anchors; None = every eligible frame.

    Returns:
        the source config.
    """
    lidar_scope = "history+initial" if policy_config.required_past_lidar_interval_us else "initial"
    camera_scope = "history+initial" if policy_config.required_past_camera_interval_us else "initial"
    return OfflineTrainingDataSourceConfig(
        garage_scene_filter=GarageSceneFilter(
            split_names=[split_name],
            future_duration_s=policy_config.trajectory_horizon_us / 1e6,
            history_duration_s=policy_config.required_history_duration_us / 1e6 or None,
            min_remaining_route_m=max(policy_config.required_target_point_distances_m, default=0.0),
            timestamp_threshold_s=timestamp_threshold_s,
            custom_anchor_filter_fns=custom_anchor_filter_fns,
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
        data_root="${oc.env:PY123D_GARAGE_DATA_ROOT}/nuplan/123D",
        cache_root=f"${{oc.env:PY123D_GARAGE_DATA_ROOT}}/nuplan/py123d_garage_cache/{cache_store}",
        served_ego_state_interval_us=100_000,
        served_camera_interval_us=100_000,
        served_lidar_interval_us=100_000,
    )


def nuplan_train(
    policy_config: AbstractPolicyConfig,
    *,
    cache_store: str,
    timestamp_threshold_s: float | None,
) -> OfflineTrainingDataSourceConfig:
    """
    The nuplan train logs on an evenly spaced anchor grid, behind the plausibility filters.

    Args:
        policy_config: the policy the source will serve.
        cache_store: name of the feature store under the dataset's cache dir.
        timestamp_threshold_s: minimum time between selected anchors; None = every eligible frame.

    Returns:
        the source config; caching and training share it.
    """
    return _nuplan(policy_config, "nuplan_train", cache_store, list(_PLAUSIBILITY_FILTER_FNS), timestamp_threshold_s)


def nuplan_val(
    policy_config: AbstractPolicyConfig,
    *,
    cache_store: str,
    timestamp_threshold_s: float | None,
) -> OfflineTrainingDataSourceConfig:
    """
    The sensor-bearing nuplan val logs on an evenly spaced anchor grid, behind the plausibility filters.

    Args:
        policy_config: the policy the source will serve.
        cache_store: name of the feature store under the dataset's cache dir.
        timestamp_threshold_s: minimum time between selected anchors; None = every eligible frame.

    Returns:
        the source config; caching and training share it.
    """
    return _nuplan(policy_config, "nuplan_val", cache_store, list(_PLAUSIBILITY_FILTER_FNS), timestamp_threshold_s)
