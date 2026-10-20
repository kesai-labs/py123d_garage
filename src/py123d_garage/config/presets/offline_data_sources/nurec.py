# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

from __future__ import annotations

from py123d_garage.api.abstract_offline_data_source_config import OfflineTrainingDataSourceConfig
from py123d_garage.api.abstract_policy_config import AbstractPolicyConfig
from py123d_garage.py123d_help.scene_filters import GarageSceneFilter


def _nurec(
    policy_config: AbstractPolicyConfig,
    split_name: str,
    cache_store: str,
    timestamp_threshold_s: float | None,
) -> OfflineTrainingDataSourceConfig:
    """
    Builds the nurec training source of one split for a policy.

    Args:
        policy_config: the policy the source will serve.
        split_name: name of the 123D split.
        cache_store: name of the feature store under the dataset's cache dir.
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
            required_scene_modalities=[
                "ego_state_se3",
                *[
                    f"lidar.{lidar.name.lower()}@{lidar_scope}"
                    for lidar in policy_config.required_lidars.get("nurec", [])
                ],
                *[f"camera.{camera.name.lower()}@{camera_scope}" for camera in policy_config.required_cameras["nurec"]],
            ],
        ),
        data_root="${oc.env:PY123D_GARAGE_DATA_ROOT}/nurec/123D",
        cache_root=f"${{oc.env:PY123D_GARAGE_DATA_ROOT}}/nurec/py123d_garage_cache/{cache_store}",
        served_ego_state_interval_us=100_000,
        served_camera_interval_us=100_000,
        served_lidar_interval_us=100_000,
    )


def nurec_train(
    policy_config: AbstractPolicyConfig,
    *,
    cache_store: str,
    timestamp_threshold_s: float | None,
) -> OfflineTrainingDataSourceConfig:
    """
    The nurec train logs, NuRec scenes joined with their physical-ai-av clips, anchored inside the 20 s sensor window.

    Args:
        policy_config: the policy the source will serve.
        cache_store: name of the feature store under the dataset's cache dir.
        timestamp_threshold_s: minimum time between selected anchors; None = every eligible frame.

    Returns:
        the source config; caching and training share it.
    """
    return _nurec(policy_config, "nurec_train", cache_store, timestamp_threshold_s)


def nurec_val(
    policy_config: AbstractPolicyConfig,
    *,
    cache_store: str,
    timestamp_threshold_s: float | None,
) -> OfflineTrainingDataSourceConfig:
    """
    The nurec val logs: AlpaSim's curated val suite, which the AlpaSim nurec evaluation drives.

    Args:
        policy_config: the policy the source will serve.
        cache_store: name of the feature store under the dataset's cache dir.
        timestamp_threshold_s: minimum time between selected anchors; None = every eligible frame.

    Returns:
        the source config; caching and training share it.
    """
    return _nurec(policy_config, "nurec_val", cache_store, timestamp_threshold_s)


def nurec_other(
    policy_config: AbstractPolicyConfig,
    *,
    cache_store: str,
    timestamp_threshold_s: float | None,
) -> OfflineTrainingDataSourceConfig:
    """
    The nurec logs outside AlpaSim's curated train and val suites.

    Args:
        policy_config: the policy the source will serve.
        cache_store: name of the feature store under the dataset's cache dir.
        timestamp_threshold_s: minimum time between selected anchors; None = every eligible frame.

    Returns:
        the source config; caching and training share it.
    """
    return _nurec(policy_config, "nurec_other", cache_store, timestamp_threshold_s)
