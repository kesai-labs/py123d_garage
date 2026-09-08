from __future__ import annotations

from py123d_garage.api.abstract_offline_data_source_config import OfflineTrainingDataSourceConfig
from py123d_garage.api.abstract_policy_config import AbstractPolicyConfig
from py123d_garage.py123d_help.scene_filters import GarageSceneFilter


def _physical_ai_av(
    policy_config: AbstractPolicyConfig,
    split_name: str,
    cache_store: str,
    timestamp_threshold_s: float | None,
) -> OfflineTrainingDataSourceConfig:
    lidar_scope = "history+initial" if policy_config.required_past_lidar_interval_us else "initial"
    return OfflineTrainingDataSourceConfig(
        garage_scene_filter=GarageSceneFilter(
            split_names=[split_name],
            future_duration_s=policy_config.trajectory_horizon_us / 1e6,
            history_duration_s=policy_config.required_history_duration_us / 1e6 or None,
            min_remaining_route_m=max(policy_config.required_target_point_distances_m),
            timestamp_threshold_s=timestamp_threshold_s,
            required_scene_modalities=[
                "ego_state_se3",
                *[
                    f"lidar.{lidar.name.lower()}@{lidar_scope}"
                    for lidar in policy_config.required_lidars.get("physical-ai-av", [])
                ],
                *[
                    f"camera.{camera.name.lower()}@initial"
                    for camera in policy_config.required_cameras["physical-ai-av"]
                ],
            ],
        ),
        data_root="${oc.env:PY123D_GARAGE_DATA_ROOT}/physical-ai-av/123D",
        cache_root=f"${{oc.env:PY123D_GARAGE_DATA_ROOT}}/physical-ai-av/py123d_garage_cache/{cache_store}",
        served_ego_state_interval_us=100_000,
        served_camera_interval_us=100_000,
        served_lidar_interval_us=100_000,
    )


def physical_ai_av_train(
    policy_config: AbstractPolicyConfig,
    *,
    cache_store: str,
    timestamp_threshold_s: float | None,
) -> OfflineTrainingDataSourceConfig:
    """
    The physical-ai-av train logs.

    Args:
        policy_config: the policy the source will serve.
        cache_store: name of the feature store under the dataset's cache dir.
        timestamp_threshold_s: minimum time between selected anchors; None = every eligible frame.

    Returns:
        the source config; caching and training share it.
    """
    return _physical_ai_av(
        policy_config,
        "physical-ai-av_train",
        cache_store,
        timestamp_threshold_s,
    )


def physical_ai_av_val(
    policy_config: AbstractPolicyConfig,
    *,
    cache_store: str,
    timestamp_threshold_s: float | None,
) -> OfflineTrainingDataSourceConfig:
    """
    The physical-ai-av val logs.

    Args:
        policy_config: the policy the source will serve.
        cache_store: name of the feature store under the dataset's cache dir.
        timestamp_threshold_s: minimum time between selected anchors; None = every eligible frame.

    Returns:
        the source config; caching and training share it.
    """
    return _physical_ai_av(
        policy_config,
        "physical-ai-av_val",
        cache_store,
        timestamp_threshold_s,
    )


def physical_ai_av_test(
    policy_config: AbstractPolicyConfig,
    *,
    cache_store: str,
    timestamp_threshold_s: float | None,
) -> OfflineTrainingDataSourceConfig:
    """
    The physical-ai-av test logs.

    Args:
        policy_config: the policy the source will serve.
        cache_store: name of the feature store under the dataset's cache dir.
        timestamp_threshold_s: minimum time between selected anchors; None = every eligible frame.

    Returns:
        the source config; caching and training share it.
    """
    return _physical_ai_av(
        policy_config,
        "physical-ai-av_test",
        cache_store,
        timestamp_threshold_s,
    )
