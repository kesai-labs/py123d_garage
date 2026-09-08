from __future__ import annotations

from py123d_garage.api.abstract_offline_data_source_config import OfflineTrainingDataSourceConfig
from py123d_garage.api.abstract_policy_config import AbstractPolicyConfig
from py123d_garage.py123d_help.scene_filters import GarageSceneFilter


def kesai_train(
    policy_config: AbstractPolicyConfig,
    *,
    cache_store: str,
    timestamp_threshold_s: float | None,
) -> OfflineTrainingDataSourceConfig:
    """
    The kesai train logs.

    Args:
        policy_config: the policy the source will serve.
        cache_store: name of the feature store under the dataset's cache dir.
        timestamp_threshold_s: minimum time between selected anchors; None = every eligible frame.

    Returns:
        the source config; caching and training share it.
    """
    lidar_scope = "history+initial" if policy_config.required_past_lidar_interval_us else "initial"
    return OfflineTrainingDataSourceConfig(
        garage_scene_filter=GarageSceneFilter(
            split_names=["kesai_train"],
            future_duration_s=policy_config.trajectory_horizon_us / 1e6,
            history_duration_s=policy_config.required_history_duration_us / 1e6 or None,
            min_remaining_route_m=max(policy_config.required_target_point_distances_m),
            timestamp_threshold_s=timestamp_threshold_s,
            required_scene_modalities=[
                "ego_state_se3",
                *[
                    f"lidar.{lidar.name.lower()}@{lidar_scope}"
                    for lidar in policy_config.required_lidars.get("kesai", [])
                ],
                *[f"camera.{camera.name.lower()}@initial" for camera in policy_config.required_cameras["kesai"]],
            ],
        ),
        data_root="${oc.env:PY123D_GARAGE_DATA_ROOT}/kesai/123D",
        cache_root=f"${{oc.env:PY123D_GARAGE_DATA_ROOT}}/kesai/py123d_garage_cache/{cache_store}",
        served_ego_state_interval_us=10_000,
        served_camera_interval_us=50_000,
        served_lidar_interval_us=100_000,
    )
