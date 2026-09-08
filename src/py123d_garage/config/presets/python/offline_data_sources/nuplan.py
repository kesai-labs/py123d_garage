from __future__ import annotations

from py123d_garage.api.abstract_offline_data_source_config import OfflineTrainingDataSourceConfig
from py123d_garage.api.abstract_policy_config import AbstractPolicyConfig
from py123d_garage.py123d_help.scene_filters import GarageSceneFilter


def _nuplan(
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
            custom_anchor_filter_fns=[
                "py123d_garage.py123d_help.scene_filters:plausible_ego_motion",
                "py123d_garage.py123d_help.scene_filters:plausible_route",
            ],
            timestamp_threshold_s=timestamp_threshold_s,
            required_scene_modalities=[
                "ego_state_se3",
                *[
                    f"lidar.{lidar.name.lower()}@{lidar_scope}"
                    for lidar in policy_config.required_lidars.get("nuplan", [])
                ],
                *[f"camera.{camera.name.lower()}@initial" for camera in policy_config.required_cameras["nuplan"]],
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
    The nuplan train logs.

    Args:
        policy_config: the policy the source will serve.
        cache_store: name of the feature store under the dataset's cache dir.
        timestamp_threshold_s: minimum time between selected anchors; None = every eligible frame.

    Returns:
        the source config; caching and training share it.
    """
    return _nuplan(policy_config, "nuplan_train", cache_store, timestamp_threshold_s)


def nuplan_val(
    policy_config: AbstractPolicyConfig,
    *,
    cache_store: str,
    timestamp_threshold_s: float | None,
) -> OfflineTrainingDataSourceConfig:
    """
    The sensor-bearing nuplan val logs, filtered for what the policy consumes.

    Args:
        policy_config: the policy the source will serve.
        cache_store: name of the feature store under the dataset's cache dir.
        timestamp_threshold_s: minimum time between selected anchors; None = every eligible frame.

    Returns:
        the source config; caching and training share it.
    """
    return _nuplan(policy_config, "nuplan_val", cache_store, timestamp_threshold_s)


def nuplan_test(
    policy_config: AbstractPolicyConfig,
    *,
    cache_store: str,
    timestamp_threshold_s: float | None,
) -> OfflineTrainingDataSourceConfig:
    """
    The sensor-bearing nuplan test logs, filtered for what the policy consumes.

    The navtest evaluation samples live on this split; a policy trained on it
    contaminates its navsim scores.

    Args:
        policy_config: the policy the source will serve.
        cache_store: name of the feature store under the dataset's cache dir.
        timestamp_threshold_s: minimum time between selected anchors; None = every eligible frame.

    Returns:
        the source config; caching and training share it.
    """
    return _nuplan(policy_config, "nuplan_test", cache_store, timestamp_threshold_s)
