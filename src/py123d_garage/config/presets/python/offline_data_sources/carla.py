from __future__ import annotations

from py123d_garage.api.abstract_offline_data_source_config import OfflineTrainingDataSourceConfig
from py123d_garage.api.abstract_policy_config import AbstractPolicyConfig
from py123d_garage.py123d_help.scene_filters import GarageSceneFilter


def carla(
    policy_config: AbstractPolicyConfig,
    *,
    cache_store: str,
) -> OfflineTrainingDataSourceConfig:
    """
    The CARLA logs.

    Args:
        policy_config: the policy the source will serve.
        cache_store: name of the feature store under the dataset's cache dir.

    Returns:
        the source config; caching and training share it.
    """
    lidar_scope = "history+initial" if policy_config.required_past_lidar_interval_us else "initial"
    return OfflineTrainingDataSourceConfig(
        garage_scene_filter=GarageSceneFilter(
            split_names=["normal_view"],
            future_duration_s=policy_config.trajectory_horizon_us / 1e6,
            history_duration_s=policy_config.required_history_duration_us / 1e6 or None,
            min_remaining_route_m=max(policy_config.required_target_point_distances_m),
            custom_anchor_filter_fns=["py123d_garage.py123d_help.scene_filters:has_box_detections"],
            required_scene_modalities=[
                "ego_state_se3",
                *[
                    f"lidar.{lidar.name.lower()}@{lidar_scope}"
                    for lidar in policy_config.required_lidars.get("carla", [])
                ],
                *[
                    f"{modality}.{camera.name.lower()}@initial"
                    for modality in ("camera", "camera_semantic", "camera_depth")
                    for camera in policy_config.required_cameras["carla"]
                ],
            ],
        ),
        data_root="${oc.env:PY123D_GARAGE_DATA_ROOT}/lead/123D",
        cache_root=f"${{oc.env:PY123D_GARAGE_DATA_ROOT}}/lead/py123d_garage_cache/{cache_store}",
        served_ego_state_interval_us=50_000,
        served_camera_interval_us=250_000,
        served_lidar_interval_us=50_000,
    )
