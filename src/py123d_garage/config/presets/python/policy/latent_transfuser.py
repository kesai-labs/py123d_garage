from __future__ import annotations

from py123d_garage.config.schema.policy.transfuser_config import (
    BackboneConfig,
    BevSemanticConfig,
    BoxDetectionConfig,
    CameraConfig,
    LidarConfig,
    PerspectiveConfig,
    TransfuserConfig,
)
from py123d_garage.py123d_help.misc import FRONT_CAMERAS


def ltf_nuplan() -> TransfuserConfig:
    """LTF for nuPlan, navsim and Physical-AI-AV, alone or mixed."""
    return TransfuserConfig(
        trajectory_horizon_us=4_000_000,
        trajectory_interval_us=500_000,
        required_target_point_distances_m=[45.0],
        camera_config=CameraConfig(
            input_cameras=FRONT_CAMERAS,
            image_width=1024,
            image_height=256,
        ),
        lidar_config=LidarConfig(
            bev_min_x_m=0.0,
            bev_max_x_m=64.0,
            bev_min_y_m=-32.0,
            bev_max_y_m=32.0,
            lidar_horizon_us=0,
        ),
        perspective_config=PerspectiveConfig(
            use_semantic=False,
            use_depth=False,
        ),
        backbone_config=BackboneConfig(latent=True),
    )


def ltf_carla() -> TransfuserConfig:
    """LTF for CARLA Leaderboard."""
    return TransfuserConfig(
        required_target_point_distances_m=[45.0],
        trajectory_horizon_us=2_000_000,
        trajectory_interval_us=250_000,
        camera_config=CameraConfig(
            input_cameras=FRONT_CAMERAS,
            image_width=1152,
            image_height=384,
        ),
        lidar_config=LidarConfig(
            bev_min_x_m=0.0,
            bev_max_x_m=64.0,
            bev_min_y_m=-40.0,
            bev_max_y_m=40.0,
            lidar_horizon_us=0,
        ),
        backbone_config=BackboneConfig(latent=True),
        perspective_config=PerspectiveConfig(
            use_semantic=True,
            use_depth=True,
        ),
    )


def ltf_kesai() -> TransfuserConfig:
    """LTF for KE:SAI."""
    return TransfuserConfig(
        trajectory_horizon_us=4_000_000,
        trajectory_interval_us=500_000,
        required_target_point_distances_m=[45.0],
        camera_config=CameraConfig(
            input_cameras=FRONT_CAMERAS,
            image_width=1024,
            image_height=256,
        ),
        lidar_config=LidarConfig(
            bev_min_x_m=0.0,
            bev_max_x_m=64.0,
            bev_min_y_m=-32.0,
            bev_max_y_m=32.0,
            lidar_horizon_us=0,
        ),
        bev_semantic_config=BevSemanticConfig(use_bev_semantic=False),
        box_detection_config=BoxDetectionConfig(detect_boxes=False),
        perspective_config=PerspectiveConfig(
            use_semantic=False,
            use_depth=False,
        ),
        backbone_config=BackboneConfig(latent=True),
    )
