from __future__ import annotations

from py123d_garage.config.schema.policy.transfuser_config import (
    BevSemanticConfig,
    BoxDetectionConfig,
    CameraConfig,
    LidarConfig,
    PerspectiveConfig,
    TransfuserConfig,
)
from py123d_garage.py123d_help.misc import FRONT_CAMERAS


def tf_nuplan() -> TransfuserConfig:
    """The nuplan variant, also for the nuplan+physical-ai-av mixture: 4 s horizon sampled every 0.5 s, 1920x1080 cameras."""
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
            bev_min_x_m=-32.0,
            bev_max_x_m=32.0,
            bev_min_y_m=-32.0,
            bev_max_y_m=32.0,
            lidar_interval_us=100_000,
            lidar_horizon_us=400_000,
        ),
        box_detection_config=BoxDetectionConfig(predict_box_velocity=True),
        perspective_config=PerspectiveConfig(
            use_semantic=False,
            use_depth=False,
        ),
    )


def tf_carla() -> TransfuserConfig:
    """The CARLA variant: 2 s horizon sampled every 0.25 s, 384x384 cameras."""
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
            bev_min_x_m=-32.0,
            bev_max_x_m=64.0,
            bev_min_y_m=-40.0,
            bev_max_y_m=40.0,
            # The LEAD logs capture at 20 Hz.
            lidar_interval_us=50_000,
            lidar_horizon_us=200_000,
        ),
        box_detection_config=BoxDetectionConfig(predict_box_velocity=True),
        perspective_config=PerspectiveConfig(
            use_semantic=True,
            use_depth=True,
        ),
    )


def tf_kesai() -> TransfuserConfig:
    """The kesai variant: f-theta cameras; the logs ship no maps, boxes, or semantics."""
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
            bev_min_x_m=-32.0,
            bev_max_x_m=32.0,
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
    )
