# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

from __future__ import annotations

from py123d_garage.api.abstract_policy_config import uniform_offsets_us
from py123d_garage.config.schema.policy.transfuser_config import (
    BackboneConfig,
    BevSemanticConfig,
    BoxDetectionConfig,
    CameraConfig,
    FlowMatchingConfig,
    LidarConfig,
    PathConfig,
    PerspectiveConfig,
    PlanningConfig,
    TransfuserConfig,
)
from py123d_garage.py123d_help.misc import FRONT_CENTER_CAMERA


def latent_transfuser() -> TransfuserConfig:
    """LTF for the nuPlan, Physical-AI-AV and NuRec mix, with a past camera frame and a 6.4 s plan."""
    return TransfuserConfig(
        # A plan every 0.1 s for 6.4 s.
        trajectory_offsets_us=uniform_offsets_us(100_000, 6_400_000),
        required_target_point_distances_m=[45.0],
        # The front center camera, with its frame 1 s back stacked on top.
        camera_config=CameraConfig(
            input_cameras=FRONT_CENTER_CAMERA,
            image_width=352,
            image_height=512,
            past_camera_offset_us=1_000_000,
        ),
        lidar_config=LidarConfig(
            bev_min_x_m=0.0,
            bev_max_x_m=64.0,
            bev_min_y_m=-32.0,
            bev_max_y_m=32.0,
            lidar_horizon_us=0,
        ),
        backbone_config=BackboneConfig(latent=True),
        perspective_config=PerspectiveConfig(use_semantic=False, use_depth=False),
        bev_semantic_config=BevSemanticConfig(map_z_gate_m=3.0),
        box_detection_config=BoxDetectionConfig(predict_box_velocity=True),
        # The decoder of the released checkpoints: the waypoints do not read the actions, the path reads neither.
        planning_config=PlanningConfig(
            flow_config=FlowMatchingConfig(waypoints_attend_to_actions=False),
            path_config=PathConfig(path_attends_to_waypoints=False, path_attends_to_actions=False),
        ),
    )
