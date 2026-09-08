# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI
# Adapted from DanielDauner/nav123d (Apache License 2.0).

from __future__ import annotations

from enum import IntEnum

import jaxtyping as jt
import numpy as np
import numpy.typing as npt
from py123d.datatypes import BoxDetectionSE2
from py123d.geometry import PoseSE2
from shapely import LineString, Polygon

from py123d_garage.evaluation.navsim.help.utils.pdm_constants import (
    DYNAMIC_OBJECT_LABELS,
)
from py123d_garage.evaluation.navsim.help.utils.pdm_enums import StateIndex


class CollisionType(IntEnum):
    """Enum for the types of collisions of interest."""

    STOPPED_EGO_COLLISION = 0
    STOPPED_TRACK_COLLISION = 1
    ACTIVE_FRONT_COLLISION = 2
    ACTIVE_REAR_COLLISION = 3
    ACTIVE_LATERAL_COLLISION = 4


def get_collision_type(
    state: jt.Float64[npt.NDArray[np.float64], " 11"],
    ego_polygon: Polygon,
    box_detection_se2: BoxDetectionSE2,
    box_detection_polygon: Polygon,
    stopped_speed_threshold: float = 5e-02,
) -> CollisionType:
    """
    Classify the collision between ego and the track.

    Args:
        state: ego's state array at the collision timestamp
        ego_polygon: polygon of the ego vehicle
        box_detection_se2: box detection state
        box_detection_polygon: polygon representing the box detection
        stopped_speed_threshold: threshold for 0 speed due to noise

    Returns:
        collision type
    """

    ego_speed = np.hypot(
        state[StateIndex.VELOCITY_X],
        state[StateIndex.VELOCITY_Y],
    )

    is_ego_stopped = float(ego_speed) <= stopped_speed_threshold

    center_point = box_detection_polygon.centroid
    tracked_object_center = PoseSE2(
        center_point.x,
        center_point.y,
        box_detection_se2.center_se2.yaw,
    )

    ego_rear_axle_pose: PoseSE2 = PoseSE2.from_array(
        state[StateIndex.STATE_SE2],
    )

    # Collisions at (close-to) zero ego speed
    if is_ego_stopped:
        collision_type = CollisionType.STOPPED_EGO_COLLISION

    # Collisions at (close-to) zero track speed
    elif is_track_stopped(box_detection_se2):
        collision_type = CollisionType.STOPPED_TRACK_COLLISION

    # Rear collision when both ego and track are not stopped
    elif is_agent_behind(ego_rear_axle_pose, tracked_object_center):
        collision_type = CollisionType.ACTIVE_REAR_COLLISION

    # Front bumper collision when both ego and track are not stopped
    elif LineString(
        [
            ego_polygon.exterior.coords[0],
            ego_polygon.exterior.coords[3],
        ],
    ).intersects(box_detection_polygon):
        collision_type = CollisionType.ACTIVE_FRONT_COLLISION

    # Lateral collision when both ego and track are not stopped
    else:
        collision_type = CollisionType.ACTIVE_LATERAL_COLLISION

    return collision_type


def is_track_stopped(
    box_detection_se2: BoxDetectionSE2,
    stopped_speed_threshold: float = 5e-02,
) -> bool:
    """
    Evaluates if a tracked object is stopped.

    Args:
        box_detection_se2: Box detection state
        stopped_speed_threshold: Threshold for 0 speed due to noise

    Returns:
        True if track is stopped else False.
    """
    is_stopped: bool = True
    if box_detection_se2.attributes.default_label in DYNAMIC_OBJECT_LABELS:
        assert box_detection_se2.velocity_2d is not None, (
            "Velocity information is required for dynamic objects to determine if they are stopped."
        )
        is_stopped = bool(
            box_detection_se2.velocity_2d.magnitude <= stopped_speed_threshold,
        )
    return is_stopped


def is_agent_behind(
    ego_pose_se2: PoseSE2,
    agent_pose_se2: PoseSE2,
    angle_tolerance: float = 150,
) -> bool:
    """
    Determines if an agent is behind the ego.

    Args:
        ego_pose_se2: ego's pose
        agent_pose_se2: agent's pose
        angle_tolerance: tolerance to consider if agent is behind, where zero is the heading of the ego [deg]

    Returns:
        true if agent is behind, false otherwise.
    """
    return bool(
        get_agent_relative_angle(ego_pose_se2, agent_pose_se2) > np.deg2rad(angle_tolerance),
    )


def is_agent_ahead(
    ego_pose_se2: PoseSE2,
    agent_pose_se2: PoseSE2,
    angle_tolerance: float = 30,
) -> bool:
    """
    Determines if an agent is ahead of the ego.

    Args:
        ego_pose_se2: ego's pose
        agent_pose_se2: agent's pose
        angle_tolerance: tolerance to consider if agent is ahead, where zero is the heading of the ego [deg]

    Returns:
        true if agent is ahead, false otherwise.
    """
    return bool(
        get_agent_relative_angle(ego_pose_se2, agent_pose_se2) < np.deg2rad(angle_tolerance),
    )


def get_agent_relative_angle(
    ego_pose_se2: PoseSE2,
    agent_pose_se2: PoseSE2,
) -> float:
    """
    Get the relative angle of an agent position to the ego.

    Args:
        ego_pose_se2: pose of ego
        agent_pose_se2: pose of an agent

    Returns:
        relative angle in radians.
    """
    agent_vector: npt.NDArray[np.float64] = np.array(
        [
            agent_pose_se2.x - ego_pose_se2.x,
            agent_pose_se2.y - ego_pose_se2.y,
        ],
    )
    ego_vector: npt.NDArray[np.float64] = np.array(
        [np.cos(ego_pose_se2.yaw), np.sin(ego_pose_se2.yaw)],
    )
    dot_product = np.dot(
        ego_vector,
        agent_vector / np.linalg.norm(agent_vector),
    )
    return float(np.arccos(dot_product))
