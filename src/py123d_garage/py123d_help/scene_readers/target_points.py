# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""The route target points the policy is conditioned on, read off the log's route polyline."""

from __future__ import annotations

import jaxtyping as jt
import numpy as np
import numpy.typing as npt
from py123d.api import SceneAPI
from py123d.api.scene.arrow.utils.route_utils import interpolate_route_at_arc
from py123d.datatypes import EgoStateSE3
from py123d.geometry import PoseSE2
from py123d.geometry.transform import (
    abs_to_rel_se2_array,
)

from py123d_garage.datatypes.numerics import NonNegativeFloat, PositiveFloat


class InsufficientRouteError(ValueError):
    """The log's route cannot serve the requested target-point distances."""


def get_target_points(
    scene_api: SceneAPI,
    target_point_distances_m: list[PositiveFloat],
) -> jt.Float32[npt.NDArray[np.float32], "num_points 2"]:
    """
    Interpolates the log's route polyline at the requested arc-length distances.

    Args:
        scene_api: scene interface anchored at the current frame (iteration 0)
        target_point_distances_m: arc-length sampling distances in meters

    Returns:
        target points in the current rear-axle frame
    """
    return get_target_point_poses(scene_api, target_point_distances_m)[:, :2]


def get_target_point_poses(
    scene_api: SceneAPI,
    target_point_distances_m: list[NonNegativeFloat],
    nan_past_route_end: bool = False,
) -> jt.Float32[npt.NDArray[np.float32], "num_points 3"]:
    """
    Interpolates the log's route polyline at the requested arc-length distances, with the route heading there.

    Args:
        scene_api: scene interface anchored at the current frame (iteration 0)
        target_point_distances_m: arc-length sampling distances in meters
        nan_past_route_end: return NaN instead of raising for points the route does not reach, or for all
            points without a route; the policy reads NaN as no navigation intent

    Returns:
        target points (x, y, heading of the route at the point) in the current rear-axle frame

    Raises:
        InsufficientRouteError: unless nan_past_route_end, if the log carries no route or too little remains.
    """
    if not target_point_distances_m:
        return np.zeros((0, 3), dtype=np.float32)
    initial_ego_state_se3: EgoStateSE3 | None = scene_api.get_ego_state_se3_at_iteration(0)
    assert initial_ego_state_se3 is not None, "Ego state should be available for target-point computation!"
    origin_pose_se2: PoseSE2 = initial_ego_state_se3.rear_axle_se2

    route = scene_api.get_route()
    progress_m = scene_api.get_route_progress_at_iteration(0)
    if route is None or progress_m is None:
        if nan_past_route_end:
            return np.full((len(target_point_distances_m), 3), np.nan, dtype=np.float32)
        raise InsufficientRouteError(
            "The log carries no route polyline or no route progress at the anchor "
            "frame; reconvert it with write_route enabled or backfill its sync table.",
        )
    route_metadata, polyline_arc_m, polyline_xyz = route

    distances: npt.NDArray[np.float64] = np.asarray(
        target_point_distances_m,
        dtype=np.float64,
    )
    remaining_m = route_metadata.total_arc_m - progress_m
    reached = distances <= remaining_m
    if not reached.all() and not nan_past_route_end:
        raise InsufficientRouteError(
            f"Only {remaining_m:.1f} m of route remain after the anchor frame but "
            f"target points need {distances.max():.1f} m; filter scenes with "
            f"SceneFilter.min_remaining_route_m.",
        )

    poses = np.full((len(distances), 3), np.nan, dtype=np.float32)
    if not reached.any():
        return poses
    target_point_arc_m = progress_m + distances[reached]
    absolute_points: npt.NDArray[np.float64] = interpolate_route_at_arc(
        polyline_arc_m,
        polyline_xyz,
        target_point_arc_m,
    )[:, :2]
    # The filter guarantees route only up to the farthest point, so the heading looks back, except at the route start.
    heading_from_arc_m = np.maximum(target_point_arc_m - 1.0, progress_m)
    heading_to_arc_m = np.maximum(target_point_arc_m, heading_from_arc_m + 1.0)
    route_directions = (
        interpolate_route_at_arc(polyline_arc_m, polyline_xyz, heading_to_arc_m)[:, :2]
        - interpolate_route_at_arc(polyline_arc_m, polyline_xyz, heading_from_arc_m)[:, :2]
    )
    pose_se2_array: npt.NDArray[np.float64] = np.concatenate(
        [absolute_points, np.arctan2(route_directions[:, 1], route_directions[:, 0])[:, None]],
        axis=1,
    )
    relative: npt.NDArray[np.float64] = abs_to_rel_se2_array(
        origin=origin_pose_se2,
        pose_se2_array=pose_se2_array,
    )
    poses[reached] = relative
    return poses
