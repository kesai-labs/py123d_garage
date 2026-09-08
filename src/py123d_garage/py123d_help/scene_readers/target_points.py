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

from py123d_garage.datatypes.numerics import PositiveFloat


class InsufficientRouteError(ValueError):
    """The log's route cannot serve the requested target-point distances."""


def get_target_points(
    scene_api: SceneAPI,
    target_point_distances_m: list[PositiveFloat],
) -> jt.Float32[npt.NDArray[np.float32], "num_points 2"]:
    """
    Interpolates the log's route polyline at the requested arc-length distances.

    Scenes are expected to be pre-filtered with SceneFilter.min_remaining_route_m,
    so a shortfall here means the filter and the policy config disagree.

    Args:
        scene_api: scene interface anchored at the current frame (iteration 0)
        target_point_distances_m: arc-length sampling distances in meters

    Returns:
        target points in the current rear-axle frame

    Raises:
        InsufficientRouteError: if the log carries no route (reconvert or backfill
            with write_route), or less route remains than the largest distance.
    """
    initial_ego_state_se3: EgoStateSE3 | None = scene_api.get_ego_state_se3_at_iteration(0)
    assert initial_ego_state_se3 is not None, "Ego state should be available for target-point computation!"
    origin_pose_se2: PoseSE2 = initial_ego_state_se3.rear_axle_se2

    route = scene_api.get_route()
    progress_m = scene_api.get_route_progress_at_iteration(0)
    if route is None or progress_m is None:
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
    if remaining_m < float(distances.max()):
        raise InsufficientRouteError(
            f"Only {remaining_m:.1f} m of route remain after the anchor frame but "
            f"target points need {distances.max():.1f} m; filter scenes with "
            f"SceneFilter.min_remaining_route_m.",
        )

    absolute_points: npt.NDArray[np.float64] = interpolate_route_at_arc(
        polyline_arc_m,
        polyline_xyz,
        progress_m + distances,
    )[:, :2]
    pose_se2_array: npt.NDArray[np.float64] = np.concatenate(
        [absolute_points, np.zeros((len(distances), 1))],
        axis=1,
    )
    relative: npt.NDArray[np.float64] = abs_to_rel_se2_array(
        origin=origin_pose_se2,
        pose_se2_array=pose_se2_array,
    )
    return relative[:, :2].astype(np.float32)
