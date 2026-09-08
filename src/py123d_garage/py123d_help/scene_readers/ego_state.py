"""Ego poses, interpolated and sampled around an origin timestamp."""

from __future__ import annotations

import jaxtyping as jt
import numpy as np
import numpy.typing as npt
from py123d.api import SceneAPI
from py123d.datatypes import EgoStateSE3
from py123d.geometry.transform import (
    abs_to_rel_se2_array,
)

from py123d_garage.datatypes.numerics import NonNegativeInt, PositiveInt
from py123d_garage.datatypes.trajectory import (
    TrajectorySE2,
    TrajectoryXY,
)


def interpolate_ego_se2_at(
    scene_api: SceneAPI,
    timestamps_us: jt.Int64[npt.NDArray[np.int64], " num_poses"],
    interpolation_margin_us: NonNegativeInt = 500_000,
    boundary_tolerance_us: NonNegativeInt = 0,
) -> jt.Float64[npt.NDArray[np.float64], "num_poses 3"]:
    """
    Ego rear-axle SE2 poses interpolated at the given timestamps from the recorded states.

    Args:
        scene_api: scene interface providing the recorded ego states.
        timestamps_us: absolute timestamps to interpolate at, in any order.
        interpolation_margin_us: extra window of recorded states loaded on both sides
            so the boundary timestamps can be interpolated, defaults to 500_000.
        boundary_tolerance_us: sensor streams start and end asynchronously, so a sensor
            timestamp can fall this far outside the recorded ego range and still clamp
            to the boundary pose instead of raising; defaults to 0 (strict).

    Returns:
        interpolated (x, y, yaw) poses, one per timestamp.

    Raises:
        ValueError: if fewer than two ego states cover the requested range, or a
            timestamp falls outside it by more than the boundary tolerance.
    """
    window_start_us = int(timestamps_us.min()) - interpolation_margin_us
    window_end_us = int(timestamps_us.max()) + interpolation_margin_us
    poses_se2: list[npt.NDArray[np.float64]] = []
    ego_timestamps_us: list[int] = []
    for ego_state_se3 in scene_api.get_modality_between_timestamps(  # pyright: ignore[reportUnknownMemberType]
        start_timestamp=window_start_us,
        end_timestamp=window_end_us,
        modality_type="ego_state_se3",
        inclusive="both",
    ):
        assert isinstance(ego_state_se3, EgoStateSE3), f"Expected EgoStateSE3, got {type(ego_state_se3)}"
        poses_se2.append(ego_state_se3.rear_axle_se2.array)
        ego_timestamps_us.append(ego_state_se3.timestamp.time_us)
    if len(poses_se2) < 2:
        raise ValueError(
            f"scene {scene_api.scene_uuid} of log {scene_api.log_name} records "
            f"{len(poses_se2)} ego states in [{window_start_us}, {window_end_us}] µs; "
            "interpolation needs at least two.",
        )
    trajectory = TrajectorySE2(
        pose_se2_array=np.array(poses_se2, dtype=np.float64),
        timestamps_us=np.array(ego_timestamps_us, dtype=np.int64),
    )
    return trajectory.interpolate(timestamps_us, boundary_tolerance_us=boundary_tolerance_us)


def sample_ego_se2(
    scene_api: SceneAPI,
    num_steps: PositiveInt,
    interval_us: PositiveInt,
    relative_to_anchor: bool = False,
    interpolation_margin_us: NonNegativeInt = 500_000,
) -> TrajectorySE2:
    """
    Loads the logged ego poses and resamples them onto the given grid.

    Args:
        scene_api: API providing access to the logged ego states.
        num_steps: how many poses the returned trajectory holds.
        interval_us: spacing between consecutive poses.
        relative_to_anchor: if True, return poses relative to the anchor pose, defaults to False.
        interpolation_margin_us: extra horizon of logged states to load so the final
            pose can be interpolated, defaults to 500_000.

    Returns:
        future ego trajectory resampled to the target grid, as SE2.
    """

    initial_ego_state_se3 = scene_api.get_ego_state_se3_at_iteration(0)
    assert initial_ego_state_se3 is not None, "Ego state should be available for label computation!"

    poses_se2: list[npt.NDArray[np.float64]] = []
    timestamps: list[int] = []
    for ego_state_se3 in scene_api.get_modality_between_timestamps(  # pyright: ignore[reportUnknownMemberType]
        start_timestamp=initial_ego_state_se3.timestamp.time_us,
        end_timestamp=initial_ego_state_se3.timestamp.time_us + num_steps * interval_us + interpolation_margin_us,
        modality_type="ego_state_se3",
        inclusive="both",
    ):
        assert isinstance(ego_state_se3, EgoStateSE3), f"Expected EgoStateSE3, got {type(ego_state_se3)}"
        poses_se2.append(ego_state_se3.rear_axle_se2.array)
        timestamps.append(ego_state_se3.timestamp.time_us)

    full_trajectory = TrajectorySE2(
        pose_se2_array=np.array(poses_se2, dtype=np.float64),
        timestamps_us=np.array(timestamps, dtype=np.int64),
    )
    initial_timestamp_us = initial_ego_state_se3.timestamp.time_us
    sampling_timestamps_us = initial_timestamp_us + np.arange(
        1,
        num_steps + 1,
        dtype=np.int64,
    ) * np.int64(interval_us)

    # The scene's future guarantee holds on the sync timeline; ego sampling jitter can
    # leave the last recorded state up to one period short of the last grid point.
    ego_period_us = int(np.median(np.diff(full_trajectory.timestamps_us))) if len(timestamps) > 1 else 0
    resampled_se2_array = full_trajectory.interpolate(
        sampling_timestamps_us,
        boundary_tolerance_us=ego_period_us,
    )

    if relative_to_anchor:
        resampled_se2_array = abs_to_rel_se2_array(
            origin=initial_ego_state_se3.rear_axle_se2,
            pose_se2_array=resampled_se2_array,
        )

    return TrajectorySE2(
        pose_se2_array=resampled_se2_array,
        timestamps_us=sampling_timestamps_us,
    )


def sample_ego_xy(
    scene_api: SceneAPI,
    num_steps: PositiveInt,
    interval_us: PositiveInt,
    relative_to_anchor: bool = False,
    interpolation_margin_us: NonNegativeInt = 500_000,
) -> TrajectoryXY:
    """
    Loads the logged ego positions and resamples them onto the given grid.

    Args:
        scene_api: API providing access to the logged ego states.
        num_steps: how many poses the returned trajectory holds.
        interval_us: spacing between consecutive poses.
        relative_to_anchor: if True, return positions relative to the anchor pose, defaults to False.
        interpolation_margin_us: extra horizon of logged states to load so the final
            position can be interpolated, defaults to 500_000.

    Returns:
        future ego trajectory resampled to the target grid, as xy.
    """
    trajectory = sample_ego_se2(
        scene_api=scene_api,
        num_steps=num_steps,
        interval_us=interval_us,
        relative_to_anchor=relative_to_anchor,
        interpolation_margin_us=interpolation_margin_us,
    )
    return TrajectoryXY(
        position_xy_array=trajectory.pose_se2_array[:, :2],
        timestamps_us=trajectory.timestamps_us,
    )
