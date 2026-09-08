# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI
# Adapted from DanielDauner/nav123d (Apache License 2.0).

from __future__ import annotations

import numpy as np
from py123d.datatypes import EgoStateSE2
from py123d.geometry.transform import rel_to_abs_se2_array

from py123d_garage.datatypes.trajectory import (
    TrajectorySE2,
)
from py123d_garage.evaluation.navsim.help.trajectory_grid import TrajectoryGrid


def resample_trajectory_se2(
    trajectory: TrajectorySE2,
    trajectory_grid: TrajectoryGrid,
    initial_ego_state_se2: EgoStateSE2,
    convert_to_absolute: bool = False,
    add_initial_ego_pose: bool = True,
) -> TrajectorySE2:
    """
    Resample a trajectory to the given trajectory grid.

    Args:
        trajectory: input trajectory
        trajectory_grid: target sampling grid of the resampled trajectory
        initial_ego_state_se2: initial ego state as SE2
        convert_to_absolute: if True, convert the input from ego-relative to absolute poses first, defaults to False
        add_initial_ego_pose: if True, include the initial ego pose as the first sample, defaults to True

    Returns:
        resampled trajectory as SE2
    """

    if convert_to_absolute:
        _trajectory = TrajectorySE2(
            pose_se2_array=rel_to_abs_se2_array(
                origin=initial_ego_state_se2.rear_axle_se2,
                pose_se2_array=trajectory.pose_se2_array,
            ),
            timestamps_us=trajectory.timestamps_us,
        )
    else:
        _trajectory = trajectory

    # NOTE @DanielDauner: The PDM modules expect the trajectory to start at the current ego timestamp/iteration.
    # If the first timestamp of the trajectory is larger than the ego timestamp, we concat the initial ego pose/timestamp.
    ego_timestamp = initial_ego_state_se2.timestamp.time_us
    if int(_trajectory.timestamps_us[0]) > initial_ego_state_se2.timestamp.time_us:
        initial_ego_se2_array = initial_ego_state_se2.rear_axle_se2.array
        new_se2_array = np.concatenate(
            [initial_ego_se2_array[None, ...], _trajectory.pose_se2_array],
            axis=0,
        )
        new_timestamps = np.concatenate(
            [
                np.array([initial_ego_state_se2.timestamp.time_us]),
                _trajectory.timestamps_us,
            ],
            axis=0,
        )
        _trajectory = TrajectorySE2(
            pose_se2_array=new_se2_array,
            timestamps_us=new_timestamps,
        )

    offset = 0 if add_initial_ego_pose else 1
    sampling_timestamps = ego_timestamp + np.arange(
        offset,
        trajectory_grid.num_poses + 1,
        dtype=np.int64,
    ) * np.int64(trajectory_grid.interval_us)
    resampled_se2_array = _trajectory.interpolate(sampling_timestamps)

    return TrajectorySE2(
        pose_se2_array=resampled_se2_array,
        timestamps_us=sampling_timestamps,
    )
