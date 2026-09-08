# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI
# Adapted from DanielDauner/nav123d (Apache License 2.0).

from __future__ import annotations

import numpy as np
from py123d.datatypes import EgoStateSE2
from py123d.geometry import PoseSE2Index
from py123d.geometry.transform import rel_to_abs_se2_array

from py123d_garage.datatypes.trajectory import (
    TrajectorySE2,
)
from py123d_garage.evaluation.navsim.help.trajectory_grid import TrajectoryGrid


class PDMEmergencyBrake:
    """Class for emergency brake maneuver of PDM-Closed."""

    def __init__(
        self,
        trajectory_grid: TrajectoryGrid,
        max_long_accel: float = 2.40,
        min_long_accel: float = -4.05,
    ):
        """
        Constructor for PDMEmergencyBrake.

        Args:
            trajectory_grid: Sampling parameters for final trajectory
            max_long_accel: maximum longitudinal acceleration for braking, defaults to 2.40
            min_long_accel: min longitudinal acceleration for braking, defaults to -4.05
        """

        # trajectory parameters
        self._trajectory_grid = trajectory_grid

        # braking parameters
        self._max_long_accel: float = max_long_accel  # [m/s^2]
        self._min_long_accel: float = min_long_accel  # [m/s^2]

    def generate_stop_trajectory(
        self,
        ego_state_se2: EgoStateSE2,
    ) -> TrajectorySE2:
        """
        Generates a braking trajectory that decelerates ego to zero velocity.

        Args:
            ego_state_se2: state object of ego

        Returns:
            braking trajectory as SE2 in the absolute/global frame
        """
        current_time_point = ego_state_se2.timestamp
        assert ego_state_se2.dynamic_state_se2 is not None, (
            "PDMEmergencyBraking: EgoStateSE2 must have dynamic state for trajectory generation!"
        )
        current_velocity = ego_state_se2.dynamic_state_se2.velocity_2d.x
        current_acceleration = ego_state_se2.dynamic_state_se2.acceleration_2d.x

        target_velocity = 0.0

        if current_velocity > 0.2:
            k_p = 10.0
            k_d = 0.0

            error = -current_velocity
            dt_error = -current_acceleration
            u_t = k_p * error + k_d * dt_error

            error = max(min(u_t, self._max_long_accel), self._min_long_accel)
            correcting_velocity = 11 / 10 * (current_velocity + error)

        else:
            k_p = 4
            k_d = 1

            error = target_velocity - current_velocity
            dt_error = -current_acceleration

            u_t = k_p * error + k_d * dt_error

            correcting_velocity = max(
                min(u_t, self._max_long_accel),
                self._min_long_accel,
            )

        pose_se2_array = np.zeros(
            (self._trajectory_grid.num_poses + 1, len(PoseSE2Index)),
            dtype=np.float64,
        )
        timestamps = np.zeros(
            (self._trajectory_grid.num_poses + 1,),
            dtype=np.int64,
        )

        # Propagate planned trajectory for set number of samples
        for time_idx in range(self._trajectory_grid.num_poses + 1):
            time_t = self._trajectory_grid.interval_s * time_idx
            pose_se2_array[time_idx, PoseSE2Index.X] = correcting_velocity * time_t
            timestamps[time_idx] = current_time_point.time_us + int(
                time_t * 1e6,
            )

        # Transform to absolute coordinates
        pose_se2_array = rel_to_abs_se2_array(
            ego_state_se2.center_se2,
            pose_se2_array,
        )
        return TrajectorySE2(pose_se2_array, timestamps)
