# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI
# Adapted from DanielDauner/nav123d (Apache License 2.0).

from __future__ import annotations

import jaxtyping as jt
import numpy as np
import numpy.typing as npt
from py123d.datatypes import EgoStateSE2, Timestamp

from py123d_garage.evaluation.navsim.help.simulation.batch_kinematic_bicycle import (
    BatchKinematicBicycleModel,
)
from py123d_garage.evaluation.navsim.help.simulation.batch_lqr import (
    BatchLQRTracker,
)
from py123d_garage.evaluation.navsim.help.trajectory_grid import TrajectoryGrid
from py123d_garage.evaluation.navsim.help.utils.pdm_array_representation import (
    ego_state_to_state_array,
)


class PDMSimulator:
    """Re-implementation of nuPlan's simulation pipeline. Enables batch-wise simulation."""

    def __init__(self, proposal_grid: TrajectoryGrid):
        """
        Constructor of PDMSimulator.

        Args:
            proposal_grid: Sampling parameters for proposals
        """

        # time parameters
        self.proposal_grid = proposal_grid

        # simulation objects
        self._motion_model = BatchKinematicBicycleModel()
        self._tracker = BatchLQRTracker()

    def simulate_proposals(
        self,
        states: jt.Float64[npt.NDArray[np.float64], "proposals times 11"],
        initial_ego_state: EgoStateSE2,
    ) -> jt.Float64[npt.NDArray[np.float64], "proposals sim_times 11"]:
        """
        Simulate all proposals over batch-dim.

        Args:
            states: proposal states as array
            initial_ego_state: ego-vehicle state at current iteration

        Returns:
            simulated proposal states as array
        """

        proposal_states = states[:, : self.proposal_grid.num_poses + 1]
        discretization_time = self.proposal_grid.interval_s
        velocity_profile, curvature_profile = self._tracker.compute_reference_profiles(
            proposal_states=proposal_states,
            discretization_time=discretization_time,
        )

        # state array representation for simulated vehicle states
        simulated_states = np.zeros(proposal_states.shape, dtype=np.float64)
        simulated_states[:, 0] = ego_state_to_state_array(initial_ego_state)
        simulated_timestamps = np.zeros(
            (self.proposal_grid.num_poses + 1,),
            dtype=np.int64,
        )
        simulated_timestamps[0] = initial_ego_state.timestamp.time_us

        current_time_point = Timestamp.from_us(
            initial_ego_state.timestamp.time_us,
        )
        delta_time_point = self.proposal_grid.interval_us
        sampling_time: Timestamp = Timestamp.from_us(delta_time_point)

        for time_idx in range(1, self.proposal_grid.num_poses + 1):
            # 1. Track the trajectory with controller to get commands (steering rate and acceleration)
            command_states = self._tracker.track_trajectory(
                time_idx=time_idx - 1,
                initial_states=simulated_states[:, time_idx - 1],
                proposal_states=proposal_states,
                velocity_profile=velocity_profile,
                curvature_profile=curvature_profile,
                discretization_time=discretization_time,
                ego_wheel_base=initial_ego_state.metadata.wheel_base,
            )

            # 2. Propagate the state with the motion model and commands
            simulated_states[:, time_idx] = self._motion_model.propagate_state(
                states=simulated_states[:, time_idx - 1],
                command_states=command_states,
                sampling_time=sampling_time,  # convert from us to s
                ego_metadata=initial_ego_state.metadata,
            )

            simulated_timestamps[time_idx] = current_time_point.time_us
            current_time_point = Timestamp.from_us(
                current_time_point.time_us + delta_time_point,
            )

        return simulated_states
