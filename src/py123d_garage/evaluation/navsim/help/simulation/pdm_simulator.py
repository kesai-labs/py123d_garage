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
    """Batch re-implementation of nuPlan's simulation: LQR tracking plus a kinematic bicycle model."""

    def __init__(self, proposal_grid: TrajectoryGrid):
        """
        Sets up the tracker and the motion model.

        Args:
            proposal_grid: Grid of the proposals to simulate.
        """
        self.proposal_grid = proposal_grid
        self._motion_model = BatchKinematicBicycleModel()
        self._tracker = BatchLQRTracker()

    def simulate_proposals(
        self,
        states: jt.Float64[npt.NDArray[np.float64], "proposals times 11"],
        initial_ego_state: EgoStateSE2,
    ) -> jt.Float64[npt.NDArray[np.float64], "proposals sim_times 11"]:
        """
        Simulates ego tracking each proposal from the current ego state.

        Args:
            states: Proposal states to track.
            initial_ego_state: Current ego state.

        Returns:
            The simulated ego states of each proposal over the proposal horizon.
        """
        proposal_states = states[:, : self.proposal_grid.num_poses + 1]
        discretization_time = self.proposal_grid.interval_s
        velocity_profile, curvature_profile = self._tracker.compute_reference_profiles(
            proposal_states=proposal_states,
            discretization_time=discretization_time,
        )

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
            command_states = self._tracker.track_trajectory(
                time_idx=time_idx - 1,
                initial_states=simulated_states[:, time_idx - 1],
                proposal_states=proposal_states,
                velocity_profile=velocity_profile,
                curvature_profile=curvature_profile,
                discretization_time=discretization_time,
                ego_wheel_base=initial_ego_state.metadata.wheel_base,
            )

            simulated_states[:, time_idx] = self._motion_model.propagate_state(
                states=simulated_states[:, time_idx - 1],
                command_states=command_states,
                sampling_time=sampling_time,
                ego_metadata=initial_ego_state.metadata,
            )

            simulated_timestamps[time_idx] = current_time_point.time_us
            current_time_point = Timestamp.from_us(
                current_time_point.time_us + delta_time_point,
            )

        return simulated_states
