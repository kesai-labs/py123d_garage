# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI
# Adapted from DanielDauner/nav123d (Apache License 2.0).

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum

import jaxtyping as jt
import numpy as np
import numpy.typing as npt
from py123d.geometry.utils.rotation_utils import normalize_angle

from py123d_garage.evaluation.navsim.help.simulation.batch_lqr_utils import (
    _generate_profile_from_initial_condition_and_derivatives,
    get_velocity_curvature_profiles_with_derivatives_from_poses,
)
from py123d_garage.evaluation.navsim.help.utils.pdm_enums import (
    DynamicStateIndex,
    StateIndex,
)


class LateralStateIndex(IntEnum):
    """Index mapping for the lateral dynamics state vector."""

    LATERAL_ERROR = 0  # [m] The lateral error with respect to the planner centerline at the vehicle's rear axle center.
    HEADING_ERROR = 1  # [rad] The heading error "".
    STEERING_ANGLE = 2  # [rad] The wheel angle relative to the longitudinal axis of the vehicle.


class BatchLQRTracker:
    """
    Implements an LQR tracker for a kinematic bicycle model.

    Tracker operates on a batch of proposals. Implementation directly based on the nuplan-devkit
    Link: https://github.com/motional/nuplan-devkit

    We decouple into two subsystems, longitudinal and lateral, with small angle approximations for linearization.
    We then solve two sequential LQR subproblems to find acceleration and steering rate inputs.

    Longitudinal Subsystem:
        States: [velocity]
        Inputs: [acceleration]
        Dynamics (continuous time):
            velocity_dot = acceleration

    Lateral Subsystem (After Linearization/Small Angle Approximation):
        States: [lateral_error, heading_error, steering_angle]
        Inputs: [steering_rate]
        Parameters: [velocity, curvature]
        Dynamics (continuous time):
            lateral_error_dot  = velocity * heading_error
            heading_error_dot  = velocity * (steering_angle / wheelbase_length - curvature)
            steering_angle_dot = steering_rate

    The continuous time dynamics are discretized using Euler integration and zero-order-hold on the input.
    In case of a stopping reference, we use a simplified stopping P controller instead of LQR.

    The final control inputs passed on to the motion model are:
        - acceleration
        - steering_rate

    The tracker holds only static LQR/control configuration. Per-simulation context
    (proposal trajectory, discretization time, vehicle geometry) and per-step inputs
    (current iteration, initial states) are passed as method arguments.
    """

    def __init__(
        self,
        q_longitudinal: float = 10.0,
        r_longitudinal: float = 1.0,
        q_lateral: Sequence[float] = (1.0, 10.0, 0.0),
        r_lateral: Sequence[float] = (1.0,),
        tracking_horizon: int = 10,
        jerk_penalty: float = 1e-4,
        curvature_rate_penalty: float = 1e-2,
        stopping_proportional_gain: float = 0.5,
        stopping_velocity: float = 0.2,
    ):
        """
        Constructor for LQR controller.

        Args:
            q_longitudinal: The weights for the Q matrix for the longitudinal subsystem.
            r_longitudinal: The weights for the R matrix for the longitudinal subsystem.
            q_lateral: The weights for the Q matrix for the lateral subsystem.
            r_lateral: The weights for the R matrix for the lateral subsystem.
            tracking_horizon: How many discrete time steps ahead to consider for the LQR objective.
            jerk_penalty: Penalty on jerk used when fitting the velocity profile from poses.
            curvature_rate_penalty: Penalty on curvature rate used when fitting the curvature profile from poses.
            stopping_proportional_gain: The proportional_gain term for the P controller when coming to a stop.
            stopping_velocity: [m/s] The velocity below which we are deemed to be stopping and we don't use LQR.
        """
        # Longitudinal LQR Parameters
        self._q_longitudinal: float = q_longitudinal
        self._r_longitudinal: float = r_longitudinal

        # Lateral LQR Parameters
        assert len(q_lateral) == 3, "q_lateral should have 3 elements (lateral_error, heading_error, steering_angle)."
        assert len(r_lateral) == 1, "r_lateral should have 1 element (steering_rate)."
        self._q_lateral: npt.NDArray[np.float64] = np.diag(q_lateral)
        self._r_lateral: npt.NDArray[np.float64] = np.diag(r_lateral)

        # Note we want a horizon > 1 so that steering rate actually can impact lateral/heading error in discrete time.
        assert tracking_horizon > 1, (
            "We expect the horizon to be greater than 1 - else steering_rate has no impact with Euler integration."
        )
        self._tracking_horizon = tracking_horizon

        # Velocity/Curvature Estimation Parameters
        assert jerk_penalty > 0.0, "The jerk penalty must be positive."
        assert curvature_rate_penalty > 0.0, "The curvature rate penalty must be positive."
        self._jerk_penalty = jerk_penalty
        self._curvature_rate_penalty = curvature_rate_penalty

        # Stopping Controller Parameters
        assert stopping_proportional_gain > 0, "stopping_proportional_gain has to be greater than 0."
        assert stopping_velocity > 0, "stopping_velocity has to be greater than 0."
        self._stopping_proportional_gain = stopping_proportional_gain
        self._stopping_velocity = stopping_velocity

    def compute_reference_profiles(
        self,
        proposal_states: jt.Float64[npt.NDArray[np.float64], "batch times 11"],
        discretization_time: float,
    ) -> tuple[
        jt.Float64[npt.NDArray[np.float64], "batch vels"],
        jt.Float64[npt.NDArray[np.float64], "batch vels"],
    ]:
        """
        Fit reference velocity and curvature profiles from a batch of proposal trajectories.

        Intended to be called once per simulation; the returned arrays are then passed into
        track_trajectory at each iteration.

        Args:
            proposal_states: array representation of proposals.
            discretization_time: [s] The time interval used for discretizing the continuous time dynamics.

        Returns:
            Tuple of (velocity_profile, curvature_profile).
        """
        assert discretization_time > 0.0, "The discretization_time should be positive."

        poses = proposal_states[..., StateIndex.STATE_SE2]
        (
            velocity_profile,
            _acceleration_profile,
            curvature_profile,
            _curvature_rate_profile,
        ) = get_velocity_curvature_profiles_with_derivatives_from_poses(
            discretization_time=discretization_time,
            poses=poses,
            jerk_penalty=self._jerk_penalty,
            curvature_rate_penalty=self._curvature_rate_penalty,
        )
        return velocity_profile, curvature_profile

    def track_trajectory(
        self,
        time_idx: int,
        initial_states: jt.Float64[npt.NDArray[np.float64], "batch 11"],
        proposal_states: jt.Float64[npt.NDArray[np.float64], "batch times 11"],
        velocity_profile: jt.Float64[npt.NDArray[np.float64], "batch vels"],
        curvature_profile: jt.Float64[npt.NDArray[np.float64], "batch vels"],
        discretization_time: float,
        ego_wheel_base: float,
    ) -> jt.Float64[npt.NDArray[np.float64], "batch 2"]:
        """
        Calculates the command values given the proposals to track.

        Args:
            time_idx: current time index.
            initial_states: array representation of current ego states.
            proposal_states: array representation of proposals.
            velocity_profile: precomputed reference velocity profile (see compute_reference_profiles).
            curvature_profile: precomputed reference curvature profile (see compute_reference_profiles).
            discretization_time: [s] The time interval used for discretizing the continuous time dynamics.
            ego_wheel_base: The wheel base of the ego vehicle.

        Returns:
            command values for motion model.
        """
        assert discretization_time > 0.0, "The discretization_time should be positive."

        batch_size = len(initial_states)
        (
            initial_velocity,
            initial_lateral_state_vector,
        ) = self._compute_initial_velocity_and_lateral_state(
            time_idx,
            initial_states,
            proposal_states,
        )  # (batch), (batch, 3)

        (
            reference_velocities,
            curvature_profiles,
        ) = self._compute_reference_velocity_and_curvature_slice(
            time_idx,
            velocity_profile,
            curvature_profile,
        )  # (batch), (batch, 10)

        # create output arrays
        accel_cmds = np.zeros(batch_size, dtype=np.float64)
        steering_rate_cmds = np.zeros(batch_size, dtype=np.float64)

        # 1. Stopping Controller
        should_stop_mask = np.logical_and(
            reference_velocities <= self._stopping_velocity,
            initial_velocity <= self._stopping_velocity,
        )
        stopping_accel_cmd, stopping_steering_rate_cmd = self._stopping_controller(
            initial_velocity[should_stop_mask],
            reference_velocities[should_stop_mask],
        )
        accel_cmds[should_stop_mask] = stopping_accel_cmd
        steering_rate_cmds[should_stop_mask] = stopping_steering_rate_cmd

        # 2. Regular Controller
        accel_cmds[~should_stop_mask] = self._longitudinal_lqr_controller(
            initial_velocity[~should_stop_mask],
            reference_velocities[~should_stop_mask],
            discretization_time,
        )

        velocity_profiles = _generate_profile_from_initial_condition_and_derivatives(
            initial_condition=initial_velocity[~should_stop_mask],
            derivatives=np.repeat(
                accel_cmds[~should_stop_mask, None],
                self._tracking_horizon,
                axis=-1,
            ),
            discretization_time=discretization_time,
        )[:, : self._tracking_horizon]

        steering_rate_cmds[~should_stop_mask] = self._lateral_lqr_controller(
            initial_lateral_state_vector[~should_stop_mask],
            velocity_profiles,
            curvature_profiles[~should_stop_mask],
            discretization_time,
            ego_wheel_base,
        )

        command_states = np.zeros(
            (batch_size, len(DynamicStateIndex)),
            dtype=np.float64,
        )
        command_states[:, DynamicStateIndex.ACCELERATION_X] = accel_cmds
        command_states[:, DynamicStateIndex.STEERING_RATE] = steering_rate_cmds

        return command_states

    def _compute_initial_velocity_and_lateral_state(
        self,
        time_idx: int,
        initial_values: jt.Float64[npt.NDArray[np.float64], "batch 11"],
        proposal_states: jt.Float64[npt.NDArray[np.float64], "batch times 11"],
    ) -> tuple[
        jt.Float64[npt.NDArray[np.float64], " batch"],
        jt.Float64[npt.NDArray[np.float64], "batch 3"],
    ]:
        """
        Projects the initial tracking error into the vehicle/Frenet frame and extracts initial velocity.

        Args:
            time_idx: Current time index.
            initial_values: The current state for ego.
            proposal_states: The reference trajectory we are tracking.

        Returns:
            Initial velocity [m/s] and initial lateral state.
        """
        # Get initial trajectory state.
        initial_trajectory_values = proposal_states[:, time_idx]

        # Determine initial error state.
        x_errors = initial_values[:, StateIndex.X] - initial_trajectory_values[:, StateIndex.X]
        y_errors = initial_values[:, StateIndex.Y] - initial_trajectory_values[:, StateIndex.Y]
        heading_references = initial_trajectory_values[:, StateIndex.HEADING]

        lateral_errors = -x_errors * np.sin(
            heading_references,
        ) + y_errors * np.cos(heading_references)
        heading_errors = normalize_angle(
            initial_values[:, StateIndex.HEADING] - heading_references,
        )

        # Return initial velocity and lateral state vector.
        initial_velocities = initial_values[:, StateIndex.VELOCITY_X]

        initial_lateral_state_vector = np.stack(
            [
                lateral_errors,
                heading_errors,
                initial_values[:, StateIndex.STEERING_ANGLE],
            ],
            axis=-1,
        )

        return initial_velocities, initial_lateral_state_vector

    def _compute_reference_velocity_and_curvature_slice(
        self,
        time_idx: int,
        velocity_profile: jt.Float64[npt.NDArray[np.float64], "batch vels"],
        curvature_profile: jt.Float64[npt.NDArray[np.float64], "batch vels"],
    ) -> tuple[
        jt.Float64[npt.NDArray[np.float64], " batch"],
        jt.Float64[npt.NDArray[np.float64], "batch horizon"],
    ]:
        """
        Slice precomputed reference profiles to obtain the lookahead window for the current iteration.

        Uses a lookahead time equal to ``self._tracking_horizon * discretization_time``.

        Args:
            time_idx: Current time index.
            velocity_profile: precomputed reference velocity profile.
            curvature_profile: precomputed reference curvature profile.

        Returns:
            The reference velocity [m/s] and curvature profile [rad] to track.
        """
        batch_size, num_poses = velocity_profile.shape
        reference_idx = min(time_idx + self._tracking_horizon, num_poses - 1)
        reference_velocities = velocity_profile[:, reference_idx]

        reference_curvature_profiles = np.zeros(
            (batch_size, self._tracking_horizon),
            dtype=np.float64,
        )

        reference_length = reference_idx - time_idx
        reference_curvature_profiles[:, 0:reference_length] = curvature_profile[
            :,
            time_idx:reference_idx,
        ]

        if reference_length < self._tracking_horizon:
            reference_curvature_profiles[:, reference_length:] = curvature_profile[:, reference_idx, None]

        return reference_velocities, reference_curvature_profiles

    def _stopping_controller(
        self,
        initial_velocities: jt.Float64[npt.NDArray[np.float64], " batch"],
        reference_velocities: jt.Float64[npt.NDArray[np.float64], " batch"],
    ) -> tuple[jt.Float64[npt.NDArray[np.float64], " batch"], float]:
        """
        Apply proportional controller when at near-stop conditions.

        Args:
            initial_velocities: [m/s] The current velocity of ego.
            reference_velocities: [m/s] The reference velocity to track.

        Returns:
            Acceleration [m/s^2] and zero steering_rate [rad/s] command.
        """
        accel = -self._stopping_proportional_gain * (initial_velocities - reference_velocities)
        return accel, 0.0

    def _longitudinal_lqr_controller(
        self,
        initial_velocities: jt.Float64[npt.NDArray[np.float64], " batch"],
        reference_velocities: jt.Float64[npt.NDArray[np.float64], " batch"],
        discretization_time: float,
    ) -> jt.Float64[npt.NDArray[np.float64], " batch"]:
        """
        Determines an acceleration input to minimize velocity error at a lookahead time.

        Args:
            initial_velocities: [m/s] The current velocity of ego.
            reference_velocities: [m/s] The reference velocity to track at a lookahead time.
            discretization_time: [s] The time interval used for discretizing the continuous time dynamics.

        Returns:
            Acceleration [m/s^2] command based on LQR.
        """
        # We assume that we hold the acceleration constant for the entire tracking horizon.
        # Given this, we can show the following where N = self._tracking_horizon and dt = discretization_time:
        # velocity_N equals velocity_0 + (N * dt) * acceleration

        batch_size = len(initial_velocities)

        A: npt.NDArray[np.float64] = np.ones(batch_size, dtype=np.float64)

        B: npt.NDArray[np.float64] = np.zeros(batch_size, dtype=np.float64)
        B.fill(self._tracking_horizon * discretization_time)

        g: npt.NDArray[np.float64] = np.zeros(batch_size, dtype=np.float64)

        return self._solve_one_step_longitudinal_lqr(
            initial_state=initial_velocities,
            reference_state=reference_velocities,
            A=A,
            B=B,
            g=g,
        )

    def _lateral_lqr_controller(
        self,
        initial_lateral_state_vector: jt.Float64[
            npt.NDArray[np.float64],
            "batch 3",
        ],
        velocity_profile: jt.Float64[npt.NDArray[np.float64], "batch horizon"],
        curvature_profile: jt.Float64[npt.NDArray[np.float64], "batch horizon"],
        discretization_time: float,
        ego_wheel_base: float,
    ) -> jt.Float64[npt.NDArray[np.float64], " batch"]:
        """
        Determines a steering_rate input to minimize lateral errors at a lookahead time.

        It requires a velocity sequence as a parameter to ensure linear time-varying lateral dynamics.

        Args:
            initial_lateral_state_vector: The current lateral state of ego.
            velocity_profile: [m/s] The velocity over the entire self._tracking_horizon-step lookahead.
            curvature_profile: [rad] The curvature over the entire self._tracking_horizon-step lookahead.
            discretization_time: [s] The time interval used for discretizing the continuous time dynamics.
            ego_wheel_base: The wheel base of the ego vehicle.

        Returns:
            Steering rate [rad/s] command based on LQR.
        """
        assert velocity_profile.shape[-1] == self._tracking_horizon, (
            f"The linearization velocity sequence should have length {self._tracking_horizon} "
            f"but is {len(velocity_profile)}."
        )
        assert curvature_profile.shape[-1] == self._tracking_horizon, (
            f"The linearization curvature sequence should have length {self._tracking_horizon} "
            f"but is {len(curvature_profile)}."
        )

        batch_dim = velocity_profile.shape[0]
        wheel_base = ego_wheel_base

        # Set up the lateral LQR problem using the constituent linear time-varying (affine) system dynamics.
        # Ultimately, we'll end up with the following problem structure where N = self._tracking_horizon:
        # lateral_error_N equals A @ lateral_error_0 + B @ steering_rate + g
        n_lateral_states = len(LateralStateIndex)

        identity: npt.NDArray[np.float64] = np.eye(
            n_lateral_states,
            dtype=np.float64,
        )

        in_matrix: npt.NDArray[np.float64] = np.zeros(
            (n_lateral_states, 1),
            np.float64,
        )  # no batch dim
        in_matrix[LateralStateIndex.STEERING_ANGLE] = discretization_time

        states_matrix_at_step: npt.NDArray[np.float64] = np.tile(
            identity[None, None, ...],
            [self._tracking_horizon, batch_dim, 1, 1],
        )  # (horizon, batch, 3, 3)

        states_matrix_at_step[
            :,
            :,
            LateralStateIndex.LATERAL_ERROR,
            LateralStateIndex.HEADING_ERROR,
        ] = velocity_profile.T * discretization_time

        states_matrix_at_step[
            :,
            :,
            LateralStateIndex.HEADING_ERROR,
            LateralStateIndex.STEERING_ANGLE,
        ] = velocity_profile.T * discretization_time / wheel_base

        affine_terms: npt.NDArray[np.float64] = np.zeros(
            (self._tracking_horizon, batch_dim, n_lateral_states),
            dtype=np.float64,
        )

        affine_terms[:, :, LateralStateIndex.HEADING_ERROR] = (
            -velocity_profile.T * curvature_profile.T * discretization_time
        )

        A: npt.NDArray[np.float64] = np.tile(
            identity[None, ...],
            [batch_dim, 1, 1],
        )  # (batch, 3, 3)
        B: npt.NDArray[np.float64] = np.zeros(
            (batch_dim, n_lateral_states, 1),
            dtype=np.float64,
        )  # (batch, 3, 1)
        g: npt.NDArray[np.float64] = np.zeros(
            (batch_dim, n_lateral_states),
            dtype=np.float64,
        )  # (batch, 3)

        for state_matrix_at_step, affine_term in zip(
            states_matrix_at_step,
            affine_terms,
            strict=True,
        ):
            # state_matrix_at_step has shape (batch, 3, 3)
            # affine_term has shape (batch, 3)
            A = np.einsum(
                "bij, bjk -> bik",
                state_matrix_at_step,
                A,
            )  # (batch, 3, 3)
            B = np.einsum("bij, bjk -> bik", state_matrix_at_step, B) + in_matrix  # (batch, 3, 1)
            g = np.einsum("bij, bj  -> bi", state_matrix_at_step, g) + affine_term  # (batch, 3)

        steering_rate_cmd = self._solve_one_step_lateral_lqr(
            initial_state=initial_lateral_state_vector,
            A=A,
            B=B,
            g=g,
        )

        return np.squeeze(steering_rate_cmd, axis=-1)

    def _solve_one_step_longitudinal_lqr(
        self,
        initial_state: jt.Float64[npt.NDArray[np.float64], " batch"],
        reference_state: jt.Float64[npt.NDArray[np.float64], " batch"],
        A: jt.Float64[npt.NDArray[np.float64], " batch"],
        B: jt.Float64[npt.NDArray[np.float64], " batch"],
        g: jt.Float64[npt.NDArray[np.float64], " batch"],
    ) -> jt.Float64[npt.NDArray[np.float64], " batch"]:
        """
        Uses LQR to find an optimal input to minimize tracking error in one step of dynamics.

        The dynamics are next_state = A @ initial_state + B @ input + g and our target is the reference_state.

        Args:
            initial_state: The current state.
            reference_state: The desired state in 1 step (according to A,B,g dynamics).
            A: The state dynamics matrix.
            B: The input dynamics matrix.
            g: The offset/affine dynamics term.

        Returns:
            LQR optimal input for the 1-step longitudinal problem.
        """
        state_error_zero_input = A * initial_state + g - reference_state
        inverse = -1 / (B * self._q_longitudinal * B + self._r_longitudinal)
        return inverse * B * self._q_longitudinal * state_error_zero_input

    def _solve_one_step_lateral_lqr(
        self,
        initial_state: jt.Float64[npt.NDArray[np.float64], "batch 3"],
        A: jt.Float64[npt.NDArray[np.float64], "batch 3 3"],
        B: jt.Float64[npt.NDArray[np.float64], "batch 3 1"],
        g: jt.Float64[npt.NDArray[np.float64], "batch 3"],
    ) -> jt.Float64[npt.NDArray[np.float64], "batch 1"]:
        """
        Uses LQR to find an optimal input to minimize tracking error in one step of dynamics.

        The dynamics are next_state = A @ initial_state + B @ input + g and our target is the reference_state.

        Args:
            initial_state: The current state.
            A: The state dynamics matrix.
            B: The input dynamics matrix.
            g: The offset/affine dynamics term.

        Returns:
            LQR optimal input for the 1-step lateral problem.
        """

        Q, R = self._q_lateral, self._r_lateral
        angle_diff_indices = [
            LateralStateIndex.HEADING_ERROR.value,
            LateralStateIndex.STEERING_ANGLE.value,
        ]
        BT = B.transpose(0, 2, 1)

        state_error_zero_input = np.einsum("bij, bj -> bi", A, initial_state) + g

        angle = state_error_zero_input[..., angle_diff_indices]
        state_error_zero_input[..., angle_diff_indices] = np.arctan2(
            np.sin(angle),
            np.cos(angle),
        )

        BT_x_Q = np.einsum("bij, jk -> bik", BT, Q)
        Inv = -1 / (np.einsum("bij, bji -> bi", BT_x_Q, B) + R)
        Tail = np.einsum("bij, bj -> bi", BT_x_Q, state_error_zero_input)

        return Inv * Tail
