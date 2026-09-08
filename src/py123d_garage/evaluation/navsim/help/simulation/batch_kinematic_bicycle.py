# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI
# Adapted from DanielDauner/nav123d (Apache License 2.0).

from __future__ import annotations

import copy

import jaxtyping as jt
import numpy as np
import numpy.typing as npt
from py123d.datatypes import EgoStateSE3Metadata, Timestamp
from py123d.geometry.utils.rotation_utils import normalize_angle

from py123d_garage.evaluation.navsim.help.utils.pdm_enums import (
    DynamicStateIndex,
    StateIndex,
)


def forward_integrate(
    init: jt.Float64[npt.NDArray[np.float64], " batch"],
    delta: jt.Float64[npt.NDArray[np.float64], " batch"],
    sampling_time: Timestamp,
) -> jt.Float64[npt.NDArray[np.float64], " batch"]:
    """
    Performs a simple euler integration.

    Args:
        init: Initial state
        delta: The rate of change of the state.
        sampling_time: The time duration to propagate for.

    Returns:
        The result of integration
    """
    return init + delta * sampling_time.time_s


class BatchKinematicBicycleModel:
    """Batch-wise kinematic bicycle motion model, referenced at the rear axle."""

    def __init__(
        self,
        max_steering_angle: float = np.pi / 3,
        accel_time_constant: float = 0.2,
        steering_angle_time_constant: float = 0.05,
    ):
        """
        Construct BatchKinematicBicycleModel.

        Args:
            max_steering_angle: [rad] Maximum absolute value steering angle allowed by model.
            accel_time_constant: low pass filter time constant for acceleration in s
            steering_angle_time_constant: low pass filter time constant for steering angle in s
        """
        self._max_steering_angle = max_steering_angle
        self._accel_time_constant = accel_time_constant
        self._steering_angle_time_constant = steering_angle_time_constant

    def get_state_dot(
        self,
        states: jt.Float64[npt.NDArray[np.float64], "batch 11"],
        wheel_base: float,
    ) -> jt.Float64[npt.NDArray[np.float64], "batch 11"]:
        """
        Calculates the rate of change of the state array representation.

        Args:
            states: array describing the state of the ego-vehicle
            wheel_base: The wheel base of the vehicle

        Returns:
            change rate across several state values
        """
        state_dots = np.zeros(states.shape, dtype=np.float64)

        longitudinal_speeds = states[:, StateIndex.VELOCITY_X]

        state_dots[:, StateIndex.X] = longitudinal_speeds * np.cos(
            states[:, StateIndex.HEADING],
        )
        state_dots[:, StateIndex.Y] = longitudinal_speeds * np.sin(
            states[:, StateIndex.HEADING],
        )
        state_dots[:, StateIndex.HEADING] = (
            longitudinal_speeds * np.tan(states[:, StateIndex.STEERING_ANGLE]) / wheel_base
        )

        state_dots[:, StateIndex.VELOCITY_2D] = states[
            :,
            StateIndex.ACCELERATION_2D,
        ]
        state_dots[:, StateIndex.ACCELERATION_2D] = 0.0

        state_dots[:, StateIndex.STEERING_ANGLE] = states[
            :,
            StateIndex.STEERING_RATE,
        ]

        return state_dots

    def _update_commands(
        self,
        states: jt.Float64[npt.NDArray[np.float64], "batch 11"],
        command_states: jt.Float64[npt.NDArray[np.float64], "batch 2"],
        sampling_time: Timestamp,
    ) -> jt.Float64[npt.NDArray[np.float64], "batch 11"]:
        """
        Applies a first-order control delay (low-pass filter) to acceleration and steering.

        Args:
            states: Ego state array
            command_states: The desired dynamic state (controller commands) for propagation
            sampling_time: The time duration to propagate for

        Returns:
            propagating_state including updated dynamic_state
        """

        propagating_state: jt.Float64[
            npt.NDArray[np.float64],
            "batch 11",
        ] = copy.deepcopy(states)

        dt_control = sampling_time.time_s

        accel = states[:, StateIndex.ACCELERATION_X]
        steering_angle = states[:, StateIndex.STEERING_ANGLE]

        ideal_accel_x = command_states[:, DynamicStateIndex.ACCELERATION_X]
        ideal_steering_angle = dt_control * command_states[:, DynamicStateIndex.STEERING_RATE] + steering_angle

        updated_accel_x = dt_control / (dt_control + self._accel_time_constant) * (ideal_accel_x - accel) + accel
        updated_steering_angle = (
            dt_control / (dt_control + self._steering_angle_time_constant) * (ideal_steering_angle - steering_angle)
            + steering_angle
        )
        updated_steering_rate = (updated_steering_angle - steering_angle) / dt_control

        propagating_state[:, StateIndex.ACCELERATION_X] = updated_accel_x
        propagating_state[:, StateIndex.ACCELERATION_Y] = 0.0
        propagating_state[:, StateIndex.STEERING_RATE] = updated_steering_rate

        return propagating_state

    def propagate_state(
        self,
        states: jt.Float64[npt.NDArray[np.float64], "batch 11"],
        command_states: jt.Float64[npt.NDArray[np.float64], "batch 2"],
        sampling_time: Timestamp,
        ego_metadata: EgoStateSE3Metadata,
    ) -> jt.Float64[npt.NDArray[np.float64], "batch 11"]:
        """
        Propagates the ego state array forward with the motion model.

        Args:
            states: state array representation of the ego-vehicle
            command_states: command array representation of controller
            sampling_time: time delta to propagate as Timestamp
            ego_metadata: Metadata for the ego state

        Returns:
            updated state array representation of the ego-vehicle
        """

        assert len(states) == len(command_states), "Batch size of states and command_states does not match!"

        wheel_base = ego_metadata.wheel_base
        propagating_state = self._update_commands(
            states,
            command_states,
            sampling_time,
        )
        output_state = copy.deepcopy(states)

        # Compute state derivatives
        state_dot = self.get_state_dot(propagating_state, wheel_base)

        output_state[:, StateIndex.X] = forward_integrate(
            states[:, StateIndex.X],
            state_dot[:, StateIndex.X],
            sampling_time,
        )
        output_state[:, StateIndex.Y] = forward_integrate(
            states[:, StateIndex.Y],
            state_dot[:, StateIndex.Y],
            sampling_time,
        )

        output_state[:, StateIndex.HEADING] = normalize_angle(
            forward_integrate(
                states[:, StateIndex.HEADING],
                state_dot[:, StateIndex.HEADING],
                sampling_time,
            ),
        )

        output_state[:, StateIndex.VELOCITY_X] = forward_integrate(
            states[:, StateIndex.VELOCITY_X],
            state_dot[:, StateIndex.VELOCITY_X],
            sampling_time,
        )

        # Lateral velocity is always zero in kinematic bicycle model
        output_state[:, StateIndex.VELOCITY_Y] = 0.0

        # Integrate steering angle and clip to bounds
        output_state[:, StateIndex.STEERING_ANGLE] = np.clip(
            forward_integrate(
                propagating_state[:, StateIndex.STEERING_ANGLE],
                state_dot[:, StateIndex.STEERING_ANGLE],
                sampling_time,
            ),
            -self._max_steering_angle,
            self._max_steering_angle,
        )

        output_state[:, StateIndex.ANGULAR_VELOCITY] = (
            output_state[:, StateIndex.VELOCITY_X] * np.tan(output_state[:, StateIndex.STEERING_ANGLE]) / wheel_base
        )

        output_state[:, StateIndex.ACCELERATION_2D] = state_dot[
            :,
            StateIndex.VELOCITY_2D,
        ]

        output_state[:, StateIndex.ANGULAR_ACCELERATION] = (
            output_state[:, StateIndex.ANGULAR_VELOCITY] - states[:, StateIndex.ANGULAR_VELOCITY]
        ) / sampling_time.time_s

        output_state[:, StateIndex.STEERING_RATE] = state_dot[
            :,
            StateIndex.STEERING_ANGLE,
        ]

        return output_state
