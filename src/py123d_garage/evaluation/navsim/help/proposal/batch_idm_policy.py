# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI
# Adapted from DanielDauner/nav123d (Apache License 2.0).

from __future__ import annotations

import jaxtyping as jt
import numpy as np
import numpy.typing as npt

from py123d_garage.evaluation.navsim.help.utils.pdm_enums import (
    LeadingAgentIndex,
    StateIDMIndex,
)


class BatchIDMPolicy:
    """IDM policies operating on a batch of proposals."""

    def __init__(
        self,
        fallback_target_velocity: list[float] | float = 15.0,
        speed_limit_fraction: list[float] | float | None = None,
        min_gap_to_lead_agent: list[float] | float = 1.0,
        headway_time: list[float] | float = 1.5,
        accel_max: list[float] | float = 1.5,
        decel_max: list[float] | float = 3.0,
        acceleration_exponent: float = 10,
    ):
        """
        Constructor for BatchIDMPolicy.

        Args:
            fallback_target_velocity: Desired fallback velocity in free traffic [m/s]
            speed_limit_fraction: Fraction of speed-limit desired in free traffic
            min_gap_to_lead_agent: Minimum relative distance to lead vehicle [m]
            headway_time: Desired time headway. Minimum time to the vehicle in front [s]
            accel_max: maximum acceleration [m/s^2]
            decel_max: maximum deceleration (positive value) [m/s^2]
            acceleration_exponent: acceleration exponent of IDM.
        """
        if speed_limit_fraction is None:
            speed_limit_fraction = [0.2, 0.4, 0.6, 0.8, 1.0]
        parameter_list = [
            fallback_target_velocity,
            speed_limit_fraction,
            min_gap_to_lead_agent,
            headway_time,
            accel_max,
            decel_max,
        ]
        num_parameter_policies = [len(item) for item in parameter_list if isinstance(item, list)]

        if len(num_parameter_policies) > 0:
            assert all(item == num_parameter_policies[0] for item in num_parameter_policies), (
                "BatchIDMPolicy initial parameters must be float, or lists of equal length"
            )
            num_policies = max(num_parameter_policies)
        else:
            num_policies = 1

        self._num_policies: int = num_policies
        self._fallback_target_velocities: npt.NDArray[np.float64] = np.zeros(
            (self._num_policies),
            dtype=np.float64,
        )
        self._speed_limit_fractions: npt.NDArray[np.float64] = np.zeros(
            (self._num_policies),
            dtype=np.float64,
        )
        self._min_gap_to_lead_agent: npt.NDArray[np.float64] = np.zeros(
            (self._num_policies),
            dtype=np.float64,
        )
        self._headway_time: npt.NDArray[np.float64] = np.zeros(
            (self._num_policies),
            dtype=np.float64,
        )
        self._accel_max: npt.NDArray[np.float64] = np.zeros(
            (self._num_policies),
            dtype=np.float64,
        )
        self._decel_max: npt.NDArray[np.float64] = np.zeros(
            (self._num_policies),
            dtype=np.float64,
        )
        self._acceleration_exponent: float = acceleration_exponent

        for i in range(self._num_policies):
            self._fallback_target_velocities[i] = (
                fallback_target_velocity[i] if isinstance(fallback_target_velocity, list) else fallback_target_velocity
            )
            self._speed_limit_fractions[i] = (
                speed_limit_fraction[i] if isinstance(speed_limit_fraction, list) else speed_limit_fraction
            )
            self._min_gap_to_lead_agent[i] = (
                min_gap_to_lead_agent[i] if isinstance(min_gap_to_lead_agent, list) else min_gap_to_lead_agent
            )
            self._headway_time[i] = headway_time[i] if isinstance(headway_time, list) else headway_time
            self._accel_max[i] = accel_max[i] if isinstance(accel_max, list) else accel_max
            self._decel_max[i] = decel_max[i] if isinstance(decel_max, list) else decel_max

        # lazy loaded
        self._target_velocities: npt.NDArray[np.float64] = np.zeros(
            (self._num_policies),
            dtype=np.float64,
        )

    @property
    def num_policies(self) -> int:
        """Number of IDM policies in the batch."""
        return self._num_policies

    @property
    def max_target_velocity(self) -> float:
        """Highest target velocity across policies [m/s]."""
        return float(np.max(self._target_velocities))

    def update(self, speed_limit_mps: float | None) -> None:
        """
        Recomputes target velocities from the current speed limit.

        Args:
            speed_limit_mps: speed limit of current lane [m/s]
        """

        if speed_limit_mps is not None:
            self._target_velocities = self._speed_limit_fractions * speed_limit_mps
        else:
            self._target_velocities = self._speed_limit_fractions * self._fallback_target_velocities

        assert np.all(np.isfinite(self._target_velocities)), (
            "BatchIDMPolicy: target velocities contain NaN or infinite values after update!"
        )

    def propagate(
        self,
        previous_idm_states: jt.Float64[npt.NDArray[np.float64], "batch 2"],
        leading_agent_states: jt.Float64[npt.NDArray[np.float64], "batch 3"],
        longitudinal_idcs: list[int],
        sampling_time: float,
    ) -> jt.Float64[npt.NDArray[np.float64], "batch 2"]:
        """
        Propagates IDM policies for one time-step.

        Args:
            previous_idm_states: array containing previous state
            leading_agent_states: array contains leading vehicle information
            longitudinal_idcs: indices of policies to be applied over a batch-dim
            sampling_time: time to propagate forward [s]

        Returns:
            array containing propagated state values
        """

        assert len(previous_idm_states) == len(longitudinal_idcs), (
            "PDMIDMPolicy: propagate function requires equal length of input arguments!"
        )
        assert len(leading_agent_states) == len(longitudinal_idcs), (
            "PDMIDMPolicy: propagate function requires equal length of input arguments!"
        )

        # state variables
        x_agent, v_agent = (
            previous_idm_states[:, StateIDMIndex.PROGRESS],
            previous_idm_states[:, StateIDMIndex.VELOCITY],
        )

        x_lead, v_lead, l_r_lead = (
            leading_agent_states[:, LeadingAgentIndex.PROGRESS],
            leading_agent_states[:, LeadingAgentIndex.VELOCITY],
            leading_agent_states[:, LeadingAgentIndex.LENGTH_REAR],
        )

        # parameters
        (
            target_velocity,
            min_gap_to_lead_agent,
            headway_time,
            accel_max,
            decel_max,
        ) = (
            self._target_velocities[longitudinal_idcs],
            self._min_gap_to_lead_agent[longitudinal_idcs],
            self._headway_time[longitudinal_idcs],
            self._accel_max[longitudinal_idcs],
            self._decel_max[longitudinal_idcs],
        )

        # convenience definitions
        s_star = (
            min_gap_to_lead_agent
            + v_agent * headway_time
            + (v_agent * (v_agent - v_lead)) / (2 * np.sqrt(accel_max * decel_max))
        )

        s_alpha = np.maximum(
            x_lead - x_agent - l_r_lead,
            min_gap_to_lead_agent,
        )  # clamp to avoid zero division

        # differential equations
        x_agent_dot = v_agent
        v_agent_dot = accel_max * (
            1 - (v_agent / target_velocity) ** self._acceleration_exponent - (s_star / s_alpha) ** 2
        )

        # clip values
        v_agent_dot = np.clip(v_agent_dot, -decel_max, accel_max)

        next_idm_states: npt.NDArray[np.float64] = np.zeros(
            (len(longitudinal_idcs), len(StateIDMIndex)),
            dtype=np.float64,
        )
        next_idm_states[:, StateIDMIndex.PROGRESS] = x_agent + sampling_time * x_agent_dot
        next_idm_states[:, StateIDMIndex.VELOCITY] = v_agent + sampling_time * v_agent_dot

        return next_idm_states
