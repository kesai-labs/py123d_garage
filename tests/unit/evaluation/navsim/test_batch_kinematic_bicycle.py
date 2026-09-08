"""Unit tests for the batch kinematic bicycle motion model against hand-computed rigid-body physics."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from py123d.datatypes import EgoStateSE3Metadata, Timestamp
from py123d.geometry import PoseSE3

from py123d_garage.evaluation.navsim.help.simulation.batch_kinematic_bicycle import (
    BatchKinematicBicycleModel,
)
from py123d_garage.evaluation.navsim.help.utils.pdm_enums import (
    DynamicStateIndex,
    StateIndex,
)

WHEEL_BASE: float = 3.0

METADATA: EgoStateSE3Metadata = EgoStateSE3Metadata(
    vehicle_name="test_vehicle",
    width=2.0,
    length=5.0,
    height=1.6,
    wheel_base=WHEEL_BASE,
    center_to_imu_se3=PoseSE3(1.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    rear_axle_to_imu_se3=PoseSE3(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
)


def _zero_states(n_batch: int) -> npt.NDArray[np.float64]:
    return np.zeros((n_batch, len(StateIndex)), dtype=np.float64)


def _zero_commands(n_batch: int) -> npt.NDArray[np.float64]:
    return np.zeros((n_batch, len(DynamicStateIndex)), dtype=np.float64)


def _instant_model() -> BatchKinematicBicycleModel:
    """Model without control lag, so commands apply exactly within one step."""
    return BatchKinematicBicycleModel(
        accel_time_constant=0.0,
        steering_angle_time_constant=0.0,
    )


def test_zero_steering_constant_velocity_moves_straight() -> None:
    states = _zero_states(1)
    states[0, StateIndex.VELOCITY_X] = 5.0

    output = _instant_model().propagate_state(
        states,
        _zero_commands(1),
        Timestamp.from_s(0.5),
        METADATA,
    )

    assert np.isclose(output[0, StateIndex.X], 2.5)
    assert output[0, StateIndex.Y] == 0.0
    assert output[0, StateIndex.HEADING] == 0.0
    assert np.isclose(output[0, StateIndex.VELOCITY_X], 5.0)
    assert output[0, StateIndex.VELOCITY_Y] == 0.0
    assert output[0, StateIndex.ANGULAR_VELOCITY] == 0.0


def test_constant_steering_yields_bicycle_curvature() -> None:
    """Yaw rate must equal v * tan(steering) / wheel_base, the bicycle-model curvature."""
    speed, steering = 2.0, 0.1
    states = _zero_states(1)
    states[0, StateIndex.VELOCITY_X] = speed
    states[0, StateIndex.STEERING_ANGLE] = steering

    output = _instant_model().propagate_state(
        states,
        _zero_commands(1),
        Timestamp.from_s(0.1),
        METADATA,
    )

    yaw_rate = speed * np.tan(steering) / WHEEL_BASE
    assert np.isclose(output[0, StateIndex.STEERING_ANGLE], steering)
    assert np.isclose(output[0, StateIndex.HEADING], yaw_rate * 0.1)
    assert np.isclose(output[0, StateIndex.ANGULAR_VELOCITY], yaw_rate)


def test_acceleration_command_applies_without_lag() -> None:
    states = _zero_states(1)
    states[0, StateIndex.VELOCITY_X] = 5.0
    commands = _zero_commands(1)
    commands[0, DynamicStateIndex.ACCELERATION_X] = 2.0

    output = _instant_model().propagate_state(
        states,
        commands,
        Timestamp.from_s(0.5),
        METADATA,
    )

    assert np.isclose(output[0, StateIndex.VELOCITY_X], 6.0)
    assert np.isclose(output[0, StateIndex.ACCELERATION_X], 2.0)


def test_control_lag_attenuates_acceleration_command() -> None:
    """With a nonzero time constant one step realizes only part of the commanded acceleration."""
    states = _zero_states(1)
    states[0, StateIndex.VELOCITY_X] = 5.0
    commands = _zero_commands(1)
    commands[0, DynamicStateIndex.ACCELERATION_X] = 2.0

    output = BatchKinematicBicycleModel().propagate_state(
        states,
        commands,
        Timestamp.from_s(0.5),
        METADATA,
    )

    applied_accel = output[0, StateIndex.ACCELERATION_X]
    assert 0.0 < applied_accel < 2.0
    assert np.isclose(
        output[0, StateIndex.VELOCITY_X],
        5.0 + applied_accel * 0.5,
    )


def test_steering_angle_clamps_to_maximum() -> None:
    states = _zero_states(1)
    states[0, StateIndex.VELOCITY_X] = 1.0
    states[0, StateIndex.STEERING_ANGLE] = 1.0
    commands = _zero_commands(1)
    commands[0, DynamicStateIndex.STEERING_RATE] = 10.0

    output = _instant_model().propagate_state(
        states,
        commands,
        Timestamp.from_s(0.5),
        METADATA,
    )

    assert np.isclose(output[0, StateIndex.STEERING_ANGLE], np.pi / 3)


def test_heading_wraps_at_pi() -> None:
    states = _zero_states(1)
    states[0, StateIndex.HEADING] = np.pi - 0.01
    states[0, StateIndex.VELOCITY_X] = 2.0
    states[0, StateIndex.STEERING_ANGLE] = np.arctan(
        0.2 * WHEEL_BASE / 2.0,
    )  # yaw rate 0.2 rad/s

    output = _instant_model().propagate_state(
        states,
        _zero_commands(1),
        Timestamp.from_s(0.1),
        METADATA,
    )

    assert np.isclose(output[0, StateIndex.HEADING], -np.pi + 0.01)


def test_batch_entries_propagate_independently() -> None:
    states = _zero_states(2)
    states[0, StateIndex.VELOCITY_X] = 5.0
    states[1, StateIndex.VELOCITY_X] = 2.0
    states[1, StateIndex.STEERING_ANGLE] = 0.1

    output = _instant_model().propagate_state(
        states,
        _zero_commands(2),
        Timestamp.from_s(0.1),
        METADATA,
    )

    assert np.isclose(output[0, StateIndex.X], 0.5)
    assert output[0, StateIndex.HEADING] == 0.0
    assert np.isclose(
        output[1, StateIndex.HEADING],
        2.0 * np.tan(0.1) / WHEEL_BASE * 0.1,
    )
