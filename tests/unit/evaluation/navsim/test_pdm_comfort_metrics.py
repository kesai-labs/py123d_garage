"""Unit tests for the PDM comfort metrics on synthetic motion profiles with known dynamics."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from py123d.datatypes import EgoStateSE3Metadata
from py123d.geometry import PoseSE3

from py123d_garage.evaluation.navsim.help.scoring.pdm_comfort_metrics import (
    ego_is_comfortable,
)
from py123d_garage.evaluation.navsim.help.utils.pdm_enums import StateIndex

LON_ACCEL, LAT_ACCEL, JERK, LON_JERK, YAW_ACCEL, YAW_RATE = range(6)

METADATA: EgoStateSE3Metadata = EgoStateSE3Metadata(
    vehicle_name="test_vehicle",
    width=2.0,
    length=5.0,
    height=1.6,
    wheel_base=3.0,
    center_to_imu_se3=PoseSE3(1.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    rear_axle_to_imu_se3=PoseSE3(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
)


def _constant_velocity_profile(
    n_time: int,
    dt: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Straight drive at 5 m/s: zero acceleration, jerk, and yaw motion throughout."""
    time_steps_s = np.arange(n_time, dtype=np.float64) * dt
    states = np.zeros((1, n_time, len(StateIndex)), dtype=np.float64)
    states[0, :, StateIndex.X] = 5.0 * time_steps_s
    states[0, :, StateIndex.VELOCITY_X] = 5.0
    return states, time_steps_s


def test_constant_velocity_is_comfortable() -> None:
    states, time_steps_s = _constant_velocity_profile(20, 0.1)
    assert ego_is_comfortable(states, time_steps_s, METADATA).all()


def test_excessive_lon_acceleration_fails_only_lon_accel() -> None:
    states, time_steps_s = _constant_velocity_profile(20, 0.1)
    states[0, :, StateIndex.ACCELERATION_X] = 3.0  # above the 2.40 m/s^2 bound

    result = ego_is_comfortable(states, time_steps_s, METADATA)
    expected = np.ones((1, 6), dtype=np.bool_)
    expected[0, LON_ACCEL] = False
    assert (result == expected).all()


def test_excessive_lat_acceleration_fails_only_lat_accel() -> None:
    states, time_steps_s = _constant_velocity_profile(20, 0.1)
    states[0, :, StateIndex.ACCELERATION_Y] = 6.0  # above the 4.89 m/s^2 bound

    result = ego_is_comfortable(states, time_steps_s, METADATA)
    expected = np.ones((1, 6), dtype=np.bool_)
    expected[0, LAT_ACCEL] = False
    assert (result == expected).all()


def test_steep_acceleration_ramp_fails_jerk_metrics() -> None:
    """A 10 m/s^3 ramp stays inside the acceleration bounds but breaks both jerk bounds."""
    states, time_steps_s = _constant_velocity_profile(9, 0.05)
    states[0, :, StateIndex.ACCELERATION_X] = -2.0 + 10.0 * time_steps_s

    result = ego_is_comfortable(states, time_steps_s, METADATA)
    expected = np.ones((1, 6), dtype=np.bool_)
    expected[0, JERK] = False
    expected[0, LON_JERK] = False
    assert (result == expected).all()


def test_fast_spin_fails_only_yaw_rate() -> None:
    states, time_steps_s = _constant_velocity_profile(20, 0.1)
    states[0, :, StateIndex.HEADING] = 2.0 * time_steps_s  # above 0.95 rad/s

    result = ego_is_comfortable(states, time_steps_s, METADATA)
    expected = np.ones((1, 6), dtype=np.bool_)
    expected[0, YAW_RATE] = False
    assert (result == expected).all()


def test_quadratic_heading_fails_only_yaw_accel() -> None:
    """Heading 2*t^2 gives 4 rad/s^2 yaw acceleration while yaw rate peaks below its bound."""
    states, time_steps_s = _constant_velocity_profile(9, 0.025)
    states[0, :, StateIndex.HEADING] = 2.0 * time_steps_s**2

    result = ego_is_comfortable(states, time_steps_s, METADATA)
    expected = np.ones((1, 6), dtype=np.bool_)
    expected[0, YAW_ACCEL] = False
    assert (result == expected).all()


def test_batch_entries_are_scored_independently() -> None:
    smooth, time_steps_s = _constant_velocity_profile(20, 0.1)
    spinning = smooth.copy()
    spinning[0, :, StateIndex.HEADING] = 2.0 * time_steps_s
    states = np.concatenate([smooth, spinning], axis=0)

    result = ego_is_comfortable(states, time_steps_s, METADATA)
    assert result[0].all()
    assert not result[1, YAW_RATE]
    assert result[1, :YAW_RATE].all()
