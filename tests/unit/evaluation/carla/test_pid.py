"""Unit tests for the PID controller and waypoint follower against hand-computed control values."""

from __future__ import annotations

import numpy as np
import pytest

from py123d_garage.evaluation.carla.help.pid import (
    PIDController,
    WaypointFollower,
)


def test_p_only_constant_error_gives_kp_times_error() -> None:
    controller = PIDController(k_p=2.0, k_i=0.0, k_d=0.0)
    assert controller.step(0.5) == pytest.approx(1.0)
    assert controller.step(0.5) == pytest.approx(1.0)
    assert controller.step(-0.25) == pytest.approx(-0.5)


def test_integral_is_sliding_window_mean() -> None:
    controller = PIDController(k_p=0.0, k_i=1.0, k_d=0.0, window_size=4)
    assert controller.step(1.0) == pytest.approx(0.25)
    assert controller.step(1.0) == pytest.approx(0.5)
    assert controller.step(1.0) == pytest.approx(0.75)
    assert controller.step(1.0) == pytest.approx(1.0)
    assert controller.step(1.0) == pytest.approx(1.0)


def test_derivative_is_difference_of_last_two_errors() -> None:
    controller = PIDController(k_p=0.0, k_i=0.0, k_d=2.0)
    assert controller.step(1.0) == pytest.approx(2.0)
    assert controller.step(1.0) == pytest.approx(0.0)
    assert controller.step(3.0) == pytest.approx(4.0)
    assert controller.step(0.0) == pytest.approx(-6.0)


def test_step_input_converges_without_leaving_bounds() -> None:
    """Driving an integrator plant to a unit step settles at the target with bounded overshoot."""
    controller = PIDController(k_p=1.0, k_i=0.5, k_d=0.2)
    state = 0.0
    trajectory: list[float] = []
    for _ in range(200):
        state += controller.step(1.0 - state) * 0.05
        trajectory.append(state)
    assert trajectory[-1] == pytest.approx(1.0, abs=1e-3)
    assert max(trajectory) < 1.05
    assert all(0.99 < value < 1.01 for value in trajectory[100:])


def _p_only_follower() -> WaypointFollower:
    """A follower whose two controllers are pure unit-gain P, so outputs equal their errors."""
    return WaypointFollower(
        interval_length=0.25,
        turn_kp=1.0,
        turn_ki=0.0,
        turn_kd=0.0,
        speed_kp=1.0,
        speed_ki=0.0,
        speed_kd=0.0,
    )


def test_straight_waypoints_give_zero_steer() -> None:
    waypoints = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])
    steer, throttle, brake = _p_only_follower().step(waypoints, 2.0)
    assert steer == pytest.approx(0.0)
    assert throttle == pytest.approx(0.99)
    assert brake == 0.0


def test_throttle_is_gain_times_speed_error() -> None:
    # Waypoints 0.5 m apart at 0.25 s spacing encode a desired speed of 2 m/s.
    waypoints = np.array([[0.5, 0.0], [1.0, 0.0], [1.5, 0.0], [2.0, 0.0]])
    _, throttle, brake = _p_only_follower().step(waypoints, 1.5)
    assert throttle == pytest.approx(0.5)
    assert brake == 0.0


def test_speed_error_clips_at_speed_delta_clip() -> None:
    waypoints = np.array([[1.5, 0.0], [3.0, 0.0], [4.5, 0.0], [6.0, 0.0]])
    _, throttle, brake = _p_only_follower().step(waypoints, 0.5)
    assert throttle == pytest.approx(0.99)
    assert brake == 0.0


def test_steer_aims_at_first_waypoint_beyond_aim_distance() -> None:
    """The 2 m waypoint is inside the 2.25 m slow aim distance, so the 45 degree one on the left is the aim."""
    waypoints = np.array(
        [[1.0, 0.0], [2.0, 0.0], [3.0, 3.0], [4.0, 0.0], [5.0, 0.0]],
    )
    steer, _, brake = _p_only_follower().step(waypoints, 2.0)
    assert steer == pytest.approx(-0.5)
    assert brake == 0.0


def test_steer_falls_back_to_last_waypoint() -> None:
    waypoints = np.array([[0.3, 0.0], [0.6, 0.0], [0.9, 0.0], [1.2, 1.2]])
    steer, _, brake = _p_only_follower().step(waypoints, 1.0)
    assert steer == pytest.approx(-0.5)
    assert brake == 0.0


def test_steer_clips_to_carla_range() -> None:
    # Aim point at 135 degrees to the left gives a raw command of -1.5.
    waypoints = np.array([[1.0, 0.0], [2.0, 0.0], [-3.0, 3.0], [4.0, 0.0]])
    steer, _, _ = _p_only_follower().step(waypoints, 2.0)
    assert steer == -1.0


def test_brakes_when_desired_speed_below_threshold() -> None:
    waypoints = np.full((4, 2), 1.0)
    steer, throttle, brake = _p_only_follower().step(waypoints, 2.0)
    assert brake == 1.0
    assert throttle == 0.0
    assert steer == pytest.approx(0.0)


def test_brakes_when_overspeeding() -> None:
    waypoints = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])
    _, throttle, brake = _p_only_follower().step(waypoints, 10.0)
    assert brake == 1.0
    assert throttle == 0.0


def test_standstill_zeroes_steer_but_not_throttle() -> None:
    waypoints = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 3.0], [4.0, 0.0]])
    steer, throttle, brake = _p_only_follower().step(waypoints, 0.0)
    assert steer == pytest.approx(0.0)
    assert throttle == pytest.approx(0.99)
    assert brake == 0.0


def test_rejects_interval_that_cannot_resolve_half_second() -> None:
    with pytest.raises(AssertionError):
        WaypointFollower(interval_length=1.0)
