"""Unit tests for the batch LQR tracker on synthetic straight-line and stopping trajectories."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from py123d_garage.evaluation.navsim.help.simulation.batch_lqr import (
    BatchLQRTracker,
)
from py123d_garage.evaluation.navsim.help.utils.pdm_enums import (
    DynamicStateIndex,
    StateIndex,
)

WHEEL_BASE: float = 3.0
DT: float = 0.1
N_TIME: int = 30


def _straight_proposals(speed: float) -> npt.NDArray[np.float64]:
    time_steps_s = np.arange(N_TIME, dtype=np.float64) * DT
    proposals = np.zeros((1, N_TIME, len(StateIndex)), dtype=np.float64)
    proposals[0, :, StateIndex.X] = speed * time_steps_s
    proposals[0, :, StateIndex.VELOCITY_X] = speed
    return proposals


def test_reference_profiles_recover_straight_line_motion() -> None:
    proposals = _straight_proposals(5.0)
    velocity_profile, curvature_profile = BatchLQRTracker().compute_reference_profiles(proposals, DT)

    assert np.allclose(velocity_profile, 5.0, atol=1e-3)
    assert np.allclose(curvature_profile, 0.0, atol=1e-6)


def test_on_track_ego_receives_zero_commands() -> None:
    tracker = BatchLQRTracker()
    proposals = _straight_proposals(5.0)
    velocity_profile, curvature_profile = tracker.compute_reference_profiles(
        proposals,
        DT,
    )

    commands = tracker.track_trajectory(
        0,
        proposals[:, 0].copy(),
        proposals,
        velocity_profile,
        curvature_profile,
        DT,
        WHEEL_BASE,
    )

    assert np.allclose(commands, 0.0, atol=1e-6)


def test_slow_ego_gets_lqr_optimal_acceleration() -> None:
    """
    The scalar LQR optimum is q*B*(v_ref - v0) / (q*B^2 + r) with B = horizon * dt.

    With defaults q=10, r=1, horizon=10, dt=0.1 and a 2 m/s velocity error this is 20/11.
    """
    tracker = BatchLQRTracker()
    proposals = _straight_proposals(5.0)
    velocity_profile, curvature_profile = tracker.compute_reference_profiles(
        proposals,
        DT,
    )
    initial_states = proposals[:, 0].copy()
    initial_states[0, StateIndex.VELOCITY_X] = 3.0

    commands = tracker.track_trajectory(
        0,
        initial_states,
        proposals,
        velocity_profile,
        curvature_profile,
        DT,
        WHEEL_BASE,
    )

    assert np.isclose(
        commands[0, DynamicStateIndex.ACCELERATION_X],
        20.0 / 11.0,
        atol=1e-3,
    )
    assert np.isclose(commands[0, DynamicStateIndex.STEERING_RATE], 0.0)


def test_left_offset_ego_steers_right() -> None:
    tracker = BatchLQRTracker()
    proposals = _straight_proposals(5.0)
    velocity_profile, curvature_profile = tracker.compute_reference_profiles(
        proposals,
        DT,
    )
    initial_states = proposals[:, 0].copy()
    initial_states[0, StateIndex.Y] = 1.0

    commands = tracker.track_trajectory(
        0,
        initial_states,
        proposals,
        velocity_profile,
        curvature_profile,
        DT,
        WHEEL_BASE,
    )

    assert commands[0, DynamicStateIndex.STEERING_RATE] < -1e-3
    assert np.isclose(commands[0, DynamicStateIndex.ACCELERATION_X], 0.0)


def test_stationary_reference_engages_stopping_controller() -> None:
    """Near standstill the P controller commands accel = -gain * (v0 - v_ref) and no steering."""
    tracker = BatchLQRTracker(stopping_proportional_gain=0.5)
    proposals = np.zeros((1, N_TIME, len(StateIndex)), dtype=np.float64)
    velocity_profile, curvature_profile = tracker.compute_reference_profiles(
        proposals,
        DT,
    )
    assert np.allclose(velocity_profile, 0.0, atol=1e-9)

    initial_states = np.zeros((1, len(StateIndex)), dtype=np.float64)
    initial_states[0, StateIndex.VELOCITY_X] = 0.1

    commands = tracker.track_trajectory(
        0,
        initial_states,
        proposals,
        velocity_profile,
        curvature_profile,
        DT,
        WHEEL_BASE,
    )

    assert np.isclose(commands[0, DynamicStateIndex.ACCELERATION_X], -0.05)
    assert commands[0, DynamicStateIndex.STEERING_RATE] == 0.0


def test_batch_mixes_on_track_and_offset_proposals_independently() -> None:
    tracker = BatchLQRTracker()
    proposals = np.repeat(_straight_proposals(5.0), 2, axis=0)
    velocity_profile, curvature_profile = tracker.compute_reference_profiles(
        proposals,
        DT,
    )
    initial_states = proposals[:, 0].copy()
    initial_states[1, StateIndex.Y] = 1.0

    commands = tracker.track_trajectory(
        0,
        initial_states,
        proposals,
        velocity_profile,
        curvature_profile,
        DT,
        WHEEL_BASE,
    )

    assert np.allclose(commands[0], 0.0, atol=1e-6)
    assert commands[1, DynamicStateIndex.STEERING_RATE] < -1e-3
