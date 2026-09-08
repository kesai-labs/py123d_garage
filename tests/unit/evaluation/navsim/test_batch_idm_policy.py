"""Unit tests for the batch IDM policy against hand-computed car-following behavior."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from py123d_garage.evaluation.navsim.help.proposal.batch_idm_policy import (
    BatchIDMPolicy,
)
from py123d_garage.evaluation.navsim.help.utils.pdm_enums import (
    LeadingAgentIndex,
    StateIDMIndex,
)

ACCEL_MAX: float = 1.5
DECEL_MAX: float = 3.0


def _single_policy() -> BatchIDMPolicy:
    policy = BatchIDMPolicy(
        fallback_target_velocity=15.0,
        speed_limit_fraction=1.0,
        min_gap_to_lead_agent=1.0,
        headway_time=1.5,
        accel_max=ACCEL_MAX,
        decel_max=DECEL_MAX,
    )
    policy.update(speed_limit_mps=10.0)
    return policy


def _idm_states(
    progress: float,
    velocity: float,
) -> npt.NDArray[np.float64]:
    states = np.zeros((1, len(StateIDMIndex)), dtype=np.float64)
    states[0, StateIDMIndex.PROGRESS] = progress
    states[0, StateIDMIndex.VELOCITY] = velocity
    return states


def _lead_states(
    progress: float,
    velocity: float,
    length_rear: float,
) -> npt.NDArray[np.float64]:
    states = np.zeros((1, len(LeadingAgentIndex)), dtype=np.float64)
    states[0, LeadingAgentIndex.PROGRESS] = progress
    states[0, LeadingAgentIndex.VELOCITY] = velocity
    states[0, LeadingAgentIndex.LENGTH_REAR] = length_rear
    return states


def test_free_road_accelerates_at_max_from_rest() -> None:
    next_states = _single_policy().propagate(
        _idm_states(0.0, 0.0),
        _lead_states(1000.0, 0.0, 1.0),
        [0],
        0.5,
    )

    assert np.isclose(
        next_states[0, StateIDMIndex.VELOCITY],
        ACCEL_MAX * 0.5,
        atol=1e-4,
    )
    assert next_states[0, StateIDMIndex.PROGRESS] == 0.0


def test_free_road_saturates_at_target_velocity() -> None:
    policy = _single_policy()
    lead = _lead_states(1000.0, 0.0, 1.0)
    states = _idm_states(0.0, 0.0)

    for _ in range(200):
        next_states = policy.propagate(states, lead, [0], 0.2)
        delta_v = next_states[0, StateIDMIndex.VELOCITY] - states[0, StateIDMIndex.VELOCITY]
        assert -DECEL_MAX * 0.2 - 1e-9 <= delta_v <= ACCEL_MAX * 0.2 + 1e-9
        states = next_states

    assert np.isclose(states[0, StateIDMIndex.VELOCITY], 10.0, atol=0.05)


def test_at_target_velocity_holds_speed() -> None:
    next_states = _single_policy().propagate(
        _idm_states(0.0, 10.0),
        _lead_states(1000.0, 0.0, 1.0),
        [0],
        0.5,
    )

    assert np.isclose(next_states[0, StateIDMIndex.VELOCITY], 10.0, atol=5e-3)
    assert np.isclose(next_states[0, StateIDMIndex.PROGRESS], 5.0)


def test_close_stopped_lead_brakes_at_max_deceleration() -> None:
    next_states = _single_policy().propagate(
        _idm_states(0.0, 10.0),
        _lead_states(7.0, 0.0, 2.0),  # 5 m gap
        [0],
        0.5,
    )

    assert np.isclose(
        next_states[0, StateIDMIndex.VELOCITY],
        10.0 - DECEL_MAX * 0.5,
    )
    assert np.isclose(next_states[0, StateIDMIndex.PROGRESS], 5.0)


def test_speed_limit_fractions_scale_target_velocities() -> None:
    policy = BatchIDMPolicy(
        fallback_target_velocity=15.0,
        speed_limit_fraction=[0.5, 1.0],
        min_gap_to_lead_agent=1.0,
        headway_time=1.5,
        accel_max=ACCEL_MAX,
        decel_max=DECEL_MAX,
    )
    assert policy.num_policies == 2

    policy.update(speed_limit_mps=10.0)
    assert policy.max_target_velocity == 10.0

    states = np.zeros((2, len(StateIDMIndex)), dtype=np.float64)
    states[:, StateIDMIndex.VELOCITY] = 5.0
    leads = np.zeros((2, len(LeadingAgentIndex)), dtype=np.float64)
    leads[:, LeadingAgentIndex.PROGRESS] = 1000.0
    leads[:, LeadingAgentIndex.LENGTH_REAR] = 1.0

    next_states = policy.propagate(states, leads, [0, 1], 0.5)

    assert np.isclose(
        next_states[0, StateIDMIndex.VELOCITY],
        5.0,
        atol=5e-3,
    )  # policy 0 is at its 5 m/s target
    assert next_states[1, StateIDMIndex.VELOCITY] > 5.5  # policy 1 targets 10


def test_update_without_speed_limit_uses_fallback() -> None:
    policy = BatchIDMPolicy(
        fallback_target_velocity=15.0,
        speed_limit_fraction=[0.2, 1.0],
        min_gap_to_lead_agent=1.0,
        headway_time=1.5,
        accel_max=ACCEL_MAX,
        decel_max=DECEL_MAX,
    )
    policy.update(speed_limit_mps=None)
    assert policy.max_target_velocity == 15.0
