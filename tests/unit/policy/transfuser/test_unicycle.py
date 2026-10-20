# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""
The unicycle action fit must invert the rollout on smooth trajectories and stay finite at standstill; bounded
actions must keep ordinary driving and stay inside their bounds.
"""

from __future__ import annotations

import math

import pytest
import torch

from py123d_garage.config.schema.policy.transfuser_config import ActionConfig
from py123d_garage.policy.transfuser.network.decoder.unicycle import (
    STEP_S,
    actions_to_state,
    fit_actions,
    rollout,
    state_to_actions,
)

_NUM_STEPS = 40


def _smooth_actions(accel_mps2: float, curvature_per_m: float, config: ActionConfig) -> torch.Tensor:
    """Normalized actions of a sinusoidal acceleration and a ramped curvature over the horizon."""
    time_s = torch.arange(_NUM_STEPS) * STEP_S
    acceleration = accel_mps2 * torch.sin(2 * math.pi * time_s / 4.0)
    curvature = curvature_per_m * time_s / time_s[-1]
    return torch.stack(
        [
            (acceleration - config.accel_mean_mps2) / config.accel_std_mps2,
            (curvature - config.curvature_mean_per_m) / config.curvature_std_per_m,
        ],
        dim=-1,
    )[None]


@pytest.mark.parametrize(
    ("speed_mps", "accel_mps2", "curvature_per_m"),
    [(10.0, 1.0, 0.02), (3.0, -0.5, -0.1), (25.0, 0.0, 0.0)],
)
def test_fit_reproduces_the_rollout(speed_mps: float, accel_mps2: float, curvature_per_m: float) -> None:
    config = ActionConfig(lateral_accel_actions=False, bounded=False)
    speed = torch.tensor([speed_mps])
    trajectory = rollout(_smooth_actions(accel_mps2, curvature_per_m, config), speed, config)[0]

    refit = rollout(fit_actions(trajectory, speed[0], config)[None], speed, config)[0]

    assert (refit[:, :2] - trajectory[:, :2]).norm(dim=-1).max() < 0.05
    assert (refit[:, 2] - trajectory[:, 2]).abs().max() < 0.01


def test_fit_at_standstill_is_zero() -> None:
    config = ActionConfig(lateral_accel_actions=False, bounded=False)
    actions = fit_actions(torch.zeros(_NUM_STEPS, 3), torch.tensor(0.0), config)
    assert torch.allclose(actions, torch.zeros_like(actions), atol=1e-4)


def test_zero_actions_hold_speed_and_heading() -> None:
    config = ActionConfig(
        accel_mean_mps2=0.0,
        curvature_mean_per_m=0.0,
        lateral_accel_actions=False,
        bounded=False,
    )
    poses = rollout(torch.zeros(1, _NUM_STEPS, 2), torch.tensor([8.0]), config)[0]
    time_s = (torch.arange(_NUM_STEPS) + 1) * STEP_S
    assert torch.allclose(poses[:, 0], 8.0 * time_s, atol=1e-4)
    assert torch.allclose(poses[:, 1:], torch.zeros(_NUM_STEPS, 2), atol=1e-6)


_LATERAL = ActionConfig(lateral_accel_actions=True, bounded=False)
_BOUNDED = ActionConfig(lateral_accel_actions=True, bounded=True)


def _physical(actions: torch.Tensor, speed_mps: float, config: ActionConfig) -> tuple[torch.Tensor, ...]:
    """Acceleration, curvature and the squared rolled-out speed at the start of each step."""
    acceleration = actions[..., 0] * config.accel_std_mps2 + config.accel_mean_mps2
    curvature = actions[..., 1] * config.curvature_std_per_m + config.curvature_mean_per_m
    speed = speed_mps + torch.cumsum(acceleration * STEP_S, dim=-1) - acceleration * STEP_S
    return acceleration, curvature, speed**2


@pytest.mark.parametrize("speed_mps", [0.0, 3.0, 12.0, 30.0])
def test_lateral_accel_state_inverts_to_the_actions(speed_mps: float) -> None:
    actions = _smooth_actions(1.0, 0.02, _LATERAL)
    speed = torch.tensor([speed_mps])
    torch.testing.assert_close(state_to_actions(actions_to_state(actions, speed, _LATERAL), speed, _LATERAL), actions)


@pytest.mark.parametrize(("speed_mps", "squared_speed"), [(2.0, 25.0), (20.0, 400.0)])
def test_lateral_accel_state_is_curvature_times_floored_squared_speed(speed_mps: float, squared_speed: float) -> None:
    curvature_per_m = 0.01
    actions = torch.stack(
        [
            torch.full((_NUM_STEPS,), -_LATERAL.accel_mean_mps2 / _LATERAL.accel_std_mps2),
            torch.full((_NUM_STEPS,), (curvature_per_m - _LATERAL.curvature_mean_per_m) / _LATERAL.curvature_std_per_m),
        ],
        dim=-1,
    )[None]
    state = actions_to_state(actions, torch.tensor([speed_mps]), _LATERAL)
    expected = torch.full((1, _NUM_STEPS), curvature_per_m * squared_speed / _LATERAL.lateral_accel_std_mps2)
    torch.testing.assert_close(state[..., 1], expected)


def test_bounded_state_keeps_actions_inside_half_the_bounds() -> None:
    state = torch.full((1, _NUM_STEPS, 2), 0.5)
    speed = torch.tensor([10.0])
    torch.testing.assert_close(state_to_actions(state, speed, _BOUNDED), state_to_actions(state, speed, _LATERAL))


@pytest.mark.parametrize("speed_mps", [0.0, 3.0, 12.0, 30.0])
def test_bounded_state_inverts_to_the_actions(speed_mps: float) -> None:
    actions = _smooth_actions(4.0, 0.005, _BOUNDED)
    speed = torch.tensor([speed_mps])
    torch.testing.assert_close(
        state_to_actions(actions_to_state(actions, speed, _BOUNDED), speed, _BOUNDED),
        actions,
        atol=1e-4,
        rtol=1e-4,
    )


@pytest.mark.parametrize("speed_mps", [0.0, 3.0, 30.0])
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_bounded_state_saturates_with_a_gradient(speed_mps: float, sign: float) -> None:
    state = torch.full((1, _NUM_STEPS, 2), 50.0 * sign, requires_grad=True)
    actions = state_to_actions(state, torch.tensor([speed_mps]), _BOUNDED)
    acceleration, curvature, squared_speed = _physical(actions, speed_mps, _BOUNDED)
    assert bool((acceleration >= -_BOUNDED.max_brake_mps2).all())
    assert bool((acceleration <= _BOUNDED.max_accel_mps2).all())
    assert bool((curvature.abs() <= _BOUNDED.max_lateral_accel_mps2 / 25.0 + 1e-6).all())
    assert bool((squared_speed * curvature.abs() <= _BOUNDED.max_lateral_accel_mps2 + 1e-3).all())

    actions.sum().backward()
    assert state.grad is not None
    assert bool(torch.isfinite(state.grad).all())
    assert bool((state.grad != 0).all())
