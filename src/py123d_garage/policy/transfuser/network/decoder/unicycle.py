# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI
# Adapted from NVlabs/alpamayo (Apache License 2.0).

"""
Unicycle acceleration-curvature actions: the regularized fit to a trajectory and the rollout back to poses.
"""

from __future__ import annotations

from typing import Final, cast

import jaxtyping as jt
import torch
from py123d.geometry import Point2DIndex, PoseSE2Index

from py123d_garage.config.schema.policy.transfuser_config import ActionConfig

STEP_S: Final[float] = 0.1

_YAW_SMOOTHING: Final[float] = 1e-6
_YAW_RIDGE: Final[float] = 1e-8
_SPEED_SMOOTHING: Final[float] = 1e-6
_SPEED_RIDGE: Final[float] = 1e-4
_ACCEL_SMOOTHING: Final[float] = 1e-4
_ACCEL_RIDGE: Final[float] = 1e-4
_CURVATURE_SMOOTHING: Final[float] = 1e-4
_CURVATURE_RIDGE: Final[float] = 1e-4
_CURVATURE_BOUND_PER_M: Final[float] = 0.33
# Below this speed the heading change says nothing about the curvature.
_LOW_SPEED_CURVATURE_THRESHOLD_MPS: Final[float] = 0.6
# AlpaSim's MPC switches to its kinematic model below this speed, so the lateral acceleration uses at least its square.
_LATERAL_ACCEL_FLOOR_SPEED_MPS: Final[float] = 5.0
# Bounded labels at or past a bound unsquash to this fraction of it.
_MAX_BOUND_FRACTION: Final[float] = 0.99


@torch.no_grad()
def fit_actions(
    trajectory: jt.Float[torch.Tensor, "poses 3"],
    speed_mps: jt.Float[torch.Tensor, ""],
    config: ActionConfig,
) -> jt.Float[torch.Tensor, "poses 2"]:
    """
    Fits the acceleration and curvature whose unicycle rollout from the anchor follows the trajectory.

    Args:
        trajectory: future ego poses relative to the anchor, one every STEP_S.
        speed_mps: ego speed at the anchor, where the fitted speed starts.
        config: normalization of the actions.

    Returns:
        normalized (acceleration, curvature) per step.
    """
    trajectory = trajectory.double()
    positions = torch.cat([trajectory.new_zeros(1, len(Point2DIndex)), trajectory[:, : len(Point2DIndex)]])
    yaw_steps = (
        torch.remainder(
            torch.diff(trajectory[:, PoseSE2Index.YAW], prepend=trajectory.new_zeros(1)) + torch.pi,
            2 * torch.pi,
        )
        - torch.pi
    )
    yaw = _solve_anchored(torch.cumsum(yaw_steps, dim=0), _YAW_SMOOTHING, _YAW_RIDGE, smoothing_order=3)
    speed = _fit_speed(torch.diff(positions, dim=0), yaw, speed_mps.double())
    acceleration = _solve_scaled(
        torch.ones_like(speed[1:]),
        torch.diff(speed) / STEP_S,
        _ACCEL_SMOOTHING,
        _ACCEL_RIDGE,
    )
    curvature = _solve_scaled(
        speed[:-1] * STEP_S + acceleration * STEP_S**2 / 2,
        torch.diff(yaw),
        _CURVATURE_SMOOTHING,
        _CURVATURE_RIDGE,
    ).clamp(-_CURVATURE_BOUND_PER_M, _CURVATURE_BOUND_PER_M)
    low_speed = (speed[:-1].abs() < _LOW_SPEED_CURVATURE_THRESHOLD_MPS) | (
        speed[1:].abs() < _LOW_SPEED_CURVATURE_THRESHOLD_MPS
    )
    curvature = curvature.masked_fill(low_speed, 0.0)
    return torch.stack(
        [
            (acceleration - config.accel_mean_mps2) / config.accel_std_mps2,
            (curvature - config.curvature_mean_per_m) / config.curvature_std_per_m,
        ],
        dim=-1,
    ).float()


def rollout(
    actions: jt.Float[torch.Tensor, "batch poses 2"],
    speed_mps: jt.Float[torch.Tensor, " batch"],
    config: ActionConfig,
) -> jt.Float[torch.Tensor, "batch poses 3"]:
    """
    Integrates normalized actions with unicycle dynamics from the anchor.

    Args:
        actions: normalized (acceleration, curvature) per step.
        speed_mps: ego speed at the anchor.
        config: normalization of the actions.

    Returns:
        ego poses relative to the anchor, (x, y, yaw) per step.
    """
    acceleration = actions[..., 0].float() * config.accel_std_mps2 + config.accel_mean_mps2
    curvature = actions[..., 1].float() * config.curvature_std_per_m + config.curvature_mean_per_m
    start_speed = speed_mps.float()[:, None]
    speed = torch.cat([start_speed, start_speed + torch.cumsum(acceleration * STEP_S, dim=-1)], dim=-1)
    yaw = torch.cat(
        [
            torch.zeros_like(start_speed),
            torch.cumsum(curvature * (speed[:, :-1] * STEP_S + acceleration * STEP_S**2 / 2), dim=-1),
        ],
        dim=-1,
    )
    x = torch.cumsum((speed[:, :-1] * yaw[:, :-1].cos() + speed[:, 1:] * yaw[:, 1:].cos()) * STEP_S / 2, dim=-1)
    y = torch.cumsum((speed[:, :-1] * yaw[:, :-1].sin() + speed[:, 1:] * yaw[:, 1:].sin()) * STEP_S / 2, dim=-1)
    return torch.stack([x, y, yaw[:, 1:]], dim=-1)


def physical_actions(
    actions: jt.Float[torch.Tensor, "batch poses 2"],
    speed_mps: jt.Float[torch.Tensor, " batch"],
    config: ActionConfig,
) -> jt.Float[torch.Tensor, "batch poses 2"]:
    """
    Converts normalized actions to the physical acceleration and lateral acceleration.

    Args:
        actions: normalized (acceleration, curvature) per step.
        speed_mps: ego speed at the anchor.
        config: normalization of the actions.

    Returns:
        (acceleration, lateral acceleration) per step, in m/s^2.
    """
    acceleration = actions[..., 0].float() * config.accel_std_mps2 + config.accel_mean_mps2
    curvature = actions[..., 1].float() * config.curvature_std_per_m + config.curvature_mean_per_m
    return torch.stack([acceleration, curvature * _lateral_accel_per_curvature(acceleration, speed_mps)], dim=-1)


def actions_to_state(
    actions: jt.Float[torch.Tensor, "batch poses 2"],
    speed_mps: jt.Float[torch.Tensor, " batch"],
    config: ActionConfig,
) -> jt.Float[torch.Tensor, "batch poses 2"]:
    """
    Converts normalized actions to the flow state, which holds the lateral acceleration if lateral_accel_actions.

    Args:
        actions: normalized (acceleration, curvature) per step.
        speed_mps: ego speed at the anchor.
        config: normalization of the actions.

    Returns:
        the normalized flow state per step.
    """
    if not config.lateral_accel_actions:
        return actions
    acceleration, lateral_acceleration = physical_actions(actions, speed_mps, config).unbind(-1)
    if config.bounded:
        acceleration = _unsquash(acceleration, _accel_bound(acceleration, config))
        lateral_acceleration = _unsquash(
            lateral_acceleration,
            torch.full_like(lateral_acceleration, config.max_lateral_accel_mps2),
        )
    return torch.stack(
        [
            (acceleration - config.accel_mean_mps2) / config.accel_std_mps2,
            lateral_acceleration / config.lateral_accel_std_mps2,
        ],
        dim=-1,
    )


def state_to_actions(
    state: jt.Float[torch.Tensor, "batch poses 2"],
    speed_mps: jt.Float[torch.Tensor, " batch"],
    config: ActionConfig,
) -> jt.Float[torch.Tensor, "batch poses 2"]:
    """
    Converts a flow state to normalized actions, the inverse of actions_to_state.

    Args:
        state: normalized flow state per step.
        speed_mps: ego speed at the anchor.
        config: normalization and bounds of the actions.

    Returns:
        the normalized (acceleration, curvature) per step.
    """
    if not config.lateral_accel_actions:
        return state
    acceleration = state[..., 0].float() * config.accel_std_mps2 + config.accel_mean_mps2
    lateral_acceleration = state[..., 1].float() * config.lateral_accel_std_mps2
    if config.bounded:
        acceleration = _squash(acceleration, _accel_bound(acceleration, config))
        lateral_acceleration = _squash(
            lateral_acceleration,
            torch.full_like(lateral_acceleration, config.max_lateral_accel_mps2),
        )
    curvature = lateral_acceleration / _lateral_accel_per_curvature(acceleration, speed_mps)
    return torch.stack(
        [
            (acceleration - config.accel_mean_mps2) / config.accel_std_mps2,
            (curvature - config.curvature_mean_per_m) / config.curvature_std_per_m,
        ],
        dim=-1,
    )


def _squash(
    value: jt.Float[torch.Tensor, "batch poses"],
    bound: jt.Float[torch.Tensor, "batch poses"],
) -> jt.Float[torch.Tensor, "batch poses"]:
    """
    Squashes values into their bound, unchanged up to half the bound and then a softsign tail.

    The tail's gradient decays polynomially, so it does not underflow to zero like tanh's.

    Args:
        value: values to squash.
        bound: positive bound of each value.

    Returns:
        the squashed values, strictly inside the bound.
    """
    knee = bound / 2
    excess = (value.abs() - knee).clamp(min=0)
    return value - value.sign() * excess**2 / (knee + excess)


def _unsquash(
    value: jt.Float[torch.Tensor, "batch poses"],
    bound: jt.Float[torch.Tensor, "batch poses"],
) -> jt.Float[torch.Tensor, "batch poses"]:
    """
    Inverts _squash after clipping the values to 0.99 of their bound.

    Args:
        value: squashed values.
        bound: positive bound of each value.

    Returns:
        the unsquashed values.
    """
    value = value.clamp(min=-_MAX_BOUND_FRACTION * bound, max=_MAX_BOUND_FRACTION * bound)
    knee = bound / 2
    excess = (value.abs() - knee).clamp(min=0)
    return value + value.sign() * excess**2 / (knee - excess)


def _accel_bound(
    acceleration: jt.Float[torch.Tensor, "batch poses"],
    config: ActionConfig,
) -> jt.Float[torch.Tensor, "batch poses"]:
    """
    Picks the acceleration bound, max_brake_mps2 for a deceleration and max_accel_mps2 otherwise.

    Args:
        acceleration: acceleration per step, in m/s^2.
        config: bounds of the actions.

    Returns:
        the bound of every step.
    """
    return torch.where(
        acceleration < 0,
        torch.full_like(acceleration, config.max_brake_mps2),
        torch.full_like(acceleration, config.max_accel_mps2),
    )


def _lateral_accel_per_curvature(
    acceleration: jt.Float[torch.Tensor, "batch poses"],
    speed_mps: jt.Float[torch.Tensor, " batch"],
) -> jt.Float[torch.Tensor, "batch poses"]:
    """
    Computes the factor from curvature to lateral acceleration, the squared rolled-out speed at each step's start.

    Args:
        acceleration: acceleration per step, in m/s^2.
        speed_mps: ego speed at the anchor.

    Returns:
        the squared speed per step, at least the floor speed squared.
    """
    speed = speed_mps.float()[:, None] + torch.cumsum(acceleration * STEP_S, dim=-1) - acceleration * STEP_S
    return (speed**2).clamp(min=_LATERAL_ACCEL_FLOOR_SPEED_MPS**2)


def _smoothness_gram(
    num_values: int,
    smoothing: float,
    order: int,
    reference: torch.Tensor,
) -> jt.Float64[torch.Tensor, "values values"]:
    """
    Builds the penalty matrix of the squared order-th derivative, sum_i smoothing * (d^order x_i / dt^order)^2.

    Args:
        num_values: length of the solved sequence.
        smoothing: weight of the penalty.
        order: order of the finite difference.
        reference: tensor whose dtype and device the matrix takes.

    Returns:
        the symmetric matrix D^T D, scaled by smoothing / STEP_S^(2 order).
    """
    difference = torch.eye(num_values, dtype=reference.dtype, device=reference.device)
    for _ in range(order):
        difference = torch.diff(difference, dim=0)
    return smoothing / STEP_S ** (2 * order) * difference.T @ difference


def _solve_anchored(
    target: jt.Float64[torch.Tensor, " values"],
    smoothing: float,
    ridge: float,
    smoothing_order: int,
) -> jt.Float64[torch.Tensor, " values_plus_one"]:
    """
    Smooths a sequence that starts from 0: min_x |x - target|^2 + smoothing penalty, with x_0 = 0 fixed.

    Args:
        target: the values after the fixed zero start.
        smoothing: weight of the smoothness penalty.
        ridge: weight of the |x|^2 regularizer.
        smoothing_order: order of the penalized finite difference.

    Returns:
        the smoothed sequence, the fixed zero first.
    """
    num_values = target.shape[0]
    gram = _smoothness_gram(num_values + 1, smoothing, smoothing_order, target)[1:, 1:]
    identity = torch.eye(num_values, dtype=target.dtype, device=target.device)
    solved = cast(
        torch.Tensor,
        torch.linalg.solve(identity * (1.0 + ridge) + gram, target),  # pyright: ignore[reportUnknownMemberType]
    )
    return torch.cat([target.new_zeros(1), solved])


def _solve_scaled(
    scale: jt.Float64[torch.Tensor, " values"],
    target: jt.Float64[torch.Tensor, " values"],
    smoothing: float,
    ridge: float,
) -> jt.Float64[torch.Tensor, " values"]:
    """
    Solves min_x |scale * x - target|^2 + smoothing |second difference of x|^2 + ridge |x|^2.

    Args:
        scale: per-value factor of the data term.
        target: per-value target of the data term.
        smoothing: weight of the second-order smoothness penalty.
        ridge: weight of the |x|^2 regularizer.

    Returns:
        the solved values.
    """
    num_values = target.shape[0]
    identity = torch.eye(num_values, dtype=target.dtype, device=target.device)
    lhs = torch.diag(scale**2) + _smoothness_gram(num_values, smoothing, 2, target) + ridge * identity
    return cast(torch.Tensor, torch.linalg.solve(lhs, scale * target))  # pyright: ignore[reportUnknownMemberType]


def _fit_speed(
    steps: jt.Float64[torch.Tensor, "values 2"],
    yaw: jt.Float64[torch.Tensor, " values_plus_one"],
    speed_mps: jt.Float64[torch.Tensor, ""],
) -> jt.Float64[torch.Tensor, " values_plus_one"]:
    """
    Fits the speeds whose trapezoidal integration along the yaw gives the position steps, the first speed fixed.

    Args:
        steps: position differences between consecutive poses, the anchor first.
        yaw: smoothed yaw of every pose, the anchor's 0 first.
        speed_mps: speed at the anchor.

    Returns:
        the speed of every pose, the anchor first.
    """
    num_steps = steps.shape[0]
    headings = torch.stack([yaw.cos(), yaw.sin()], dim=-1)
    design = steps.new_zeros(num_steps, 2, num_steps + 1)
    step_indices = torch.arange(num_steps, device=steps.device)
    design[step_indices, :, step_indices] = headings[:-1]
    design[step_indices, :, step_indices + 1] = headings[1:]
    design = design.reshape(2 * num_steps, num_steps + 1)
    gram = design.T @ design + _smoothness_gram(num_steps + 1, _SPEED_SMOOTHING, 3, steps)
    rhs = design[:, 1:].T @ (2 / STEP_S * steps).reshape(-1) - gram[1:, 0] * speed_mps
    identity = torch.eye(num_steps, dtype=steps.dtype, device=steps.device)
    solved = cast(
        torch.Tensor,
        torch.linalg.solve(gram[1:, 1:] + _SPEED_RIDGE * identity, rhs),  # pyright: ignore[reportUnknownMemberType]
    )
    return torch.cat([speed_mps[None], solved])
