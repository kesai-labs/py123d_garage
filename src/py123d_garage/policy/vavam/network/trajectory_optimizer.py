# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation
# Copyright (c) 2025 Brayden Zhang

"""
Smooths predicted trajectories with a classical optimizer under comfort limits.

Adapted from https://drive.google.com/file/d/1u-Hmpc304HySIXZwrptJnPQTT3fbC1Jz/view.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from py123d_garage.datatypes.numerics import WrappedSE2Array
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of trajectory optimization."""

    # Optimized trajectory [N, 3] of (x, y, heading), or the input trajectory on failure.
    trajectory: np.ndarray
    # Whether the optimizer converged.
    success: bool
    # Number of optimizer iterations; None on failure.
    iterations: int | None = None
    # Cost of the optimized trajectory; None on failure.
    final_cost: float | None = None
    # Optimizer or failure message.
    message: str = ""


@dataclass
class VehicleConstraints:
    """Kinematic bounds and the PDMS comfort limits the optimizer enforces."""

    # Maximum per-pose deviation from the original x and y in meters.
    max_deviation: float = 2.0
    # Maximum per-pose deviation from the original heading in radians (about 30 degrees).
    max_heading_change: float = 0.5236
    # Speed limit in m/s; not used by the optimizer.
    max_speed: float = 15.0
    # Acceleration limit in m/s^2; not used by the optimizer.
    max_accel: float = 5.0

    # Maximum absolute yaw rate in rad/s.
    max_abs_yaw_rate: float = 0.95
    # Maximum absolute yaw acceleration in rad/s^2.
    max_abs_yaw_acc: float = 1.93
    # Maximum longitudinal acceleration in m/s^2.
    max_lon_acc_pos: float = 4.89
    # Minimum (most negative) longitudinal acceleration in m/s^2.
    max_lon_acc_neg: float = -4.05
    # Maximum absolute longitudinal jerk in m/s^3.
    max_abs_lon_jerk: float = 8.37


class TrajectoryOptimizer:
    """Refines predicted trajectories for smoothness and comfort while staying close to them."""

    def __init__(
        self,
        smoothness_weight: float = 1.0,
        deviation_weight: float = 0.1,
        comfort_weight: float = 2.0,
        max_iterations: int = 100,
        enable_frenet_retiming: bool = True,
        retime_alpha: float = 0.25,
    ):
        """
        Initializes the trajectory optimizer.

        Args:
            smoothness_weight: weight of the second-difference smoothness cost.
            deviation_weight: weight of the deviation from the original trajectory.
            comfort_weight: weight of the soft comfort penalty.
            max_iterations: maximum optimizer iterations.
            enable_frenet_retiming: whether to start from waypoints redistributed along the path.
            retime_alpha: retiming strength in [0, 1]; higher front-loads more distance.
        """
        self.smoothness_weight = smoothness_weight
        self.deviation_weight = deviation_weight
        self.comfort_weight = comfort_weight
        self.max_iterations = max_iterations
        self.enable_frenet_retiming = bool(enable_frenet_retiming)
        self.retime_alpha = float(np.clip(retime_alpha, 0.0, 1.0))

    def optimize(
        self,
        trajectory: WrappedSE2Array,
        time_step: float = 0.5,
        vehicle_constraints: VehicleConstraints | None = None,
        retime_in_frenet: bool | None = None,
        retime_alpha: float | None = None,
    ) -> OptimizationResult:
        """
        Optimizes a trajectory for smoothness and comfort.

        Args:
            trajectory: original trajectory [N, 3] of (x, y, heading) in the ego frame.
            time_step: time between waypoints in seconds.
            vehicle_constraints: kinematic and comfort limits; None uses the defaults.
            retime_in_frenet: override of enable_frenet_retiming.
            retime_alpha: override of the retiming strength.

        Returns:
            the optimized trajectory with metadata, or the original one with success False.
        """
        if trajectory is None or len(trajectory) < 2:
            return OptimizationResult(
                trajectory=trajectory if trajectory is not None else np.array([]),
                success=False,
                message="Trajectory too short to optimize",
            )

        trajectory = np.asarray(trajectory, dtype=np.float64)

        if trajectory.shape[1] < 3:
            return OptimizationResult(
                trajectory=trajectory,
                success=False,
                message="Trajectory must have shape [N, 3] (x, y, heading)",
            )

        original_trajectory = trajectory.copy()
        constraints = vehicle_constraints or VehicleConstraints()

        use_retiming = (
            self.enable_frenet_retiming
            if retime_in_frenet is None
            else bool(retime_in_frenet)
        )
        alpha = (
            self.retime_alpha
            if retime_alpha is None
            else float(np.clip(retime_alpha, 0.0, 1.0))
        )

        if use_retiming:
            try:
                initial_guess = self._retime_along_path(
                    original_trajectory, alpha=alpha,
                )
            except Exception as e:
                logger.debug("Retiming failed, using original: %s", e)
                initial_guess = original_trajectory
        else:
            initial_guess = original_trajectory

        cost_fn = self._create_cost_function(
            original_trajectory, time_step, constraints,
        )

        bounds = self._create_bounds(original_trajectory, constraints)

        opt_constraints = self._create_constraints(
            original_trajectory, time_step, constraints, enforce_comfort=True,
        )

        x0 = initial_guess.flatten()

        try:
            # L-BFGS-B respects only the bounds; SLSQP with the constraints is the fallback.
            result = minimize(
                cost_fn,
                x0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": self.max_iterations, "ftol": 1e-6},
            )

            if not result.success:
                result = minimize(
                    cost_fn,
                    x0,
                    method="SLSQP",
                    bounds=bounds,
                    constraints=opt_constraints,
                    options={"maxiter": self.max_iterations, "ftol": 1e-6},
                )

            if result.success:
                optimized = result.x.reshape(-1, 3)
                return OptimizationResult(
                    trajectory=optimized,
                    success=True,
                    iterations=result.nit,
                    final_cost=float(result.fun),
                    message=str(result.message),
                )
            else:
                logger.warning("Optimization did not converge: %s", result.message)
                return OptimizationResult(
                    trajectory=original_trajectory,
                    success=False,
                    message=f"Optimization failed: {result.message}",
                )

        except Exception as e:
            logger.warning("Optimization error: %s", e)
            return OptimizationResult(
                trajectory=original_trajectory,
                success=False,
                message=f"Optimization error: {e}",
            )

    def _create_cost_function(
        self,
        original_trajectory: np.ndarray,
        time_step: float,
        constraints: VehicleConstraints,
    ):
        """
        Builds the weighted sum of smoothness, deviation and comfort costs.

        Args:
            original_trajectory: trajectory [N, 3] to stay close to.
            time_step: time between waypoints in seconds.
            constraints: comfort limits of the soft penalty.

        Returns:
            the cost of a flattened [N * 3] trajectory.
        """

        def cost_function(x: np.ndarray) -> float:
            """Returns the total cost of a flattened trajectory."""
            traj = x.reshape(-1, 3)

            smoothness_cost = self._compute_smoothness_cost(traj)

            deviation_cost = float(np.sum((traj - original_trajectory) ** 2))

            comfort_penalty = self._compute_comfort_penalty(
                traj, time_step, constraints,
            )

            total = (
                self.smoothness_weight * smoothness_cost
                + self.deviation_weight * deviation_cost
                + self.comfort_weight * comfort_penalty
            )
            return total

        return cost_function

    def _compute_smoothness_cost(self, trajectory: np.ndarray) -> float:
        """
        Sums the squared second differences of position and wrapped heading.

        Args:
            trajectory: trajectory [N, 3] of (x, y, heading).

        Returns:
            the smoothness cost.
        """
        if trajectory.shape[0] < 3:
            return 0.0

        p = trajectory[:, :2]
        ddp = p[2:] - 2 * p[1:-1] + p[:-2]
        pos_smooth = float(np.sum(np.linalg.norm(ddp, axis=1) ** 2))

        yaw = trajectory[:, 2]
        dyaw1 = np.arctan2(np.sin(yaw[1:] - yaw[:-1]), np.cos(yaw[1:] - yaw[:-1]))
        if len(dyaw1) < 2:
            return pos_smooth
        dyaw2 = np.arctan2(
            np.sin(dyaw1[1:] - dyaw1[:-1]), np.cos(dyaw1[1:] - dyaw1[:-1]),
        )
        yaw_smooth = float(np.sum(dyaw2**2))

        return pos_smooth + yaw_smooth

    def _compute_comfort_penalty(
        self,
        trajectory: np.ndarray,
        dt: float,
        constraints: VehicleConstraints,
    ) -> float:
        """
        Penalizes squared exceedances of the comfort limits.

        Args:
            trajectory: trajectory [N, 3] of (x, y, heading).
            dt: time between waypoints in seconds.
            constraints: comfort limits.

        Returns:
            the comfort penalty, 0 when every limit holds.
        """
        if trajectory.shape[0] < 4:
            return 0.0

        yaw_rate, yaw_acc, lon_a, lon_j = self._compute_kinematics(trajectory, dt)

        def over_abs(arr: np.ndarray, limit: float) -> np.ndarray:
            """Returns how far the absolute values exceed the limit."""
            return np.maximum(0.0, np.abs(arr) - limit)

        def over_upper(arr: np.ndarray, limit: float) -> np.ndarray:
            """Returns how far the values exceed the upper limit."""
            return np.maximum(0.0, arr - limit)

        def over_lower(arr: np.ndarray, limit: float) -> np.ndarray:
            """Returns how far the values fall below the lower limit."""
            return np.maximum(0.0, limit - arr)

        p_yaw_rate = over_abs(yaw_rate, constraints.max_abs_yaw_rate)
        p_yaw_acc = over_abs(yaw_acc, constraints.max_abs_yaw_acc)
        p_lon_a = np.sqrt(
            over_upper(lon_a, constraints.max_lon_acc_pos) ** 2
            + over_lower(lon_a, constraints.max_lon_acc_neg) ** 2,
        )
        p_lon_j = over_abs(lon_j, constraints.max_abs_lon_jerk)

        penalty = (
            np.mean(p_yaw_rate**2)
            + np.mean(p_yaw_acc**2)
            + 1.5 * np.mean(p_lon_a**2)
            + 1.5 * np.mean(p_lon_j**2)
        )
        return float(penalty)

    def _compute_kinematics(
        self, trajectory: np.ndarray, dt: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Differentiates a trajectory into yaw and longitudinal kinematics.

        Args:
            trajectory: trajectory [N, 3] of (x, y, heading).
            dt: time between waypoints in seconds, floored at 1 ms.

        Returns:
            the yaw rate, yaw acceleration, longitudinal acceleration and longitudinal jerk, each [N].
        """
        dt = max(float(dt), 1e-3)
        x = trajectory[:, 0]
        y = trajectory[:, 1]
        yaw = trajectory[:, 2]

        vx = np.gradient(x, dt)
        vy = np.gradient(y, dt)

        lon_v = vx * np.cos(yaw) + vy * np.sin(yaw)
        lon_a = np.gradient(lon_v, dt)
        lon_j = np.gradient(lon_a, dt)

        yaw_rate = np.gradient(np.unwrap(yaw), dt)
        yaw_acc = np.gradient(yaw_rate, dt)

        return yaw_rate, yaw_acc, lon_a, lon_j

    def _create_bounds(
        self,
        original_trajectory: np.ndarray,
        constraints: VehicleConstraints,
    ) -> List[Tuple[float, float]]:
        """
        Bounds every pose to a box around the original pose.

        Args:
            original_trajectory: trajectory [N, 3] to stay close to.
            constraints: the allowed position and heading deviation.

        Returns:
            the (low, high) bounds of the flattened [N * 3] variables.
        """
        bounds = []
        for i in range(len(original_trajectory)):
            bounds.extend(
                [
                    (
                        original_trajectory[i, 0] - constraints.max_deviation,
                        original_trajectory[i, 0] + constraints.max_deviation,
                    ),
                    (
                        original_trajectory[i, 1] - constraints.max_deviation,
                        original_trajectory[i, 1] + constraints.max_deviation,
                    ),
                    (
                        original_trajectory[i, 2] - constraints.max_heading_change,
                        original_trajectory[i, 2] + constraints.max_heading_change,
                    ),
                ],
            )
        return bounds

    def _create_constraints(
        self,
        original_trajectory: np.ndarray,
        time_step: float,
        constraints: VehicleConstraints,
        enforce_comfort: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Builds the SLSQP constraints: fixed start and end poses and, optionally, hard comfort limits.

        Args:
            original_trajectory: trajectory [N, 3] whose endpoints stay fixed.
            time_step: time between waypoints in seconds.
            constraints: comfort limits.
            enforce_comfort: whether to add the comfort limits as inequality constraints.

        Returns:
            the scipy constraint dicts.
        """
        opt_constraints = []

        opt_constraints.append(
            {"type": "eq", "fun": lambda x: x[:3] - original_trajectory[0]},
        )

        opt_constraints.append(
            {"type": "eq", "fun": lambda x: x[-3:] - original_trajectory[-1]},
        )

        if enforce_comfort:

            def comfort_margins(z: np.ndarray) -> np.ndarray:
                """Returns the margin to each comfort limit, non-negative when it holds."""
                traj = z.reshape(-1, 3)
                yaw_rate, yaw_acc, lon_a, lon_j = self._compute_kinematics(
                    traj, time_step,
                )
                margins = [
                    constraints.max_abs_yaw_rate - np.max(np.abs(yaw_rate)),
                    constraints.max_abs_yaw_acc - np.max(np.abs(yaw_acc)),
                    constraints.max_lon_acc_pos - np.max(lon_a),
                    np.min(lon_a) - constraints.max_lon_acc_neg,
                    constraints.max_abs_lon_jerk - np.max(np.abs(lon_j)),
                ]
                return np.array(margins, dtype=np.float64)

            opt_constraints.append({"type": "ineq", "fun": comfort_margins})

        return opt_constraints

    def _polyline_arclen(self, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the cumulative arc length along a 2D polyline.

        Args:
            pts: polyline points [N, 2].

        Returns:
            the arc length at every point [N] and the segment lengths [N - 1].
        """
        if pts.shape[0] < 2:
            return np.zeros((pts.shape[0],), dtype=np.float64), np.zeros(
                (0,), dtype=np.float64,
            )
        diffs = pts[1:] - pts[:-1]
        seg_len = np.hypot(diffs[:, 0], diffs[:, 1])
        s = np.zeros((pts.shape[0],), dtype=np.float64)
        s[1:] = np.cumsum(seg_len)
        return s, seg_len

    def _sample_polyline_by_s(
        self, pts: np.ndarray, s: np.ndarray, s_query: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Samples positions and segment headings of a 2D polyline at query arc lengths.

        Args:
            pts: polyline points [N, 2].
            s: arc length at every point [N].
            s_query: arc lengths to sample [M], clipped to the polyline.

        Returns:
            the sampled positions [M, 2] and headings [M].
        """
        N = pts.shape[0]
        if N == 0:
            return np.zeros((0, 2)), np.zeros((0,))

        total_s = s[-1] if len(s) > 0 else 0.0
        if total_s <= 1e-6:
            xy = np.repeat(pts[:1], repeats=len(s_query), axis=0)
            yaw = np.zeros((len(s_query),), dtype=np.float64)
            if N >= 2:
                d = pts[1] - pts[0]
                if np.hypot(d[0], d[1]) > 1e-9:
                    yaw[:] = np.arctan2(d[1], d[0])
            return xy, yaw

        xy = np.zeros((len(s_query), 2), dtype=np.float64)
        yaw = np.zeros((len(s_query),), dtype=np.float64)
        i = 0

        for k, sq in enumerate(s_query):
            sq = float(np.clip(sq, 0.0, total_s))
            while i < N - 2 and s[i + 1] < sq:
                i += 1

            ds = s[i + 1] - s[i]
            if ds <= 1e-12:
                j = i
                while j < N - 2 and (s[j + 1] - s[j]) <= 1e-12:
                    j += 1
                dvec = pts[j + 1] - pts[j] if j < N - 1 else pts[i + 1] - pts[i]
                xy[k] = pts[i]
                yaw[k] = (
                    np.arctan2(dvec[1], dvec[0])
                    if np.hypot(dvec[0], dvec[1]) > 1e-12
                    else (yaw[k - 1] if k > 0 else 0.0)
                )
            else:
                u = (sq - s[i]) / ds
                p0, p1 = pts[i], pts[i + 1]
                dvec = p1 - p0
                xy[k] = p0 + u * dvec
                yaw[k] = np.arctan2(dvec[1], dvec[0])

        return xy, yaw

    def _retime_along_path(
        self, trajectory: np.ndarray, alpha: float = 0.25,
    ) -> np.ndarray:
        """
        Redistributes the waypoints along the path with an ease-out warp, keeping the endpoints fixed.

        Args:
            trajectory: trajectory [N, 3] of (x, y, yaw).
            alpha: retiming strength in [0, 1]; higher front-loads more distance.

        Returns:
            the retimed trajectory [N, 3].
        """
        if trajectory is None or trajectory.shape[0] < 2:
            return trajectory

        pts = np.ascontiguousarray(trajectory[:, :2], dtype=np.float64)
        s, _ = self._polyline_arclen(pts)
        total = s[-1] if len(s) > 0 else 0.0

        if total <= 1e-6:
            return trajectory

        N = trajectory.shape[0]
        t = np.linspace(0.0, 1.0, N, dtype=np.float64)

        beta = 1.0 + 4.0 * float(np.clip(alpha, 0.0, 1.0))
        w = 1.0 - np.power(1.0 - t, beta)
        s_new = total * w

        xy_new, yaw_new = self._sample_polyline_by_s(pts, s, s_new)

        out = trajectory.copy()
        out[:, 0:2] = xy_new
        out[:, 2] = yaw_new

        out[0] = trajectory[0]
        out[-1] = trajectory[-1]

        return out


def add_heading_to_trajectory(xy_trajectory: np.ndarray) -> WrappedSE2Array:
    """
    Adds a heading column from consecutive position deltas, the first measured from the ego origin.

    Args:
        xy_trajectory: positions [N, 2].

    Returns:
        the trajectory [N, 3] of (x, y, heading), heading wrapped to [-pi, pi).
    """
    if xy_trajectory is None or len(xy_trajectory) == 0:
        return np.zeros((0, 3), dtype=np.float64)

    xy = np.asarray(xy_trajectory, dtype=np.float64)
    if xy.ndim == 1:
        xy = xy.reshape(1, -1)

    N = xy.shape[0]
    headings = np.zeros(N, dtype=np.float64)

    prev_pos = np.array([0.0, 0.0])
    for i in range(N):
        delta = xy[i] - prev_pos
        dist = np.hypot(delta[0], delta[1])
        if dist > 1e-4:
            headings[i] = np.arctan2(delta[1], delta[0])
        else:
            headings[i] = headings[i - 1] if i > 0 else 0.0
        prev_pos = xy[i]

    # arctan2 reaches +pi; wrap to [-pi, pi) so the heading matches the garage convention.
    headings = (headings + np.pi) % (2 * np.pi) - np.pi
    return np.column_stack([xy, headings])
