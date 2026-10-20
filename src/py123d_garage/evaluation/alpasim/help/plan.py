# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""The driver's plan in the rollout's local frame, and its sampling onto the serve grid of its own clock."""

from __future__ import annotations

import dataclasses
import math
from typing import Final

import jaxtyping as jt
import numpy as np
import numpy.typing as npt
from py123d.geometry import Point2DIndex, PoseSE2Index

from py123d_garage.datatypes.numerics import NonNegativeInt, PositiveInt

SERVE_INTERVAL_US: Final[int] = 100_000
SERVE_HORIZON_US: Final[int] = 5_000_000


@dataclasses.dataclass(frozen=True)
class CachedPlan:
    """One prediction fixed in the rollout's local frame; drive re-slices it every tick."""

    # Runtime time the plan was made at, which its first sample (the anchor pose) belongs to.
    created_time_us: int
    # Sample times in seconds after created_time_us, starting at 0.
    times_s: jt.Float64[npt.NDArray[np.float64], " num_samples"]
    # Sample positions in the local frame.
    positions_xy: jt.Float64[npt.NDArray[np.float64], "num_samples 2"]
    # Sample yaws in the local frame, in radians.
    yaws: jt.Float64[npt.NDArray[np.float64], " num_samples"]


@dataclasses.dataclass(frozen=True)
class PlanSamples:
    """The plan sampled on the serve grid from a request time on."""

    timestamps_us: jt.Int64[npt.NDArray[np.int64], " num_samples"]
    positions_xy: jt.Float64[npt.NDArray[np.float64], "num_samples 2"]
    yaws: jt.Float64[npt.NDArray[np.float64], " num_samples"]


def rig_to_local_xy(
    polyline_rig_xy: jt.Float64[npt.NDArray[np.float64], "num_points 2"],
    ego_pose_local: jt.Float64[npt.NDArray[np.float64], " 4"],
) -> jt.Float64[npt.NDArray[np.float64], "num_points 2"]:
    """
    Transforms points from the ego's rig frame into the local frame.

    Args:
        polyline_rig_xy: the points in the rig frame.
        ego_pose_local: the (x, y, z, yaw) of the ego in the local frame.

    Returns:
        the points in the local frame.
    """
    x, y, _, yaw = ego_pose_local
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=np.float64)
    return polyline_rig_xy @ rotation.T + [x, y]


def make_plan(
    poses_rig: jt.Float64[npt.NDArray[np.float64], "num_poses 3"],
    ego_pose_local: jt.Float64[npt.NDArray[np.float64], " 4"],
    time_now_us: NonNegativeInt,
    offsets_us: list[PositiveInt],
) -> CachedPlan:
    """
    Fixes the rig-frame prediction in the local frame, anchored at the current pose.

    Args:
        poses_rig: the planned (x, y, yaw) poses in the rig frame.
        ego_pose_local: the (x, y, z, yaw) of the ego in the local frame.
        time_now_us: the runtime's current time.
        offsets_us: the time offset of each planned pose.

    Returns:
        the cached plan.
    """
    origin_x, origin_y, _, origin_yaw = ego_pose_local
    positions_xy = np.vstack(
        [
            [[origin_x, origin_y]],
            rig_to_local_xy(poses_rig[:, : len(Point2DIndex)], ego_pose_local),
        ],
    )
    yaws = np.concatenate(
        [[origin_yaw], origin_yaw + poses_rig[:, PoseSE2Index.YAW]],
    )
    times_s = np.concatenate([[0.0], np.array(offsets_us, dtype=np.float64) * 1e-6])
    return CachedPlan(
        created_time_us=time_now_us,
        times_s=times_s,
        positions_xy=positions_xy,
        yaws=yaws,
    )


def sample_plan(plan: CachedPlan, time_now_us: int) -> PlanSamples | None:
    """
    Samples the plan from now on, on the 10 Hz grid of its own clock, so no served point moves between inferences.

    Args:
        plan: the cached plan.
        time_now_us: the runtime's current time.

    Returns:
        the samples, or None once fewer than two grid points of the plan remain.
    """
    serve_interval_s = SERVE_INTERVAL_US / 1e6
    elapsed_s = max(0.0, (time_now_us - plan.created_time_us) / 1e6)
    horizon_end_s = min(float(plan.times_s[-1]), elapsed_s + SERVE_HORIZON_US / 1e6)
    first_step = math.ceil(elapsed_s / serve_interval_s - 1e-9)
    last_step = math.floor(horizon_end_s / serve_interval_s + 1e-9)
    if last_step - first_step < 1:
        return None
    sample_times_s = np.arange(first_step, last_step + 1, dtype=np.float64) * serve_interval_s
    xs = np.interp(sample_times_s, plan.times_s, plan.positions_xy[:, 0])
    ys = np.interp(sample_times_s, plan.times_s, plan.positions_xy[:, 1])
    yaws = np.interp(sample_times_s, plan.times_s, np.unwrap(plan.yaws))
    timestamps_us = plan.created_time_us + np.round(sample_times_s * 1e6).astype(np.int64)
    return PlanSamples(timestamps_us=timestamps_us, positions_xy=np.stack([xs, ys], axis=1), yaws=yaws)
