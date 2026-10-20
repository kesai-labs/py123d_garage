# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import jaxtyping as jt
import numpy as np
import numpy.typing as npt
import torch
from py123d.geometry import PolylineSE2
from py123d.geometry.geometry_index import PoseSE2Index
from py123d.geometry.utils.rotation_utils import normalize_angle
from scipy.interpolate import interp1d
from torch import Tensor

from py123d_garage.datatypes.numerics import NonNegativeInt, PositiveFloat, WrappedAngleRadTensor, WrappedSE2Array

DEFAULT_MIN_YAW_DISPLACEMENT_M = 0.01


@dataclass(frozen=True, eq=False)
class TrajectorySE2:
    """Timestamped trajectory of SE2 poses (x, y, yaw), with the yaw wrapped into [-pi, pi)."""

    # One (x, y, yaw) pose per row.
    pose_se2_array: jt.Float64[npt.NDArray[np.float64], "num_poses 3"]
    # Absolute timestamp of each pose in microseconds.
    timestamps_us: jt.Int64[npt.NDArray[np.int64], " num_poses"]

    def __post_init__(self) -> None:
        """Wraps the yaw column, then freezes both arrays against later writes."""
        wrapped_pose_se2_array = self.pose_se2_array.copy()
        wrapped_pose_se2_array[:, PoseSE2Index.YAW] = normalize_angle(
            wrapped_pose_se2_array[:, PoseSE2Index.YAW],
        )
        wrapped_pose_se2_array.setflags(write=False)
        frozen_timestamps_us = self.timestamps_us.copy()
        frozen_timestamps_us.setflags(write=False)
        object.__setattr__(self, "pose_se2_array", wrapped_pose_se2_array)
        object.__setattr__(self, "timestamps_us", frozen_timestamps_us)

    def to_se2(self) -> TrajectorySE2:
        """
        Returns the trajectory as SE2, which it already is.

        Returns:
            this trajectory.
        """
        return self

    @property
    def polyline_se2(self) -> PolylineSE2:
        """The trajectory poses as a PolylineSE2."""
        return PolylineSE2.from_array(self.pose_se2_array)

    def interpolate(
        self,
        timestamp: int | np.int64 | npt.NDArray[np.int64],
        boundary_tolerance_us: NonNegativeInt = 0,
    ) -> WrappedSE2Array:
        """
        Interpolates the trajectory poses at the given timestamp(s).

        Args:
            timestamp: absolute timestamp(s) in microseconds to sample at.
            boundary_tolerance_us: how far outside the recorded range a timestamp may fall and still clamp to the
                boundary pose; 0 is strict.

        Returns:
            interpolated (x, y, yaw) pose array; a single-pose trajectory yields that pose for every timestamp.

        Raises:
            ValueError: if a timestamp falls outside the recorded range by more than the tolerance.
        """
        t_origin = self.timestamps_us[0]
        t_min_i, t_max_i = self.timestamps_us[0], self.timestamps_us[-1]

        query_i = np.asarray(timestamp, dtype=np.int64)
        if np.any((query_i < t_min_i - boundary_tolerance_us) | (query_i > t_max_i + boundary_tolerance_us)):
            raise ValueError(
                f"TrajectorySE2.interpolate received timestamps beyond the "
                f"{boundary_tolerance_us} µs tolerance around the recorded "
                f"range [{int(t_min_i)}, {int(t_max_i)}] µs.",
            )
        query_i = np.clip(query_i, t_min_i, t_max_i)
        if len(self.timestamps_us) == 1:
            return np.broadcast_to(self.pose_se2_array[0], (*query_i.shape, len(PoseSE2Index))).copy()

        timestamps_s = (self.timestamps_us - t_origin).astype(np.float64) * 1e-6
        query_s = (query_i - t_origin).astype(np.float64) * 1e-6

        unwrapped_pose_se2_array = self.pose_se2_array.copy()
        unwrapped_pose_se2_array[:, PoseSE2Index.YAW] = np.unwrap(
            unwrapped_pose_se2_array[:, PoseSE2Index.YAW],
            axis=0,
        )
        interpolator = interp1d(
            timestamps_s,
            unwrapped_pose_se2_array,
            axis=0,
            bounds_error=False,
            fill_value=0.0,
        )
        result = cast(npt.NDArray[np.float64], interpolator(query_s))
        result[..., PoseSE2Index.YAW] = normalize_angle(
            result[..., PoseSE2Index.YAW],
        )
        return result


@dataclass(frozen=True, eq=False)
class TrajectoryXY:
    """Timestamped trajectory of (x, y) positions in the ego frame."""

    # One (x, y) position per row.
    position_xy_array: jt.Float64[npt.NDArray[np.float64], "num_poses 2"]
    # Absolute timestamp of each position in microseconds.
    timestamps_us: jt.Int64[npt.NDArray[np.int64], " num_poses"]

    def __post_init__(self) -> None:
        """Freezes both arrays against later writes."""
        frozen_position_xy_array = self.position_xy_array.copy()
        frozen_position_xy_array.setflags(write=False)
        frozen_timestamps_us = self.timestamps_us.copy()
        frozen_timestamps_us.setflags(write=False)
        object.__setattr__(self, "position_xy_array", frozen_position_xy_array)
        object.__setattr__(self, "timestamps_us", frozen_timestamps_us)

    def to_se2(
        self,
        min_displacement_m: PositiveFloat = DEFAULT_MIN_YAW_DISPLACEMENT_M,
    ) -> TrajectorySE2:
        """
        Lifts the path into SE2, recovering yaw as the heading of the path.

        Args:
            min_displacement_m: shortest step whose heading is trusted; shorter steps keep the last trusted heading,
                or 0.0 before the first.

        Returns:
            the trajectory as SE2, with yaw derived from the path.
        """
        yaw_wrapped_rad = cast(
            npt.NDArray[np.float64],
            derive_yaw_from_positions(
                torch.from_numpy(self.position_xy_array),  # pyright: ignore[reportUnknownMemberType]
                min_displacement_m,
            )
            .numpy()  # pyright: ignore[reportUnknownMemberType]
            .astype(np.float64),
        )
        return TrajectorySE2(
            pose_se2_array=np.concatenate(
                [self.position_xy_array, yaw_wrapped_rad[:, None]],
                axis=1,
            ),
            timestamps_us=self.timestamps_us,
        )


def derive_yaw_from_positions(
    positions_xy: jt.Float[Tensor, "*batch poses 2"],
    min_displacement_m: PositiveFloat = DEFAULT_MIN_YAW_DISPLACEMENT_M,
) -> WrappedAngleRadTensor:
    """
    Recovers yaw as the heading of the path.

    Args:
        positions_xy: ego-frame (x, y) positions.
        min_displacement_m: shortest step whose heading is trusted; shorter steps keep the last trusted heading,
            or 0.0 before the first.

    Returns:
        the heading at every position.
    """
    origin_xy = torch.zeros_like(positions_xy[..., :1, :])
    previous_xy = torch.cat([origin_xy, positions_xy[..., :-1, :]], dim=-2)
    deltas_xy = positions_xy - previous_xy

    # atan2 reaches +pi but normalize_angle never does; wrap so both agree.
    step_yaw_wrapped_rad = torch.atan2(deltas_xy[..., 1], deltas_xy[..., 0])
    step_yaw_wrapped_rad = (step_yaw_wrapped_rad + torch.pi) % (2 * torch.pi) - torch.pi
    is_trusted = torch.hypot(deltas_xy[..., 0], deltas_xy[..., 1]) >= min_displacement_m

    indices = torch.arange(
        step_yaw_wrapped_rad.shape[-1],
        device=step_yaw_wrapped_rad.device,
    ).expand_as(is_trusted)
    last_trusted = torch.cummax(torch.where(is_trusted, indices, -1), dim=-1).values
    return torch.where(
        last_trusted >= 0,
        step_yaw_wrapped_rad.gather(-1, last_trusted.clamp(min=0)),
        torch.zeros_like(step_yaw_wrapped_rad),
    )
