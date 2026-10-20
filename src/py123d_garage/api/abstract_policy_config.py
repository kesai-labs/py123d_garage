# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

from __future__ import annotations

import abc
from dataclasses import dataclass

from omegaconf import MISSING
from py123d.datatypes import CameraID, LidarID

from py123d_garage.datatypes.numerics import NonNegativeInt, PositiveFloat, PositiveInt


@dataclass
class AbstractPolicyConfig(abc.ABC):
    """Policy protocol configuration."""

    # Times past the anchor the predicted trajectory holds a pose at, strictly increasing.
    trajectory_offsets_us: list[PositiveInt] = MISSING

    # Distances along the intended route at which the target points are placed.
    required_target_point_distances_m: list[PositiveFloat] = MISSING

    @property
    @abc.abstractmethod
    def required_history_duration_us(self) -> NonNegativeInt:
        """How far back the policy reads ego state, camera and lidar; zero means only the anchor frame."""

    @property
    def required_cameras(self) -> dict[str, list[CameraID]]:
        """Cameras the policy ingests, per dataset; empty = the policy reads no camera."""
        return {}

    @property
    def required_lidars(self) -> dict[str, list[LidarID]]:
        """Lidars the policy ingests, per dataset; empty = the policy reads no lidar."""
        return {}

    @property
    def required_past_ego_state_interval_us(self) -> PositiveInt | None:
        """Spacing of the past ego states the policy consumes; None = current/anchor state only."""
        return None

    @property
    def required_past_camera_interval_us(self) -> PositiveInt | None:
        """Spacing of the past camera frames the policy consumes; None = current/anchor frame only."""
        return None

    @property
    def required_past_lidar_interval_us(self) -> PositiveInt | None:
        """Spacing of the past lidar sweeps the policy consumes; None = current/anchor sweep only."""
        return None

    @property
    def trajectory_num_steps(self) -> PositiveInt:
        """How many poses the predicted trajectory holds."""
        return len(self.trajectory_offsets_us)

    @property
    def trajectory_horizon_us(self) -> PositiveInt:
        """How far past the anchor the last predicted pose sits."""
        return self.trajectory_offsets_us[-1]


def uniform_offsets_us(
    interval_us: PositiveInt,
    horizon_us: PositiveInt,
    start_us: NonNegativeInt = 0,
) -> list[PositiveInt]:
    """
    Builds one evenly spaced block of offsets for trajectory_offsets_us.

    Args:
        interval_us: spacing between consecutive offsets.
        horizon_us: the last offset of the block.
        start_us: the offset the block starts after, defaults to 0.

    Returns:
        the block's offsets, the first at start_us + interval_us and the last at horizon_us.
    """
    return list(range(start_us + interval_us, horizon_us + 1, interval_us))
