# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""The sampling grid PDM scores trajectories and proposals on."""

from __future__ import annotations

from dataclasses import dataclass

from py123d_garage.datatypes.numerics import PositiveFloat, PositiveInt


@dataclass(frozen=True)
class TrajectoryGrid:
    """
    The timeline a trajectory is sampled on.

    The first pose is at interval_us and the last at horizon_us.
    """

    # Number of poses, excluding the current pose.
    num_poses: PositiveInt
    # Time of the last pose after the current one, in microseconds.
    horizon_us: PositiveInt
    # Time between consecutive poses, in microseconds.
    interval_us: PositiveInt

    def __post_init__(self) -> None:
        """Rejects a grid whose three numbers disagree or are not positive."""
        if min(self.num_poses, self.horizon_us, self.interval_us) < 1:
            raise ValueError(
                f"num_poses {self.num_poses}, horizon_us {self.horizon_us} and interval_us {self.interval_us} must be positive.",
            )
        if self.num_poses * self.interval_us != self.horizon_us:
            raise ValueError(
                f"horizon_us {self.horizon_us} is not num_poses {self.num_poses} * interval_us {self.interval_us}.",
            )

    @classmethod
    def from_horizon_and_interval(cls, horizon_us: PositiveInt, interval_us: PositiveInt) -> TrajectoryGrid:
        """
        Builds the grid covering horizon_us with poses interval_us apart.

        Args:
            horizon_us: Time of the last pose, in microseconds.
            interval_us: Time between consecutive poses, in microseconds.

        Returns:
            The grid with num_poses deduced.
        """
        if horizon_us % interval_us:
            raise ValueError(
                f"horizon_us {horizon_us} is not a multiple of interval_us {interval_us}.",
            )
        return cls(
            num_poses=horizon_us // interval_us,
            horizon_us=horizon_us,
            interval_us=interval_us,
        )

    @classmethod
    def from_num_poses_and_interval(cls, num_poses: PositiveInt, interval_us: PositiveInt) -> TrajectoryGrid:
        """
        Builds the grid of num_poses poses interval_us apart.

        Args:
            num_poses: Number of poses.
            interval_us: Time between consecutive poses, in microseconds.

        Returns:
            The grid with horizon_us deduced.
        """
        return cls(
            num_poses=num_poses,
            horizon_us=num_poses * interval_us,
            interval_us=interval_us,
        )

    @property
    def interval_s(self) -> PositiveFloat:
        """
        Time between consecutive poses.

        Returns:
            The interval in seconds.
        """
        return self.interval_us * 1e-6
