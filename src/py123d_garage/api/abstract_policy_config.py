from __future__ import annotations

import abc
from dataclasses import dataclass

from omegaconf import MISSING
from py123d.datatypes import CameraID, LidarID

from py123d_garage.datatypes.numerics import NonNegativeInt, PositiveFloat, PositiveInt


@dataclass
class AbstractPolicyConfig(abc.ABC):
    """Policy protocol configuration."""

    # Planning horizon.
    trajectory_horizon_us: PositiveInt = MISSING

    # Planning interval.
    trajectory_interval_us: PositiveInt = MISSING

    # Distances along the intended route at which the target points are placed.
    required_target_point_distances_m: list[PositiveFloat] = MISSING

    @property
    @abc.abstractmethod
    def required_history_duration_us(self) -> NonNegativeInt:
        """How far back the policy requires its history to cover: ego state, camera and lidar. Zero means the policy only needs the current/anchor state."""

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
        """
        How many poses the predicted trajectory holds, derived from the physical fields.

        Returns:
            the number of poses, the first at trajectory_interval_us and the last at trajectory_horizon_us.

        Raises:
            ValueError: if the horizon is not a whole number of intervals.
        """
        if self.trajectory_horizon_us % self.trajectory_interval_us:
            raise ValueError(
                f"trajectory_horizon_us {self.trajectory_horizon_us} is not a multiple of "
                f"trajectory_interval_us {self.trajectory_interval_us}.",
            )
        return self.trajectory_horizon_us // self.trajectory_interval_us

    @property
    def trajectory_interval_s(self) -> PositiveFloat:
        """Time between consecutive poses of the predicted trajectory in seconds."""
        return self.trajectory_interval_us * 1e-6
