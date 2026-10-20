# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

from __future__ import annotations

from dataclasses import dataclass, field

from py123d_garage.api.abstract_offline_data_source_config import OfflineEvaluationDataSourceConfig
from py123d_garage.datatypes.numerics import NonNegativeFloat, NonNegativeInt, PositiveFloat, PositiveInt


@dataclass
class AbstractBenchmarkConfig:
    """Benchmark protocol configuration."""

    # Global RNG seed.
    seed: int = 0

    # Offline data sources whose scenes the benchmark scores.
    benchmark_offline_data_sources: list[OfflineEvaluationDataSourceConfig] = field(
        default_factory=list,
    )

    @property
    def required_trajectory_horizon_us(self) -> PositiveInt | None:
        """Future window the benchmark scores, which the policy's trajectory must cover; None = no fixed window."""
        return None

    @property
    def max_history_duration_us(self) -> NonNegativeInt | None:
        """Farthest past the benchmark lets a policy read; None = no limit."""
        return None

    @property
    def scored_scene_future_duration_us(self) -> PositiveInt | None:
        """Logged future a scored scene needs past its anchor; None = the policy's trajectory horizon, for its labels."""
        return None

    @property
    def scored_scene_needs_route(self) -> bool:
        """
        Whether every scored scene must carry route out to the policy's farthest target point.

        False serves the target points past the route's end as NaN, which the policy reads as no navigation intent.
        """
        return True

    @property
    def min_served_target_point_distance_m(self) -> NonNegativeFloat | None:
        """Nearest distance the benchmark can place a target point at; None = no limit."""
        return None

    @property
    def max_served_target_point_distance_m(self) -> PositiveFloat | None:
        """Farthest distance the benchmark can place a target point at; None = no limit."""
        return None

    @property
    def served_ego_state_interval_us(self) -> PositiveInt | None:
        """Time interval the benchmark's protocol serves past ego states on; None = declared per data source or ego state is absent."""
        return None

    @property
    def served_camera_interval_us(self) -> PositiveInt | None:
        """Time interval the benchmark's protocol serves camera frames on; None = declared per data source or camera is absent."""
        return None

    @property
    def served_lidar_interval_us(self) -> PositiveInt | None:
        """Time interval the benchmark's protocol serves lidar on; None = declared per data source or LiDAR is absent."""
        return None
