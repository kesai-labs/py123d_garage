from __future__ import annotations

from dataclasses import dataclass, field

from py123d_garage.api.abstract_offline_data_source_config import OfflineEvaluationDataSourceConfig
from py123d_garage.datatypes.numerics import NonNegativeFloat, NonNegativeInt, PositiveFloat, PositiveInt


@dataclass
class AbstractBenchmarkConfig:
    """Benchmark protocol configuration."""

    # Global RNG seed.
    seed: int = 0

    # --- Open-loop benchmark configuration ---
    benchmark_offline_data_sources: dict[str, OfflineEvaluationDataSourceConfig] = field(
        default_factory=dict,
    )

    # --- Closed-loop benchmark configuration ---
    @property
    def required_trajectory_horizon_us(self) -> PositiveInt | None:
        """Future window the benchmark scores; the policy's trajectory must cover it. None = no fixed window required."""
        return None

    @property
    def max_history_duration_us(self) -> NonNegativeInt | None:
        """Farthest past the benchmark lets a policy read; None = no limit."""
        return None

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
