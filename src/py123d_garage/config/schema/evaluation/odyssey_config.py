# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import override

from py123d_garage.api.abstract_benchmark_config import AbstractBenchmarkConfig
from py123d_garage.config.schema.policy.policy_config import EvaluationPolicyConfig
from py123d_garage.datatypes.numerics import NonNegativeFloat, PositiveFloat, PositiveInt


@dataclass
class OdysseyBenchmarkConfig(AbstractBenchmarkConfig):
    """Config of the planner the Odyssey runtime calls every simulation tick."""

    # -- Config objects --

    # The policy under evaluation; evaluation_checkpoint_files select its weights.
    policy_config: EvaluationPolicyConfig = field(default_factory=EvaluationPolicyConfig)

    # -- Atomic settings --

    # Sim time between inference runs, the stored plan serving the ticks between.
    replan_interval_us: int = 500_000

    # -- The Odyssey protocol, hard-coded --

    @property
    @override
    def min_served_target_point_distance_m(self) -> NonNegativeFloat:
        """The route centerline starts at the ego's projection onto the route."""
        return 0.0

    @property
    @override
    def max_served_target_point_distance_m(self) -> PositiveFloat:
        """The route centerline reaches 120 m ahead of the ego."""
        return 120.0

    @property
    @override
    def served_ego_state_interval_us(self) -> PositiveInt:
        """The simulator ticks at 10 Hz and calls the planner every tick."""
        return 100_000

    @property
    @override
    def served_camera_interval_us(self) -> PositiveInt:
        """Every tick renders the declared cameras."""
        return 100_000

    @property
    @override
    def served_lidar_interval_us(self) -> PositiveInt | None:
        """Odyssey renders no lidar."""
        return None
