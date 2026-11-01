# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import override

from py123d_garage.api.abstract_benchmark_config import AbstractBenchmarkConfig
from py123d_garage.config.schema.policy.policy_config import EvaluationPolicyConfig
from py123d_garage.datatypes.numerics import PositiveInt


@dataclass
class HugsimBenchmarkConfig(AbstractBenchmarkConfig):
    """Config of the agent HUGSIM asks for a plan every simulation step."""

    # -- Config objects --

    # The policy under evaluation; evaluation_checkpoint_files select its weights.
    policy_config: EvaluationPolicyConfig = field(default_factory=EvaluationPolicyConfig)

    # -- Atomic settings --

    # Sim time between inference runs, the stored plan serving the steps between.
    replan_interval_us: int = 500_000

    # -- The HUGSIM protocol, hard-coded --

    @property
    @override
    def required_trajectory_horizon_us(self) -> PositiveInt:
        """HUGSIM tracks and scores a 4 s plan, served from a plan up to one replan interval old."""
        return 4_000_000 + self.replan_interval_us

    @property
    @override
    def served_ego_state_interval_us(self) -> PositiveInt:
        """The simulator steps every 0.25 s and asks for a plan every step."""
        return 250_000

    @property
    @override
    def served_camera_interval_us(self) -> PositiveInt:
        """Every step renders the cameras."""
        return 250_000

    @property
    @override
    def served_lidar_interval_us(self) -> PositiveInt | None:
        """HUGSIM renders no lidar."""
        return None
