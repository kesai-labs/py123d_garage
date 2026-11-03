# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import override

from py123d_garage.api.abstract_benchmark_config import AbstractBenchmarkConfig
from py123d_garage.config.schema.policy.policy_config import EvaluationPolicyConfig
from py123d_garage.datatypes.numerics import PositiveInt


@dataclass
class NavsafeBenchmarkConfig(AbstractBenchmarkConfig):
    """Config of the agent NavSafe asks for a plan every replan."""

    # -- Config objects --

    # The policy under evaluation; evaluation_checkpoint_files select its weights.
    policy_config: EvaluationPolicyConfig = field(default_factory=EvaluationPolicyConfig)

    # -- The NavSafe protocol, hard-coded --

    @property
    @override
    def required_trajectory_horizon_us(self) -> PositiveInt:
        """NavSafe tracks a 4 s plan of 8 poses until the next replan."""
        return 4_000_000

    @property
    @override
    def served_ego_state_interval_us(self) -> PositiveInt:
        """NavSafe simulates at 10 Hz and calls the policy every 5th frame."""
        return 500_000

    @property
    @override
    def served_camera_interval_us(self) -> PositiveInt:
        """Every policy call carries the cameras."""
        return 500_000

    @property
    @override
    def served_lidar_interval_us(self) -> PositiveInt | None:
        """NavSafe renders no lidar for the policy."""
        return None
