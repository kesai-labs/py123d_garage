# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from typing_extensions import override

from py123d_garage.api.abstract_benchmark_config import AbstractBenchmarkConfig
from py123d_garage.config.schema.evaluation.parallel_offline_evaluation_config import ParallelizationConfig
from py123d_garage.config.schema.policy.policy_config import EvaluationPolicyConfig
from py123d_garage.datatypes.numerics import NonNegativeInt, PositiveInt


@dataclass
class NavsimBenchmarkConfig(AbstractBenchmarkConfig):
    """Config of the navsim entry point that PDM-scores a policy over the navtest scenes."""

    # -- Config objects --

    # The policy under evaluation; evaluation_checkpoint_files select its weights.
    policy_config: EvaluationPolicyConfig = field(default_factory=EvaluationPolicyConfig)
    # Sharding across processes and the resources of this shard.
    parallelization_config: ParallelizationConfig = field(
        default_factory=ParallelizationConfig,
    )

    # -- Atomic settings --

    # Writes the policy's rendered view of every scored scene into <run dir>/visualizations.
    save_visualizations: bool = False
    # Writes the planned trajectory of every scene into <run dir>/trajectories, e.g. to score it with the official devkit.
    save_trajectories: bool = False

    # -- The navtest protocol, hard-coded --

    # Future window each scored scene needs past its anchor: the 4 s PDM horizon plus the TTC metric's 1 s lookahead.
    navsim_scoring_future_duration_us: ClassVar[int] = 5_000_000

    @property
    @override
    def required_trajectory_horizon_us(self) -> PositiveInt:
        """Future window the PDM score simulates (see PDMMetric's sampling)."""
        return 4_000_000

    @property
    @override
    def max_history_duration_us(self) -> NonNegativeInt:
        """Navsim's AgentInput protocol: the current frame plus 1.5 s of past at 2 Hz."""
        return 1_500_000

    @property
    @override
    def scored_scene_future_duration_us(self) -> PositiveInt:
        """The scoring window with the TTC lookahead; the plan beyond it is not scored, so its labels are not needed."""
        return self.navsim_scoring_future_duration_us

    @property
    @override
    def scored_scene_needs_route(self) -> bool:
        """Every navtest frame is scored, also the ones whose route ends before the farthest target point."""
        return False
