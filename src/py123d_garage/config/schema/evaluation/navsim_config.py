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
    """The navsim evaluate entry point: PDM-score a policy over the navtest scenes."""

    # -- Config objects --

    # The policy under evaluation; evaluation_checkpoint_file selects its weights.
    policy_config: EvaluationPolicyConfig = field(default_factory=EvaluationPolicyConfig)
    # Sharding across processes and the resources of this shard.
    parallelization_config: ParallelizationConfig = field(
        default_factory=ParallelizationConfig,
    )

    # -- Atomic settings --

    # Writes the policy's rendered view of every scored scene into <run dir>/visualizations.
    save_visualizations: bool = False

    # -- The navtest protocol, hard-coded --

    # Future window each scored scene carries past its anchor: the PDM horizon
    # plus the TTC metric's one-second lookahead. The presets author it into
    # their sources' filters.
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
