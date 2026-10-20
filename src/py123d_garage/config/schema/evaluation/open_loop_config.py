# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

from __future__ import annotations

from dataclasses import dataclass, field

from py123d_garage.api.abstract_benchmark_config import AbstractBenchmarkConfig
from py123d_garage.config.schema.evaluation.parallel_offline_evaluation_config import ParallelizationConfig
from py123d_garage.config.schema.policy.policy_config import EvaluationPolicyConfig


@dataclass
class OpenLoopBenchmarkConfig(AbstractBenchmarkConfig):
    """Config of the open-loop evaluation over offline scenes."""

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
