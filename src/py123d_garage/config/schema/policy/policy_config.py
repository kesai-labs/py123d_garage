# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

from __future__ import annotations

from dataclasses import dataclass

from omegaconf import MISSING

from py123d_garage.config.schema.policy.transfuser_config import TransfuserConfig


@dataclass
class PolicyConfig:
    """The policy to build; each policy reads its own config block."""

    # -- Config objects --

    # TransFuser's block, read when target names the TransfuserPolicy.
    transfuser_config: TransfuserConfig | None = None

    # -- Atomic settings --

    # module:Class path of the AbstractPolicy implementation.
    target: str = "py123d_garage.policy.transfuser.transfuser_policy:TransfuserPolicy"


@dataclass
class EvaluationPolicyConfig(PolicyConfig):
    """The policy under evaluation, with the checkpoint and sensor rig files it loads."""

    # Checkpoint files: one loads the policy's weights, several average an ensemble of one recipe.
    evaluation_checkpoint_files: list[str] = MISSING

    # Sensor rig YAML saved next to the checkpoint, from which the benchmark builds its served rig; edit it to change the rig.
    evaluation_sensor_rig_file: str | None = None
