# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""The evaluated policy from its checkpoints: one policy, or the ensemble averaging several of one recipe."""

from __future__ import annotations

from typing import cast

from py123d_garage.api.abstract_policy import AbstractPolicy, AnyPolicy
from py123d_garage.api.ensemble_policy import EnsemblePolicy
from py123d_garage.common.config_help import build_from_string
from py123d_garage.config.schema.policy.policy_config import EvaluationPolicyConfig


def build_evaluation_policy(policy_config: EvaluationPolicyConfig) -> AnyPolicy:
    """
    The policy of the config with its checkpoint loaded, or the ensemble of them when several checkpoints are given.

    Args:
        policy_config: the evaluated policy; each checkpoint loads into its own copy of the policy.

    Returns:
        the initialized policy, on the CPU.
    """
    members: list[AnyPolicy] = []
    for checkpoint_file in policy_config.evaluation_checkpoint_files:
        member = cast(AnyPolicy, build_from_string(policy_config, AbstractPolicy))
        member.initialize(checkpoint_file)
        members.append(member)
    if len(members) == 1:
        return members[0]
    return EnsemblePolicy(members)
