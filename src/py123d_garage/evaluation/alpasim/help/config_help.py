# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

from __future__ import annotations

import logging

import torch

from py123d_garage.api.abstract_policy import AnyPolicy
from py123d_garage.config.schema.evaluation.alpasim_config import AlpasimBenchmarkConfig
from py123d_garage.evaluation.help.policy_building import build_evaluation_policy

LOG = logging.getLogger(__name__)


def build_policy(config: AlpasimBenchmarkConfig, device: torch.device) -> AnyPolicy:
    """
    Builds the evaluated policy, checks it against the benchmark and moves it to the device.

    Args:
        config: the benchmark config naming the policy and its checkpoints.
        device: the device to run the policy on.

    Returns:
        the policy in eval mode.
    """
    policy = build_evaluation_policy(config.policy_config)
    policy.verify_contract(benchmark_config=config)
    LOG.info(f"Driving {config.policy_config.target} with {config.policy_config.evaluation_checkpoint_files}")
    return policy.to(device).eval()
