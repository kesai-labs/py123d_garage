from __future__ import annotations

import logging
from typing import cast

import torch

from py123d_garage.api.abstract_policy import AbstractPolicy, AnyPolicy
from py123d_garage.common.config_help import build_from_string
from py123d_garage.config.schema.evaluation.alpasim_config import AlpasimBenchmarkConfig

LOG = logging.getLogger(__name__)


def build_policy(config: AlpasimBenchmarkConfig, device: torch.device) -> AnyPolicy:
    policy = cast("AnyPolicy", build_from_string(config.policy_config, AbstractPolicy))
    policy.verify_contract(benchmark_config=config)
    policy.initialize(config.policy_config.evaluation_checkpoint_file)
    LOG.info(f"Driving {config.policy_config.target} with {config.policy_config.evaluation_checkpoint_file}")
    return policy.to(device).eval()
