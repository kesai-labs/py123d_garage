# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""Sharing a sample's context among its noise draws gives the flow losses and gradients of one copy per draw."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from py123d_garage.config.schema.policy.policy_config import PolicyConfig
from py123d_garage.config.schema.policy.transfuser_config import TransfuserConfig
from py123d_garage.policy.transfuser.transfuser_policy import TransfuserPolicy

from .test_flow_planning_decoder import _config, _inputs, _labels


def _shared(config: TransfuserConfig, shared: bool, masked: bool) -> TransfuserConfig:
    planning = config.planning_config
    flow = dataclasses.replace(
        planning.flow_config,
        shared_draw_context=shared,
        # Target points that the waypoints and actions may not see: a cross-attention mask.
        waypoints_and_actions_attend_to_target_points=not masked,
    )
    return dataclasses.replace(config, planning_config=dataclasses.replace(planning, flow_config=flow))


def _losses_and_gradients(
    shared: bool,
    masked: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    torch.manual_seed(0)
    config = _shared(_config(), shared, masked)
    policy = TransfuserPolicy(PolicyConfig(transfuser_config=config))
    policy.initialize()
    # Eval: no dropout, so both runs see the same network; autograd still makes the forward return the context.
    policy.eval()
    torch.manual_seed(1)
    inputs = _inputs(config)
    predictions = policy(*inputs)
    torch.manual_seed(2)  # the same noise draws
    losses = policy.compute_loss(_labels(config), predictions)
    torch.stack(list(losses.values())).sum().backward()
    gradients = {name: p.grad.clone() for name, p in policy.named_parameters() if p.grad is not None}
    return {name: loss.detach() for name, loss in losses.items()}, gradients


# Without and with a cross-attention mask on the target points.
@pytest.mark.parametrize("masked", [False, True])
def test_matches_one_context_copy_per_draw(masked: bool) -> None:
    losses, gradients = _losses_and_gradients(True, masked)
    reference_losses, reference_gradients = _losses_and_gradients(False, masked)
    assert losses.keys() == reference_losses.keys()
    for name, loss in losses.items():
        torch.testing.assert_close(loss, reference_losses[name], rtol=1e-5, atol=1e-6, msg=name)
    assert gradients.keys() == reference_gradients.keys()
    scale = max(g.abs().max().item() for g in reference_gradients.values())
    for name, gradient in gradients.items():
        torch.testing.assert_close(gradient, reference_gradients[name], rtol=1e-4, atol=1e-6 * scale, msg=name)
