# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""The fused key, query and value projection of the fusion transformers gives the unpatched outputs and gradients."""

from __future__ import annotations

import copy

import torch

from py123d_garage.policy.transfuser.network.encoder.transfuser import SelfAttention, fuse_fusion_qkv


def test_matches_the_unpatched_attention() -> None:
    torch.manual_seed(0)
    reference = torch.nn.Sequential(SelfAttention(64, 4, 0.1, 0.1), SelfAttention(64, 4, 0.1, 0.1)).double()
    patched = copy.deepcopy(reference)
    fuse_fusion_qkv(patched)
    assert patched.state_dict().keys() == reference.state_dict().keys()
    tokens = torch.randn(3, 40, 64, dtype=torch.float64)
    weights = torch.randn(3, 40, 64, dtype=torch.float64)
    outputs: list[torch.Tensor] = []
    for module in (patched, reference):
        # Both draw the same dropout masks from the same seed.
        torch.manual_seed(1)
        out = module(tokens)
        (out * weights).sum().backward()
        outputs.append(out.detach())
    torch.testing.assert_close(outputs[0], outputs[1])
    for (name, parameter), reference_parameter in zip(patched.named_parameters(), reference.parameters(), strict=True):
        assert parameter.grad is not None, name
        assert reference_parameter.grad is not None, name
        torch.testing.assert_close(parameter.grad, reference_parameter.grad, msg=name)
