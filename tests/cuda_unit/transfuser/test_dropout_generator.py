# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""
The fused dropout ops leave the CUDA generator alone at dropout zero, so a patched and an unpatched policy draw the same
noise from the same seed, and advance it with dropout.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch import nn

if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)
pytest.importorskip("py123d_garage_cuda")

from py123d_garage_cuda.transfuser.registry import unsupported_reason

_ops = torch.ops.py123d_garage_cuda


def _adaptive_norm() -> Callable[[float], None]:
    x = torch.randn(2, 64, 256, device="cuda")
    branch = torch.randn(2, 64, 256, device="cuda").bfloat16()
    modulation = torch.randn(2, 768, device="cuda").bfloat16()
    norm = nn.LayerNorm(256).cuda()
    gate, shift, scale = modulation[:, :256], modulation[:, 256:512], modulation[:, 512:]
    return lambda p: _ops.planner_adaptive_norm(x, branch, gate, shift, scale, norm.weight, norm.bias, p, 1e-5)


def _attention() -> Callable[[float], None]:
    qkv = torch.randn(2, 64, 768, device="cuda").bfloat16()
    q_norm, k_norm = nn.LayerNorm(32).cuda(), nn.LayerNorm(32).cuda()
    norms = (q_norm.weight, q_norm.bias, k_norm.weight, k_norm.bias)
    return lambda p: _ops.planner_attention(qkv, qkv, 0, 256, 512, 8, *norms, None, p, 1e-5, True)


def _swiglu_linear() -> Callable[[float], None]:
    x = torch.randn(64, 128, device="cuda").bfloat16()
    weight = (torch.randn(208, 128, device="cuda") / 16).bfloat16()
    bias = torch.randn(208, device="cuda").bfloat16()
    return lambda p: _ops.planner_swiglu_linear(x, weight, bias, p)


def _swiglu_feed_forward() -> Callable[[float], None]:
    x = torch.randn(64, 128, device="cuda").bfloat16()
    w1 = (torch.randn(208, 128, device="cuda") / 16).bfloat16()
    b1 = torch.randn(208, device="cuda").bfloat16()
    w2 = (torch.randn(256, 104, device="cuda") / 16).bfloat16()
    b2 = torch.randn(256, device="cuda").bfloat16()
    return lambda p: _ops.planner_swiglu_feed_forward(x, w1, b1, w2, b2, p)


DROPOUT_OPS: dict[str, Callable[[], Callable[[float], None]]] = {
    "planner_adaptive_norm": _adaptive_norm,
    "planner_attention": _attention,
    "planner_swiglu_linear": _swiglu_linear,
    "planner_swiglu_feed_forward": _swiglu_feed_forward,
}


@pytest.mark.parametrize("op", sorted(DROPOUT_OPS))
def test_the_generator_moves_only_with_dropout(op: str) -> None:
    reason = unsupported_reason(op)
    if reason is not None:
        pytest.skip(reason)
    torch.manual_seed(0)
    run = DROPOUT_OPS[op]()
    torch.cuda.synchronize()
    before = torch.cuda.get_rng_state()
    run(0.0)
    assert torch.equal(torch.cuda.get_rng_state(), before)
    run(0.1)
    assert not torch.equal(torch.cuda.get_rng_state(), before)
