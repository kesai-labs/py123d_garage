# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""
torch.library.opcheck over every op of torch.ops.py123d_garage_cuda: schemas, fake implementations, autograd
registration and the dispatcher agree; an op without example inputs here fails the sweep.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch import nn

if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)
pytest.importorskip("py123d_garage_cuda")

from py123d_garage_cuda.transfuser.registry import family_of, unsupported_reason
from py123d_garage_cuda.transfuser.resnet import HALO_ENGINE

_ops = torch.ops.py123d_garage_cuda
_NAMESPACE = "py123d_garage_cuda::"


def _channels_last(*shape: int, dtype: torch.dtype = torch.bfloat16, grad: bool = False) -> torch.Tensor:
    tensor = torch.randn(*shape, device="cuda").to(dtype).contiguous(memory_format=torch.channels_last)
    return tensor.requires_grad_(grad)


def _dense(*shape: int, dtype: torch.dtype = torch.bfloat16, grad: bool = False) -> torch.Tensor:
    return torch.randn(*shape, device="cuda").to(dtype).requires_grad_(grad)


def _depthwise_conv() -> tuple[object, ...]:
    return _channels_last(2, 32, 12, 20, grad=True), _dense(32, 1, 7, 7, grad=True), _dense(32, grad=True)


def _depthwise_conv_backward() -> tuple[object, ...]:
    return _channels_last(2, 32, 12, 20), _channels_last(2, 32, 12, 20), _dense(32, 1, 7, 7)


def _fusion_upsample_add() -> tuple[object, ...]:
    return _channels_last(2, 16, 15, 21, grad=True), _channels_last(2, 16, 5, 7, grad=True)


def _fusion_upsample_add_backward() -> tuple[object, ...]:
    return _channels_last(2, 16, 15, 21), 5, 7


def _stem_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(2, 3, 38, 30, device="cuda").to(torch.bfloat16)
    padded = nn.functional.pad(x, (0, 0, 0, 0, 0, 1)).contiguous(memory_format=torch.channels_last)
    return padded, (torch.randn(64, 3, 7, 7, device="cuda") * 0.1).to(torch.bfloat16)


def _resnet_stem_conv() -> tuple[object, ...]:
    padded, weight = _stem_inputs()
    return padded, weight.requires_grad_()


def _resnet_stem_conv_weight_grad() -> tuple[object, ...]:
    padded, _ = _stem_inputs()
    return _channels_last(2, 64, 19, 15), padded, 3


def _batch_norm_finalize() -> tuple[object, ...]:
    padded, weight = _stem_inputs()
    _, partial = _ops.resnet_stem_conv(padded, weight)
    return partial, torch.rand(64, device="cuda") + 0.5, torch.randn(64, device="cuda"), 1e-5


def _stem_pool_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    padded, weight = _stem_inputs()
    y, partial = _ops.resnet_stem_conv(padded, weight)
    norm_weight = (torch.rand(64, device="cuda") + 0.5).requires_grad_()
    norm_bias = torch.randn(64, device="cuda").requires_grad_()
    values = _ops.batch_norm_finalize(partial, norm_weight.detach(), norm_bias.detach(), 1e-5)
    return y.requires_grad_(), values, norm_weight, norm_bias


def _resnet_stem_pool() -> tuple[object, ...]:
    return _stem_pool_inputs()


def _resnet_stem_pool_backward() -> tuple[object, ...]:
    y, values, norm_weight, norm_bias = _stem_pool_inputs()
    out, offsets = _ops.resnet_stem_pool(y.detach(), values, norm_weight.detach(), norm_bias.detach())
    return torch.randn_like(out), y.detach(), offsets, values, norm_weight.detach()


def _pinned_conv() -> tuple[object, ...]:
    x, weight = _channels_last(2, 64, 16, 8, grad=True), _channels_last(64, 64, 3, 3, grad=True)
    return x, weight, [1, 1], [1, 1], [HALO_ENGINE] * 3, [], [], []


def _pinned_conv_data_grad() -> tuple[object, ...]:
    return _channels_last(2, 64, 16, 8), _channels_last(64, 64, 3, 3), [2, 64, 16, 8], [1, 1], [1, 1], HALO_ENGINE, []


def _pinned_conv_weight_grad() -> tuple[object, ...]:
    return _channels_last(2, 64, 16, 8), _channels_last(2, 64, 16, 8), [64, 64, 3, 3], [1, 1], [1, 1], -1, []


def _regnet_grouped_conv() -> tuple[object, ...]:
    return _channels_last(2, 72, 12, 10, grad=True), _dense(72, 24, 3, 3, dtype=torch.float32, grad=True), 1


def _regnet_grouped_conv_backward() -> tuple[object, ...]:
    return _channels_last(2, 72, 12, 10), _channels_last(2, 72, 12, 10), _dense(72, 24, 3, 3, dtype=torch.float32), 1


def _bn_relu_inputs(grad: bool) -> tuple[object, ...]:
    y = _channels_last(2, 72, 12, 10, grad=grad)
    scale = (torch.rand(72, device="cuda") + 0.5).requires_grad_(grad)
    shift = (torch.randn(72, device="cuda") * 0.5).requires_grad_(grad)
    center = y.detach().float().mean((0, 2, 3))
    weight = (torch.randn(72, 24, 3, 3, device="cuda") / 8).requires_grad_(grad)
    return y, scale, shift, center, weight, 1


def _regnet_bn_relu_grouped_conv() -> tuple[object, ...]:
    return _bn_relu_inputs(grad=True)


def _regnet_bn_relu_grouped_conv_backward() -> tuple[object, ...]:
    return _channels_last(2, 72, 12, 10), *_bn_relu_inputs(grad=False)


def _attention_inputs() -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], torch.Tensor]:
    q_src, kv_src = _dense(2, 50, 64), _dense(2, 37, 128)
    norms = [torch.randn(32, device="cuda") * 0.3 + float(index % 2 == 0) for index in range(4)]
    mask = torch.rand(50, 37, device="cuda") > 0.4
    mask[:, 0] = True
    return q_src, kv_src, norms, mask


def _planner_attention() -> tuple[object, ...]:
    q_src, kv_src, norms, mask = _attention_inputs()
    leaves = [tensor.requires_grad_() for tensor in (q_src, kv_src, *norms)]
    return leaves[0], leaves[1], 0, 0, 64, 2, *leaves[2:], mask, 0.1, 1e-5, False


def _planner_attention_backward() -> tuple[object, ...]:
    q_src, kv_src, norms, mask = _attention_inputs()
    out, lse, keep_bits = _ops.planner_attention(q_src, kv_src, 0, 0, 64, 2, *norms, mask, 0.1, 1e-5, False)
    return torch.randn_like(out), q_src, kv_src, 0, 0, 64, 2, *norms, mask, 0.1, keep_bits, 1e-5, out, lse, False


def _planner_attention_backward_shared_bytes() -> tuple[object, ...]:
    return 192, 192, True, True


def _planner_dropout_keep() -> tuple[object, ...]:
    q_src, kv_src, norms, mask = _attention_inputs()
    _, _, keep_bits = _ops.planner_attention(q_src, kv_src, 0, 0, 64, 2, *norms, mask, 0.1, 1e-5, False)
    return keep_bits, 2, 2, 50, 37, mask


def _adaptive_norm_inputs(grad: bool) -> tuple[object, ...]:
    x = _dense(2, 5, 256, dtype=torch.float32, grad=grad)
    branch = _dense(2, 5, 256, grad=grad)
    modulation = _dense(2, 3 * 256, grad=grad)
    gate, shift, scale = modulation[:, :256], modulation[:, 256:512], modulation[:, 512:]
    weight = _dense(256, dtype=torch.float32, grad=grad)
    bias = _dense(256, dtype=torch.float32, grad=grad)
    return x, branch, gate, shift, scale, weight, bias


def _planner_adaptive_norm() -> tuple[object, ...]:
    return *_adaptive_norm_inputs(grad=True), 0.1, 1e-5


def _planner_adaptive_norm_backward() -> tuple[object, ...]:
    x, branch, gate, shift, scale, weight, bias = _adaptive_norm_inputs(grad=False)
    x_out, y, stats, rng = _ops.planner_adaptive_norm(x, branch, gate, shift, scale, weight, bias, 0.1, 1e-5)
    return (
        torch.randn_like(x_out),
        torch.randn_like(y),
        x_out,
        stats,
        branch,
        gate,
        shift,
        scale,
        weight,
        bias,
        0.1,
        rng,
        1e-5,
    )


def _swiglu_inputs(grad: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = _dense(4, 64, grad=grad)
    w1 = (torch.randn(32, 64, device="cuda") / 8).to(torch.bfloat16).requires_grad_(grad)
    b1 = _dense(32, grad=grad)
    w2 = (torch.randn(256, 16, device="cuda") / 4).to(torch.bfloat16).requires_grad_(grad)
    b2 = _dense(256, grad=grad)
    return x, w1, b1, w2, b2


def _planner_swiglu_linear() -> tuple[object, ...]:
    x, w1, b1, _, _ = _swiglu_inputs(grad=True)
    return x, w1, b1, 0.1


def _planner_swiglu_linear_backward() -> tuple[object, ...]:
    x, w1, b1, _, _ = _swiglu_inputs(grad=False)
    hidden, gate_value, rng = _ops.planner_swiglu_linear(x, w1, b1, 0.1)
    return torch.randn_like(hidden), x, w1, gate_value, 0.1, rng


def _planner_swiglu_feed_forward() -> tuple[object, ...]:
    return *_swiglu_inputs(grad=True), 0.1


def _planner_swiglu_feed_forward_backward() -> tuple[object, ...]:
    x, w1, b1, w2, b2 = _swiglu_inputs(grad=False)
    out, hidden, gate_value, rng = _ops.planner_swiglu_feed_forward(x, w1, b1, w2, b2, 0.1)
    return torch.randn_like(out), x, w1, w2, hidden, gate_value, 0.1, rng


EXAMPLES: dict[str, Callable[[], tuple[object, ...]]] = {
    "depthwise_conv": _depthwise_conv,
    "depthwise_conv_backward": _depthwise_conv_backward,
    "fusion_upsample_add": _fusion_upsample_add,
    "fusion_upsample_add_backward": _fusion_upsample_add_backward,
    "resnet_stem_conv": _resnet_stem_conv,
    "resnet_stem_conv_weight_grad": _resnet_stem_conv_weight_grad,
    "batch_norm_finalize": _batch_norm_finalize,
    "resnet_stem_pool": _resnet_stem_pool,
    "resnet_stem_pool_backward": _resnet_stem_pool_backward,
    "pinned_conv": _pinned_conv,
    "pinned_conv_data_grad": _pinned_conv_data_grad,
    "pinned_conv_weight_grad": _pinned_conv_weight_grad,
    "regnet_grouped_conv": _regnet_grouped_conv,
    "regnet_grouped_conv_backward": _regnet_grouped_conv_backward,
    "regnet_bn_relu_grouped_conv": _regnet_bn_relu_grouped_conv,
    "regnet_bn_relu_grouped_conv_backward": _regnet_bn_relu_grouped_conv_backward,
    "planner_attention": _planner_attention,
    "planner_attention_backward": _planner_attention_backward,
    "planner_attention_backward_shared_bytes": _planner_attention_backward_shared_bytes,
    "planner_dropout_keep": _planner_dropout_keep,
    "planner_adaptive_norm": _planner_adaptive_norm,
    "planner_adaptive_norm_backward": _planner_adaptive_norm_backward,
    "planner_swiglu_linear": _planner_swiglu_linear,
    "planner_swiglu_linear_backward": _planner_swiglu_linear_backward,
    "planner_swiglu_feed_forward": _planner_swiglu_feed_forward,
    "planner_swiglu_feed_forward_backward": _planner_swiglu_feed_forward_backward,
}


def _registered_ops() -> list[str]:
    return sorted(
        name[len(_NAMESPACE) :] for name in torch._C._dispatch_get_all_op_names() if name.startswith(_NAMESPACE)
    )  # pyright: ignore[reportPrivateUsage]


def test_every_registered_op_has_example_inputs_and_a_registry_entry() -> None:
    ops = _registered_ops()
    assert ops, "no ops registered"
    assert sorted(set(ops) - set(EXAMPLES)) == [], "ops without example inputs"
    assert sorted(set(EXAMPLES) - set(ops)) == [], "examples for ops that are not registered"
    for op in ops:
        family_of(op)


@pytest.mark.parametrize("op", sorted(EXAMPLES))
def test_opcheck(op: str) -> None:
    reason = unsupported_reason(op)
    if reason is not None:
        pytest.skip(reason)
    torch.manual_seed(0)
    torch.library.opcheck(getattr(_ops, op).default, EXAMPLES[op]())
