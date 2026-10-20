# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""
ResNet's fused batch norm -> ReLU -> max-pool op over the stem convolution's output and batch statistics,
with its fake implementations and autograd.
"""

from __future__ import annotations

from typing import cast

import jaxtyping as jt
import torch
from torch import Tensor

_ops = torch.ops.py123d_garage_cuda


def resnet_stem_pool(
    y: jt.Float[Tensor, "batch channels height width"],
    values: jt.Float[Tensor, "5 channels"],
    weight: jt.Float[Tensor, " channels"],
    bias: jt.Float[Tensor, " channels"],
) -> tuple[
    jt.Float[Tensor, "batch channels out_height out_width"],
    jt.Int8[Tensor, "batch channels out_height out_width"],
]:
    """
    Computes max_pool2d(relu(batch_norm(y)), 3, 2, 1) from the given batch statistics of y, the normalized values
    rounded to y's dtype before the maximum as eager torch does.

    Args:
        y: the channels-last bf16 stem convolution output, C a multiple of 8.
        values: the statistics of y from batch_norm_finalize; the gradient flows through them all the same.
        weight: the batch norm's weight, already folded into values.
        bias: the batch norm's bias, already folded into values.

    Returns:
        the channels-last pooled output and the int8 window position kh * 3 + kw of each first maximum.
    """
    return _ops.resnet_stem_pool(y, values, weight, bias)


@torch.library.register_fake("py123d_garage_cuda::resnet_stem_pool")
def _(y: Tensor, values: Tensor, weight: Tensor, bias: Tensor) -> tuple[Tensor, Tensor]:
    """Fake resnet_stem_pool: the channels-last pooled output and the int8 argmax offsets."""
    batch, channels, height, width = y.shape
    size = (batch, channels, (height - 1) // 2 + 1, (width - 1) // 2 + 1)
    return (
        y.new_empty(size).contiguous(memory_format=torch.channels_last),
        y.new_empty(size, dtype=torch.int8).contiguous(memory_format=torch.channels_last),
    )


@torch.library.register_fake("py123d_garage_cuda::resnet_stem_pool_backward")
def _(grad: Tensor, y: Tensor, offsets: Tensor, values: Tensor, weight: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Fake resnet_stem_pool_backward: the gradients of y, weight and bias."""
    return torch.empty_like(y, memory_format=torch.channels_last), torch.empty_like(weight), torch.empty_like(weight)


def _pool_setup_context(
    ctx: torch.autograd.function.FunctionCtx,
    inputs: tuple[object, ...],
    output: tuple[Tensor, Tensor],
) -> None:
    """Saves y, the argmax offsets, values and weight for the backward; the offsets get no gradient."""
    y, values, weight, _ = cast("tuple[Tensor, Tensor, Tensor, Tensor]", inputs)
    ctx.save_for_backward(y, output[1], values, weight)  # pyright: ignore[reportAttributeAccessIssue]
    ctx.mark_non_differentiable(output[1])
    ctx.set_materialize_grads(False)


def _pool_backward(ctx: torch.autograd.function.FunctionCtx, *grads: Tensor | None) -> tuple[Tensor | None, ...]:
    """Gradients of y, weight and bias through resnet_stem_pool_backward."""
    y, offsets, values, weight = ctx.saved_tensors  # pyright: ignore[reportAttributeAccessIssue]
    grad = grads[0]
    assert grad is not None
    dy, dweight, dbias = _ops.resnet_stem_pool_backward(grad, y, offsets, values, weight)
    return dy, None, dweight, dbias


torch.library.register_autograd(
    "py123d_garage_cuda::resnet_stem_pool",
    _pool_backward,
    setup_context=_pool_setup_context,
)
