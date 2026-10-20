# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""
TransFuser's fusion residual op, x + bilinear_upsample(grid) in one pass, with its fake implementation
and autograd, and the backbone patch that swaps it in.
"""

from __future__ import annotations

from typing import cast

import jaxtyping as jt
import torch
from torch import Tensor, nn

from py123d_garage_cuda.transfuser.registry import note_fallback, supports, unsupported_reason

_ops = torch.ops.py123d_garage_cuda
# Channels per thread of the kernels.
_VECTOR = 8


def fusion_upsample_add(
    x: jt.Float[Tensor, "batch channels height width"],
    grid: jt.Float[Tensor, "batch channels grid_height grid_width"],
) -> jt.Float[Tensor, "batch channels height width"]:
    """
    Computes x + F.interpolate(grid, size=x.shape[2:], mode="bilinear", align_corners=False) in fp32 arithmetic.

    Args:
        x: the channels-last bf16 map, channels a multiple of 8.
        grid: the channels-last bf16 grid.

    Returns:
        the channels-last bf16 sum.
    """
    return _ops.fusion_upsample_add(x, grid)


@torch.library.register_fake("py123d_garage_cuda::fusion_upsample_add")
def _(x: Tensor, grid: Tensor) -> Tensor:
    """Fake fusion_upsample_add: the channels-last sum."""
    return torch.empty_like(x, memory_format=torch.channels_last)


@torch.library.register_fake("py123d_garage_cuda::fusion_upsample_add_backward")
def _(grad: Tensor, grid_height: int, grid_width: int) -> Tensor:
    """Fake fusion_upsample_add_backward: the channels-last gradient of the grid."""
    return grad.new_empty(grad.shape[0], grad.shape[1], grid_height, grid_width).contiguous(
        memory_format=torch.channels_last,
    )


def _setup_context(ctx: torch.autograd.function.FunctionCtx, inputs: tuple[object, ...], output: object) -> None:
    """Keeps the grid's size for the backward."""
    grid = cast(Tensor, inputs[1])
    ctx.grid_size = (grid.shape[2], grid.shape[3])  # pyright: ignore[reportAttributeAccessIssue]


def _backward(ctx: torch.autograd.function.FunctionCtx, grad: Tensor) -> tuple[Tensor, Tensor | None]:
    """The gradient passes to x unchanged and sums back onto the grid."""
    grid_height, grid_width = ctx.grid_size  # pyright: ignore[reportAttributeAccessIssue]
    if not ctx.needs_input_grad[1]:  # pyright: ignore[reportAttributeAccessIssue]
        return grad, None
    grad = grad.contiguous(memory_format=torch.channels_last)
    return grad, _ops.fusion_upsample_add_backward(grad, grid_height, grid_width)


torch.library.register_autograd("py123d_garage_cuda::fusion_upsample_add", _backward, setup_context=_setup_context)


def upsample_add(
    x: jt.Float[Tensor, "batch channels height width"],
    grid: jt.Float[Tensor, "batch channels grid_height grid_width"],
) -> jt.Float[Tensor, "batch channels height width"] | None:
    """
    Runs fusion_upsample_add under bf16 autocast on a GPU that runs it, for a channels-last x with channels a multiple
    of 8 and the grid's batch; a shared latent BEV stem's map of batch one is expanded to it.

    Args:
        x: the map.
        grid: the grid to up-sample onto x.

    Returns:
        the sum, or None where the op does not apply.
    """
    if not (x.is_cuda and torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == torch.bfloat16):
        note_fallback("fusion_upsample_add", "needs bf16 autocast on CUDA")
        return None
    if x.dtype != torch.bfloat16 or x.shape[1] % _VECTOR or not x.is_contiguous(memory_format=torch.channels_last):
        note_fallback(
            "fusion_upsample_add",
            lambda: (
                f"needs a bf16 channels-last map with channels a multiple of {_VECTOR}; got {x.dtype}, "
                f"{x.shape[1]} channels, channels-last {x.is_contiguous(memory_format=torch.channels_last)}"
            ),
        )
        return None
    if x.shape[0] == 1 and grid.shape[0] > 1:
        x = x.expand(grid.shape[0], -1, -1, -1).contiguous(memory_format=torch.channels_last)
    if x.shape[:2] != grid.shape[:2]:
        note_fallback(
            "fusion_upsample_add",
            lambda: f"map and grid differ in batch or channels: {tuple(x.shape[:2])} and {tuple(grid.shape[:2])}",
        )
        return None
    if not supports("fusion_upsample_add", x.device):
        note_fallback("fusion_upsample_add", lambda: str(unsupported_reason("fusion_upsample_add", x.device)))
        return None
    grid = grid.to(torch.bfloat16).contiguous(memory_format=torch.channels_last)
    return fusion_upsample_add(x, grid)


def patch_fusion_upsample_add(backbone: nn.Module) -> None:
    """
    Routes the bilinear up-sample and residual add of every TransFuser fusion stage inside backbone through
    fusion_upsample_add, in place; parameters and state_dict keys stay the same.

    Args:
        backbone: the module to patch.
    """
    for module in backbone.modules():
        if hasattr(module, "transformers") and hasattr(module, "fuse_features"):
            setattr(module, "fused_upsample_add", upsample_add)  # noqa: B010
