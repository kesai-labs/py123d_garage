# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""
The flow planner's adaptive-norm op, residual, gate, dropout, norm and modulation in one pass, with
its fake implementation and autograd, and the patch that swaps it into the planning decoder.
"""

from __future__ import annotations

from typing import cast

import jaxtyping as jt
import torch
from torch import Tensor, nn

from py123d_garage_cuda.transfuser.registry import note_fallback, supports, unsupported_reason

_ops = torch.ops.py123d_garage_cuda
_NORM_DIM = 256


def planner_adaptive_norm(
    x: jt.Float[Tensor, "batch queries 256"],
    branch: jt.Float[Tensor, "batch queries 256"] | None,
    gate: jt.Float[Tensor, "batch 256"] | None,
    shift: jt.Float[Tensor, "batch 256"] | None,
    scale: jt.Float[Tensor, "batch 256"] | None,
    norm: nn.LayerNorm,
    dropout_p: float,
) -> tuple[Tensor, Tensor]:
    """
    Computes x_out = x + (1 + gate) dropout(branch) and y = norm(x_out) (1 + scale) + shift in one pass.

    Args:
        x: the fp32 residual stream.
        branch: the bf16 sublayer output added to x, or None to only normalize x.
        gate: the bf16 gate per batch row, with branch.
        shift: the bf16 shift per batch row, or None (with scale) to skip the modulation.
        scale: the bf16 scale per batch row.
        norm: the LayerNorm.
        dropout_p: dropout probability of the branch; the mask is drawn from the CUDA generator.

    Returns:
        x_out (x itself without a branch) and y, bf16 with a modulation, else fp32.
    """
    x_out, y, _, _ = _ops.planner_adaptive_norm(
        x,
        branch,
        gate,
        shift,
        scale,
        norm.weight,
        norm.bias,
        dropout_p,
        norm.eps,
    )
    return (x if branch is None else x_out), y


@torch.library.register_fake("py123d_garage_cuda::planner_adaptive_norm")
def _(
    x: Tensor,
    branch: Tensor | None,
    gate: Tensor | None,
    shift: Tensor | None,
    scale: Tensor | None,
    weight: Tensor,
    bias: Tensor,
    dropout_p: float,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Fake planner_adaptive_norm: x_out (empty without a branch), y, the rows' (mean, rstd) and the dropout stream."""
    x_out = torch.empty_like(x) if branch is not None else x.new_empty(0)
    y = torch.empty_like(x, dtype=torch.bfloat16 if scale is not None else torch.float32)
    return x_out, y, x.new_empty(*x.shape[:-1], 2), x.new_empty(2, dtype=torch.int64)


@torch.library.register_fake("py123d_garage_cuda::planner_adaptive_norm_backward")
def _(
    grad_x_out: Tensor | None,
    grad_y: Tensor,
    normed: Tensor,
    stats: Tensor,
    branch: Tensor | None,
    gate: Tensor | None,
    shift: Tensor | None,
    scale: Tensor | None,
    weight: Tensor,
    bias: Tensor,
    dropout_p: float,
    rng: Tensor,
    eps: float,
) -> tuple[Tensor, ...]:
    """Fake planner_adaptive_norm_backward: the gradients of x, branch, gate, shift, scale, weight and bias."""
    modulation_grads = tuple(
        normed.new_empty(*tensor.shape, dtype=torch.bfloat16)
        if tensor is not None
        else normed.new_empty(0, dtype=torch.bfloat16)
        for tensor in (branch, gate, shift, scale)
    )
    return torch.empty_like(normed), *modulation_grads, torch.empty_like(weight), torch.empty_like(bias)


def _adaptive_norm_setup_context(
    ctx: torch.autograd.function.FunctionCtx,
    inputs: tuple[object, ...],
    output: object,
) -> None:
    """Saves the norm's input, the stats, the branch, gate, shift, scale, norm params and the dropout stream."""
    x, branch, gate, shift, scale, weight, bias, dropout_p, eps = cast(
        "tuple[Tensor, Tensor | None, Tensor | None, Tensor | None, Tensor | None, Tensor, Tensor, float, float]",
        inputs,
    )
    x_out, _, stats, rng = cast("tuple[Tensor, Tensor, Tensor, Tensor]", output)
    ctx.save_for_backward(x if branch is None else x_out, stats, branch, gate, shift, scale, weight, bias, rng)  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]
    ctx.arguments = (dropout_p, eps)  # pyright: ignore[reportAttributeAccessIssue]


def _adaptive_norm_backward(
    ctx: torch.autograd.function.FunctionCtx,
    grad_x_out: Tensor | None,
    grad_y: Tensor,
    _grad_stats: Tensor | None,
    _grad_rng: Tensor | None,
) -> tuple[Tensor | None, ...]:
    """Gradients of x, branch, gate, shift, scale, weight and bias through planner_adaptive_norm_backward."""
    normed, stats, branch, gate, shift, scale, weight, bias, rng = ctx.saved_tensors  # pyright: ignore[reportAttributeAccessIssue]
    dropout_p, eps = ctx.arguments  # pyright: ignore[reportAttributeAccessIssue]
    grads = _ops.planner_adaptive_norm_backward(
        grad_x_out if branch is not None else None,
        grad_y,
        normed,
        stats,
        branch,
        gate,
        shift,
        scale,
        weight,
        bias,
        dropout_p,
        rng,
        eps,
    )
    grad_x, grad_branch, grad_gate, grad_shift, grad_scale, grad_weight, grad_bias = grads
    return (
        grad_x,
        grad_branch if branch is not None else None,
        grad_gate if gate is not None else None,
        grad_shift if shift is not None else None,
        grad_scale if scale is not None else None,
        grad_weight,
        grad_bias,
        None,
        None,
    )


torch.library.register_autograd(
    "py123d_garage_cuda::planner_adaptive_norm",
    _adaptive_norm_backward,
    setup_context=_adaptive_norm_setup_context,
)


def adaptive_norm(
    x: jt.Float[Tensor, "batch queries dim"],
    residual: tuple[Tensor, Tensor, nn.Dropout] | None,
    norm: nn.LayerNorm,
    shift: jt.Float[Tensor, "batch dim"] | None,
    scale: jt.Float[Tensor, "batch dim"] | None,
) -> tuple[Tensor, Tensor]:
    """
    Runs a planning decoder's residual, gate, residual dropout, norm and modulation through planner_adaptive_norm, or
    through torch where the op does not apply (not bf16 autocast on CUDA, not fp32 rows of 256, no affine norm, a GPU
    that does not run it).

    Args:
        x: the residual stream.
        residual: the sublayer output, its gate per batch row and its residual dropout, or None.
        norm: the LayerNorm.
        shift: the modulation's shift per batch row, or None (with scale) for a plain norm.
        scale: the modulation's scale per batch row.

    Returns:
        the residual stream after the residual, and the normalized (and modulated) rows.
    """
    applies = False
    if not (x.is_cuda and torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == torch.bfloat16):
        note_fallback("planner_adaptive_norm", "needs bf16 autocast on CUDA")
    elif not (x.dtype == torch.float32 and x.dim() == 3 and x.shape[-1] == _NORM_DIM):
        note_fallback(
            "planner_adaptive_norm",
            lambda: f"needs an fp32 [batch, rows, {_NORM_DIM}] residual stream; got {x.dtype} {tuple(x.shape)}",
        )
    elif norm.weight is None or norm.bias is None or norm.weight.dtype != torch.float32:
        note_fallback("planner_adaptive_norm", "needs a LayerNorm with fp32 weight and bias")
    elif not supports("planner_adaptive_norm", x.device):
        note_fallback("planner_adaptive_norm", lambda: str(unsupported_reason("planner_adaptive_norm", x.device)))
    else:
        applies = True
    if not applies:
        if residual is not None:
            branch, gate, dropout = residual
            x = x + (1.0 + gate[:, None]) * dropout(branch)
        y = norm(x)
        if scale is not None and shift is not None:
            y = y * (1.0 + scale[:, None]) + shift[:, None]
        return x, y
    dtype = torch.bfloat16
    branch, gate, dropout_p = None, None, 0.0
    if residual is not None:
        branch, gate, dropout = residual[0].to(dtype), residual[1].to(dtype), residual[2]
        dropout_p = dropout.p if dropout.training else 0.0
    if scale is not None and shift is not None:
        shift, scale = shift.to(dtype), scale.to(dtype)
    return planner_adaptive_norm(x.contiguous(), branch, gate, shift, scale, norm, dropout_p)


def patch_planner_adaptive_norm(module: nn.Module) -> None:
    """
    Routes every residual, gate, residual dropout, norm and modulation of the adaptive-norm planning decoders inside
    module through planner_adaptive_norm.

    Args:
        module: the module to patch.
    """
    for child in module.modules():
        layers = getattr(child, "layers", None)
        if isinstance(layers, nn.ModuleList) and all(hasattr(layer, "modulation") for layer in layers):
            setattr(child, "fused_adaptive_norm", adaptive_norm)  # noqa: B010
