# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""
The flow planner's query/key-norm attention and SwiGLU feed-forward ops with their fake implementations
and autograd, and the patches that swap them, with the adaptive norm of adaptive_norm, into the planning decoder.
"""

from __future__ import annotations

from typing import cast

import jaxtyping as jt
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from py123d_garage_cuda.transfuser.registry import note_fallback, supports, unsupported_reason

_ops = torch.ops.py123d_garage_cuda
_HEAD_DIM = 32
# Keys per softmax chunk of the kernels; sets the layout of the dropout keep bits.
_CHUNK = 64
_MAX_KEYS = 256
# The fused SwiGLU GEMM holds K in 64-wide blocks, at most four.
_SWIGLU_K_BLOCK = 64
_SWIGLU_MAX_K = 256


@torch.compiler.assume_constant_result
def _attention_backward_shared_bytes(queries: int, keys: int, dropout: bool, masked: bool) -> int:
    """
    Computes the dynamic shared memory per block planner_attention_backward needs for a shape, as a constant under
    torch.compile.

    Args:
        queries: the query count.
        keys: the key count.
        dropout: whether the attention drops out.
        masked: whether there is a mask.

    Returns:
        the bytes.
    """
    return _ops.planner_attention_backward_shared_bytes(queries, keys, dropout, masked)


def planner_attention(
    q_src: jt.Float[Tensor, "batch queries q_columns"],
    kv_src: jt.Float[Tensor, "batch keys kv_columns"],
    q_col: int,
    k_col: int,
    v_col: int,
    heads: int,
    q_norm: nn.LayerNorm,
    k_norm: nn.LayerNorm,
    mask: jt.Bool[Tensor, "queries keys"] | None,
    dropout_p: float,
    shared: bool,
) -> jt.Float[Tensor, "batch queries hidden"]:
    """
    Computes dropout(softmax(q_norm(q) k_norm(k)^T / sqrt(32), mask)) v per head, for head dim 32 and at most 256 keys.

    Args:
        q_src: bf16 rows holding q at columns [q_col, q_col + 32 heads).
        kv_src: bf16 rows holding k at k_col and v at v_col.
        q_col: first query column.
        k_col: first key column.
        v_col: first value column.
        heads: attention heads of 32 dims.
        q_norm: LayerNorm over each head's queries.
        k_norm: LayerNorm over each head's keys.
        mask: True where a query may attend; None attends everywhere.
        dropout_p: attention dropout probability; the mask is drawn from the CUDA generator.
        shared: q_src and kv_src are one tensor holding q, k and v (self-attention); its gradient comes back whole.

    Returns:
        the bf16 attention output, the heads side by side (hidden = 32 heads).
    """
    out, _, _ = _ops.planner_attention(
        q_src,
        kv_src,
        q_col,
        k_col,
        v_col,
        heads,
        q_norm.weight,
        q_norm.bias,
        k_norm.weight,
        k_norm.bias,
        mask,
        dropout_p,
        q_norm.eps,
        shared,
    )
    return out


@torch.library.register_fake("py123d_garage_cuda::planner_attention")
def _(
    q_src: Tensor,
    kv_src: Tensor,
    q_col: int,
    k_col: int,
    v_col: int,
    heads: int,
    q_weight: Tensor,
    q_bias: Tensor,
    k_weight: Tensor,
    k_bias: Tensor,
    mask: Tensor | None,
    dropout_p: float,
    eps: float,
    shared: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    """Fake planner_attention: the output, the per-row log-sum-exp and the dropout keep bits."""
    batch, queries, keys = q_src.shape[0], q_src.shape[1], kv_src.shape[1]
    out = q_src.new_empty(batch, queries, heads * _HEAD_DIM)
    words = batch * heads * -(-queries // 16) * -(-keys // _CHUNK) * 32 if dropout_p > 0.0 else 0
    return out, q_src.new_empty(batch, heads, queries, dtype=torch.float32), q_src.new_empty(words, dtype=torch.int32)


@torch.library.register_fake("py123d_garage_cuda::planner_attention_backward")
def _(
    grad: Tensor,
    q_src: Tensor,
    kv_src: Tensor,
    q_col: int,
    k_col: int,
    v_col: int,
    heads: int,
    q_weight: Tensor,
    q_bias: Tensor,
    k_weight: Tensor,
    k_bias: Tensor,
    mask: Tensor | None,
    dropout_p: float,
    keep_bits: Tensor,
    eps: float,
    out: Tensor,
    lse: Tensor,
    shared: bool,
) -> tuple[Tensor, ...]:
    """Fake planner_attention_backward: the gradients of q_src, kv_src (empty if shared) and the four norm params."""
    dq_src = torch.empty(q_src.shape, dtype=q_src.dtype, device=q_src.device)
    dkv_src = kv_src.new_empty(0) if shared else torch.empty(kv_src.shape, dtype=kv_src.dtype, device=kv_src.device)
    norm_grads = tuple(q_weight.new_empty(_HEAD_DIM) for _ in range(4))
    return dq_src, dkv_src, *norm_grads


@torch.library.register_fake("py123d_garage_cuda::planner_dropout_keep")
def _(keep_bits: Tensor, batch: int, heads: int, queries: int, keys: int, mask: Tensor | None) -> Tensor:
    """Fake planner_dropout_keep: the dense bool keep mask."""
    return keep_bits.new_empty(batch, heads, queries, keys, dtype=torch.bool)


def _setup_context(ctx: torch.autograd.function.FunctionCtx, inputs: tuple[object, ...], output: object) -> None:
    """Saves the inputs, the output, the log-sum-exp and the keep bits for the backward."""
    tensors = cast("tuple[Tensor, ...]", inputs[:2] + inputs[6:11])
    out, lse, keep_bits = cast("tuple[Tensor, Tensor, Tensor]", output)
    ctx.save_for_backward(*tensors, keep_bits, out, lse)  # pyright: ignore[reportAttributeAccessIssue]
    # q_col, k_col, v_col, heads, then dropout_p, eps, shared.
    ctx.arguments = (*inputs[2:6], *inputs[11:])  # pyright: ignore[reportAttributeAccessIssue]


def _backward(
    ctx: torch.autograd.function.FunctionCtx,
    grad: Tensor,
    _grad_lse: Tensor | None,
    _grad_keep_bits: Tensor | None,
) -> tuple[object, ...]:
    """Gradients of q_src, kv_src and the norm parameters through planner_attention_backward."""
    q_src, kv_src, q_weight, q_bias, k_weight, k_bias, mask, keep_bits, out, lse = ctx.saved_tensors  # pyright: ignore[reportAttributeAccessIssue]
    q_col, k_col, v_col, heads, dropout_p, eps, shared = ctx.arguments  # pyright: ignore[reportAttributeAccessIssue]
    dq_src, dkv_src, dq_weight, dq_bias, dk_weight, dk_bias = _ops.planner_attention_backward(
        grad,
        q_src,
        kv_src,
        q_col,
        k_col,
        v_col,
        heads,
        q_weight,
        q_bias,
        k_weight,
        k_bias,
        mask,
        dropout_p,
        keep_bits,
        eps,
        out,
        lse,
        shared,
    )
    # A shared source gets its whole gradient through q_src; autograd sums the two slots of the one tensor.
    return (
        dq_src,
        None if shared else dkv_src,
        None,
        None,
        None,
        None,
        dq_weight,
        dq_bias,
        dk_weight,
        dk_bias,
        None,
        None,
        None,
        None,
    )


torch.library.register_autograd("py123d_garage_cuda::planner_attention", _backward, setup_context=_setup_context)


def planner_swiglu_linear(
    x: jt.Float[Tensor, "*batch in_features"],
    weight: jt.Float[Tensor, "double_width in_features"],
    bias: jt.Float[Tensor, " double_width"],
    dropout_p: float,
) -> jt.Float[Tensor, "*batch width"]:
    """
    Computes dropout(silu(gate) * value) of [gate | value] = x weight^T + bias, the feed-forward's first projection and
    activation, in one GEMM.

    Args:
        x: the bf16 input, in_features a multiple of 64 up to 256.
        weight: the bf16 weight of 2 width rows, width a multiple of 8; the gate rows, then the value rows.
        bias: the bf16 bias.
        dropout_p: dropout probability of the activation; the mask is drawn from the CUDA generator.

    Returns:
        the bf16 activation.
    """
    hidden, _, _ = _ops.planner_swiglu_linear(x, weight, bias, dropout_p)
    return hidden


@torch.library.register_fake("py123d_garage_cuda::planner_swiglu_linear")
def _(x: Tensor, weight: Tensor, bias: Tensor, dropout_p: float) -> tuple[Tensor, Tensor, Tensor]:
    """Fake planner_swiglu_linear: the activation, the saved [rows, 2 width] gate and value and the dropout stream."""
    rows = x.numel() // x.shape[-1]
    hidden = x.new_empty(*x.shape[:-1], weight.shape[0] // 2)
    return hidden, x.new_empty(rows, weight.shape[0]), x.new_empty(2, dtype=torch.int64)


@torch.library.register_fake("py123d_garage_cuda::planner_swiglu_linear_backward")
def _(
    grad: Tensor,
    x: Tensor,
    weight: Tensor,
    gate_value: Tensor,
    dropout_p: float,
    rng: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Fake planner_swiglu_linear_backward: the gradients of x, weight and bias."""
    return torch.empty_like(x), torch.empty_like(weight), weight.new_empty(weight.shape[0])


def _swiglu_setup_context(ctx: torch.autograd.function.FunctionCtx, inputs: tuple[object, ...], output: object) -> None:
    """Saves x, weight, the gate and value, the dropout stream and dropout_p for the backward."""
    x, weight, _, dropout_p = cast("tuple[Tensor, Tensor, Tensor, float]", inputs)
    _, gate_value, rng = cast("tuple[Tensor, Tensor, Tensor]", output)
    ctx.save_for_backward(x, weight, gate_value, rng)  # pyright: ignore[reportAttributeAccessIssue]
    ctx.dropout_p = dropout_p  # pyright: ignore[reportAttributeAccessIssue]


def _swiglu_backward(
    ctx: torch.autograd.function.FunctionCtx,
    grad: Tensor,
    _grad_gate_value: Tensor | None,
    _grad_rng: Tensor | None,
) -> tuple[Tensor | None, ...]:
    """Gradients of x, weight and bias through planner_swiglu_linear_backward."""
    x, weight, gate_value, rng = ctx.saved_tensors  # pyright: ignore[reportAttributeAccessIssue]
    grad_x, grad_weight, grad_bias = _ops.planner_swiglu_linear_backward(
        grad,
        x,
        weight,
        gate_value,
        ctx.dropout_p,  # pyright: ignore[reportAttributeAccessIssue]
        rng,
    )
    return grad_x, grad_weight, grad_bias, None


torch.library.register_autograd(
    "py123d_garage_cuda::planner_swiglu_linear",
    _swiglu_backward,
    setup_context=_swiglu_setup_context,
)


def swiglu_linear(
    x: jt.Float[Tensor, "*batch in_features"],
    weight: jt.Float[Tensor, "double_width in_features"],
    bias: jt.Float[Tensor, " double_width"],
    dropout_p: float,
) -> jt.Float[Tensor, "*batch width"] | None:
    """
    Runs planner_swiglu_linear in bf16 under bf16 autocast on a GPU that runs it, with aligned halves and in_features.

    Args:
        x: the input.
        weight: the weight, gate rows then value rows.
        bias: the bias.
        dropout_p: dropout probability of the activation.

    Returns:
        the bf16 activation, or None where the op does not apply.
    """
    if not (x.is_cuda and torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == torch.bfloat16):
        note_fallback("planner_swiglu_linear", "needs bf16 autocast on CUDA")
        return None
    if weight.shape[0] % 16 or x.shape[-1] % _SWIGLU_K_BLOCK or x.shape[-1] > _SWIGLU_MAX_K:
        note_fallback(
            "planner_swiglu_linear",
            lambda: (
                f"needs output rows a multiple of 16 and in_features a multiple of {_SWIGLU_K_BLOCK} up to "
                f"{_SWIGLU_MAX_K}; got {weight.shape[0]} rows and {x.shape[-1]} in_features"
            ),
        )
        return None
    if not supports("planner_swiglu_linear", x.device):
        note_fallback("planner_swiglu_linear", lambda: str(unsupported_reason("planner_swiglu_linear", x.device)))
        return None
    dtype = torch.bfloat16
    return planner_swiglu_linear(x.to(dtype), weight.to(dtype), bias.to(dtype), dropout_p)


def planner_swiglu_feed_forward(
    x: jt.Float[Tensor, "*batch in_features"],
    w1: jt.Float[Tensor, "double_width in_features"],
    b1: jt.Float[Tensor, " double_width"],
    w2: jt.Float[Tensor, "out_features width"],
    b2: jt.Float[Tensor, " out_features"],
    dropout_p: float,
) -> jt.Float[Tensor, "*batch out_features"]:
    """
    Computes the feed-forward hidden w2^T + b2 of hidden = dropout(silu(gate) * value), [gate | value] = x w1^T + b1;
    the backward computes the activation's gradient in the epilogue of the second projection's GEMM.

    Args:
        x: the bf16 input, in_features a multiple of 64 up to 256.
        w1: the bf16 first weight of 2 width rows, width a multiple of 8; the gate rows, then the value rows.
        b1: the bf16 first bias.
        w2: the bf16 second weight, 256 out_features.
        b2: the bf16 second bias.
        dropout_p: dropout probability of the activation; the mask is drawn from the CUDA generator.

    Returns:
        the bf16 output.
    """
    out, _, _, _ = _ops.planner_swiglu_feed_forward(x, w1, b1, w2, b2, dropout_p)
    return out


@torch.library.register_fake("py123d_garage_cuda::planner_swiglu_feed_forward")
def _(
    x: Tensor,
    w1: Tensor,
    b1: Tensor,
    w2: Tensor,
    b2: Tensor,
    dropout_p: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Fake planner_swiglu_feed_forward: the output, the activation, the saved [gate | value] and the dropout stream."""
    rows = x.numel() // x.shape[-1]
    out = x.new_empty(*x.shape[:-1], w2.shape[0])
    hidden = x.new_empty(*x.shape[:-1], w1.shape[0] // 2)
    return out, hidden, x.new_empty(rows, w1.shape[0]), x.new_empty(2, dtype=torch.int64)


@torch.library.register_fake("py123d_garage_cuda::planner_swiglu_feed_forward_backward")
def _(
    grad: Tensor,
    x: Tensor,
    w1: Tensor,
    w2: Tensor,
    hidden: Tensor,
    gate_value: Tensor,
    dropout_p: float,
    rng: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Fake planner_swiglu_feed_forward_backward: the gradients of x, w1, b1, w2 and b2."""
    return (
        torch.empty_like(x),
        torch.empty_like(w1),
        w1.new_empty(w1.shape[0]),
        torch.empty_like(w2),
        w2.new_empty(w2.shape[0]),
    )


def _feed_forward_setup_context(
    ctx: torch.autograd.function.FunctionCtx,
    inputs: tuple[object, ...],
    output: object,
) -> None:
    """Saves x, both weights, the activation, the gate and value, the dropout stream and dropout_p for the backward."""
    x, w1, _, w2, _, dropout_p = cast("tuple[Tensor, Tensor, Tensor, Tensor, Tensor, float]", inputs)
    _, hidden, gate_value, rng = cast("tuple[Tensor, Tensor, Tensor, Tensor]", output)
    ctx.save_for_backward(x, w1, w2, hidden, gate_value, rng)  # pyright: ignore[reportAttributeAccessIssue]
    ctx.dropout_p = dropout_p  # pyright: ignore[reportAttributeAccessIssue]


def _feed_forward_backward(
    ctx: torch.autograd.function.FunctionCtx,
    grad: Tensor,
    _grad_hidden: Tensor | None,
    _grad_gate_value: Tensor | None,
    _grad_rng: Tensor | None,
) -> tuple[Tensor | None, ...]:
    """Gradients of x, w1, b1, w2 and b2 through planner_swiglu_feed_forward_backward."""
    x, w1, w2, hidden, gate_value, rng = ctx.saved_tensors  # pyright: ignore[reportAttributeAccessIssue]
    grads = _ops.planner_swiglu_feed_forward_backward(
        grad,
        x,
        w1,
        w2,
        hidden,
        gate_value,
        ctx.dropout_p,  # pyright: ignore[reportAttributeAccessIssue]
        rng,
    )
    return (*grads, None)


torch.library.register_autograd(
    "py123d_garage_cuda::planner_swiglu_feed_forward",
    _feed_forward_backward,
    setup_context=_feed_forward_setup_context,
)


def swiglu_feed_forward(
    x: jt.Float[Tensor, "*batch in_features"],
    w1: jt.Float[Tensor, "double_width in_features"],
    b1: jt.Float[Tensor, " double_width"],
    w2: jt.Float[Tensor, "out_features width"],
    b2: jt.Float[Tensor, " out_features"],
    dropout_p: float,
) -> jt.Float[Tensor, "*batch out_features"] | None:
    """
    Runs planner_swiglu_feed_forward in bf16 under bf16 autocast on a GPU that runs it, with aligned shapes and 256
    outputs.

    Args:
        x: the input.
        w1: the first weight, gate rows then value rows.
        b1: the first bias.
        w2: the second weight.
        b2: the second bias.
        dropout_p: dropout probability of the activation.

    Returns:
        the bf16 output, or None where the op does not apply.
    """
    if not (x.is_cuda and torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == torch.bfloat16):
        note_fallback("planner_swiglu_feed_forward", "needs bf16 autocast on CUDA")
        return None
    if w1.shape[0] % 16 or x.shape[-1] % _SWIGLU_K_BLOCK or x.shape[-1] > _SWIGLU_MAX_K:
        note_fallback(
            "planner_swiglu_feed_forward",
            lambda: (
                f"needs first-layer rows a multiple of 16 and in_features a multiple of {_SWIGLU_K_BLOCK} up to "
                f"{_SWIGLU_MAX_K}; got {w1.shape[0]} rows and {x.shape[-1]} in_features"
            ),
        )
        return None
    if w2.shape[0] != _SWIGLU_MAX_K or w2.shape[1] != w1.shape[0] // 2:
        note_fallback(
            "planner_swiglu_feed_forward",
            lambda: f"needs a second weight of [{_SWIGLU_MAX_K}, {w1.shape[0] // 2}]; got {list(w2.shape)}",
        )
        return None
    if not supports("planner_swiglu_feed_forward", x.device):
        note_fallback(
            "planner_swiglu_feed_forward",
            lambda: str(unsupported_reason("planner_swiglu_feed_forward", x.device)),
        )
        return None
    dtype = torch.bfloat16
    return planner_swiglu_feed_forward(
        x.to(dtype),
        w1.to(dtype),
        b1.to(dtype),
        w2.to(dtype).contiguous(),
        b2.to(dtype),
        dropout_p,
    )


def qk_norm_attention(
    attention: nn.MultiheadAttention,
    q_norm: nn.LayerNorm,
    k_norm: nn.LayerNorm,
    query: jt.Float[Tensor, "batch queries dim"],
    memory: jt.Float[Tensor, "batch keys dim"],
    attention_mask: jt.Bool[Tensor, "queries keys"] | None,
    dropout_p: float,
) -> jt.Float[Tensor, "batch queries dim"] | None:
    """
    Runs the input projections and normalized attention of py123d_garage's qk_norm_attention through planner_attention.

    Args:
        attention: the attention module whose input projection is used.
        q_norm: LayerNorm over each head's queries.
        k_norm: LayerNorm over each head's keys.
        query: the batch-first queries.
        memory: the batch-first keys and values; the same tensor as query for self-attention.
        attention_mask: True where a query may attend, or None.
        dropout_p: attention dropout probability.

    Returns:
        the attention output before the output projection, or None where the op does not apply (not bf16 autocast on
        CUDA, head dim not 32, too many keys, a GPU without the shared memory its backward needs).
    """
    dim = query.shape[-1]
    heads = cast(int, attention.num_heads)
    if not (query.is_cuda and torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == torch.bfloat16):
        note_fallback("planner_attention", "needs bf16 autocast on CUDA")
        return None
    if dim != heads * _HEAD_DIM or memory.shape[1] > _MAX_KEYS or query.shape[0] != memory.shape[0]:
        note_fallback(
            "planner_attention",
            lambda: (
                f"needs head dim {_HEAD_DIM}, at most {_MAX_KEYS} keys and one batch for query and memory; "
                f"got width {dim} for {heads} heads, {memory.shape[1]} keys, batches {query.shape[0]} and {memory.shape[0]}"
            ),
        )
        return None
    if q_norm.weight is None or q_norm.bias is None or k_norm.weight is None or k_norm.bias is None:
        note_fallback("planner_attention", "needs query and key norms with weight and bias")
        return None
    backward_bytes = _attention_backward_shared_bytes(
        query.shape[1],
        memory.shape[1],
        dropout_p > 0.0,
        attention_mask is not None,
    )
    if not supports("planner_attention", query.device, backward_bytes):
        note_fallback(
            "planner_attention",
            lambda: str(unsupported_reason("planner_attention", query.device, backward_bytes)),
        )
        return None
    mask = attention_mask.contiguous() if attention_mask is not None else None
    if query is memory:
        qkv = F.linear(query, attention.in_proj_weight, attention.in_proj_bias)
        return planner_attention(qkv, qkv, 0, dim, 2 * dim, heads, q_norm, k_norm, mask, dropout_p, shared=True)
    q = F.linear(query, attention.in_proj_weight[:dim], attention.in_proj_bias[:dim])
    kv = F.linear(memory, attention.in_proj_weight[dim:], attention.in_proj_bias[dim:])
    return planner_attention(q, kv, 0, 0, dim, heads, q_norm, k_norm, mask, dropout_p, shared=False)


def patch_planner_attention(module: nn.Module) -> None:
    """
    Routes the query/key-norm attention of every planning decoder layer inside module through planner_attention, in
    place; parameters and state_dict keys stay the same.

    Args:
        module: the module to patch.
    """
    for child in module.modules():
        if hasattr(child, "self_q_norm"):
            setattr(child, "fused_qk_norm_attention", qk_norm_attention)  # noqa: B010


def patch_planner_swiglu(module: nn.Module) -> None:
    """
    Routes the feed-forward's first projection and activation of every planning decoder layer inside module through
    planner_swiglu_linear.

    Args:
        module: the module to patch.
    """
    for child in module.modules():
        if hasattr(child, "self_q_norm"):
            setattr(child, "fused_swiglu_linear", swiglu_linear)  # noqa: B010


def patch_planner_feed_forward(module: nn.Module) -> None:
    """
    Routes the whole feed-forward of every planning decoder layer inside module through planner_swiglu_feed_forward.

    Args:
        module: the module to patch.
    """
    for child in module.modules():
        if hasattr(child, "self_q_norm"):
            setattr(child, "fused_feed_forward", swiglu_feed_forward)  # noqa: B010
