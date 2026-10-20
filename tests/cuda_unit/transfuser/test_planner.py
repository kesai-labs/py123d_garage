# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

"""
The planner's query/key-norm attention, SwiGLU and adaptive-norm ops match a float64 reference in values and every
gradient, with and without masks and dropout (the reference drops what the op's own keep mask drops), as closely as
torch's bf16 path does; the attention repeats bitwise for a CUDA generator state, the SwiGLU ops run on a fresh thread,
and the fused adaptive-norm decoder matches the unfused one.
"""

from __future__ import annotations

import copy
import math
import threading

import pytest
import torch
import torch.nn.functional as F
from torch import nn

if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)
py123d_garage_cuda = pytest.importorskip("py123d_garage_cuda")

from py123d_garage_cuda.transfuser.registry import unsupported_reason  # noqa: E402

_ops = torch.ops.py123d_garage_cuda
_SWIGLU_UNSUPPORTED = unsupported_reason("planner_swiglu_linear")
_ADAPTIVE_NORM_UNSUPPORTED = unsupported_reason("planner_adaptive_norm")
_swiglu = pytest.mark.skipif(_SWIGLU_UNSUPPORTED is not None, reason=_SWIGLU_UNSUPPORTED or "")
_adaptive_norm = pytest.mark.skipif(_ADAPTIVE_NORM_UNSUPPORTED is not None, reason=_ADAPTIVE_NORM_UNSUPPORTED or "")


def _skip_unless_attention_runs(queries: int, keys: int, dropout_p: float, masked: bool) -> None:
    """Skips where this device lacks the shared memory the backward needs for the shape."""
    backward_bytes = _ops.planner_attention_backward_shared_bytes(queries, keys, dropout_p > 0.0, masked)
    reason = unsupported_reason("planner_attention", shared_memory_bytes=backward_bytes)
    if reason is not None:
        pytest.skip(reason)


def _planner_masks(steps: int, path_points: int, context: int, target_points: int) -> tuple[torch.Tensor, ...]:
    """
    The midtrain planner's self mask and a cross mask hiding the target points from the waypoints and actions.

    In the self mask the waypoints, actions and path read neither other group, and all read the path.
    """
    queries = 2 * steps + path_points
    self_mask = torch.ones(queries, queries, dtype=torch.bool, device="cuda")
    self_mask[:steps, steps : 2 * steps] = False
    self_mask[steps : 2 * steps, :steps] = False
    self_mask[2 * steps :, : 2 * steps] = False
    cross_mask = torch.ones(queries, context, dtype=torch.bool, device="cuda")
    cross_mask[: 2 * steps, context - target_points :] = False
    return self_mask, cross_mask


def _norms(seed: int) -> tuple[nn.LayerNorm, nn.LayerNorm]:
    torch.manual_seed(seed)
    q_norm, k_norm = nn.LayerNorm(32).cuda(), nn.LayerNorm(32).cuda()
    for norm in (q_norm, k_norm):
        nn.init.normal_(norm.weight, 1.0, 0.3)
        nn.init.normal_(norm.bias, 0.0, 0.3)
    return q_norm, k_norm


def _reference(
    q_src: torch.Tensor,
    kv_src: torch.Tensor,
    columns: tuple[int, int, int],
    heads: int,
    norms: list[torch.Tensor],
    mask: torch.Tensor | None,
    keep: torch.Tensor | None,
    dropout_p: float,
    eps: float = 1e-5,
) -> torch.Tensor:
    q_col, k_col, v_col = columns
    batch, queries, dim = q_src.shape[0], q_src.shape[1], 32 * heads

    def heads_of(source: torch.Tensor, col: int) -> torch.Tensor:
        return source[..., col : col + dim].unflatten(-1, (heads, 32)).transpose(1, 2)

    q = F.layer_norm(heads_of(q_src, q_col), (32,), norms[0], norms[1], eps)
    k = F.layer_norm(heads_of(kv_src, k_col), (32,), norms[2], norms[3], eps)
    v = heads_of(kv_src, v_col)
    scores = q @ k.transpose(-1, -2) / math.sqrt(32)
    if mask is not None:
        scores = scores.masked_fill(~mask, -math.inf)
    probabilities = scores.softmax(-1)
    if keep is not None:
        probabilities = probabilities * keep / (1.0 - dropout_p)
    return (probabilities @ v).transpose(1, 2).reshape(batch, queries, dim)


def _torch_bf16(
    q_src: torch.Tensor,
    kv_src: torch.Tensor,
    columns: tuple[int, int, int],
    heads: int,
    norms: list[torch.Tensor],
    mask: torch.Tensor | None,
) -> torch.Tensor:
    """The model's own path: fp32 norms, then bf16 attention."""
    q_col, k_col, v_col = columns
    dim = 32 * heads

    def heads_of(source: torch.Tensor, col: int) -> torch.Tensor:
        return source[..., col : col + dim].unflatten(-1, (heads, 32)).transpose(1, 2)

    q = F.layer_norm(heads_of(q_src, q_col).float(), (32,), norms[0], norms[1]).bfloat16()
    k = F.layer_norm(heads_of(kv_src, k_col).float(), (32,), norms[2], norms[3]).bfloat16()
    out = F.scaled_dot_product_attention(q, k, heads_of(kv_src, v_col), attn_mask=mask)
    return out.transpose(1, 2).flatten(2)


def _relative_error(actual: torch.Tensor | None, expected: torch.Tensor | None, scale: float = 0.0) -> float:
    assert actual is not None
    assert expected is not None
    return ((actual.double() - expected.double()).norm() / max(expected.double().norm().item(), scale, 1e-30)).item()


def _norm_errors(params: list[torch.Tensor], references: list[torch.Tensor]) -> dict[str, float]:
    # A bias shared by every key shifts each score row by a constant, which softmax ignores: the key-norm bias
    # gradient is zero up to rounding, so it is measured against the key-norm weight gradient.
    key_scale = (references[2].grad.double().norm().item()) if references[2].grad is not None else 0.0
    return {
        name: _relative_error(param.grad, reference.grad, key_scale if name == "kb" else 0.0)
        for name, param, reference in zip(("qw", "qb", "kw", "kb"), params, references, strict=True)
    }


# (batch, queries, keys, heads, shared, mask kind): the planner's self- and cross-attention shapes, then odd sizes; the
# unmasked shapes with key and query counts multiples of 64 run the Hopper backward.
CASES = [
    (3, 192, 192, 8, True, "planner"),
    (3, 192, 192, 8, True, None),
    (2, 128, 64, 4, False, None),
    (2, 768, 76, 8, False, "planner"),
    (2, 768, 76, 8, False, None),
    (2, 50, 37, 2, False, "random"),
    (3, 23, 23, 3, True, "random"),
]


def _case_inputs(
    batch: int,
    queries: int,
    keys: int,
    heads: int,
    shared: bool,
    mask_kind: str | None,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int, int], torch.Tensor | None]:
    torch.manual_seed(1)
    dim = 32 * heads
    if shared:
        q_src = (torch.randn(batch, queries, 3 * dim, device="cuda") * 2 + 0.3).bfloat16()
        kv_src, columns = q_src, (0, dim, 2 * dim)
    else:
        q_src = (torch.randn(batch, queries, dim, device="cuda") * 2 + 0.3).bfloat16()
        kv_src = (torch.randn(batch, keys, 2 * dim, device="cuda") * 2 - 0.2).bfloat16()
        columns = (0, 0, dim)
    mask = None
    if mask_kind == "planner":
        self_mask, cross_mask = _planner_masks(64, 64, keys, 3)
        mask = self_mask if shared else cross_mask.repeat(queries // 192, 1)
    elif mask_kind == "random":
        mask = torch.rand(queries, keys, device="cuda") > 0.4
        mask[:, 0] = True
    return q_src, kv_src, columns, mask


def _run_op(
    q_src: torch.Tensor,
    kv_src: torch.Tensor,
    columns: tuple[int, int, int],
    heads: int,
    norms: list[torch.Tensor],
    mask: torch.Tensor | None,
    dropout_p: float,
    shared: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The output and the forward's keep bits."""
    out, _, keep_bits = _ops.planner_attention(
        q_src,
        kv_src,
        *columns,
        heads,
        *norms,
        mask,
        dropout_p,
        1e-5,
        shared,
    )
    return out, keep_bits


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("dropout_p", [0.0, 0.1])
def test_matches_float64_reference(case: tuple[int, int, int, int, bool, str | None], dropout_p: float) -> None:
    batch, queries, keys, heads, shared, mask_kind = case
    _skip_unless_attention_runs(queries, keys, dropout_p, mask_kind is not None)
    q_src, kv_src, columns, mask = _case_inputs(batch, queries, keys, heads, shared, mask_kind)
    q_norm, k_norm = _norms(2)
    sources = [q_src] if shared else [q_src, kv_src]
    leaves = [tensor.detach().clone().requires_grad_() for tensor in sources]
    norms = [param.detach().clone().requires_grad_() for param in (*q_norm.parameters(), *k_norm.parameters())]
    out, keep_bits = _run_op(leaves[0], leaves[-1], columns, heads, norms, mask, dropout_p, shared)
    keep = None
    if dropout_p > 0:
        keep = _ops.planner_dropout_keep(keep_bits, batch, heads, queries, keys, mask)
        attended = torch.ones_like(keep) if mask is None else mask.expand_as(keep)
        # Within 4 standard deviations of the keep rate of independent draws.
        tolerance = 4 * math.sqrt(dropout_p * (1 - dropout_p) / attended.sum().item())
        assert abs(keep[attended].float().mean().item() - (1 - dropout_p)) < tolerance

    reference_leaves = [tensor.detach().double().requires_grad_() for tensor in sources]
    reference_norms = [param.detach().double().requires_grad_() for param in norms]
    reference = _reference(
        reference_leaves[0],
        reference_leaves[-1],
        columns,
        heads,
        reference_norms,
        mask,
        keep,
        dropout_p,
    )

    torch.manual_seed(3)
    grad = torch.randn_like(reference).bfloat16()
    (out.float() * grad.float()).sum().backward()
    (reference * grad.double()).sum().backward()
    errors = {"out": _relative_error(out, reference)}
    for index, (leaf, reference_leaf) in enumerate(zip(leaves, reference_leaves, strict=True)):
        errors[f"source{index}"] = _relative_error(leaf.grad, reference_leaf.grad)
    errors |= _norm_errors(norms, reference_norms)

    if dropout_p == 0.0:
        # As close as torch's own bf16 path.
        torch_leaves = [tensor.detach().clone().requires_grad_() for tensor in sources]
        torch_norms = [param.detach().clone().requires_grad_() for param in norms]
        torch_out = _torch_bf16(torch_leaves[0], torch_leaves[-1], columns, heads, torch_norms, mask)
        (torch_out.float() * grad.float()).sum().backward()
        torch_errors = {"out": _relative_error(torch_out, reference)}
        for index, (leaf, reference_leaf) in enumerate(zip(torch_leaves, reference_leaves, strict=True)):
            torch_errors[f"source{index}"] = _relative_error(leaf.grad, reference_leaf.grad)
        torch_errors |= _norm_errors(torch_norms, reference_norms)
        for name, error in errors.items():
            assert error <= 1.5 * torch_errors[name] + 2e-3, (name, errors, torch_errors)
    else:
        for name, error in errors.items():
            assert error < 1.5e-2, (name, errors)


@pytest.mark.parametrize("mask_kind", ["planner", None])
def test_repeats_bitwise_for_a_generator_state(mask_kind: str | None) -> None:
    _skip_unless_attention_runs(192, 192, 0.1, mask_kind is not None)
    q_src, _, columns, mask = _case_inputs(2, 192, 192, 8, True, mask_kind)
    q_norm, k_norm = _norms(4)
    results = []
    for _ in range(2):
        torch.cuda.manual_seed(42)
        leaf = q_src.detach().clone().requires_grad_()
        norms = [param.detach().clone().requires_grad_() for param in (*q_norm.parameters(), *k_norm.parameters())]
        out, _ = _run_op(leaf, leaf, columns, 8, norms, mask, 0.1, True)
        out.backward(torch.ones_like(out))
        results.append([out, leaf.grad, *(param.grad for param in norms)])
    for first, second in zip(*results, strict=True):
        assert torch.equal(first, second)


def test_new_call_new_mask() -> None:
    q_src, _, columns, _ = _case_inputs(1, 64, 64, 2, True, None)
    norms = [param.detach() for norm in _norms(5) for param in norm.parameters()]
    keeps = []
    for _ in range(2):
        _, keep_bits = _run_op(q_src, q_src, columns, 2, norms, None, 0.5, True)
        keeps.append(_ops.planner_dropout_keep(keep_bits, 1, 2, 64, 64, None))
    assert 0.4 < (keeps[0] != keeps[1]).float().mean().item() < 0.6


def test_opcheck() -> None:
    _skip_unless_attention_runs(50, 37, 0.1, True)
    q_src, kv_src, columns, mask = _case_inputs(2, 50, 37, 2, False, "random")
    q_norm, k_norm = _norms(5)
    norms = [param.detach().clone().requires_grad_() for param in (*q_norm.parameters(), *k_norm.parameters())]
    torch.library.opcheck(
        _ops.planner_attention.default,
        (q_src.requires_grad_(), kv_src.requires_grad_(), *columns, 2, *norms, mask, 0.1, 1e-5, False),
    )


def _swiglu_linear_path(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    keep: torch.Tensor,
    dropout_p: float,
) -> torch.Tensor:
    gate, value = F.linear(x, weight, bias).chunk(2, dim=-1)
    return F.silu(gate) * value * keep / (1.0 - dropout_p)


# (width, hidden width, in_features): the planner's padded feed-forward, then a narrow one with a partial tile.
SWIGLU_CASES = [(1376, 1365, 256), (104, 100, 128)]


@_swiglu
@pytest.mark.parametrize("case", SWIGLU_CASES)
@pytest.mark.parametrize("dropout_p", [0.0, 0.1])
def test_swiglu_linear_matches_float64_reference(case: tuple[int, int, int], dropout_p: float) -> None:
    torch.manual_seed(6)
    width, hidden_width, in_features = case
    x = torch.randn(3, 50, in_features, device="cuda").bfloat16()
    weight = (torch.randn(2, width, in_features, device="cuda") / in_features**0.5).bfloat16()
    bias = (torch.randn(2, width, device="cuda") / 4).bfloat16()
    # Padding rows past the hidden width are zero in both halves, as in the padded planner feed-forward.
    weight[:, hidden_width:] = 0
    bias[:, hidden_width:] = 0
    weight, bias = weight.flatten(0, 1), bias.flatten()
    leaves = [tensor.detach().clone().requires_grad_() for tensor in (x, weight, bias)]
    hidden, gate_value, _ = _ops.planner_swiglu_linear(*leaves, dropout_p)
    # Dropped elements are the op's exact zeros; random inputs leave no other zero products.
    gate, value = gate_value.detach().double().view(*hidden.shape[:-1], -1).chunk(2, dim=-1)
    keep = (hidden.detach() != 0) | (gate * value == 0)
    if dropout_p > 0:
        assert abs(keep[..., :hidden_width].double().mean().item() - (1 - dropout_p)) < 0.01

    grad = torch.randn_like(hidden).bfloat16()
    (hidden.float() * grad.float()).sum().backward()
    reference_x, reference_weight, reference_bias = (
        tensor.detach().double().requires_grad_() for tensor in (x, weight, bias)
    )
    reference_leaves = [reference_x, reference_weight, reference_bias]
    reference = _swiglu_linear_path(reference_x, reference_weight, reference_bias, keep.double(), dropout_p)
    (reference * grad.double()).sum().backward()
    torch_x, torch_weight, torch_bias = (tensor.detach().clone().requires_grad_() for tensor in (x, weight, bias))
    torch_leaves = [torch_x, torch_weight, torch_bias]
    torch_hidden = _swiglu_linear_path(torch_x, torch_weight, torch_bias, keep.bfloat16(), dropout_p)
    (torch_hidden.float() * grad.float()).sum().backward()

    pairs: dict[str, tuple[torch.Tensor | None, ...]] = {"hidden": (hidden, torch_hidden, reference)}
    for name, leaf, torch_leaf, reference_leaf in zip(
        ("x", "weight", "bias"),
        leaves,
        torch_leaves,
        reference_leaves,
        strict=True,
    ):
        pairs[name] = (leaf.grad, torch_leaf.grad, reference_leaf.grad)
    for name, (actual, torch_actual, expected) in pairs.items():
        error, torch_error = _relative_error(actual, expected), _relative_error(torch_actual, expected)
        assert error <= 1.5 * torch_error + 2e-3, (name, error, torch_error)
    assert leaves[1].grad is not None
    assert not leaves[1].grad.view(2, width, in_features)[:, hidden_width:].any()


def _feed_forward_path(
    x: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    keep: torch.Tensor,
    dropout_p: float,
) -> torch.Tensor:
    return F.linear(_swiglu_linear_path(x, w1, b1, keep, dropout_p), w2, b2)


@_swiglu
@pytest.mark.parametrize("case", SWIGLU_CASES)
@pytest.mark.parametrize("dropout_p", [0.0, 0.1])
def test_swiglu_feed_forward_matches_float64_reference(case: tuple[int, int, int], dropout_p: float) -> None:
    torch.manual_seed(7)
    width, hidden_width, in_features = case
    x = torch.randn(3, 50, in_features, device="cuda").bfloat16()
    w1 = (torch.randn(2, width, in_features, device="cuda") / in_features**0.5).bfloat16()
    b1 = (torch.randn(2, width, device="cuda") / 4).bfloat16()
    # Padding rows past the hidden width are zero, as in the padded planner feed-forward.
    w1[:, hidden_width:] = 0
    b1[:, hidden_width:] = 0
    w1, b1 = w1.flatten(0, 1), b1.flatten()
    w2 = (torch.randn(256, width, device="cuda") / width**0.5).bfloat16()
    w2[:, hidden_width:] = 0
    b2 = (torch.randn(256, device="cuda") / 4).bfloat16()
    tensors = (x, w1, b1, w2, b2)
    leaves = [tensor.detach().clone().requires_grad_() for tensor in tensors]
    out, hidden, gate_value, _ = _ops.planner_swiglu_feed_forward(*leaves, dropout_p)
    gate, value = gate_value.detach().double().view(*hidden.shape[:-1], -1).chunk(2, dim=-1)
    keep = (hidden.detach() != 0) | (gate * value == 0)
    grad = torch.randn_like(out).bfloat16()
    (out.float() * grad.float()).sum().backward()
    reference_leaves = [tensor.detach().double().requires_grad_() for tensor in tensors]
    reference_x, reference_w1, reference_b1, reference_w2, reference_b2 = reference_leaves
    reference = _feed_forward_path(
        reference_x,
        reference_w1,
        reference_b1,
        reference_w2,
        reference_b2,
        keep.double(),
        dropout_p,
    )
    (reference * grad.double()).sum().backward()
    torch_leaves = [tensor.detach().clone().requires_grad_() for tensor in tensors]
    torch_x, torch_w1, torch_b1, torch_w2, torch_b2 = torch_leaves
    torch_out = _feed_forward_path(torch_x, torch_w1, torch_b1, torch_w2, torch_b2, keep.bfloat16(), dropout_p)
    (torch_out.float() * grad.float()).sum().backward()
    pairs: dict[str, tuple[torch.Tensor | None, ...]] = {"out": (out, torch_out, reference)}
    for name, leaf, torch_leaf, reference_leaf in zip(
        ("x", "w1", "b1", "w2", "b2"),
        leaves,
        torch_leaves,
        reference_leaves,
        strict=True,
    ):
        pairs[name] = (leaf.grad, torch_leaf.grad, reference_leaf.grad)
    for name, (actual, torch_actual, expected) in pairs.items():
        error, torch_error = _relative_error(actual, expected), _relative_error(torch_actual, expected)
        assert error <= 1.5 * torch_error + 2e-3, (name, error, torch_error)
    assert leaves[1].grad is not None
    assert not leaves[1].grad.view(2, width, in_features)[:, hidden_width:].any()


@_swiglu
def test_swiglu_feed_forward_opcheck() -> None:
    x = torch.randn(4, 64, device="cuda").bfloat16().requires_grad_()
    w1 = torch.randn(32, 64, device="cuda").bfloat16().requires_grad_()
    b1 = torch.randn(32, device="cuda").bfloat16().requires_grad_()
    w2 = torch.randn(256, 16, device="cuda").bfloat16().requires_grad_()
    b2 = torch.randn(256, device="cuda").bfloat16().requires_grad_()
    torch.library.opcheck(_ops.planner_swiglu_feed_forward.default, (x, w1, b1, w2, b2, 0.1))


@_swiglu
def test_swiglu_linear_opcheck() -> None:
    x = torch.randn(4, 64, device="cuda").bfloat16().requires_grad_()
    weight = torch.randn(32, 64, device="cuda").bfloat16().requires_grad_()
    bias = torch.randn(32, device="cuda").bfloat16().requires_grad_()
    torch.library.opcheck(_ops.planner_swiglu_linear.default, (x, weight, bias, 0.1))


@_swiglu
def test_swiglu_ops_on_a_fresh_thread() -> None:
    # The second warm-up's freed blocks serve the thread's allocations and without dropout no generator call binds the
    # device's context, so the first tensor-map encode is the thread's first CUDA call, as on an autograd device thread;
    # unbound, it failed with CUDA error 201.
    torch.manual_seed(8)
    x = torch.randn(150, 128, device="cuda").bfloat16()
    w1 = (torch.randn(208, 128, device="cuda") / 16).bfloat16()
    b1 = (torch.randn(208, device="cuda") / 4).bfloat16()
    w2 = (torch.randn(256, 104, device="cuda") / 16).bfloat16()
    b2 = (torch.randn(256, device="cuda") / 4).bfloat16()
    grad = torch.randn(150, 256, device="cuda").bfloat16()

    def run() -> tuple[torch.Tensor, ...]:
        out, hidden, gate_value, rng = _ops.planner_swiglu_feed_forward(x, w1, b1, w2, b2, 0.0)
        grads = _ops.planner_swiglu_feed_forward_backward(grad, x, w1, w2, hidden, gate_value, 0.0, rng)
        return out, hidden, gate_value, *grads, _ops.planner_swiglu_linear(x, w1, b1, 0.1)[0].isfinite().all()

    expected = run()
    run()
    torch.cuda.synchronize()
    outcomes: list[tuple[torch.Tensor, ...] | BaseException] = []

    def target() -> None:
        try:
            outcomes.append(run())
            torch.cuda.synchronize()
        except BaseException as error:
            outcomes.append(error)

    thread = threading.Thread(target=target)
    thread.start()
    thread.join()
    outcome = outcomes[0]
    assert not isinstance(outcome, BaseException), outcome
    for actual, reference in zip(outcome, expected, strict=True):
        assert torch.equal(actual, reference)


def _adaptive_norm_path(
    x: torch.Tensor,
    branch: torch.Tensor | None,
    modulation: torch.Tensor,
    norm: list[torch.Tensor],
    keep: torch.Tensor,
    dropout_p: float,
    modulated: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """x_out and y of the module's math in x's dtype; 1 + gate and 1 + scale round to bf16 as in the model."""
    dim = x.shape[-1]
    gate, shift, scale = (modulation[:, k * dim : (k + 1) * dim, None].transpose(1, 2) for k in range(3))
    if branch is not None:
        x = x + (1 + gate).bfloat16().to(x.dtype) * branch.to(x.dtype) * keep / (1.0 - dropout_p)
    y = F.layer_norm(x, (dim,), norm[0], norm[1])
    if modulated:
        y = y * (1 + scale).bfloat16().to(x.dtype) + shift.to(x.dtype)
    return x, y


@_adaptive_norm
@pytest.mark.parametrize(("residual", "modulated"), [(True, True), (False, True), (True, False)])
@pytest.mark.parametrize("dropout_p", [0.0, 0.1])
def test_adaptive_norm_matches_float64_reference(residual: bool, modulated: bool, dropout_p: float) -> None:
    torch.manual_seed(8)
    batch, queries, dim = 6, 192, 256
    x = torch.randn(batch, queries, dim, device="cuda") * 3 + 0.5
    branch = torch.randn(batch, queries, dim, device="cuda").bfloat16() if residual else None
    # gate, shift and scale as row-strided slices of one modulation, as the decoder's chunks are.
    modulation = (torch.randn(batch, 4 * dim, device="cuda") * 0.3).bfloat16()
    norm = [torch.randn(dim, device="cuda") * 0.2 + 1, torch.randn(dim, device="cuda") * 0.1]
    leaves = [tensor.detach().clone().requires_grad_() if tensor is not None else None for tensor in (x, branch)]
    modulation_leaf = modulation.detach().clone().requires_grad_()
    norm_leaves = [param.detach().clone().requires_grad_() for param in norm]
    gate, shift, scale = (modulation_leaf[:, k * dim : (k + 1) * dim] for k in range(3))
    x_out, y, _, _ = _ops.planner_adaptive_norm(
        leaves[0],
        leaves[1],
        gate if residual else None,
        shift if modulated else None,
        scale if modulated else None,
        *norm_leaves,
        dropout_p,
        1e-5,
    )
    keep = torch.ones_like(x)
    if residual and dropout_p > 0:
        # Dropped elements leave x unchanged; random inputs leave no other unchanged element.
        keep = (x_out.detach() != x).float()
        assert abs(keep.mean().item() - (1 - dropout_p)) < 0.01
    grad_y, grad_x = torch.randn_like(x).to(y.dtype), torch.randn_like(x)
    loss = (y.float() * grad_y.float()).sum() + ((x_out * grad_x).sum() if residual else 0.0)
    loss.backward()

    def leaves_of(dtype: torch.dtype) -> tuple[list[torch.Tensor | None], torch.Tensor, list[torch.Tensor]]:
        return (
            [tensor.detach().to(dtype).requires_grad_() if tensor is not None else None for tensor in (x, branch)],
            modulation.detach().clone().requires_grad_(),
            [param.detach().to(dtype).requires_grad_() for param in norm],
        )

    errors: dict[str, list[float]] = {}
    expected: list[torch.Tensor | None] = []
    for dtype in (torch.float64, torch.float32):
        path_leaves, path_modulation, path_norm = leaves_of(dtype)
        assert path_leaves[0] is not None
        path_x, path_y = _adaptive_norm_path(
            path_leaves[0],
            path_leaves[1],
            path_modulation.to(dtype),
            path_norm,
            keep.to(dtype),
            dropout_p,
            modulated,
        )
        if dtype == torch.float32 and modulated:
            path_y = path_y.bfloat16()
        path_loss = (path_y.to(dtype) * grad_y.to(dtype)).sum() + (
            (path_x * grad_x.to(dtype)).sum() if residual else 0.0
        )
        path_loss.backward()
        results = [
            path_y,
            path_x if residual else None,
            *(leaf.grad if leaf is not None else None for leaf in path_leaves),
        ]
        results += [path_modulation.grad, *(param.grad for param in path_norm)]
        if dtype == torch.float64:
            expected = results
            continue
        actual = [y, x_out if residual else None, *(leaf.grad if leaf is not None else None for leaf in leaves)]
        actual += [modulation_leaf.grad, *(param.grad for param in norm_leaves)]
        for name, ours, torch_result, reference in zip(
            ("y", "x_out", "x", "branch", "modulation", "weight", "bias"),
            actual,
            results,
            expected,
            strict=True,
        ):
            if reference is not None:
                errors[name] = [_relative_error(ours, reference), _relative_error(torch_result, reference)]
    for name, (error, torch_error) in errors.items():
        assert error <= 1.5 * torch_error + 2e-3, (name, errors)


@_adaptive_norm
def test_adaptive_norm_opcheck() -> None:
    torch.manual_seed(9)
    x = torch.randn(2, 5, 256, device="cuda").requires_grad_()
    branch = torch.randn(2, 5, 256, device="cuda").bfloat16().requires_grad_()
    modulation = torch.randn(2, 3 * 256, device="cuda").bfloat16().requires_grad_()
    gate, shift, scale = modulation[:, :256], modulation[:, 256:512], modulation[:, 512:]
    weight = torch.randn(256, device="cuda").requires_grad_()
    bias = torch.randn(256, device="cuda").requires_grad_()
    torch.library.opcheck(
        _ops.planner_adaptive_norm.default,
        (x, branch, gate, shift, scale, weight, bias, 0.1, 1e-5),
    )


@_adaptive_norm
def test_fused_adaptive_norm_decoder_matches_unfused() -> None:
    from py123d_garage_cuda.transfuser.adaptive_norm import patch_planner_adaptive_norm
    from py123d_garage_cuda.transfuser.planner import patch_planner_attention, patch_planner_swiglu

    from py123d_garage.policy.transfuser.network.decoder.flow_planning_decoder import (
        AdaptiveNormDecoderLayer,
        AdaptiveNormTransformerDecoder,
    )

    torch.manual_seed(10)
    decoder = AdaptiveNormTransformerDecoder(256, 8, 2).cuda().eval()
    for layer in decoder.modules():
        if isinstance(layer, AdaptiveNormDecoderLayer):
            nn.init.normal_(layer.modulation.weight, 0.0, 0.02)
            nn.init.normal_(layer.modulation.bias, 0.0, 0.1)
    fused, unfused, reference = copy.deepcopy(decoder), copy.deepcopy(decoder), copy.deepcopy(decoder).double()
    for patched in (fused, unfused):
        patch_planner_attention(patched)
        patch_planner_swiglu(patched)
    patch_planner_adaptive_norm(fused)
    batch, queries, tokens = 4, 192, 73
    inputs = [torch.randn(batch, queries, 256), torch.randn(batch, tokens, 256), torch.randn(batch, 256)]
    self_mask, cross_mask = _planner_masks(64, 64, tokens, 3)
    grad = torch.randn(batch, queries, 256, device="cuda")

    def run(model: nn.Module, dtype: torch.dtype) -> list[torch.Tensor | None]:
        leaves = [tensor.to("cuda", dtype).requires_grad_() for tensor in inputs]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype == torch.float32):
            out = model(*leaves, self_mask, cross_mask)
        (out.to(dtype) * grad.to(dtype)).sum().backward()
        return [out, *(leaf.grad for leaf in leaves), *(param.grad for param in model.parameters())]

    expected = run(reference, torch.float64)
    names = ["out", "tgt", "memory", "conditioning"] + [name for name, _ in decoder.named_parameters()]
    for name, ours, theirs, reference_result in zip(
        names,
        run(fused, torch.float32),
        run(unfused, torch.float32),
        expected,
        strict=True,
    ):
        # The key-norm biases shift every score of a row alike: their exact gradient is zero.
        if "k_norm.bias" in name:
            continue
        error, unfused_error = _relative_error(ours, reference_result), _relative_error(theirs, reference_result)
        assert error <= 1.5 * unfused_error + 2e-3, (name, error, unfused_error)
