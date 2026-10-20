# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 KE:SAI

from __future__ import annotations

from collections.abc import Callable
from typing import Final, cast

import jaxtyping as jt
import torch
import torch.nn.functional as F
from torch import nn
from typing_extensions import override

# bf16 elements per 64 bytes: hidden rows of whole 32-byte memory sectors, which the GEMM writing [gate | value] needs
# to store at full speed.
_HIDDEN_ALIGNMENT: Final[int] = 32

# The hooks a planning kernel patch may set on a layer, each returning None where it does not apply to the inputs.
# (attention, q_norm, k_norm, query, memory, attention_mask, dropout_p) to the attention output before the output
# projection.
FusedQkNormAttention = Callable[
    [nn.MultiheadAttention, nn.LayerNorm, nn.LayerNorm, torch.Tensor, torch.Tensor, torch.Tensor | None, float],
    torch.Tensor | None,
]
# (x, weight1, bias1, dropout_p) to the feed-forward's hidden activation after its dropout.
FusedSwigluLinear = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, float], torch.Tensor | None]
# (x, weight1, bias1, weight2, bias2, dropout_p) to the feed-forward output before the residual dropout.
FusedFeedForward = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float],
    torch.Tensor | None,
]


class PlanningDecoderLayer(nn.TransformerDecoderLayer):
    """Pre-norm transformer decoder layer with per-head query and key norms, a SwiGLU feed-forward and optional gates."""

    fused_qk_norm_attention: FusedQkNormAttention | None = None
    fused_swiglu_linear: FusedSwigluLinear | None = None
    fused_feed_forward: FusedFeedForward | None = None

    def __init__(self, d_model: int, nhead: int, gated_attention: bool = False) -> None:
        """
        Initializes the layer with its query and key norms, SwiGLU projections and optional attention gates.

        Args:
            d_model: width of the queries and the memory.
            nhead: attention heads of both attention blocks.
            gated_attention: gate both attention outputs by a sigmoid of a projection of their queries.
        """
        super().__init__(d_model, nhead, batch_first=True, norm_first=True)  # pyright: ignore[reportUnknownMemberType]
        head_dim = d_model // nhead
        self.self_q_norm = nn.LayerNorm(head_dim)
        self.self_k_norm = nn.LayerNorm(head_dim)
        self.cross_q_norm = nn.LayerNorm(head_dim)
        self.cross_k_norm = nn.LayerNorm(head_dim)
        hidden_dim = 2 * self.linear1.out_features // 3
        self.linear1 = nn.Linear(d_model, 2 * hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, d_model)
        self.self_attn_gate = nn.Linear(d_model, d_model) if gated_attention else None
        self.cross_attn_gate = nn.Linear(d_model, d_model) if gated_attention else None

    @override
    def _sa_block(
        self,
        x: jt.Float[torch.Tensor, "batch queries dim"],
        attn_mask: jt.Bool[torch.Tensor, "queries queries"] | None,
        key_padding_mask: torch.Tensor | None,
        is_causal: bool = False,
        dropout: bool = True,
    ) -> jt.Float[torch.Tensor, "batch queries dim"]:
        """Self-attention with normalized queries and keys; dropout False leaves the residual dropout to the caller."""
        assert key_padding_mask is None
        assert not is_causal
        out = qk_norm_attention(
            self.self_attn,
            self.self_q_norm,
            self.self_k_norm,
            x,
            x,
            self.self_attn_gate,
            attn_mask,
            self.fused_qk_norm_attention,
        )
        return self.dropout1(out) if dropout else out

    @override
    def _mha_block(
        self,
        x: jt.Float[torch.Tensor, "batch queries dim"],
        mem: jt.Float[torch.Tensor, "memory_batch tokens dim"],
        attn_mask: jt.Bool[torch.Tensor, "queries tokens"] | None,
        key_padding_mask: torch.Tensor | None,
        is_causal: bool = False,
        dropout: bool = True,
    ) -> jt.Float[torch.Tensor, "batch queries dim"]:
        """
        Cross-attention with normalized queries and keys, where consecutive query rows may share one memory row; dropout
        False leaves the residual dropout to the caller.
        """
        assert key_padding_mask is None
        assert not is_causal
        batch, num_queries, dim = x.shape
        draws = batch // mem.shape[0]
        assert draws * mem.shape[0] == batch, f"{batch} query rows for {mem.shape[0]} memory rows"
        if draws > 1:
            # A sample's noise draws attend as one row of draws x queries, so its memory is projected once.
            x = x.reshape(mem.shape[0], draws * num_queries, dim)
            attn_mask = attn_mask.repeat(draws, 1) if attn_mask is not None else None
        out = qk_norm_attention(
            self.multihead_attn,
            self.cross_q_norm,
            self.cross_k_norm,
            x,
            mem,
            self.cross_attn_gate,
            attn_mask,
            self.fused_qk_norm_attention,
        )
        out = out.reshape(batch, num_queries, dim)
        return self.dropout2(out) if dropout else out

    @override
    def _ff_block(
        self,
        x: jt.Float[torch.Tensor, "batch queries dim"],
        dropout: bool = True,
    ) -> jt.Float[torch.Tensor, "batch queries dim"]:
        """SwiGLU feed-forward; dropout False leaves the residual dropout to the caller."""
        hidden_dim = self.linear2.in_features
        padding = -hidden_dim % _HIDDEN_ALIGNMENT
        if not padding:
            gate, value = self.linear1(x).chunk(2, dim=-1)
            out = self.linear2(self.dropout(F.silu(gate) * value))
            return self.dropout3(out) if dropout else out
        # Zero padding aligns the bf16 hidden rows for cuBLAS; silu(0) * 0 = 0 adds nothing and gets no gradient.
        weight1 = F.pad(self.linear1.weight.view(2, hidden_dim, -1), (0, 0, 0, padding)).flatten(0, 1)
        bias1 = F.pad(self.linear1.bias.view(2, hidden_dim), (0, padding)).flatten()
        weight2 = F.pad(self.linear2.weight, (0, padding))
        dropout_p = self.dropout.p if self.training else 0.0
        out = (
            self.fused_feed_forward(x, weight1, bias1, weight2, self.linear2.bias, dropout_p)
            if self.fused_feed_forward is not None
            else None
        )
        if out is not None:
            return self.dropout3(out) if dropout else out
        hidden = (
            self.fused_swiglu_linear(x, weight1, bias1, dropout_p) if self.fused_swiglu_linear is not None else None
        )
        if hidden is None:
            gate, value = F.linear(x, weight1, bias1).chunk(2, dim=-1)
            hidden = self.dropout(F.silu(gate) * value)
        out = F.linear(hidden, weight2, self.linear2.bias)
        return self.dropout3(out) if dropout else out


def qk_norm_attention(
    attention: nn.MultiheadAttention,
    q_norm: nn.LayerNorm,
    k_norm: nn.LayerNorm,
    query: jt.Float[torch.Tensor, "batch queries dim"],
    memory: jt.Float[torch.Tensor, "batch tokens dim"],
    gate: nn.Linear | None = None,
    attention_mask: jt.Bool[torch.Tensor, "queries tokens"] | None = None,
    fused: FusedQkNormAttention | None = None,
) -> jt.Float[torch.Tensor, "batch queries dim"]:
    """
    Runs multi-head attention with the per-head queries and keys normalized before the product.

    Args:
        attention: the batch-first block whose packed input projection and output projection are used.
        q_norm: norm over each head's queries.
        k_norm: norm over each head's keys.
        query: the queries.
        memory: the tokens attended to.
        gate: projection of the queries whose sigmoid scales the attention output elementwise; None leaves it as is.
        attention_mask: True where a query may attend to a token; None lets every query attend to every token.
        fused: a kernel patch's projections, norms and attention in one, taking over where it returns a tensor.

    Returns:
        the attention output, projected back to the model width.
    """
    batch, num_queries, dim = query.shape
    num_heads = cast(int, attention.num_heads)
    dropout_p = attention.dropout if attention.training else 0.0
    out = fused(attention, q_norm, k_norm, query, memory, attention_mask, dropout_p) if fused is not None else None
    if out is not None:
        if gate is not None:
            out = out * torch.sigmoid(gate(query))
        return attention.out_proj(out)

    def heads(x: torch.Tensor) -> torch.Tensor:
        """
        Splits the model width into heads.

        Args:
            x: tokens at the model width.

        Returns:
            the tokens per head, heads before tokens.
        """
        return x.view(batch, x.shape[1], num_heads, dim // num_heads).transpose(1, 2)

    # One projection GEMM per input instead of one per output.
    if query is memory:
        q, k, v = F.linear(query, attention.in_proj_weight, attention.in_proj_bias).chunk(3, dim=-1)
    else:
        q = F.linear(query, attention.in_proj_weight[:dim], attention.in_proj_bias[:dim])
        k, v = F.linear(memory, attention.in_proj_weight[dim:], attention.in_proj_bias[dim:]).chunk(2, dim=-1)
    q, k, v = q_norm(heads(q)), k_norm(heads(k)), heads(v)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, dropout_p=dropout_p)
    out = out.transpose(1, 2).reshape(batch, num_queries, dim)
    if gate is not None:
        out = out * torch.sigmoid(gate(query))
    return attention.out_proj(out)
