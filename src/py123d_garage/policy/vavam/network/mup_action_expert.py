# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Valeo
# Adapted from valeoai/VideoActionModel (MIT License).

"""muP action expert: a transformer that denoises action sequences, built from affine-free LayerNorms and a MuReadout."""

import math

import torch
import torch.nn as nn
from einops import rearrange, repeat
from py123d_garage.policy.vavam.network.mup_readout import MuReadout, normal_
from torch import Tensor


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal embedding of the diffusion step."""

    def __init__(self, dim: int) -> None:
        """
        Stores the embedding width.

        Args:
            dim: embedding width.
        """
        super().__init__()
        self.dim = dim

    def forward(
        self,
        t: Tensor,
        max_period: float = 10000.0,
    ) -> Tensor:
        """
        Embeds the diffusion steps.

        Args:
            t: diffusion steps shaped [n].
            max_period: longest sinusoid period.

        Returns:
            the embeddings shaped [n, dim].
        """
        half_dim = self.dim // 2
        emb = math.log(max_period) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb).to(dtype=t.dtype)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ActionEncoder(nn.Module):
    """Embeds noisy actions with the high-level command, the horizon position and the diffusion step."""

    def __init__(
        self,
        action_dim: int,
        action_hidden_dim: int,
        action_horizon: int,
        number_high_level_command: int,
        max_period: float = 10000.0,
        bias: bool = False,
    ) -> None:
        """
        Creates the encoder layers.

        Args:
            action_dim: width of one action.
            action_hidden_dim: width of the action embedding and of the denoiser.
            action_horizon: number of actions per sequence, unrelated to the video context length.
            number_high_level_command: number of high-level commands.
            max_period: longest period of the diffusion step embedding.
            bias: whether the linear layers have a bias.
        """
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.action_hidden_dim = action_hidden_dim

        self.linear_1 = nn.Linear(action_dim, action_hidden_dim, bias=bias)
        self.linear_2 = nn.Linear(2 * action_hidden_dim, action_hidden_dim, bias=bias)
        self.nonlinearity = nn.SiLU()
        self.linear_3 = nn.Linear(action_hidden_dim, action_hidden_dim, bias=bias)

        self.command_embedding = nn.Embedding(number_high_level_command, action_hidden_dim)
        self.action_positional_embedding = nn.Embedding(action_horizon, action_hidden_dim)

        self.max_period = max_period
        self.diffusion_step_embedding = SinusoidalPosEmb(action_hidden_dim)

    def forward(
        self,
        actions: Tensor,
        high_level_command: Tensor,
        diffusion_step: Tensor,
    ) -> Tensor:
        """
        Embeds the actions.

        Args:
            actions: noisy actions shaped [batch, context, horizon, action_dim].
            high_level_command: command ids shaped [batch, context].
            diffusion_step: diffusion steps shaped [batch, context].

        Returns:
            the embeddings shaped [batch, context, horizon, action_hidden_dim].
        """
        bs, context_length, horizon, _ = actions.size()

        action_emb = self.linear_1(actions)
        command_emb = self.command_embedding(high_level_command)
        command_emb = repeat(command_emb, "b t d -> b t h d", h=horizon)
        action_pos_emb = self.action_positional_embedding(
            torch.arange(horizon, device=actions.device),
        )
        action_pos_emb = repeat(action_pos_emb, "h d -> b t h d", b=bs, t=context_length)
        action_emb = action_emb + command_emb + action_pos_emb

        diffusion_step = rearrange(diffusion_step, "b t -> (b t)")
        diffusion_step_emb = self.diffusion_step_embedding(
            diffusion_step, self.max_period,
        )
        diffusion_step_emb = rearrange(diffusion_step_emb, "(b t) d -> b t d", b=bs)
        diffusion_step_emb = repeat(diffusion_step_emb, "b t d -> b t h d", h=horizon).type_as(action_emb)

        action_emb = torch.cat([diffusion_step_emb, action_emb], dim=-1)
        action_emb = self.nonlinearity(self.linear_2(action_emb))
        action_emb = self.linear_3(action_emb)
        return action_emb


class MLP(nn.Module):
    """Feed-forward layer of a transformer block."""

    def __init__(self, dim_model: int, hidden_dim_mult: int = 4) -> None:
        """
        Creates the two projections.

        Args:
            dim_model: model width.
            hidden_dim_mult: hidden width as a multiple of the model width.
        """
        super().__init__()
        self.c_fc = nn.Linear(dim_model, hidden_dim_mult * dim_model, bias=False)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(hidden_dim_mult * dim_model, dim_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        Applies the feed-forward layer.

        Args:
            x: features shaped [batch, seq, dim_model].

        Returns:
            the features with the same shape.
        """
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class SelfAttention(nn.Module):
    """Bidirectional multi-head self-attention with muP scaling."""

    def __init__(self, dim_model: int, attn_dim: int, dim_heads: int, attn_scale: float) -> None:
        """
        Creates the projections.

        Args:
            dim_model: model width.
            attn_dim: total width of the attention heads.
            dim_heads: width of one attention head.
            attn_scale: multiplier on the muP attention scale.
        """
        super().__init__()
        assert attn_dim % dim_heads == 0, "dim_model must be divisible by dim_heads"

        self.dim_heads = dim_heads
        self.dim_model = dim_model
        self.attn_dim = attn_dim

        self.attn_scale = attn_scale
        # Identity hooks for muP coordinate checks.
        self.attn_score = nn.Identity()
        self.query = nn.Identity()
        self.key = nn.Identity()
        self.value = nn.Identity()

        self.c_attn = nn.Linear(dim_model, 3 * self.attn_dim, bias=False)

        self.c_proj = nn.Linear(self.attn_dim, dim_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        Applies self-attention over the whole sequence.

        Args:
            x: features shaped [batch, seq, dim_model].

        Returns:
            the attended features with the same shape.
        """
        x = self.c_attn(x)

        q, k, v = rearrange(x, "b seq (n nb_heads dim_heads) -> n b nb_heads seq dim_heads", n=3, dim_heads=self.dim_heads)

        q = self.query(q)
        k = self.key(k)
        v = self.value(v)

        # muP scales attention by 1/dim_heads instead of 1/sqrt(dim_heads).
        attn_scaling = self.attn_scale / v.size(-1)

        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            scale=attn_scaling,
        )

        y = rearrange(y, "b nb_heads seq dim_head -> b seq (nb_heads dim_head)")

        y = self.c_proj(y)

        return y


class Block(nn.Module):
    """Pre-norm transformer block of the action expert."""

    def __init__(
        self,
        dim_model: int,
        attn_dim: int,
        dim_heads: int,
        attn_scale: float,
        mlp_dim_mult: int = 4,
    ) -> None:
        """
        Creates the block layers.

        Args:
            dim_model: model width.
            attn_dim: total width of the attention heads.
            dim_heads: width of one attention head.
            attn_scale: multiplier on the muP attention scale.
            mlp_dim_mult: hidden width of the MLP as a multiple of the model width.
        """
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim_model, elementwise_affine=False)
        self.attn = SelfAttention(dim_model, attn_dim, dim_heads, attn_scale)
        self.ln_2 = nn.LayerNorm(dim_model, elementwise_affine=False)
        self.mlp = MLP(dim_model, mlp_dim_mult)

    def forward(self, x: Tensor) -> Tensor:
        """
        Applies the block.

        Args:
            x: features shaped [batch, seq, dim_model].

        Returns:
            the features with the same shape.
        """
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MupActionExpert(nn.Module):
    """Transformer that predicts the flow velocity of noisy action sequences."""

    def __init__(
        self,
        *,
        embedding_dim: int = 256,
        attention_dim: int = 384,
        action_dim: int = 2,
        action_horizon: int = 6,
        number_high_level_command: int = 3,
        max_period: float = 10000.0,
        nb_layers: int = 12,
        dim_heads: int = 64,
        mlp_dim_mult: int = 4,
        init_std: float = 0.02,
        output_scale: float = 1.0,
        attn_scale: float = 1.0,
    ) -> None:
        """
        Creates and initializes the model.

        Args:
            embedding_dim: model width.
            attention_dim: total width of the attention heads.
            action_dim: width of one action.
            action_horizon: number of actions per sequence.
            number_high_level_command: number of high-level commands.
            max_period: longest period of the diffusion step embedding.
            nb_layers: number of transformer blocks.
            dim_heads: width of one attention head.
            mlp_dim_mult: hidden width of the MLP as a multiple of the model width.
            init_std: standard deviation of the weight init.
            output_scale: input multiplier of the action readout.
            attn_scale: multiplier on the muP attention scale.
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.attention_dim = attention_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.number_high_level_command = number_high_level_command
        self.init_std = init_std
        self.attn_scale = attn_scale
        self.output_scale = output_scale
        self.mlp_dim_mult = mlp_dim_mult
        self.nb_layers = nb_layers

        self.action_encoder = ActionEncoder(
            action_dim=action_dim,
            action_hidden_dim=embedding_dim,
            action_horizon=action_horizon,
            number_high_level_command=number_high_level_command,
            max_period=max_period,
        )

        self.transformer = nn.ModuleDict(
            {
                "h": nn.ModuleList(
                    [
                        Block(
                            embedding_dim,
                            attention_dim,
                            dim_heads,
                            attn_scale,
                            mlp_dim_mult,
                        )
                        for _ in range(nb_layers)
                    ],
                ),
                "ln_f": nn.LayerNorm(embedding_dim, elementwise_affine=False),
            },
        )

        self.action_decoder = MuReadout(embedding_dim, action_dim, bias=False, output_mult=output_scale)

        # This runs before set_base_shapes, so a proper muP init re-applies _init_weights after it.
        self.apply(self._init_weights)

        print("number of non-embedding parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Counts the model's parameters.

        Args:
            non_embedding: unused; every parameter is counted.

        Returns:
            the number of parameters.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_c_proj_residual(self, module: nn.Module) -> None:
        """
        Scales the init std of the residual output projections by 1/sqrt(2 * nb_layers), as in GPT-2.

        Args:
            module: attention or MLP module whose c_proj weights are re-initialized.
        """
        depth_std = self.init_std * (2 * self.nb_layers) ** -0.5

        for p_name, p in module.named_parameters():
            if p_name.endswith("c_proj.weight"):
                if hasattr(p, "infshape"):
                    normal_(p, mean=0.0, std=depth_std)
                else:
                    p.data.normal_(mean=0.0, std=depth_std)

    def _init_weights(self, module: nn.Module) -> None:
        """
        Initializes one module; model.apply visits children before parents, so a parent's rule overrides its layers' init.

        Args:
            module: module to initialize.
        """
        if isinstance(module, MuReadout):
            # Zero-init readout, see https://arxiv.org/abs/2404.05728 section 4.2.6.
            module.weight.data.zero_()

        elif isinstance(module, nn.Linear):

            if hasattr(module.weight, "infshape"):
                normal_(module.weight, mean=0.0, std=self.init_std)
            else:
                module.weight.data.normal_(mean=0.0, std=self.init_std)

            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.init_std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                module.bias.data.zero_()
            if module.weight is not None:
                module.weight.data.fill_(1.0)

        elif isinstance(module, SelfAttention):
            # muP zero-inits the query rows of c_attn so attention starts uniform.
            fanout, _ = module.c_attn.weight.shape
            assert fanout % 3 == 0

            module.c_attn.weight.data[: fanout // 3, :] = 0

            self._init_c_proj_residual(module)

        elif isinstance(module, MLP):
            self._init_c_proj_residual(module)

    def forward(self, noisy_actions: Tensor, high_level_command: Tensor, t: Tensor) -> Tensor:
        """
        Predicts the flow velocity of the noisy actions without visual context.

        Args:
            noisy_actions: noisy actions shaped [batch, context, horizon, action_dim].
            high_level_command: command ids shaped [batch, context].
            t: diffusion steps shaped [batch, context].

        Returns:
            the predicted velocity with the shape of noisy_actions.
        """
        action_embeds = self.action_encoder(actions=noisy_actions, high_level_command=high_level_command, diffusion_step=t)
        action_embeds = rearrange(action_embeds, "b t h d -> b (t h) d")

        for block in self.transformer.h:
            action_embeds = block(action_embeds)
        action_embeds = self.transformer.ln_f(action_embeds)

        denoised_actions = self.action_decoder(action_embeds)

        denoised_actions = rearrange(denoised_actions, "b (t h) d -> b t h d", t=noisy_actions.size(1))

        return denoised_actions
