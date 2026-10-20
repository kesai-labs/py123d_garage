# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Valeo
# Adapted from valeoai/VideoActionModel (MIT License).

"""
muP GPT2 video model that predicts the next visual token, modified from https://github.com/karpathy/nanoGPT/blob/master/model.py.

It uses affine-free LayerNorms (see https://arxiv.org/abs/2404.05728), MuReadout heads and the muP normal_ init.
"""

import os
from collections import OrderedDict
from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from py123d_garage.policy.vavam.network.mup_readout import MuReadout, MuSharedReadout, normal_, set_base_shapes
from torch import Tensor
from tqdm import tqdm

from py123d_garage.policy.vavam.network.prepare_token_sequence import compute_position_indices


class KVCache(nn.Module):
    """
    Preallocated key/value cache of one attention layer, adapted from karpathy/nano-llama31.

    Every sample holds the same number of tokens, and a pass either recomputes the whole cache with the causal mask or feeds one token without a mask.
    """

    def __init__(
        self,
        batch_size: int,
        seq_length: int,
        n_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        layer_idx: int = 0,
    ) -> None:
        """
        Allocates the zeroed cache buffers.

        Args:
            batch_size: number of samples.
            seq_length: maximum number of cached tokens.
            n_kv_heads: number of attention heads.
            head_dim: width of one attention head.
            dtype: dtype of the buffers.
            device: device of the buffers.
            layer_idx: index of the layer the cache belongs to.
        """
        super().__init__()
        cache_shape = (batch_size, n_kv_heads, seq_length, head_dim)
        self.register_buffer("cache_k", torch.zeros(cache_shape, dtype=dtype, device=device))
        self.register_buffer("cache_v", torch.zeros(cache_shape, dtype=dtype, device=device))
        self.start_pos = 0
        self.layer_idx = layer_idx

    def reset(self) -> None:
        """Zeroes the cache and restarts it at position 0."""
        self.cache_k.zero_()
        self.cache_v.zero_()
        self.start_pos = 0

    def update(self, xk: Tensor, xv: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Appends new keys and values and returns everything cached so far.

        Args:
            xk: new keys shaped [batch, heads, seq, head_dim].
            xv: new values shaped [batch, heads, seq, head_dim].

        Returns:
            all cached keys and values, including the new ones.
        """
        seqlen = xk.size(2)
        self.cache_k[:, :, self.start_pos : self.start_pos + seqlen] = xk
        self.cache_v[:, :, self.start_pos : self.start_pos + seqlen] = xv
        xk_cached = self.cache_k[:, :, : self.start_pos + seqlen]
        xv_cached = self.cache_v[:, :, : self.start_pos + seqlen]
        self.start_pos += seqlen
        return xk_cached, xv_cached


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


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with muP scaling, an explicit mask and an optional KV cache."""

    def __init__(self, dim_model: int, dim_heads: int, block_size: int, attn_scale: float) -> None:
        """
        Creates the projections.

        Args:
            dim_model: model width.
            dim_heads: width of one attention head.
            block_size: maximum sequence length, used only by the slow attention fallback.
            attn_scale: multiplier on the muP attention scale.
        """
        super().__init__()
        assert dim_model % dim_heads == 0, "dim_model must be divisible by dim_heads"

        self.dim_heads = dim_heads
        self.dim_model = dim_model
        self.nb_heads = dim_model // dim_heads

        self.attn_scale = attn_scale
        # Identity hooks for muP coordinate checks.
        self.attn_score = nn.Identity()
        self.query = nn.Identity()
        self.key = nn.Identity()
        self.value = nn.Identity()

        self.c_attn = nn.Linear(dim_model, 3 * dim_model, bias=False)

        self.c_proj = nn.Linear(dim_model, dim_model, bias=False)

        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

        # Set by MupGPT2._setup_kv_cache during generation.
        self.cache = None

    def forward(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        """
        Applies masked self-attention, reading and extending the KV cache when one is set.

        Args:
            x: features shaped [batch, seq, dim_model].
            attn_mask: boolean attention mask, or None for full attention.

        Returns:
            the attended features shaped [batch, seq, dim_model].
        """
        x = self.c_attn(x)

        q, k, v = rearrange(x, "b seq (n nb_heads dim_heads) -> n b nb_heads seq dim_heads", n=3, dim_heads=self.dim_heads)

        if self.cache is not None:
            k, v = self.cache.update(k, v)

        q = self.query(q)
        k = self.key(k)
        v = self.value(v)

        # muP scales attention by 1/dim_heads instead of 1/sqrt(dim_heads).
        attn_scaling = self.attn_scale / v.size(-1)

        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=False,
            scale=attn_scaling,
        )

        y = rearrange(y, "b nb_heads seq dim_head -> b seq (nb_heads dim_head)")

        y = self.c_proj(y)

        return y


class Block(nn.Module):
    """Pre-norm transformer block of the video model."""

    def __init__(
        self,
        dim_model: int,
        dim_heads: int,
        block_size: int,
        attn_scale: float,
        mlp_dim_mult: int = 4,
    ) -> None:
        """
        Creates the block layers.

        Args:
            dim_model: model width.
            dim_heads: width of one attention head.
            block_size: maximum sequence length.
            attn_scale: multiplier on the muP attention scale.
            mlp_dim_mult: hidden width of the MLP as a multiple of the model width.
        """
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim_model, elementwise_affine=False)
        self.attn = CausalSelfAttention(dim_model, dim_heads, block_size, attn_scale)
        self.ln_2 = nn.LayerNorm(dim_model, elementwise_affine=False)
        self.mlp = MLP(dim_model, mlp_dim_mult)

    def forward(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        """
        Applies the block.

        Args:
            x: features shaped [batch, seq, dim_model].
            attn_mask: boolean attention mask, or None for full attention.

        Returns:
            the features with the same shape.
        """
        x = x + self.attn(self.ln_1(x), attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class MupGPT2(nn.Module):
    """GPT2 over visual tokens, laid out to load the released pre-trained weights."""

    def __init__(
        self,
        embedding_dim: int = 256,
        nb_layers: int = 12,
        dim_heads: int = 2,
        mlp_dim_mult: int = 4,
        vocabulary_size: int = 1104,
        nb_timesteps: int = 8,
        nb_tokens_per_timestep: int = 352,
        init_std: float = 0.02,
        output_tied: bool = True,
        output_scale: float = 1.0,
        attn_scale: float = 1.0,
    ) -> None:
        """
        Creates and initializes the model.

        Args:
            embedding_dim: model width.
            nb_layers: number of transformer blocks.
            dim_heads: width of one attention head.
            mlp_dim_mult: hidden width of the MLP as a multiple of the model width.
            vocabulary_size: number of entries in the tokenizer codebook.
            nb_timesteps: maximum number of frames in one sequence.
            nb_tokens_per_timestep: number of tokens per frame.
            init_std: standard deviation of the weight init.
            output_tied: whether the readout shares the token embedding weights.
            output_scale: input multiplier of the readout.
            attn_scale: multiplier on the muP attention scale.
        """
        super().__init__()
        assert vocabulary_size is not None, "vocabulary_size must be provided"
        assert nb_tokens_per_timestep is not None, "nb_tokens_per_timestep must be provided"
        assert nb_timesteps is not None, "nb_timesteps must be provided"

        self.embedding_dim = embedding_dim
        self.init_std = init_std
        self.attn_scale = attn_scale
        self.output_scale = output_scale
        self.mlp_dim_mult = mlp_dim_mult
        self.nb_layers = nb_layers
        self.output_tied = output_tied

        self.nb_timesteps = nb_timesteps
        self.nb_tokens_per_timestep = nb_tokens_per_timestep
        self.block_size = nb_timesteps * nb_tokens_per_timestep

        self.transformer = nn.ModuleDict(
            {
                "wie": nn.Embedding(vocabulary_size, embedding_dim),  # token embeddings
                "wse": nn.Embedding(nb_tokens_per_timestep, embedding_dim),  # spatial position embeddings
                "wte": nn.Embedding(nb_timesteps, embedding_dim),  # temporal position embeddings
                "h": nn.ModuleList(
                    [
                        Block(
                            embedding_dim,
                            dim_heads,
                            self.block_size,
                            attn_scale,
                            mlp_dim_mult,
                        )
                        for _ in range(nb_layers)
                    ],
                ),
                "ln_f": nn.LayerNorm(embedding_dim, elementwise_affine=False),
            },
        )

        if output_tied:
            self.lm_head = MuSharedReadout(self.transformer.wie.weight, bias=False, output_mult=output_scale)
        else:
            self.lm_head = MuReadout(embedding_dim, vocabulary_size, bias=False, output_mult=output_scale)

        # This runs before set_base_shapes, so a proper muP init re-applies _init_weights after it.
        self.apply(self._init_weights)

        print("number of non-embedding parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Counts the parameters; untied token embeddings are never counted.

        Args:
            non_embedding: whether to leave out the position embeddings.

        Returns:
            the number of parameters.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wse.weight.numel()
            n_params -= self.transformer.wte.weight.numel()
        if not self.output_tied:
            n_params -= self.transformer.wie.weight.numel()
        return n_params

    def _init_c_proj_residual(self, module: nn.Module) -> None:
        """
        Scales the init std of the residual output projections by 1/sqrt(2 * nb_layers), as in GPT-2.

        Args:
            module: attention or MLP module whose c_proj weights are re-initialized.
        """
        depth_std = self.init_std * (2 * self.nb_layers) ** -0.5

        for p_name, p in module.named_parameters():
            if p_name.endswith(".c_proj.weight"):
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
        if isinstance(module, MuReadout) and not self.output_tied:
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

        elif isinstance(module, CausalSelfAttention):
            # muP zero-inits the query rows of c_attn so attention starts uniform.
            fanout, _ = module.c_attn.weight.shape
            assert fanout % 3 == 0

            module.c_attn.weight.data[: fanout // 3, :] = 0

            self._init_c_proj_residual(module)

        elif isinstance(module, MLP):
            self._init_c_proj_residual(module)

    def _get_emb(self, token_sequence: Tensor, spatial_positions: Tensor, temporal_positions: Tensor) -> Tensor:
        """
        Sums the token, temporal and spatial embeddings.

        Args:
            token_sequence: token ids shaped [batch, seq].
            spatial_positions: spatial position ids shaped [batch, seq].
            temporal_positions: temporal position ids shaped [batch, seq].

        Returns:
            the embeddings shaped [batch, seq, embedding_dim].
        """
        assert (
            spatial_positions.max() < self.nb_tokens_per_timestep
        ), f"spatial_positions.max()={spatial_positions.max()} >= self.nb_tokens_per_timestep={self.nb_tokens_per_timestep}"
        assert (
            temporal_positions.max() < self.nb_timesteps
        ), f"temporal_positions.max()={temporal_positions.max()} >= self.nb_timesteps={self.nb_timesteps}"

        spatial_pos_emb = self.transformer.wse(spatial_positions)
        temporal_pos_emb = self.transformer.wte(temporal_positions)

        tok_emb = self.transformer.wie(token_sequence)

        return tok_emb + temporal_pos_emb + spatial_pos_emb

    @torch.no_grad()
    def get_intermediate_layers(
        self,
        token_sequence: Tensor,
        stop_layer_idx: int,
    ) -> Tensor:
        """
        Runs the causal transformer up to a given layer and returns its features per token.

        Args:
            token_sequence: token ids shaped [batch, frames, height, width].
            stop_layer_idx: index of the last layer to run.

        Returns:
            the features shaped [batch, frames, height, width, embedding_dim].
        """
        token_sequence = token_sequence[:, -self.block_size :]
        bs, context_timesteps, height, width = token_sequence.shape
        token_sequence = rearrange(token_sequence, "b t h w -> b (t h w)")

        positions = compute_position_indices(bs, context_timesteps, height, width)
        spatial_positions = positions["spatial_positions"]
        temporal_positions = positions["temporal_positions"]

        x = self._get_emb(token_sequence, spatial_positions, temporal_positions)

        seqlen = token_sequence.size(1)
        attn_mask = torch.tril(torch.ones(seqlen, seqlen, device=token_sequence.device, dtype=torch.bool))

        for layer_idx, block in enumerate(self.transformer.h):
            x = block(x, attn_mask)
            if layer_idx == len(self.transformer.h) - 1:
                x = self.transformer.ln_f(x)
            if layer_idx == stop_layer_idx:
                break

        return rearrange(x, "b (t h w) d -> b t h w d", t=context_timesteps, h=height, w=width)

    def forward(
        self,
        token_sequence: Tensor,
        spatial_positions: Tensor,
        temporal_positions: Tensor,
        inference: bool = False,
        use_kv_cache: bool = False,
    ) -> Tensor:
        """
        Predicts next-token logits for a flattened visual token sequence.

        Args:
            token_sequence: token ids shaped [batch, seq].
            spatial_positions: spatial position of each token, e.g. [0, 1, 2, 3, 0, 1, 2, 3].
            temporal_positions: temporal position of each token, e.g. [0, 0, 0, 0, 1, 1, 1, 1].
            inference: whether to return the logits of the last token only.
            use_kv_cache: whether one new token is fed against the KV cache, without a mask.

        Returns:
            the logits shaped [batch, seq, vocabulary_size], or [batch, 1, vocabulary_size] at inference.
        """
        x = self._get_emb(token_sequence, spatial_positions, temporal_positions)

        seqlen = token_sequence.size(1)
        if inference and use_kv_cache:
            attn_mask = None
            assert seqlen == 1, "inference with KV cache only supports single token forward"
        else:
            attn_mask = torch.tril(torch.ones(seqlen, seqlen, device=token_sequence.device, dtype=torch.bool))

        for block in self.transformer.h:
            x = block(x, attn_mask)
        emb_out = self.transformer.ln_f(x)

        if not inference:
            img_logits = self.lm_head(emb_out)
        else:
            img_logits = self.lm_head(emb_out[:, [-1], :])  # the list index keeps the sequence dimension

        return img_logits

    def _setup_kv_cache(self, batch_size: int) -> None:
        """
        Attaches an empty KV cache sized to the maximum context to every attention layer.

        Args:
            batch_size: number of samples.
        """
        for layer_idx, block in enumerate(self.transformer.h):
            layer = block.attn
            cache = KVCache(
                batch_size=batch_size,
                seq_length=self.block_size,
                n_kv_heads=layer.nb_heads,
                head_dim=layer.dim_heads,
                dtype=layer.c_attn.weight.dtype,
                device=layer.c_attn.weight.device,
                layer_idx=layer_idx,
            )
            layer.cache = cache

    def _reset_kv_cache(self) -> None:
        """Empties the KV cache of every layer."""
        for block in self.transformer.h:
            block.attn.cache.reset()

    def _clear_kv_cache(self) -> None:
        """Removes the KV cache from every layer."""
        for block in self.transformer.h:
            block.attn.cache = None

    def _sample_next_token(self, logits: Tensor, temperature: float, topk_sampler: int) -> Tensor:
        """
        Samples the next token from the last position's logits with top-k sampling.

        Args:
            logits: logits shaped [batch, seq, vocabulary_size].
            temperature: sampling temperature.
            topk_sampler: number of most likely tokens to sample from.

        Returns:
            the sampled token ids shaped [batch, 1].
        """
        logits = logits[:, -1, :] / temperature
        tokens_probs = torch.softmax(logits, dim=-1)
        topk_probs, topk_tokens = torch.topk(tokens_probs, topk_sampler, dim=-1, sorted=False)
        topk_renormalized_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        next_token_idx = torch.multinomial(topk_renormalized_probs, num_samples=1)
        next_token = torch.gather(topk_tokens, -1, next_token_idx)
        return next_token

    def _process_generated_frame(self, context: Tensor, height: int, width: int) -> Tensor:
        """
        Extracts the last generated frame from the context as a token grid.

        Args:
            context: flattened token ids shaped [batch, seq].
            height: height of a token frame.
            width: width of a token frame.

        Returns:
            the last frame shaped [batch, height, width].
        """
        frame = context[:, -self.nb_tokens_per_timestep :]
        frame = rearrange(frame, "b (h w) -> b h w", h=height, w=width)
        return frame

    @torch.no_grad()
    def forward_inference(
        self,
        number_of_future_frames: int,
        burnin_visual_tokens: Tensor,
        temperature: float = 1.0,
        topk_sampler: int = 1,
        use_kv_cache: bool = True,
        verbose: int | bool = 0,
    ) -> Tensor:
        """
        Generates future token frames autoregressively, one token at a time.

        Args:
            number_of_future_frames: number of frames to generate.
            burnin_visual_tokens: context token ids shaped [batch, frames, height, width].
            temperature: sampling temperature.
            topk_sampler: number of most likely tokens to sample from.
            use_kv_cache: whether to cache keys and values between tokens.
            verbose: 1 shows a progress bar per frame, 2 also one per token.

        Returns:
            the generated frames shaped [batch, number_of_future_frames, height, width].
        """
        verbose = int(verbose)

        bs, _, height, width = burnin_visual_tokens.shape
        context = rearrange(burnin_visual_tokens, "b t h w -> b (t h w)")

        def _count_nb_frames(context: Tensor) -> int:
            """Counts the frames in a flattened token context."""
            return context.size(1) // (height * width)

        positions = compute_position_indices(bs, self.nb_timesteps, height, width)

        context = context[:, -self.block_size :]
        spatial_positions = positions["spatial_positions"]
        temporal_positions = positions["temporal_positions"]

        generated_frames = torch.zeros(
            bs, number_of_future_frames, height, width, device=burnin_visual_tokens.device, dtype=burnin_visual_tokens.dtype,
        )

        if use_kv_cache:
            self._setup_kv_cache(bs)
            kv_cache_was_reset = True

        for frame_idx in tqdm(
            range(number_of_future_frames), f"Generating {number_of_future_frames} frames", disable=verbose < 1,
        ):

            # The context keeps at most nb_timesteps - 1 frames to leave room for the generated one.
            if _count_nb_frames(context) > self.nb_timesteps - 1:
                context = context[:, -((self.nb_timesteps - 1) * self.nb_tokens_per_timestep) :]
                # Learned position embeddings rule out rolling the cache, so it is recomputed.
                if use_kv_cache:
                    self._reset_kv_cache()
                    kv_cache_was_reset = True

            for _ in tqdm(
                range(self.nb_tokens_per_timestep),
                f"AR generation of {self.nb_tokens_per_timestep} tokens",
                disable=verbose < 2,
                leave=False,
                position=1,
            ):
                tokens_in_context = context.size(1)
                tmp_ctx = context
                tmp_spatial_positions = spatial_positions[:, :tokens_in_context]
                tmp_temporal_positions = temporal_positions[:, :tokens_in_context]
                if use_kv_cache and not kv_cache_was_reset:
                    tmp_ctx = tmp_ctx[:, -1:]
                    tmp_spatial_positions = tmp_spatial_positions[:, -1:]
                    tmp_temporal_positions = tmp_temporal_positions[:, -1:]

                logits = self.forward(
                    tmp_ctx,
                    tmp_spatial_positions,
                    tmp_temporal_positions,
                    inference=True,
                    use_kv_cache=use_kv_cache and not kv_cache_was_reset,
                )
                next_tokens = self._sample_next_token(logits, temperature, topk_sampler)
                context = torch.cat([context, next_tokens], dim=1)
                kv_cache_was_reset = False

            generated_frames[:, frame_idx] = self._process_generated_frame(context, height, width)

        if use_kv_cache:
            self._clear_kv_cache()
        return generated_frames


def load_pretrained_gpt(checkpoint_path: str, device: torch.device | str = "cuda", tempdir: Optional[str] = None) -> MupGPT2:
    """
    Loads a frozen MupGPT2 in eval mode from a Lightning or DeepSpeed checkpoint.

    Args:
        checkpoint_path: checkpoint file, or a DeepSpeed checkpoint directory.
        device: device to move the model to.
        tempdir: parent directory for the temporary fused DeepSpeed checkpoint.

    Returns:
        the loaded model.
    """
    if os.path.isdir(checkpoint_path):
        # A directory is a DeepSpeed checkpoint.
        import tempfile

        from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict

        with tempfile.TemporaryDirectory(dir=tempdir) as tmpdirname:
            ckpt = convert_zero_checkpoint_to_fp32_state_dict(
                checkpoint_path,
                os.path.join(tmpdirname, "fused_ckpt.pt"),
            )
    else:
        ckpt = torch.load(checkpoint_path, map_location="cpu")

    config = ckpt["hyper_parameters"]["network"].copy()
    config.pop("_target_")
    gpt = MupGPT2(**config)
    state_dict = OrderedDict()
    for k, v in ckpt["state_dict"].items():
        state_dict[k.replace("network.", "")] = v

    gpt.load_state_dict(state_dict)
    set_base_shapes(gpt, ckpt["hyper_parameters"]["mup_base_shapes"], rescale_params=False)
    _ = gpt.eval()
    _ = gpt.to(device)
    _ = gpt.requires_grad_(False)

    return gpt
