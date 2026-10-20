# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Valeo
# Adapted from valeoai/VideoActionModel (MIT License).

"""Joint model that runs the GPT2 video model and the action expert through shared attention."""

from typing import Dict, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from torch import BoolTensor, FloatTensor, LongTensor

from py123d_garage.policy.vavam.network.mup_action_expert import Block as ActionBlock
from py123d_garage.policy.vavam.network.mup_action_expert import MupActionExpert
from py123d_garage.policy.vavam.network.mup_action_expert import SelfAttention as ActionAttention
from py123d_garage.policy.vavam.network.mup_gpt2 import Block as GPTBlock
from py123d_garage.policy.vavam.network.mup_gpt2 import CausalSelfAttention as GPTAttention
from py123d_garage.policy.vavam.network.mup_gpt2 import MupGPT2
from py123d_garage.policy.vavam.network.prepare_token_sequence import prepare_token_sequence

EmbedsDict = Dict[str, FloatTensor]
InputsDict = Dict[str, FloatTensor | LongTensor]
OutputDict = Dict[str, FloatTensor]


class KVCache:
    """
    Per-layer keys and values of the video model for one batch at a fixed context length.

    It does not roll the context, so it must be reset for each new batch.
    """

    def __init__(self, num_layers: int) -> None:
        """
        Creates an empty cache.

        Args:
            num_layers: number of transformer layers to cache.
        """
        super().__init__()
        self.num_layers = num_layers
        self.reset()

    def reset(self) -> None:
        """Empties the cache of all layers for a new batch."""
        self.k_cache = [None for _ in range(self.num_layers)]
        self.v_cache = [None for _ in range(self.num_layers)]
        self._filled = [False for _ in range(self.num_layers)]

    @property
    def filled(self) -> bool:
        """Whether every layer of the cache has been filled."""
        return all(self._filled)

    def update(self, k_cache: FloatTensor, v_cache: FloatTensor, layer_idx: int) -> None:
        """
        Stores the keys and values of one layer.

        Args:
            k_cache: keys of the video tokens.
            v_cache: values of the video tokens.
            layer_idx: index of the layer.
        """
        self.k_cache[layer_idx] = k_cache
        self.v_cache[layer_idx] = v_cache
        self._filled[layer_idx] = True

    def get(self, layer_idx: int) -> Tuple[FloatTensor, FloatTensor]:
        """
        Reads the keys and values of one layer.

        Args:
            layer_idx: index of the layer.

        Returns:
            the cached keys and values.
        """
        assert self._filled[layer_idx], "Cache not filled"
        return self.k_cache[layer_idx], self.v_cache[layer_idx]


class JointModel(nn.Module):
    """Bridges the GPT2 video model and the action expert: shared attention per layer, then separate FFNs."""

    def __init__(
        self,
        gpt: MupGPT2,
        action_expert: MupActionExpert,
    ) -> None:
        """
        Wraps the two models, which must have the same number of layers.

        Args:
            gpt: GPT2 video model.
            action_expert: action expert model.
        """
        super().__init__()
        self.gpt = gpt
        self.action_expert = action_expert

        self.num_hidden_layers = len(self.gpt.transformer.h)
        assert self.num_hidden_layers == len(
            self.action_expert.transformer.h,
        ), "The GPT and action expert must have the same number of layers"

        self.kv_cache = None

    def init_kv_cache(self) -> None:
        """Creates the KV cache for inference."""
        self.kv_cache = KVCache(self.num_hidden_layers)

    def cleanup_kv_cache(self) -> None:
        """Empties the KV cache for a new batch."""
        self.kv_cache.reset()

    @property
    def use_kv_cache(self) -> None:
        """Whether to read the video keys and values from the filled cache instead of recomputing them."""
        return (self.kv_cache is not None) and self.kv_cache.filled and (not self.training)

    @property
    def fill_kv_cache(self) -> None:
        """Whether to store the video keys and values computed in this pass."""
        return (self.kv_cache is not None) and (not self.training)

    def _visual_tokens_to_embeds(self, visual_tokens: LongTensor) -> FloatTensor:
        """
        Embeds the visual tokens with the GPT2 token, temporal and spatial embeddings.

        Args:
            visual_tokens: token ids shaped [batch, time, height, width].

        Returns:
            the embeddings shaped [batch, time * height * width, dim].
        """
        sequence_data = prepare_token_sequence(visual_tokens)
        spatial_pos_emb = self.gpt.transformer.wse(sequence_data["spatial_positions"])
        temporal_pos_emb = self.gpt.transformer.wte(sequence_data["temporal_positions"])
        tok_emb = self.gpt.transformer.wie(sequence_data["token_sequence"])
        visual_embeds = tok_emb + temporal_pos_emb + spatial_pos_emb
        return visual_embeds

    def _noisy_action_to_embeds(
        self, noisy_action: FloatTensor, high_level_command: LongTensor, t: FloatTensor,
    ) -> FloatTensor:
        """
        Embeds the noisy actions with the action expert's encoder.

        Args:
            noisy_action: noisy actions shaped [batch, time, horizon, action_dim].
            high_level_command: command ids shaped [batch, time].
            t: flow time shaped [batch, time].

        Returns:
            the embeddings shaped [batch, time * horizon, dim].
        """
        action_embeds = self.action_expert.action_encoder(
            actions=noisy_action, high_level_command=high_level_command, diffusion_step=t,
        )
        action_embeds = rearrange(action_embeds, "b t h d -> b (t h) d")
        return action_embeds

    def _forward_mutual_attention(
        self,
        attn_mask: BoolTensor,
        visual_attn_input: FloatTensor | None,
        action_attn_input: FloatTensor,
        gpt_attention: GPTAttention,
        action_attention: ActionAttention,
        layer_idx: int,
    ) -> Tuple[FloatTensor, FloatTensor]:
        """
        Runs one attention over the visual and action tokens; the visual outputs equal plain GPT2.

        Args:
            attn_mask: joint attention mask, ignored when the KV cache is read.
            visual_attn_input: normalized visual embeddings; None when the KV cache is read.
            action_attn_input: normalized action embeddings.
            gpt_attention: attention layer of the GPT2 block.
            action_attention: attention layer of the action expert block.
            layer_idx: index of the layer.

        Returns:
            the visual outputs (None when the KV cache is read) and the action outputs.
        """
        if self.use_kv_cache:
            visual_k, visual_v = self.kv_cache.get(layer_idx)
        else:
            visual_q, visual_k, visual_v = rearrange(
                gpt_attention.c_attn(visual_attn_input),
                "b seq (n nb_heads dim_heads) -> n b nb_heads seq dim_heads",
                n=3,
                dim_heads=gpt_attention.dim_heads,
            )
            if self.fill_kv_cache:
                self.kv_cache.update(visual_k, visual_v, layer_idx)

        action_q, action_k, action_v = rearrange(
            action_attention.c_attn(action_attn_input),
            "b seq (n nb_heads dim_heads) -> n b nb_heads seq dim_heads",
            n=3,
            dim_heads=action_attention.dim_heads,
        )
        # muP scales attention by 1/dim_heads instead of 1/sqrt(dim_heads).
        attn_scaling = action_attention.attn_scale / action_v.size(-1)

        if self.use_kv_cache:
            q = action_q
            attn_mask = None  # only the action tokens remain, which use full attention at inference time
        else:
            q = torch.cat([visual_q, action_q], dim=-2)
        k = torch.cat([visual_k, action_k], dim=-2)
        v = torch.cat([visual_v, action_v], dim=-2)

        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=False,
            scale=attn_scaling,
        )
        y = rearrange(y, "b nb_heads seq dim_head -> b seq (nb_heads dim_head)")

        if self.use_kv_cache:
            action_y = y
            visual_y = None
        else:
            visual_y, action_y = torch.split(y, [visual_q.size(-2), action_q.size(-2)], dim=1)
            visual_y = gpt_attention.c_proj(visual_y)
        action_y = action_attention.c_proj(action_y)
        return visual_y, action_y

    def _forward_single_layer(
        self,
        attention_mask: BoolTensor,
        embeds_all: EmbedsDict,
        layer_idx: int,
    ) -> EmbedsDict:
        """
        Runs one joint layer: the shared attention, then each model's own FFN.

        Args:
            attention_mask: joint attention mask.
            embeds_all: visual and action embeddings.
            layer_idx: index of the layer.

        Returns:
            the updated visual and action embeddings.
        """
        visual_embeds = embeds_all["visual_embeds"]
        action_embeds = embeds_all["action_embeds"]

        gpt_block: GPTBlock = self.gpt.transformer.h[layer_idx]
        action_block: ActionBlock = self.action_expert.transformer.h[layer_idx]
        gpt_attention: GPTAttention = gpt_block.attn
        action_attention: ActionAttention = action_block.attn

        if not self.use_kv_cache:
            visual_attn_input = gpt_block.ln_1(visual_embeds)
        else:
            visual_attn_input = None
        action_attn_input = action_block.ln_1(action_embeds)

        visual_attn_output, action_attn_output = self._forward_mutual_attention(
            attention_mask, visual_attn_input, action_attn_input, gpt_attention, action_attention, layer_idx,
        )

        if not self.use_kv_cache:
            visual_embeds = visual_embeds + visual_attn_output
            visual_embeds = visual_embeds + gpt_block.mlp(gpt_block.ln_2(visual_embeds))

        action_embeds = action_embeds + action_attn_output
        action_embeds = action_embeds + action_block.mlp(action_block.ln_2(action_embeds))

        return {"visual_embeds": visual_embeds, "action_embeds": action_embeds}

    def forward(
        self,
        attention_mask: BoolTensor,
        inputs_all: InputsDict,
        return_gpt: bool = False,
    ) -> OutputDict:
        """
        Embeds the visual tokens and noisy actions, runs all joint layers and decodes the action velocities.

        Args:
            attention_mask: joint attention mask.
            inputs_all: visual_tokens, noisy_actions, high_level_command and diffusion_step.
            return_gpt: whether to also return the GPT2 embeddings and logits.

        Returns:
            the decoded actions and action embeddings, plus the visual embeddings and logits when asked.
        """
        if not self.use_kv_cache:
            visual_embeds = self._visual_tokens_to_embeds(inputs_all["visual_tokens"])
        else:
            visual_embeds = None

        action_embeds = self._noisy_action_to_embeds(
            inputs_all["noisy_actions"], inputs_all["high_level_command"], inputs_all["diffusion_step"],
        )

        embeds_all = {"visual_embeds": visual_embeds, "action_embeds": action_embeds}
        for layer_idx in range(self.num_hidden_layers):
            embeds_all = self._forward_single_layer(attention_mask, embeds_all, layer_idx)

        action_embeds = self.action_expert.transformer.ln_f(embeds_all["action_embeds"])

        denoised_actions = self.action_expert.action_decoder(action_embeds)

        denoised_actions = rearrange(denoised_actions, "b (t h) d -> b t h d", t=inputs_all["noisy_actions"].size(1))
        action_embeds = rearrange(action_embeds, "b (t h) d -> b t h d", t=inputs_all["noisy_actions"].size(1))
        outputs = {"actions": denoised_actions, "actions_embeds": action_embeds}

        if return_gpt:
            visual_embeds = self.gpt.transformer.ln_f(embeds_all["visual_embeds"])
            visual_logits = self.gpt.lm_head(visual_embeds)
            outputs["visual_embeds"] = visual_embeds
            outputs["visual_logits"] = visual_logits

        return outputs
